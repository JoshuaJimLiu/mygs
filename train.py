#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
import csv
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ==============================================================================
# Misc helpers
# ==============================================================================
def gt_to_gray3(x: torch.Tensor, *, weights=(0.299, 0.587, 0.114)) -> torch.Tensor:
    """
    Convert GT image to 3-channel grayscale (gray repeated in RGB).
    Supports:
      - CHW: (C,H,W) where C=1 or 3
      - NCHW: (N,C,H,W) where C=1 or 3
    Input expected in [0,1] (but we don't enforce it).
    Output dtype/device matches input.
    """
    if x.dim() == 3:
        c, h, w = x.shape
        if c == 1:
            return x.repeat(3, 1, 1)
        if c != 3:
            raise ValueError(f"Expected C=1 or 3 for CHW, got {x.shape}")
        r, g, b = x[0:1], x[1:2], x[2:3]
        y = weights[0] * r + weights[1] * g + weights[2] * b  # (1,H,W)
        return y.repeat(3, 1, 1)

    if x.dim() == 4:
        n, c, h, w = x.shape
        if c == 1:
            return x.repeat(1, 3, 1, 1)
        if c != 3:
            raise ValueError(f"Expected C=1 or 3 for NCHW, got {x.shape}")
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = weights[0] * r + weights[1] * g + weights[2] * b  # (N,1,H,W)
        return y.repeat(1, 3, 1, 1)

    raise ValueError(f"Expected CHW or NCHW tensor, got dim={x.dim()}, shape={tuple(x.shape)}")

def _sanitize_expname(name: str) -> str:
    """
    Filesystem-friendly experiment name.
      - strip
      - replace separators/spaces with '_'
      - keep only [A-Za-z0-9._-], others -> '_'
      - collapse multiple '_'
      - strip leading/trailing '_'
    """
    name = (name or "").strip()
    if not name:
        return ""
    name = name.replace("\\", "_").replace("/", "_").replace(" ", "_")
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def resolve_model_path(args):
    """
    Make expname *actually* affect the output folder by resolving args.model_path
    BEFORE ModelParams.extract() (so dataset.model_path also picks it up).
    """
    if getattr(args, "model_path", ""):
        # user already specified --model_path, don't override
        return

    unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
    uid10 = unique_str[:10]

    exp = _sanitize_expname(getattr(args, "expname", ""))
    folder = f"{exp}_{uid10}" if exp else uid10
    args.model_path = os.path.join("./output", folder)


# ==============================================================================
# Logging / PSNR curve
# ==============================================================================
def save_psnr_curve(psnr_records, out_dir: str, tb_writer=None):
    if not psnr_records:
        print("[WARN] No PSNR records collected. Skip plotting.")
        return

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "psnr_test_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "psnr"])
        for it, v in psnr_records:
            w.writerow([int(it), float(v)])

    try:
        import matplotlib.pyplot as plt

        xs = [it for it, _ in psnr_records]
        ys = [v for _, v in psnr_records]

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("Iteration")
        plt.ylabel("PSNR (test)")
        plt.title("Test PSNR vs Iteration")
        plt.grid(True)

        png_path = os.path.join(out_dir, "psnr_test_curve.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved PSNR curve CSV: {csv_path}")
        print(f"[INFO] Saved PSNR curve PNG: {png_path}")

        if tb_writer is not None:
            from PIL import Image
            import numpy as np
            img = Image.open(png_path).convert("RGB")
            arr = np.asarray(img).astype("uint8")
            arr_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tb_writer.add_image("curves/test_psnr_curve", arr_t, global_step=xs[-1])

    except Exception as e:
        print(f"[WARN] Failed to plot PSNR curve: {e}")
        print(f"[INFO] But CSV has been saved: {csv_path}")


def prepare_output_and_logger(dataset):
    print("Output folder:", dataset.model_path)
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w", encoding="utf-8") as f:
        f.write(str(Namespace(**vars(dataset))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


# ==============================================================================
# Stage-2 (finetune) optimizer
# ==============================================================================
def setup_stage2_optimizer(
    gaussians: GaussianModel,
    opt,
    feature_lr_scale: float = 0.1,
    train_opacity: bool = False,
    opacity_lr_scale: float = 0.1,
    force_dc_sh: bool = False,
):
    """
    Stage-2: recreate optimizer from scratch (clears Adam moments) BUT KEEP ALL param groups
    (xyz/scaling/rotation/opacity/f_dc/f_rest...) so densify/prune won't crash.

    Strategy:
      1) call gaussians.training_setup(opt) to build the original optimizer (all groups exist)
      2) freeze geometry grads by default + set geometry lrs to 0 (will be restored after warmup if enabled)
      3) scale SH lrs; opacity lr depends on train_opacity flag
    """
    # (1) rebuild the default optimizer so param groups are complete (fix "missing xyz")
    gaussians.training_setup(opt)

    # make sure every group has base_lr for later restore / warmup
    for g in gaussians.optimizer.param_groups:
        if "base_lr" not in g:
            g["base_lr"] = float(g.get("lr", 0.0))

    # (2) freeze geometry grads by default
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    # (3) SH grads
    gaussians._features_dc.requires_grad_(True)
    if hasattr(gaussians, "_features_rest") and gaussians._features_rest is not None:
        gaussians._features_rest.requires_grad_(not force_dc_sh)

    # opacity grads
    gaussians._opacity.requires_grad_(bool(train_opacity))

    # adjust lrs by group name (names follow the official repo: xyz, f_dc, f_rest, opacity, scaling, rotation)
    for g in gaussians.optimizer.param_groups:
        name = g.get("name", "")

        # geometry groups: keep base_lr but start with lr=0 (warmup phase frozen)
        if name in ("xyz", "scaling", "rotation"):
            # base_lr already stored above (original lr)
            g["lr"] = 0.0
            continue

        # SH groups: scale
        if name == "f_dc":
            g["lr"] = float(g["base_lr"]) * float(feature_lr_scale)
            g["base_lr"] = float(g["lr"])
            continue

        if name == "f_rest":
            if force_dc_sh:
                g["lr"] = 0.0
                g["base_lr"] = 0.0
            else:
                g["lr"] = float(g["base_lr"]) * float(feature_lr_scale)
                g["base_lr"] = float(g["lr"])
            continue

        # opacity group: optional
        if name == "opacity":
            if train_opacity:
                g["lr"] = float(g["base_lr"]) * float(opacity_lr_scale)
                g["base_lr"] = float(g["lr"])
            else:
                g["lr"] = 0.0
                g["base_lr"] = 0.0
            continue

    print(
        f"[STAGE2] Optimizer rebuilt (all groups kept). "
        f"feature_lr_scale={feature_lr_scale}, train_opacity={train_opacity}, "
        f"opacity_lr_scale={opacity_lr_scale}, force_dc_sh={force_dc_sh}. "
        f"Geometry lr set to 0 during warmup (will restore if stage2_densify_after_warmup)."
    )


def stage2_apply_warmup_lr(gaussians: GaussianModel, warmup_iters: int, t: int,
                          warmup_group_names=("f_dc", "f_rest", "opacity")):
    """
    Linear warmup for stage2 learning rate (only for SH/opacity groups).
    IMPORTANT: uses stored 'base_lr' to avoid shrinking LR repeatedly.
    """
    if warmup_iters <= 0:
        return
    w = min(1.0, max(0.0, (t + 1) / float(warmup_iters)))
    for g in gaussians.optimizer.param_groups:
        if g.get("name", "") not in warmup_group_names:
            continue
        base_lr = float(g.get("base_lr", g.get("lr", 0.0)))
        g["lr"] = base_lr * w


# ==============================================================================
# Losses (RGB only)
# ==============================================================================
def compute_rgb_loss(image_c, gt_c, opt):
    Ll1 = l1_loss(image_c, gt_c)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_c, gt_c)), Ll1


# ==============================================================================
# Image saving helper (test preds)
# ==============================================================================
def _save_chw_png(path: str, chw: torch.Tensor):
    """
    Save CHW float tensor in [0,1] as PNG. Supports C=1/3.
    """
    from PIL import Image

    x = chw.detach()
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.clamp(0.0, 1.0)
    arr = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # HWC uint8
    Image.fromarray(arr).save(path)


# ==============================================================================
# Training
# ==============================================================================
def training(
    dataset, opt, pipe,
    testing_iterations, saving_iterations, checkpoint_iterations,
    checkpoint, debug_from,
    two_stage: bool,
    rgb_finetune_iters: int,
    stage2_feature_lr_scale: float,
    stage2_warmup_iters: int,
    stage2_only: bool,
    stage2_train_opacity: bool,
    stage2_opacity_lr_scale: float,
    stage2_densify_after_warmup: bool,
    save_test_preds: bool,
    test_pred_dirname: str,
    force_dc_sh: bool,
):
    """
    Modes:
      1) single-stage (default): RGB loss for all iters
      2) --two_stage: stage1 normal training + stage2 finetune (optimizer reset + freeze geometry),
                      BOTH stages still use RGB loss
      3) --stage2_only: start directly in stage2 (use with --start_checkpoint)

    Stage1 GT is forced to 3-ch grayscale (gt_to_gray3).
    Stage2 uses original RGB GT.

    NEW:
      --stage2_densify_after_warmup: enable densify/prune ONLY after warmup;
      and restore geometry lrs/requires_grad so densify works and actually helps.
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"[INFO] Restored checkpoint: {checkpoint}, first_iter={first_iter}")

    # Force DC SH: never increase SH degree; keep active_sh_degree at 0 if exists
    if force_dc_sh:
        if hasattr(gaussians, "active_sh_degree"):
            try:
                gaussians.active_sh_degree = 0
            except Exception:
                pass
        if hasattr(gaussians, "_features_rest") and gaussians._features_rest is not None:
            gaussians._features_rest.requires_grad_(False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    psnr_test_records = []

    rgb_finetune_iters = max(0, int(rgb_finetune_iters))
    if two_stage:
        if rgb_finetune_iters <= 0 or rgb_finetune_iters >= opt.iterations:
            print("[WARN] two_stage enabled but rgb_finetune_iters invalid; falling back to single-stage.")
            two_stage = False
            stage2_start_iter = None
        else:
            stage2_start_iter = opt.iterations - rgb_finetune_iters + 1  # 1-based
    else:
        stage2_start_iter = None

    stage2_initialized = False
    stage2_geometry_enabled = False  # NEW: geometry lr/grad restored after warmup if densify enabled
    if stage2_only:
        stage2_start_iter = first_iter

    # where to save test preds
    test_pred_dir = os.path.join(scene.model_path, test_pred_dirname)
    if save_test_preds:
        os.makedirs(test_pred_dir, exist_ok=True)
        print(f"[INFO] Will save test preds to: {test_pred_dir}")

    print(
        f"[INFO] mode: "
        f"{'stage2_only' if stage2_only else ('two_stage' if two_stage else 'single_stage')}, "
        f"rgb_finetune_iters={rgb_finetune_iters}, "
        f"stage2_feature_lr_scale={stage2_feature_lr_scale}, stage2_warmup_iters={stage2_warmup_iters}, "
        f"stage2_train_opacity={stage2_train_opacity}, stage2_opacity_lr_scale={stage2_opacity_lr_scale}, "
        f"stage2_densify_after_warmup={stage2_densify_after_warmup}, "
        f"save_test_preds={save_test_preds}, "
        f"force_dc_sh={force_dc_sh}"
    )

    for iteration in range(first_iter, opt.iterations + 1):
        # Stage selection
        in_stage2 = (stage2_start_iter is not None) and (iteration >= stage2_start_iter)
        cur_stage = "stage2" if in_stage2 else "stage1"

        # Stage2 warmup status
        if in_stage2 and stage2_start_iter is not None:
            t2 = iteration - stage2_start_iter  # 0-based
            stage2_warmup_done = (stage2_warmup_iters <= 0) or (t2 >= stage2_warmup_iters)
        else:
            t2 = 0
            stage2_warmup_done = False

        # Stage2 init (once)
        if in_stage2 and (iteration == stage2_start_iter) and (not stage2_initialized):
            setup_stage2_optimizer(
                gaussians, opt,
                feature_lr_scale=stage2_feature_lr_scale,
                train_opacity=stage2_train_opacity,
                opacity_lr_scale=stage2_opacity_lr_scale,
                force_dc_sh=force_dc_sh,
            )
            stage2_initialized = True

        # If stage2 densify is requested, only enable AFTER warmup.
        # Also restore geometry lr/grad so densification has "xyz" group and meaningful grads.
        stage2_densify_ok = (
            in_stage2 and stage2_initialized and stage2_densify_after_warmup and stage2_warmup_done
        )
        if stage2_densify_ok and (not stage2_geometry_enabled):
            # restore geometry lrs from base_lr
            for g in gaussians.optimizer.param_groups:
                if g.get("name", "") in ("xyz", "scaling", "rotation"):
                    g["lr"] = float(g.get("base_lr", g.get("lr", 0.0)))
            # enable geometry grads so viewspace_points has grad (densify stats relies on it)
            gaussians._xyz.requires_grad_(True)
            gaussians._scaling.requires_grad_(True)
            gaussians._rotation.requires_grad_(True)

            stage2_geometry_enabled = True
            print(f"[STAGE2] Warmup done @iter {iteration}. Geometry lr/grad restored; densify/prune enabled.")

        # If stage2 and densify is NOT enabled (or still warming up), keep geometry frozen
        if in_stage2 and stage2_initialized and (not stage2_geometry_enabled):
            gaussians._xyz.requires_grad_(False)
            gaussians._scaling.requires_grad_(False)
            gaussians._rotation.requires_grad_(False)
            for g in gaussians.optimizer.param_groups:
                if g.get("name", "") in ("xyz", "scaling", "rotation"):
                    g["lr"] = 0.0

        # GUI
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        # LR schedule:
        # - stage1: original behavior
        # - stage2: only when geometry is enabled (after warmup + flag)
        if (not in_stage2) or stage2_geometry_enabled:
            gaussians.update_learning_rate(iteration)

        # SH degree schedule
        if (not force_dc_sh) and (iteration % 1000 == 0):
            gaussians.oneupSHdegree()

        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        # GT
        gt_image = viewpoint_cam.original_image.cuda()
        if gt_image.shape[0] == 1:
            gt_image = gt_image.repeat(3, 1, 1)

        image_c = torch.clamp(image, 0.0, 1.0)
        gt_c = torch.clamp(gt_image, 0.0, 1.0)

        # Stage1 uses gray3 GT; Stage2 uses RGB GT
        if not in_stage2:
            gt_c = gt_to_gray3(gt_c)

        # RGB loss ALWAYS
        loss, Ll1 = compute_rgb_loss(image_c, gt_c, opt)

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Stage2 warmup (only affects SH/opacity groups)
            if stage2_initialized and in_stage2 and (not stage2_warmup_done):
                stage2_apply_warmup_lr(gaussians, warmup_iters=stage2_warmup_iters, t=t2)

            ema_loss_for_log = 0.4 * float(loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}", "stage": cur_stage})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene, render,
                (pipe, background),
                psnr_test_records,
                save_test_preds=save_test_preds,
                test_pred_dir=test_pred_dir,
                in_stage2=in_stage2
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification / prune:
            # - stage1: original constraint (iteration < opt.densify_until_iter)
            # - stage2: only when stage2_geometry_enabled (warmup done + flag)
            do_densify = ((not in_stage2) and (iteration < opt.densify_until_iter)) or stage2_geometry_enabled
            if do_densify:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )

    save_psnr_curve(psnr_test_records, scene.model_path, tb_writer)


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    psnr_test_records,
                    save_test_preds: bool,
                    test_pred_dir: str,
                    in_stage2: bool = False
                    ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', float(Ll1.item()), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', float(loss.item()), iteration)
        tb_writer.add_scalar('iter_time', float(elapsed), iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test_val = 0.0

                for cam_idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if not in_stage2:
                        gt_image = gt_to_gray3(gt_image)

                    if gt_image.shape[0] == 1:
                        gt_image = gt_image.repeat(3, 1, 1)

                    # Save test preds: test_{iter}_{camIdx}.png
                    if save_test_preds and config["name"] == "test":
                        try:
                            out_name = f"test_{int(iteration)}_{int(cam_idx)}.png"
                            out_path = os.path.join(test_pred_dir, out_name)
                            _save_chw_png(out_path, image)
                        except Exception as e:
                            print(f"[WARN] Failed to save test pred (iter={iteration}, cam={cam_idx}): {e}")

                    if tb_writer and (cam_idx < 5):
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration
                            )

                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test_val += psnr(image, gt_image).mean().double()

                psnr_test_val /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test_val}")

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test_val, iteration)

                if config['name'] == 'test':
                    psnr_test_records.append((int(iteration), float(psnr_test_val)))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--expname", type=str, default="",
                        help="Experiment name prefix for output folder (e.g., stage2_rgb_room1).")

    # Two-stage controls
    parser.add_argument("--two_stage", action="store_true",
                        help="Stage1 normal training + Stage2 finetune (optimizer reset + freeze geometry). "
                             "Both stages still use RGB loss (Stage1 GT is gray3, Stage2 GT is RGB).")
    parser.add_argument("--rgb_finetune_iters", type=int, default=1000,
                        help="Number of final iterations to run Stage2 finetune in two-stage mode.")
    parser.add_argument("--stage2_only", action="store_true",
                        help="Do only Stage2 finetune from the beginning (use with --start_checkpoint).")

    # Keep your intent: stage2 SH lr should usually be smaller (default 0.1)
    parser.add_argument("--stage2_feature_lr_scale", type=float, default=0.1,
                        help="Stage2 SH feature lr scale (multiply original). Default 0.1.")
    parser.add_argument("--stage2_warmup_iters", type=int, default=100,
                        help="Warmup iterations for stage2 SH lr (0 disables).")

    # NEW: enable densify/prune after warmup in stage2
    parser.add_argument("--stage2_densify_after_warmup", action="store_true",
                        help="If set, enable densify+prune in stage2 ONLY after warmup; "
                             "also restores geometry lr/grad so densification won't crash and can help.")

    # Stage2 train opacity
    parser.add_argument("--stage2_train_opacity", action="store_true",
                        help="In stage2, also train opacity.")
    parser.add_argument("--stage2_opacity_lr_scale", type=float, default=0.2,
                        help="Stage2 opacity lr scale (multiply original). Default 0.2.")

    # Force DC SH
    parser.add_argument("--force_DC_SH", action="store_true",
                        help="Force using DC-only SH (never increase SH degree; f_rest lr=0 in stage2).")

    # Save test preds
    parser.add_argument("--save_test_preds", action="store_true",
                        help="If set, save test predictions at each test iteration as test_{iter}_{camIdx}.png")
    parser.add_argument("--test_pred_dirname", type=str, default="test_preds",
                        help="Subfolder name under output folder for saved test predictions.")

    # Other original args
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    resolve_model_path(args)

    args.save_iterations.append(args.iterations)
    test_iterations = (
        [x for x in range(0, args.iterations + 1, 2000)]
        + [x for x in range(max(0, args.iterations - args.rgb_finetune_iters), args.iterations + 1, 100)]
    )

    print("Optimizing " + str(args.model_path))
    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    training(
        dataset,
        opt,
        pipe,
        test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.two_stage,
        args.rgb_finetune_iters,
        args.stage2_feature_lr_scale,
        args.stage2_warmup_iters,
        args.stage2_only,
        args.stage2_train_opacity,
        args.stage2_opacity_lr_scale,
        args.stage2_densify_after_warmup,
        save_test_preds=args.save_test_preds,
        test_pred_dirname=args.test_pred_dirname,
        force_dc_sh=args.force_DC_SH,
    )

    print("\nTraining complete.")
