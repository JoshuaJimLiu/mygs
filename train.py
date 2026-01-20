# train.py
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
# Color helpers
# ==============================================================================
def rgb_to_luma(rgb_chw: torch.Tensor) -> torch.Tensor:
    """
    rgb_chw: (3,H,W) in [0,1]
    return:  (1,H,W) luma (PIL-like weights)
    """
    r = rgb_chw[0:1]
    g = rgb_chw[1:2]
    b = rgb_chw[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


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
# Stage-2 stabilization helpers
# ==============================================================================
@torch.no_grad()
def project_sh_to_gray(gaussians: GaussianModel):
    """
    Make SH colors grayscale (R=G=B) by averaging RGB channels per coeff.
    Helps reduce chroma shock when switching to RGB loss.
    """
    if hasattr(gaussians, "_features_dc") and gaussians._features_dc.numel() > 0:
        dc = gaussians._features_dc.data
        m = dc.mean(dim=-1, keepdim=True)
        dc.copy_(m.expand_as(dc))

    if hasattr(gaussians, "_features_rest") and gaussians._features_rest.numel() > 0:
        rst = gaussians._features_rest.data
        m = rst.mean(dim=-1, keepdim=True)
        rst.copy_(m.expand_as(rst))


def setup_stage2_optimizer(
    gaussians: GaussianModel,
    opt,
    feature_lr_scale: float = 0.1,
    do_project_gray: bool = True,
    train_opacity: bool = False,
    opacity_lr_scale: float = 0.1,
):
    """
    Stage-2: freeze geometry (xyz/scale/rot), train SH (and optionally opacity).
    Recreate optimizer from scratch (clears Adam moments).
    """
    if do_project_gray:
        project_sh_to_gray(gaussians)

    # Freeze geometry
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    # Opacity: optional
    gaussians._opacity.requires_grad_(bool(train_opacity))

    # Train SH parameters
    gaussians._features_dc.requires_grad_(True)
    gaussians._features_rest.requires_grad_(True)

    lr_dc = float(opt.feature_lr) * float(feature_lr_scale)
    lr_rest = float(opt.feature_lr / 20.0) * float(feature_lr_scale)

    param_groups = [
        {"params": [gaussians._features_dc], "lr": lr_dc, "base_lr": lr_dc, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": lr_rest, "base_lr": lr_rest, "name": "f_rest"},
    ]

    if train_opacity:
        lr_op = float(opt.opacity_lr) * float(opacity_lr_scale)
        param_groups.append({"params": [gaussians._opacity], "lr": lr_op, "base_lr": lr_op, "name": "opacity"})

    gaussians.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    print(
        f"[STAGE2] Recreated optimizer: "
        f"lr_dc={lr_dc:g}, lr_rest={lr_rest:g}, "
        f"train_opacity={train_opacity}, opacity_lr={float(opt.opacity_lr) * float(opacity_lr_scale):g}, "
        f"project_gray={do_project_gray}"
    )


def stage2_apply_warmup_lr(gaussians: GaussianModel, warmup_iters: int, t: int):
    """
    Linear warmup for stage2 learning rate (per param group).
    IMPORTANT: uses stored 'base_lr' to avoid the common bug of shrinking LR repeatedly.
    """
    if warmup_iters <= 0:
        return
    w = min(1.0, max(0.0, (t + 1) / float(warmup_iters)))
    for g in gaussians.optimizer.param_groups:
        base_lr = float(g.get("base_lr", g["lr"]))
        g["lr"] = base_lr * w


# ==============================================================================
# Losses
# ==============================================================================
def compute_rgb_loss(image_c, gt_c, opt):
    Ll1 = l1_loss(image_c, gt_c)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_c, gt_c)), Ll1


def compute_gray_loss_equiv(image_c, gt_c, opt):
    """
    Gray loss (luma) computed from RGB images.
    Returns total_loss, Ll1(gray)
    """
    pred_y = rgb_to_luma(image_c)  # (1,H,W)
    gt_y = rgb_to_luma(gt_c)       # (1,H,W)
    pred_y3 = pred_y.repeat(3, 1, 1)
    gt_y3 = gt_y.repeat(3, 1, 1)
    Ll1 = l1_loss(pred_y3, gt_y3)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_y3, gt_y3))
    return loss, Ll1


# ==============================================================================
# Training
# ==============================================================================
def training(
    dataset, opt, pipe,
    testing_iterations, saving_iterations, checkpoint_iterations,
    checkpoint, debug_from,
    color_loss: str,
    two_stage: bool,
    rgb_finetune_iters: int,
    stage2_feature_lr_scale: float,
    stage2_warmup_iters: int,
    stage2_project_gray: bool,
    stage2_only: bool,
    stage2_train_opacity: bool,
    stage2_opacity_lr_scale: float,
):
    """
    Modes:
      1) single-stage: --color_loss {rgb,gray}
      2) --two_stage: gray-loss first, then RGB-loss for last --rgb_finetune_iters
      3) --stage2_only: start directly in stage2 (use with --start_checkpoint)
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
            print("[WARN] two_stage enabled but rgb_finetune_iters invalid; falling back to single-stage rgb.")
            two_stage = False
            stage2_start_iter = None
        else:
            stage2_start_iter = opt.iterations - rgb_finetune_iters + 1  # 1-based
    else:
        stage2_start_iter = None

    stage2_initialized = False
    if stage2_only:
        stage2_start_iter = first_iter

    print(
        f"[INFO] mode: "
        f"{'stage2_only' if stage2_only else ('two_stage' if two_stage else 'single_stage')}, "
        f"color_loss(single)={color_loss}, rgb_finetune_iters={rgb_finetune_iters}, "
        f"stage2_lr_scale={stage2_feature_lr_scale}, stage2_warmup_iters={stage2_warmup_iters}, "
        f"stage2_project_gray={stage2_project_gray}, "
        f"stage2_train_opacity={stage2_train_opacity}, stage2_opacity_lr_scale={stage2_opacity_lr_scale}"
    )
    for iteration in range(first_iter, opt.iterations + 1):
        # Stage selection
        if stage2_start_iter is not None and iteration >= stage2_start_iter:
            cur_stage = "rgb"
        else:
            cur_stage = "gray" if two_stage else color_loss

        # Stage2 init (once)
        if (stage2_start_iter is not None) and (iteration == stage2_start_iter) and (not stage2_initialized):
            setup_stage2_optimizer(
                gaussians, opt,
                feature_lr_scale=stage2_feature_lr_scale,
                do_project_gray=stage2_project_gray,
                train_opacity=stage2_train_opacity,
                opacity_lr_scale=stage2_opacity_lr_scale,
            )
            stage2_initialized = True

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

        # LR schedule for xyz group (only meaningful when xyz is in optimizer)
        gaussians.update_learning_rate(iteration)

        # SH degree schedule
        if iteration % 1000 == 0 and 0: # use DC
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

        # Loss
        if cur_stage == "rgb" or 1:
            loss, Ll1 = compute_rgb_loss(image_c, gt_c, opt)

            if (stage2_start_iter is not None) and (iteration == stage2_start_iter):
                gray_loss_equiv, gray_l1 = compute_gray_loss_equiv(image_c, gt_c, opt)
                print(
                    f"[STAGE2-DEBUG @iter {iteration}] "
                    f"gray_equiv_loss={float(gray_loss_equiv):.6f} (L1y={float(gray_l1):.6f}), "
                    f"rgb_loss={float(loss):.6f} (L1rgb={float(Ll1):.6f}), "
                    f"pred[min,max]=({float(image_c.min()):.4f},{float(image_c.max()):.4f}), "
                    f"gt[min,max]=({float(gt_c.min()):.4f},{float(gt_c.max()):.4f})"
                )
        elif cur_stage == "gray":
            loss, Ll1 = compute_gray_loss_equiv(image_c, gt_c, opt)
        else:
            raise ValueError(f"Unknown stage/loss mode: {cur_stage}")

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Stage2 warmup
            in_stage2 = stage2_initialized and (stage2_start_iter is not None) and (iteration >= stage2_start_iter)
            if in_stage2:
                t2 = iteration - stage2_start_iter  # 0-based
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
                psnr_test_records
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification: disable in stage2 (features/opacity-only finetune)
            if (not in_stage2) and (iteration < opt.densify_until_iter):
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
                    psnr_test_records):
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
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if gt_image.shape[0] == 1:
                        gt_image = gt_image.repeat(3, 1, 1)

                    if tb_writer and (idx < 5):
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
                        help="Experiment name prefix for output folder (e.g., grayAsRGB_SH_room1).")

    # Two-stage controls
    parser.add_argument("--two_stage", action="store_true",
                        help="Run gray-loss first, then RGB-loss for the last --rgb_finetune_iters.")
    parser.add_argument("--rgb_finetune_iters", type=int, default=1000,
                        help="Number of final iterations to finetune with RGB loss in two-stage mode.")
    parser.add_argument("--stage2_only", action="store_true",
                        help="Do only 'RGB finetune' mode from the beginning (use with --start_checkpoint). "
                             "This recreates optimizer and trains SH (and optional opacity) only.")
    parser.add_argument("--stage2_feature_lr_scale", type=float, default=0.1,
                        help="Stage2 feature LR = original feature_lr * this scale (default 0.1).")
    parser.add_argument("--stage2_warmup_iters", type=int, default=100,
                        help="Warmup iterations for stage2 LR (0 disables).")
    parser.add_argument("--stage2_project_gray", action="store_true",
                        help="Before stage2, project SH colors to grayscale (R=G=B) to reduce loss spike.")

    # NEW: stage2 train opacity
    parser.add_argument("--stage2_train_opacity", action="store_true",
                        help="In stage2, also train opacity (often helps prevent RGB finetune from flying).")
    parser.add_argument("--stage2_opacity_lr_scale", type=float, default=0.2,
                        help="Stage2 opacity LR = original opacity_lr * this scale (default 0.2).")

    # Single-stage fallback
    parser.add_argument("--color_loss", type=str, choices=["rgb", "gray"], default="rgb",
                        help="Single-stage training loss: RGB or gray(luma). Ignored in --two_stage/--stage2_only.")

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
    test_iterations = [x for x in range(0, args.iterations + 1, 500)]

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
        args.color_loss,
        args.two_stage,
        args.rgb_finetune_iters,
        args.stage2_feature_lr_scale,
        args.stage2_warmup_iters,
        args.stage2_project_gray,
        args.stage2_only,
        args.stage2_train_opacity,
        args.stage2_opacity_lr_scale,
    )

    print("\nTraining complete.")
