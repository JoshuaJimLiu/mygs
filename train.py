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
import torch.nn.functional as F
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


# ---------------------------
# Helper: RGB -> Luma (Y)  (PIL-like weights)
# ---------------------------
def rgb_to_luma(rgb_chw: torch.Tensor) -> torch.Tensor:
    """
    rgb_chw: (3,H,W) in [0,1]
    return:  (1,H,W) luma
    """
    r = rgb_chw[0:1]
    g = rgb_chw[1:2]
    b = rgb_chw[2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------------------------
# Helper: Sobel gradients
# ---------------------------
def sobel_grad_1ch(y_1hw: torch.Tensor) -> torch.Tensor:
    """
    y_1hw: (1,H,W)
    return: (2,H,W) => [gx, gy]
    """
    assert y_1hw.dim() == 3 and y_1hw.shape[0] == 1
    y = y_1hw.unsqueeze(0)  # (N=1,C=1,H,W)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)

    gx = F.conv2d(y, kx, padding=1)  # (1,1,H,W)
    gy = F.conv2d(y, ky, padding=1)  # (1,1,H,W)
    g = torch.cat([gx, gy], dim=1)   # (1,2,H,W)
    return g.squeeze(0)              # (2,H,W)


def sobel_grad_3ch(rgb_3hw: torch.Tensor) -> torch.Tensor:
    """
    rgb_3hw: (3,H,W)
    return: (6,H,W) => [gx_r,gy_r,gx_g,gy_g,gx_b,gy_b]
    """
    assert rgb_3hw.dim() == 3 and rgb_3hw.shape[0] == 3
    out = []
    for c in range(3):
        gc = sobel_grad_1ch(rgb_3hw[c:c+1])
        out.append(gc)
    return torch.cat(out, dim=0)  # (6,H,W)


def robust_norm01(x_hw: torch.Tensor, q: float = 0.995, eps: float = 1e-8) -> torch.Tensor:
    """
    x_hw: (H,W) or (1,H,W)
    return: same shape, normalized to [0,1] by quantile
    """
    x = x_hw.float()
    if x.dim() == 3:
        x_flat = x.reshape(-1)
    else:
        x_flat = x.reshape(-1)
    scale = torch.quantile(x_flat, q).clamp(min=eps)
    return (x / scale).clamp(0.0, 1.0)


def chroma_edge_weight_map(gt_rgb_3hw: torch.Tensor, edge_quantile: float = 0.995) -> torch.Tensor:
    """
    估计“灰度会丢信息”的区域：chroma 的梯度大（颜色边缘明显）
    gt_rgb_3hw: (3,H,W) in [0,1]
    return: w_hw: (H,W) in [0,1]
    """
    # chroma = rgb - y
    y = rgb_to_luma(gt_rgb_3hw)            # (1,H,W)
    chroma = gt_rgb_3hw - y.repeat(3, 1, 1)  # (3,H,W)

    # sobel grad per channel (6,H,W)
    g = sobel_grad_3ch(chroma)  # (6,H,W)

    # magnitude across channels and directions
    # gx/gy per channel -> sqrt(sum(g^2))
    g2 = (g * g).sum(dim=0)  # (H,W)
    mag = torch.sqrt(g2 + 1e-8)  # (H,W)

    # robust normalize to [0,1]
    w = robust_norm01(mag, q=edge_quantile).detach()  # detach: weight is from GT only
    return w  # (H,W)


def weighted_l1(pred: torch.Tensor, gt: torch.Tensor, w_hw: torch.Tensor) -> torch.Tensor:
    """
    pred/gt: (C,H,W)
    w_hw: (H,W) in [0,1]
    returns weighted mean absolute error
    """
    assert pred.shape == gt.shape
    assert w_hw.dim() == 2
    w = w_hw.unsqueeze(0)  # (1,H,W)
    err = (pred - gt).abs()
    # normalize by mean weight to keep scale stable
    denom = w.mean().clamp(min=1e-6)
    return (err * w).mean() / denom


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
            w.writerow([it, float(v)])

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


def training(dataset, opt, pipe,
             testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from,
             color_loss: str,
             grad_weight: float,
             chroma_edge_weight: float,
             edge_quantile: float):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    psnr_test_records = []

    print(f"[INFO] color_loss={color_loss}, grad_weight={grad_weight}, chroma_edge_weight={chroma_edge_weight}, edge_quantile={edge_quantile}")

    for iteration in range(first_iter, opt.iterations + 1):
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

        gaussians.update_learning_rate(iteration)

        # Only grow SH when training in RGB (your original intent)
        
        # some versions do not use much shs
        if iteration % 1000 == 0:
            if color_loss == "rgb":
                gaussians.oneupSHdegree()
            elif gaussians.active_sh_degree <= 3:
                gaussians.oneupSHdegree()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        gt_image = viewpoint_cam.original_image.cuda()

        # clamp for stable loss/grad
        image_c = torch.clamp(image, 0.0, 1.0)
        gt_c = torch.clamp(gt_image, 0.0, 1.0)

        if color_loss == "rgb":
            Ll1 = l1_loss(image_c, gt_c)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_c, gt_c))

        elif color_loss == "gray":
            pred_y = rgb_to_luma(image_c)   # (1,H,W)
            gt_y = rgb_to_luma(gt_c)        # (1,H,W)

            pred_y3 = pred_y.repeat(3, 1, 1)
            gt_y3 = gt_y.repeat(3, 1, 1)

            # --- chroma edge weight map (from GT only) ---
            w_edge = None
            if chroma_edge_weight > 0.0:
                w_edge = chroma_edge_weight_map(gt_c, edge_quantile=edge_quantile)  # (H,W)
            # L1 on Y (optionally weighted)
            if w_edge is None:
                Ll1 = l1_loss(pred_y3, gt_y3)
            else:
                # weight Y error on pixels where chroma edges are strong
                Ll1 = weighted_l1(pred_y3, gt_y3, (1.0 + chroma_edge_weight * w_edge))

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_y3, gt_y3))

            # --- gradient loss on Y (Sobel) ---
            if grad_weight > 0.0:
                gp = sobel_grad_1ch(pred_y)  # (2,H,W)
                gg = sobel_grad_1ch(gt_y)    # (2,H,W)
                diff = (gp - gg).abs()       # (2,H,W)

                if w_edge is None:
                    loss_g = diff.mean()
                else:
                    w = (1.0 + chroma_edge_weight * w_edge).unsqueeze(0)  # (1,H,W)
                    denom = w.mean().clamp(min=1e-6)
                    loss_g = (diff * w).mean() / denom

                loss = loss + grad_weight * loss_g

        else:
            raise ValueError(f"Unknown color_loss: {color_loss}")

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, color loss: {color_loss}"})
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
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (unchanged) — loss weighting/grad will naturally influence the grads used inside stats
            if iteration < opt.densify_until_iter:
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
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )

    save_psnr_curve(psnr_test_records, scene.model_path, tb_writer)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs,
                    psnr_test_records):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test_val))

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

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument(
        "--color_loss",
        type=str,
        choices=["rgb", "gray"],
        default="gray",
        help="Training loss computed in RGB or grayscale luma-Y."
    )

    # NEW: gradient loss weight on Y (Sobel)
    parser.add_argument(
        "--grad_weight",
        type=float,
        default=0.0,
        help="Add Sobel gradient loss on Y with this weight (useful in gray mode)."
    )

    # NEW: emphasize regions where grayscale loses info (strong chroma edges)
    parser.add_argument(
        "--chroma_edge_weight",
        type=float,
        default=0.0,
        help="Reweight Y/grad losses by (1 + chroma_edge_weight * chroma_edge_map)."
    )

    # NEW: robust normalization quantile for edge map
    parser.add_argument(
        "--edge_quantile",
        type=float,
        default=0.995,
        help="Quantile used to normalize chroma edge magnitude to [0,1]."
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    test_iterations = [x for x in range(0, args.iterations + 1, 1000)]
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.color_loss,
        args.grad_weight,
        args.chroma_edge_weight,
        args.edge_quantile
    )

    print("\nTraining complete.")
