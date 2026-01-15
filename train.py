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


# ---------------------------
# Helper: RGB -> Luma (Y) Image.convert() Rev. 601
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


def save_psnr_curve(psnr_records, out_dir: str, tb_writer=None):
    """
    psnr_records: list of (iteration:int, psnr:float)
    write:
      - psnr_test_curve.csv
      - psnr_test_curve.png
    """
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

    # Plot
    try:
        import matplotlib.pyplot as plt

        xs = [it for it, _ in psnr_records]
        ys = [v for _, v in psnr_records]

        plt.figure()
        plt.plot(xs, ys)  # don't set colors (use defaults)
        plt.xlabel("Iteration")
        plt.ylabel("PSNR (test)")
        plt.title("Test PSNR vs Iteration")
        plt.grid(True)

        png_path = os.path.join(out_dir, "psnr_test_curve.png")
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Saved PSNR curve CSV: {csv_path}")
        print(f"[INFO] Saved PSNR curve PNG: {png_path}")

        # Optional: log to TensorBoard as image
        if tb_writer is not None:
            # re-open image and log
            from PIL import Image
            import numpy as np
            img = Image.open(png_path).convert("RGB")
            arr = np.asarray(img).astype("uint8")
            # TB expects CHW float in [0,1]
            arr_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            tb_writer.add_image("curves/test_psnr_curve", arr_t, global_step=xs[-1])

    except Exception as e:
        print(f"[WARN] Failed to plot PSNR curve: {e}")
        print(f"[INFO] But CSV has been saved: {csv_path}")


def training(dataset, opt, pipe,
             testing_iterations, saving_iterations, checkpoint_iterations,
             checkpoint, debug_from,
             color_loss: str):
    first_iter = 0

    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if color_loss == "gray":
        print("Using grayscale (luma-Y) training loss.")
        bg_color = [0] if dataset.white_background else [1] # automatically deduce channels in cuda backend
    else:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

        print("Using RGB training loss.")
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # record test PSNR at each test iteration
    psnr_test_records = []

    print(f'color loss mode: {color_loss}')

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

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and color_loss == "rgb":
            gaussians.oneupSHdegree()

        # Pick a random Camera
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

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        if color_loss == "rgb":
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        elif color_loss == "gray":
            # compute luma, then repeat to 3 channels for SSIM safety
            pred_y = rgb_to_luma(torch.clamp(image, 0.0, 1.0))         # (1,H,W)
            gt_y = rgb_to_luma(torch.clamp(gt_image, 0.0, 1.0))        # (1,H,W)
            pred_y3 = pred_y.repeat(3, 1, 1)                           # (3,H,W)
            gt_y3 = gt_y.repeat(3, 1, 1)                               # (3,H,W)

            Ll1 = l1_loss(pred_y3, gt_y3)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(pred_y3, gt_y3))
        else:
            raise ValueError(f"Unknown color_loss: {color_loss}")

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, color loss: {color_loss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
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

            # Densification
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

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                )

    # After training: save curve
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

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
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
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                # record test PSNR curve
                if config['name'] == 'test':
                    psnr_test_records.append((int(iteration), float(psnr_test)))

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

    # NEW: RGB vs Gray training loss switch
    parser.add_argument(
        "--color_loss",
        type=str,
        choices=["rgb", "gray"],
        default="gray",
        help="Training loss computed in RGB (default) or grayscale luma-Y."
    )

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    test_iterations = [x for x in range(0, args.iterations + 1, 500)]
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
        args.color_loss
    )

    print("\nTraining complete.")
