/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// Infer channels (1 or 3) from a 1D tensor of shape (C,)
static inline int inferChannelsFromBackground(const torch::Tensor& background) {
    if (background.numel() == 0) {
        AT_ERROR("background must be non-empty (shape: [C])");
    }
    if (background.ndimension() != 1) {
        AT_ERROR("background must be a 1D tensor of shape (C,)");
    }
    const int C = (int)background.size(0);
    if (C != 1 && C != 3) {
        AT_ERROR("Only C=1 or C=3 are supported for background");
    }
    return C;
}

// Validate sh layout and get M.
// Supported sh layouts:
//   - empty tensor -> sh_ptr=nullptr, M=0
//   - (P, M, C) where C matches channels
//   - (P, M) only allowed when channels==1 (grayscale SH)
static inline int inferMAndCheckSH(const torch::Tensor& sh, int P, int channels) {
    if (sh.numel() == 0) return 0;

    if (sh.ndimension() == 3) {
        if ((int)sh.size(0) != P) AT_ERROR("sh must have shape (P, M, C)");
        if ((int)sh.size(2) != channels) AT_ERROR("sh last dim (C) must match channels");
        return (int)sh.size(1);
    }
    if (sh.ndimension() == 2) {
        if (channels != 1) AT_ERROR("sh with shape (P, M) is only valid when channels==1");
        if ((int)sh.size(0) != P) AT_ERROR("sh must have shape (P, M) for grayscale");
        return (int)sh.size(1);
    }

    AT_ERROR("sh must be empty, or have shape (P, M, C) or (P, M)");
    return 0;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug)
{
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }

    const int P = (int)means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    const int channels = inferChannelsFromBackground(background);

    // If colors is provided (non-empty), it must be (P, channels)
    if (colors.numel() != 0) {
        if (colors.ndimension() != 2 || (int)colors.size(0) != P || (int)colors.size(1) != channels) {
            AT_ERROR("colors must have shape (P, C) where C matches background channels, or be empty");
        }
    }

    const int M = inferMAndCheckSH(sh, P, channels);

    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({channels, H, W}, 0.0, float_opts);
    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if (P != 0)
    {
        // Optional pointers: pass nullptr when tensor is empty.
        const float* sh_ptr = (sh.numel() != 0) ? sh.contiguous().data_ptr<float>() : nullptr;
        const float* colors_ptr = (colors.numel() != 0) ? colors.contiguous().data_ptr<float>() : nullptr;
        const float* cov3D_ptr = (cov3D_precomp.numel() != 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;

        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P, degree, M, channels,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh_ptr,
            colors_ptr,
            opacity.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_ptr,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data_ptr<float>(),
            radii.contiguous().data_ptr<int>(),
            debug);
    }

    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug) 
{
    const int P = (int)means3D.size(0);

    if (dL_dout_color.ndimension() != 3) {
        AT_ERROR("dL_dout_color must have shape (C, H, W)");
    }

    const int channels = (int)dL_dout_color.size(0);
    if (channels != 1 && channels != 3) {
        AT_ERROR("Only C=1 or C=3 are supported for dL_dout_color");
    }

    // Background consistency check (optional but helpful)
    if (background.numel() != 0) {
        const int bgC = inferChannelsFromBackground(background);
        if (bgC != channels) {
            AT_ERROR("background channels must match dL_dout_color channels");
        }
    }

    const int H = (int)dL_dout_color.size(1);
    const int W = (int)dL_dout_color.size(2);

    const int M = inferMAndCheckSH(sh, P, channels);

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());

    // Keep original behavior: always return (P, channels) grads for Gaussian colors
    torch::Tensor dL_dcolors = torch::zeros({P, channels}, means3D.options());

    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());

    // Keep original behavior: always have internal cov grad buffer of size (P, 6)
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());

    // Keep original behavior: return (P, M, channels)
    torch::Tensor dL_dsh = torch::zeros({P, M, channels}, means3D.options());

    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

    if (P != 0)
    {
        const float* sh_ptr = (sh.numel() != 0) ? sh.contiguous().data_ptr<float>() : nullptr;
        const float* colors_ptr = (colors.numel() != 0) ? colors.contiguous().data_ptr<float>() : nullptr;
        const float* cov3D_ptr = (cov3D_precomp.numel() != 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;

        CudaRasterizer::Rasterizer::backward(
            P, degree, M, R, channels,
            background.contiguous().data_ptr<float>(),
            W, H,
            means3D.contiguous().data_ptr<float>(),
            sh_ptr,
            colors_ptr,
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_ptr,
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            campos.contiguous().data_ptr<float>(),
            tan_fovx,
            tan_fovy,
            radii.contiguous().data_ptr<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data_ptr<float>(),
            dL_dmeans2D.contiguous().data_ptr<float>(),
            dL_dconic.contiguous().data_ptr<float>(),
            dL_dopacity.contiguous().data_ptr<float>(),
            dL_dcolors.contiguous().data_ptr<float>(),
            dL_dmeans3D.contiguous().data_ptr<float>(),
            dL_dcov3D.contiguous().data_ptr<float>(),
            dL_dsh.contiguous().data_ptr<float>(),
            dL_dscales.contiguous().data_ptr<float>(),
            dL_drotations.contiguous().data_ptr<float>(),
            debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix)
{
    const int P = (int)means3D.size(0);

    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

    if (P != 0)
    {
        CudaRasterizer::Rasterizer::markVisible(P,
            means3D.contiguous().data_ptr<float>(),
            viewmatrix.contiguous().data_ptr<float>(),
            projmatrix.contiguous().data_ptr<float>(),
            present.contiguous().data_ptr<bool>());
    }

    return present;
}
