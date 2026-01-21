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

// ------------------------------------------------------------
// Small helpers
// ------------------------------------------------------------
static inline int infer_channels_from_background(const torch::Tensor& background) {
	// background is expected to be shape [C] (C=1 or 3)
	if (background.numel() == 1) return 1;
	if (background.numel() == 3) return 3;
	AT_ERROR("background must have numel()==1 or 3, got numel=", background.numel());
}

static inline void check_cuda_contig_float(const torch::Tensor& t, const char* name) {
	if (!t.is_cuda()) AT_ERROR(name, " must be a CUDA tensor");
	if (t.scalar_type() != at::kFloat) AT_ERROR(name, " must be float32");
	if (!t.is_contiguous()) {
		// we will call contiguous() before data_ptr() anyway, but shape checks should still be done on original
		// (contiguous() does not change shape)
	}
}

static inline void check_cuda_contig_int32(const torch::Tensor& t, const char* name) {
	if (!t.is_cuda()) AT_ERROR(name, " must be a CUDA tensor");
	if (t.scalar_type() != at::kInt) AT_ERROR(name, " must be int32");
}

static inline void check_channels_1_or_3(int channels) {
	if (!(channels == 1 || channels == 3)) {
		AT_ERROR("channels must be 1 or 3, got ", channels);
	}
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

// ============================================================
// Forward
// ============================================================
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,        // [C] C=1 or 3
	const torch::Tensor& means3D,           // [P,3]
	const torch::Tensor& colors,            // [P,C] or empty (numel==0) if not used
	const torch::Tensor& opacity,           // [P,1] or [P]
	const torch::Tensor& scales,            // [P,3]
	const torch::Tensor& rotations,         // [P,4]
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,     // [P,6] or empty
	const torch::Tensor& viewmatrix,        // [4,4] or [16]
	const torch::Tensor& projmatrix,        // [4,4] or [16]
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,                // either [P,M,C] for C=3 OR [P,M] for C=1 OR empty
	const int degree,
	const torch::Tensor& campos,            // [3]
	const bool prefiltered,
	const bool debug)
{
	check_cuda_contig_float(background, "background");
	check_cuda_contig_float(means3D, "means3D");
	check_cuda_contig_float(opacity, "opacity");
	check_cuda_contig_float(scales, "scales");
	check_cuda_contig_float(rotations, "rotations");
	check_cuda_contig_float(viewmatrix, "viewmatrix");
	check_cuda_contig_float(projmatrix, "projmatrix");
	check_cuda_contig_float(campos, "campos");
	// colors / cov3D_precomp / sh may be empty, check only when numel>0
	if (colors.numel() > 0) check_cuda_contig_float(colors, "colors");
	if (cov3D_precomp.numel() > 0) check_cuda_contig_float(cov3D_precomp, "cov3D_precomp");
	if (sh.numel() > 0) check_cuda_contig_float(sh, "sh");

	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int channels = infer_channels_from_background(background);
	check_channels_1_or_3(channels);

	const int P = (int)means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	// basic sanity
	if (campos.numel() != 3) AT_ERROR("campos must have numel()==3");
	if (opacity.numel() != P && opacity.numel() != P * 1) {
		// allow [P] or [P,1]
		// If opacity is [P,1], numel==P. If [P], numel==P too, so this check mostly just ensures P matches.
	}

	// --- colors_precomp shape: [P,C] if provided ---
	const bool has_colors_precomp = (colors.numel() > 0);
	if (has_colors_precomp) {
		if (colors.ndimension() != 2 || colors.size(0) != P || colors.size(1) != channels) {
			AT_ERROR("colors must have shape (P, C). Expected (", P, ", ", channels, "), got (",
					 colors.sizes(), ")");
		}
	}

	// --- sh shape contract ---
	// If sh is provided:
	//   channels==3: sh should be [P, M, 3] (or [P, M, 3] contiguous)
	//   channels==1: sh should be [P, M]  (scalar coeffs)
	int M = 0;
	const bool has_sh = (sh.numel() > 0);
	if (has_sh) {
		if (channels == 3) {
			if (sh.ndimension() != 3 || sh.size(0) != P || sh.size(2) != 3) {
				AT_ERROR("For channels=3, sh must have shape (P, M, 3). Got sizes=", sh.sizes());
			}
			M = (int)sh.size(1);
		} else { // channels == 1
			if (sh.ndimension() != 2 || sh.size(0) != P) {
				AT_ERROR("For channels=1, sh must have shape (P, M). Got sizes=", sh.sizes());
			}
			M = (int)sh.size(1);
		}
	}

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	// out_color: [C,H,W]
	torch::Tensor out_color = torch::full({channels, H, W}, 0.0, float_opts);
	torch::Tensor radii = torch::full({P}, 0, int_opts);

	// Buffers
	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0) {
		// cov3D_precomp pointer can be nullptr if empty
		const float* cov_ptr = (cov3D_precomp.numel() > 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;

		// colors_precomp pointer can be nullptr if empty
		const float* colors_ptr = has_colors_precomp ? colors.contiguous().data_ptr<float>() : nullptr;

		// sh pointer can be nullptr if empty
		const float* sh_ptr = has_sh ? sh.contiguous().data_ptr<float>() : nullptr;

		// background must be length channels
		// NOTE: background is [C], contiguous float
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degree, M,
			channels, // <- runtime dispatch
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			opacity.contiguous().data_ptr<float>(),
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov_ptr,
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			campos.contiguous().data_ptr<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data_ptr<float>(),
			radii.contiguous().data_ptr<int>(),
			debug
		);
	}

	return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

// ============================================================
// Backward
// ============================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& background,         // [C]
	const torch::Tensor& means3D,            // [P,3]
	const torch::Tensor& radii,              // [P]
	const torch::Tensor& colors,             // [P,C] or empty
	const torch::Tensor& scales,             // [P,3]
	const torch::Tensor& rotations,          // [P,4]
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,      // [P,6] or empty
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,      // [C,H,W]
	const torch::Tensor& sh,                 // see forward
	const int degree,
	const torch::Tensor& campos,             // [3]
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug)
{
	check_cuda_contig_float(background, "background");
	check_cuda_contig_float(means3D, "means3D");
	check_cuda_contig_int32(radii, "radii");
	check_cuda_contig_float(scales, "scales");
	check_cuda_contig_float(rotations, "rotations");
	check_cuda_contig_float(viewmatrix, "viewmatrix");
	check_cuda_contig_float(projmatrix, "projmatrix");
	check_cuda_contig_float(dL_dout_color, "dL_dout_color");
	check_cuda_contig_float(campos, "campos");
	if (colors.numel() > 0) check_cuda_contig_float(colors, "colors");
	if (cov3D_precomp.numel() > 0) check_cuda_contig_float(cov3D_precomp, "cov3D_precomp");
	if (sh.numel() > 0) check_cuda_contig_float(sh, "sh");

	const int channels = infer_channels_from_background(background);
	check_channels_1_or_3(channels);

	const int P = (int)means3D.size(0);

	if (dL_dout_color.ndimension() != 3 || dL_dout_color.size(0) != channels) {
		AT_ERROR("dL_dout_color must have shape (C,H,W) with C==channels. Expected C=",
				 channels, ", got sizes=", dL_dout_color.sizes());
	}
	const int H = (int)dL_dout_color.size(1);
	const int W = (int)dL_dout_color.size(2);

	// colors_precomp shape if provided
	const bool has_colors_precomp = (colors.numel() > 0);
	if (has_colors_precomp) {
		if (colors.ndimension() != 2 || colors.size(0) != P || colors.size(1) != channels) {
			AT_ERROR("colors must have shape (P, C). Expected (", P, ", ", channels, "), got ", colors.sizes());
		}
	}

	// SH layout + M
	int M = 0;
	const bool has_sh = (sh.numel() > 0);
	if (has_sh) {
		if (channels == 3) {
			if (sh.ndimension() != 3 || sh.size(0) != P || sh.size(2) != 3) {
				AT_ERROR("For channels=3, sh must have shape (P, M, 3). Got sizes=", sh.sizes());
			}
			M = (int)sh.size(1);
		} else {
			if (sh.ndimension() != 2 || sh.size(0) != P) {
				AT_ERROR("For channels=1, sh must have shape (P, M). Got sizes=", sh.sizes());
			}
			M = (int)sh.size(1);
		}
	}

	// Allocate grads with channel-aware shapes
	torch::Tensor dL_dmeans3D   = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D   = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors    = torch::zeros({P, channels}, means3D.options());
	torch::Tensor dL_dconic     = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dopacity   = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dcov3D     = torch::zeros({P, 6}, means3D.options());
	torch::Tensor dL_dscales    = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	// dL_dsh depends on channels:
	//  - channels==3: [P,M,3]
	//  - channels==1: [P,M]
	torch::Tensor dL_dsh;
	if (has_sh) {
		if (channels == 3) dL_dsh = torch::zeros({P, M, 3}, means3D.options());
		else              dL_dsh = torch::zeros({P, M}, means3D.options());
	} else {
		// keep an empty tensor for API compatibility
		dL_dsh = torch::empty({0}, means3D.options());
	}

	if (P != 0) {
		const float* cov_ptr = (cov3D_precomp.numel() > 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr;
		const float* colors_ptr = has_colors_precomp ? colors.contiguous().data_ptr<float>() : nullptr;
		const float* sh_ptr = has_sh ? sh.contiguous().data_ptr<float>() : nullptr;

		CudaRasterizer::Rasterizer::backward(
			P, degree, M, R,
			channels, // <- runtime dispatch
			background.contiguous().data_ptr<float>(),
			W, H,
			means3D.contiguous().data_ptr<float>(),
			sh_ptr,
			colors_ptr,
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			cov_ptr,
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
			has_sh ? dL_dsh.contiguous().data_ptr<float>() : nullptr,
			dL_dscales.contiguous().data_ptr<float>(),
			dL_drotations.contiguous().data_ptr<float>(),
			debug
		);
	}

	return std::make_tuple(
		dL_dmeans2D,
		dL_dcolors,
		dL_dopacity,
		dL_dmeans3D,
		dL_dcov3D,
		dL_dsh,
		dL_dscales,
		dL_drotations
	);
}

// ============================================================
// markVisible (unchanged)
// ============================================================
torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix)
{
	const int P = means3D.size(0);

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0) {
		CudaRasterizer::Rasterizer::markVisible(P,
			means3D.contiguous().data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(),
			projmatrix.contiguous().data_ptr<float>(),
			present.contiguous().data_ptr<bool>());
	}

	return present;
}
