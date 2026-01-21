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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// NOTE about layouts:
	// - channels == 3:
	//     shs:            float* storing P * M * 3 floats (treated as glm::vec3[M] per point)
	//     colors_precomp: float* storing P * 3
	//     colors:         float* storing P * 3
	//     bg_color:       float* length 3
	//     clamped:        bool*  length P * 3 (per-channel clamp flags)
	//
	// - channels == 1:
	//     shs:            float* storing P * M (scalar SH per point)
	//     colors_precomp: float* storing P
	//     colors:         float* storing P
	//     bg_color:       float* length 1
	//     clamped:        bool*  length P (scalar clamp flag)

	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		int P, int D, int M,
		int channels,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		int channels,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color);
}


#endif
