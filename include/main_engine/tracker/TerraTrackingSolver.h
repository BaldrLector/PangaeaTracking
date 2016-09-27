#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <math.h>

#include <iostream>

#include <cuda_runtime.h>

#include "../utils/cudaUtil.h"

extern "C" {
#include "third_party/Opt.h"
}


template <class type> type* createDeviceBuffer(const std::vector<type>& v) {
	type* d_ptr;
	cutilSafeCall(cudaMalloc(&d_ptr, sizeof(type)*v.size()));

	cutilSafeCall(cudaMemcpy(d_ptr, v.data(), sizeof(type)*v.size(), cudaMemcpyHostToDevice));
	return d_ptr;
}

class TerraTrackingSolver
{
private:
	int* d_headX;
	int* d_headY;

	int* d_tailX;
	int* d_tailY;

	int edgeCount;

	Opt_State*		m_optimizerState;
	Opt_Problem*	m_problem;
	Opt_Plan*		m_plan;

	unsigned int m_numUnknown;
public:
	TerraTrackingSolver(unsigned int img_width, unsigned int img_height, 
		unsigned int vertexCount, unsigned int E, 
		const int* d_xCoords, const int* d_offsets, 
		const std::string& terraFile, const std::string& optName);

	~TerraTrackingSolver();

	void solve(
		float3* d_templateVertexPos,
		float* d_templateVertexColor,
		float* d_image, float* d_gradX_image, float* d_gradY_image,
		float3* d_meshTrans,
		float3* d_meshRot,
		float3 d_camRot,
		float3 d_camTrans,
		float3* d_prevMeshTrans,
		int* d_visibility,
		float f_x, float f_y, float u_x, float u_y,
		unsigned int nNonLinearIterations,
		unsigned int nLinearIterations,
		float w_photometric, float w_tv, float w_arap, float w_tempdeform);
};