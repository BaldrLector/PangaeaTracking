#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <math.h>

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

class TerraTrackingSolver {

public:
	TerraTrackingSolver(unsigned int vertexCount, unsigned int E, 
		const int* d_xCoords, const int* d_offsets, 
		const std::string& terraFile, const std::string& optName);

	~TerraTrackingSolver();

	void solveGN(
		float3* d_vertexPosFloat3,
		float3* d_anglesFloat3,
		float3* d_vertexPosFloat3Urshape,
		float3* d_vertexPosTargetFloat3,
		unsigned int nNonLinearIterations,
		unsigned int nLinearIterations,
		float weightFit,
		float weightReg);

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
};
