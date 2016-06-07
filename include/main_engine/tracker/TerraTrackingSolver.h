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


class TerraTrackingSolver_Rigid : public TerraTrackingSolver
{
public:
	void solveGN(
		double3* d_templateVertexPos,
		double3* d_templateVertexColor,
		double3* d_meshTrans,
		double3* d_camRot,
		double3* d_camTrans,
		double3* d_prevCamTrans,
		int* d_visibility,
		double f_x, double f_y, double u_x, double u_y,
		unsigned int nNonLinearIterations,
		unsigned int nLinearIterations,
		double w_photometric,
		double w_temptrans);
};

class TerraTrackingSolver_NonRigid : public TerraTrackingSolver
{
public:
	void solveGN(
		float3* d_vertexPosFloat3,
		float3* d_anglesFloat3,
		float3* d_vertexPosFloat3Urshape,
		float3* d_vertexPosTargetFloat3,
		unsigned int nNonLinearIterations,
		unsigned int nLinearIterations,
		float weightFit,
		float weightReg);
};