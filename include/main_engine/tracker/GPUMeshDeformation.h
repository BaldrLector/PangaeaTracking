#pragma once

#include <iostream>
#include <cuda_runtime.h>

#include "./TerraTtackingSolver.h"
#include "./Mesh.h"

class GPUMeshDeformation
{
public:
	GPUMeshDeformation(PangaeaMeshData* templateMesh, double* camPose,
		MeshDeformation* meshTrans, MeshDeformation* meshRot, 
		double* prevCamPose, MeshDeformation* prevMeshTrans);

	void setConstraints(float alpha);

	void resetGPUMemory();

	~GPUMeshDeformation();

	SimpleMesh* solve();

	void copyResultToCPUFromFloat3();

private:

	PangaeaMeshData* templateMesh;
	double* camPose,
	MeshDeformation* meshTrans;
	MeshDeformation* meshRot, 
	MeshDeformation* prevMeshTrans;
	double* prevCamPose;

	float3* d_anglesFloat3;
	float3*	d_vertexPosTargetFloat3;
	float3*	d_vertexPosFloat3;
	float3*	d_vertexPosFloat3Urshape;
	int*	d_numNeighbours;
	int*	d_neighbourIdx;
	int* 	d_neighbourOffset;

	TerraTtackingSolver* m_optWarpingSolver;

	std::vector<int>				m_constraintsIdx;
	std::vector<std::vector<float>>	m_constraintsTarget;
};
