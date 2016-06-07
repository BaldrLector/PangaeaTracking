#pragma once

#include <iostream>
#include <cuda_runtime.h>

#include "./TerraTrackingSolver.h"
#include "./Mesh.h"

class GPUMeshDeformation
{
public:
	GPUMeshDeformation();

	GPUMeshDeformation(PangaeaMeshData* _templateMesh, double* _camPose,
		MeshDeformation* _meshTrans, MeshDeformation* _meshRot, 
		double* _prevCamPose, MeshDeformation* _prevMeshTrans,
		bool* _visibility, std::string _rigidEnergyFilePath, 
		std::string _nonRigidEnergyFilePath);

	void init(PangaeaMeshData* _templateMesh, double* _camPose,
		MeshDeformation* _meshTrans, MeshDeformation* _meshRot, 
		double* _prevCamPose, MeshDeformation* _prevMeshTrans,
		bool* _visibility, std::string _rigidEnergyFilePath, 
		std::string _nonRigidEnergyFilePath);

	void setIntrinsicMatrix(const CameraInfo* pCamera);

	void setEnergyWeights(double _w_photometric, double _w_tv, double _w_arap,
		double _w_tempdeform, double _w_temptrans);

	void setNumIterations(unsigned int _num_iter, 
		unsigned int _nonlinear_num_iter, unsigned int _linear_num_iter);

	void resetGPUMemory();

	~GPUMeshDeformation();

	void solve();

	void copyResultsFromDevice();

private:

	PangaeaMeshData* templateMesh;
	double*	camPose;
	MeshDeformation* meshTrans;
	MeshDeformation* meshRot;
	MeshDeformation* prevMeshTrans;
	double*	prevCamPose;
	bool *visibility;

	double 	f_x, f_y, u_x, u_y;

	double3 *d_templateVertexPos;
	double3 *d_meshTrans;
	double3 *d_meshRot;
	double3 d_camRot;
	double3 d_camTrans;

	double3 *d_prevMeshTrans;
	double3 d_prevCamTrans;	

	int*	d_numNeighbours;
	int*	d_neighbourIdx;
	int* 	d_neighbourOffset;

	double3 *h_templateVertexPos;
	double3 *h_meshTrans;
	double3 *h_meshRot;
	double3 h_camRot;
	double3 h_camTrans;

	double3 *h_prevMeshTrans;
	double3 h_prevCamTrans;	

	int*	h_numNeighbours;
	int*	h_neighbourIdx;
	int*	h_neighbourOffset;

	double 	w_photometric;
	double 	w_tv;
	double 	w_arap;
	double 	w_tempdeform;
	double 	w_temptrans;

	unsigned int num_iter;
	unsigned int nonlinear_num_iter;
	unsigned int linear_num_iter;

	TerraTrackingSolver_Rigid* m_optTrackingSolver_Rigid;
	TerraTrackingSolver_NonRigid* m_optTrackingSolver_NonRigid;
};
