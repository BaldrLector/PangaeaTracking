#pragma once

#include "./ImagePyramid.h"
#include "./TerraTrackingSolver.h"
#include "./Mesh.h"

class GPUMeshDeformation
{
public:
	GPUMeshDeformation();

	GPUMeshDeformation(const PangaeaMeshData* _templateMesh, 
		const CameraInfo* pCamera, 
		const std::string _rigidEnergyFilePath,
		const std::string _nonRigidEnergyFilePath);

	~GPUMeshDeformation();

	void setData(ImageLevel* _pFrame, 
		double* _camPose, MeshDeformation* _meshTrans, 
		MeshDeformation* _meshRot, 
		double* _prevCamPose, MeshDeformation* _prevMeshTrans,
		vector<bool>* _visibility);

	void setEnergyWeights(double _w_photometric, double _w_tv, double _w_arap,
		double _w_tempdeform, double _w_temptrans);

	void setNumIterations(	
		unsigned int _rigid_num_iter, 
		unsigned int _rigid_nonlinear_num_iter, 
		unsigned int _rigid_linear_num_iter,
		unsigned int _non_rigid_num_iter, 
		unsigned int _non_rigid_nonlinear_num_iter, 
		unsigned int _non_rigid_linear_num_iter);

	void solve();

private:

	unsigned int N;	//number of vertices
	unsigned int E;	//number of edges

	unsigned int img_width;
	unsigned int img_height;

	const PangaeaMeshData* templateMesh;

	ImageLevel* pFrame; 

	double*	camPose;
	MeshDeformation* meshTrans;
	MeshDeformation* meshRot;
	MeshDeformation* prevMeshTrans;
	double*	prevCamPose;
	vector<bool> *visibility;

	double 	f_x, f_y, u_x, u_y;

	float3 *d_templateVertexPos;
	float 	*d_templateVertexColor;
	float 	*d_image;
	float 	*d_gradX_image;
	float 	*d_gradY_image;

	float3 *d_meshTrans;
	float3 *d_meshRot;
	float3 *d_camRot;
	float3 *d_camTrans;
	int*	d_visibility;

	float3 *d_prevMeshTrans;
	float3 *d_prevCamTrans;	

	int*	d_numNeighbours;
	int*	d_neighbourIdx;
	int* 	d_neighbourOffset;

	float3 *h_meshTrans;
	float3 *h_meshRot;
	float3 *h_camRot;
	float3 *h_camTrans;
	int*	h_visibility;

	float3 *h_prevMeshTrans;
	float3 *h_prevCamTrans;

	float 	w_photometric;
	float 	w_tv;
	float 	w_arap;
	float 	w_tempdeform;
	float 	w_temptrans;

	unsigned int rigid_num_iter = 5;
	unsigned int rigid_nonlinear_num_iter = 10;
	unsigned int rigid_linear_num_iter = 200;

	unsigned int non_rigid_num_iter = 5;
	unsigned int non_rigid_nonlinear_num_iter = 10;
	unsigned int non_rigid_linear_num_iter = 200;

	TerraTrackingSolver_Rigid* m_optTrackingSolver_Rigid;
	TerraTrackingSolver_NonRigid* m_optTrackingSolver_NonRigid;

	void allocateMemory();
	void setIntrinsicParameters(const CameraInfo* pCamera);
	void copyTemplateToGPUMemory();
	void copyDataToGPUMemory();
	void copyResultsFromDevice();

};
