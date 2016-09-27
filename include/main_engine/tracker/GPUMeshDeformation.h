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
		const std::string _nonRigidEnergyFilePath);

	~GPUMeshDeformation();

	void setFrame(ImageLevel* _pFrame);

	void setRigidRotTrans(const double* _camPose);

	void setVisibility(const vector<bool>* _visibility);

	void setEnergyWeights(double _w_photometric, double _w_tv, double _w_arap,
		double _w_tempdeform, double _w_temptrans);

	void setNumIterations(	
		unsigned int _num_iter, 
		unsigned int _nonlinear_num_iter, 
		unsigned int _linear_num_iter);

	void solve();

	void copyResultsFromDevice(MeshDeformation* _meshTrans, 
		MeshDeformation* _meshRot);

private:

	unsigned int N;	//number of vertices
	unsigned int E;	//number of edges

	unsigned int img_width;
	unsigned int img_height;

	cv::Mat_<float>	h_image;
	cv::Mat_<float>	h_gradX_image;
	cv::Mat_<float>	h_gradY_image;

	float 	f_x, f_y, u_x, u_y;

	float3 *d_templateVertexPos;
	float 	*d_templateVertexGray;

	float 	*d_image;
	float 	*d_gradX_image;
	float 	*d_gradY_image;

	float3 *d_meshTrans;
	float3 *d_meshRot;
	float3 *d_camRot;
	float3 *d_camTrans;
	int*	d_visibility;

	float3 *d_prevMeshTrans;	

	int*	d_numNeighbours;
	int*	d_neighbourIdx;
	int* 	d_neighbourOffset;

	float3 *h_meshTrans;
	float3 *h_meshRot;
	int*	h_visibility;

	float3	camRot;
	float3	camTrans;

	float 	w_photometric;
	float 	w_tv;
	float 	w_arap;
	float 	w_tempdeform;
	float 	w_temptrans;

	unsigned int num_iter = 5;
	unsigned int nonlinear_num_iter = 5;
	unsigned int linear_num_iter = 10;

	TerraTrackingSolver* m_optTrackingSolver;

	void allocateMemory();
	void setIntrinsicParameters(const CameraInfo* pCamera);
	void initUnknowns();
	void copyTemplateToGPUMemory(const PangaeaMeshData* _templateMesh);

};
