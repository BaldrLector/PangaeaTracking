#include "main_engine/tracker/GPUMeshDeformation.h"

GPUMeshDeformation::GPUMeshDeformation()
{

}

GPUMeshDeformation::GPUMeshDeformation(const PangaeaMeshData* _templateMesh, 
	const CameraInfo* pCamera, 
	const std::string _rigidEnergyFilePath,
	const std::string _nonRigidEnergyFilePath) : 
	templateMesh(_templateMesh)
{
	N = (unsigned int)templateMesh->numVertices;
	E = (unsigned int)(templateMesh->numVertices + templateMesh->numFaces);

	img_width = (unsigned int)pCamera->width;
	img_height = (unsigned int)pCamera->height;

	allocateMemory();

	setIntrinsicParameters(pCamera);

	copyTemplateToGPUMemory();
		
   	m_optTrackingSolver_Rigid = new TerraTrackingSolver_Rigid(N, 2 * E, 
   		d_neighbourIdx, d_neighbourOffset, _rigidEnergyFilePath, 
   		"gaussNewtonGPU");

	m_optTrackingSolver_NonRigid = new TerraTrackingSolver_NonRigid(N, 2 * E, 
		d_neighbourIdx, d_neighbourOffset, _nonRigidEnergyFilePath, 
		"gaussNewtonGPU");
}

GPUMeshDeformation::~GPUMeshDeformation()
{
	cutilSafeCall(cudaFree(d_templateVertexPos));
	cutilSafeCall(cudaFree(d_templateVertexColor));
	cutilSafeCall(cudaFree(d_image));
	cutilSafeCall(cudaFree(d_gradX_image));
	cutilSafeCall(cudaFree(d_gradY_image));

	cutilSafeCall(cudaFree(d_meshTrans));
	cutilSafeCall(cudaFree(d_meshRot));

	cutilSafeCall(cudaFree(d_camRot));
	cutilSafeCall(cudaFree(d_camTrans));

	cutilSafeCall(cudaFree(d_visibility));

	cutilSafeCall(cudaFree(d_prevMeshTrans));
	cutilSafeCall(cudaFree(d_prevCamTrans));

	cutilSafeCall(cudaFree(d_numNeighbours));
	cutilSafeCall(cudaFree(d_neighbourIdx));
	cutilSafeCall(cudaFree(d_neighbourOffset));

	SAFE_DELETE(m_optTrackingSolver_Rigid);
	SAFE_DELETE(m_optTrackingSolver_NonRigid);

	SafeDeleteArray(h_meshTrans);
	SafeDeleteArray(h_meshRot);

	SAFE_DELETE(h_camRot);
	SAFE_DELETE(h_camTrans);
	
	SafeDeleteArray(h_visibility);
	
	SafeDeleteArray(h_prevMeshTrans);
	SAFE_DELETE(h_prevCamTrans);
}

void GPUMeshDeformation::setData(ImageLevel* _pFrame, 
	double* _camPose, MeshDeformation* _meshTrans, MeshDeformation* _meshRot, 
	double* _prevCamPose, MeshDeformation* _prevMeshTrans,
	vector<bool>* _visibility)
{
	pFrame = _pFrame;

	camPose = _camPose;
	meshTrans = _meshTrans;
	meshRot = _meshRot;
	visibility = _visibility;

	prevCamPose = _prevCamPose;
	prevMeshTrans = _prevMeshTrans;

	copyDataToGPUMemory();
} 

void GPUMeshDeformation::setEnergyWeights(
	double _w_photometric, 
	double _w_tv, 
	double _w_arap,
	double _w_tempdeform, 
	double _w_temptrans)
{
	w_photometric = _w_photometric;
	w_tv = _w_tv;
	w_arap = _w_arap;
	w_tempdeform = _w_tempdeform;
	w_temptrans = _w_temptrans;
}

void GPUMeshDeformation::setNumIterations(
	unsigned int _rigid_num_iter, unsigned int _rigid_nonlinear_num_iter, 
	unsigned int _rigid_linear_num_iter,
	unsigned int _non_rigid_num_iter, unsigned int _non_rigid_nonlinear_num_iter, 
	unsigned int _non_rigid_linear_num_iter)
{
	rigid_num_iter = _rigid_num_iter;
	rigid_nonlinear_num_iter = _rigid_nonlinear_num_iter;
	rigid_linear_num_iter = _rigid_linear_num_iter;

	non_rigid_num_iter = _non_rigid_num_iter;
	non_rigid_nonlinear_num_iter = _non_rigid_nonlinear_num_iter;
	non_rigid_linear_num_iter = _non_rigid_linear_num_iter;
}

void GPUMeshDeformation::solve()
{
	// copyResultsFromDevice();

	// resetGPUMemory();

	std::cout << "//////////// RIGID (OPT) ///////////////" << std::endl;
	for (unsigned int i = 1; i < rigid_num_iter; i++)
	{
		std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
		m_optTrackingSolver_Rigid->solveGN(
			d_templateVertexPos, d_templateVertexColor,
			d_image, d_gradX_image, d_gradY_image,
			d_meshTrans, d_camRot, d_camTrans, d_prevCamTrans, d_visibility,
			f_x, f_y, u_x, u_y,
			rigid_nonlinear_num_iter, rigid_linear_num_iter,
			w_photometric, w_temptrans);
	}

	std::cout << "//////////// NON RIGID (OPT) ///////////////" << std::endl;
	for (unsigned int i = 1; i < non_rigid_num_iter; i++)
	{
		std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
		m_optTrackingSolver_NonRigid->solveGN(
			d_templateVertexPos, d_templateVertexColor,
			d_image, d_gradX_image, d_gradY_image,
			d_meshTrans, d_meshRot,
			d_camRot, d_camTrans,
			d_prevMeshTrans,
			d_visibility,
			f_x, f_y, u_x, u_y,
			non_rigid_nonlinear_num_iter,
			rigid_nonlinear_num_iter,
			w_photometric, w_tv, w_arap, w_tempdeform);
	}

	copyResultsFromDevice();
}

void GPUMeshDeformation::copyResultsFromDevice()
{
	cutilSafeCall(cudaMemcpy(h_meshTrans, d_meshTrans, 
		sizeof(double3)*N, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_meshRot, d_meshRot, 
		sizeof(double3)*N, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&h_camRot, &d_camRot, 
		sizeof(double3), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&h_camTrans, &d_camTrans,
		sizeof(double3), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < N; i++)
	{
		(*meshTrans)[i][0] = h_meshTrans[i].x;
		(*meshTrans)[i][1] = h_meshTrans[i].y;
		(*meshTrans)[i][2] = h_meshTrans[i].z;

		(*meshRot)[i][0] = h_meshRot[i].x;
		(*meshRot)[i][1] = h_meshRot[i].y;
		(*meshRot)[i][2] = h_meshRot[i].z;
	}

	camPose[0] = h_camRot->x;
	camPose[1] = h_camRot->y;
	camPose[2] = h_camRot->z;
	camPose[3] = h_camTrans->x;
	camPose[4] = h_camTrans->y;
	camPose[5] = h_camTrans->z;
}

void GPUMeshDeformation::allocateMemory()
{
	cutilSafeCall(cudaMalloc(&d_templateVertexPos, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_templateVertexColor, sizeof(double)*N));
	cutilSafeCall(cudaMalloc(&d_image, sizeof(double)*img_width*img_height));
	cutilSafeCall(cudaMalloc(&d_gradX_image, sizeof(double)*img_width*img_height));
	cutilSafeCall(cudaMalloc(&d_gradY_image, sizeof(double)*img_width*img_height));

	cutilSafeCall(cudaMalloc(&d_meshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_meshRot, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_camRot, sizeof(double3)));
	cutilSafeCall(cudaMalloc(&d_camTrans, sizeof(double3)));
	cutilSafeCall(cudaMalloc(&d_visibility, sizeof(uchar)*N));

	cutilSafeCall(cudaMalloc(&d_prevMeshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_prevCamTrans, sizeof(double3)));

	cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
	cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));

	h_meshTrans = new double3[N];
	h_meshRot = new double3[N];
	h_camRot = new double3;
	h_camTrans = new double3;
	h_visibility = new uchar[N];
	h_prevMeshTrans = new double3[N];
	h_prevCamTrans = new double3;
}

void GPUMeshDeformation::setIntrinsicParameters(const CameraInfo* pCamera)
{
	f_x = pCamera->KK[0][0];
	f_y = pCamera->KK[1][1];
	u_x = pCamera->KK[0][2];
	u_y = pCamera->KK[1][2];
}

void GPUMeshDeformation::copyTemplateToGPUMemory()
{
	double3 *h_templateVertexPos = new double3[N];
	double 	*h_templateVertexColor= new double[N]; 

	int* 	h_numNeighbours   = new int[N];
	int* 	h_neighbourIdx	  = new int[2*E];
	int* 	h_neighbourOffset = new int[N+1];

	unsigned int count = 0;
	unsigned int offset = 0;
	h_neighbourOffset[0] = 0;

	const vector<vector<unsigned int> >& meshNeighbors = templateMesh->adjVerticesInd;
	for (unsigned int i = 0; i < N; i++)
	{
		const vector<double> &v = templateMesh->vertices[i];
		h_templateVertexPos[i] = make_double3(v[0], v[1], v[2]);

		const double &gray = templateMesh->grays[i];
		h_templateVertexColor[i] = gray;

		unsigned int valance = (unsigned int)meshNeighbors[i].size();
		h_numNeighbours[count] = valance;

		for (unsigned int j = 0; j < valance; j++)
		{
			h_neighbourIdx[offset] = meshNeighbors[i][j];
			offset++;
		}

		h_neighbourOffset[count + 1] = offset;

		count++;
	}
	
	cutilSafeCall(cudaMemcpy(d_templateVertexPos, h_templateVertexPos, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_templateVertexColor, h_templateVertexColor, 
		sizeof(double)*N, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, 
		sizeof(int)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, 
		sizeof(int)* 2 * E, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, 
		sizeof(int)*(N + 1), cudaMemcpyHostToDevice));

	delete[] h_templateVertexPos;
	delete[] h_templateVertexColor;

	delete[] h_numNeighbours;
	delete[] h_neighbourIdx;
	delete[] h_neighbourOffset;
}

void GPUMeshDeformation::copyDataToGPUMemory()
{
	for (unsigned int i = 0; i < N; i++)
	{
		const vector<double> &v_trans = (*meshTrans)[i];
		h_meshTrans[i] = make_double3(v_trans[0], v_trans[1], v_trans[2]);

		const vector<double> &v_rot = (*meshTrans)[i];
		h_meshRot[i] = make_double3(v_rot[0], v_rot[1], v_rot[2]);

		const vector<double> &v_prev_trans = (*prevMeshTrans)[i];
		h_prevMeshTrans[i] = make_double3(v_prev_trans[0], v_prev_trans[1], v_prev_trans[2]);

		h_visibility[i] = (unsigned char)(*visibility)[i];
	}

	h_camRot->x = camPose[0];
	h_camRot->y = camPose[1];
	h_camRot->z = camPose[2];
	h_camTrans->x = camPose[3];
	h_camTrans->y = camPose[4];
	h_camTrans->z = camPose[5];
	h_prevCamTrans->x = prevCamPose[3];
	h_prevCamTrans->y = prevCamPose[4];
	h_prevCamTrans->z = prevCamPose[5];

	// double 	*h_image = new double[img_width*img_height];
	// double* p;
	// unsigned int count = 0;
	// for (unsigned int i = 0; i < img_height; i++)
	// {
	// 	p = image->ptr<double>(i);
	// 	for (unsigned int j = 0; j < img_width; j++)
	// 	{
	// 		h_image[count] = p[j];
	// 		count++;
	// 	}
	// }
	// cutilSafeCall(cudaMemcpy(d_image, h_image, 
	// 	sizeof(double)*N, cudaMemcpyHostToDevice));	
	// delete[] h_image;

	cutilSafeCall(cudaMemcpy(d_image, pFrame->grayImage.data, 
		sizeof(double)*N, cudaMemcpyHostToDevice));	
	cutilSafeCall(cudaMemcpy(d_gradX_image, pFrame->gradXImage.data, 
		sizeof(double)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_gradY_image, pFrame->gradYImage.data, 
		sizeof(double)*N, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(d_meshTrans, h_meshTrans, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));	
	cutilSafeCall(cudaMemcpy(d_meshRot, h_meshRot, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_prevMeshTrans, h_prevMeshTrans, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_visibility, h_visibility, 
		sizeof(uchar)*N, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(&d_camRot, &h_camRot, 
		sizeof(double3), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_camTrans, &h_camTrans, 
		sizeof(double3), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_prevCamTrans, &h_prevCamTrans, 
		sizeof(double3), cudaMemcpyHostToDevice));
}