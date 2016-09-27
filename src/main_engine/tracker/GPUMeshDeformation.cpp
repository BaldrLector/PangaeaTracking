#include "main_engine/tracker/GPUMeshDeformation.h"

GPUMeshDeformation::GPUMeshDeformation()
{

}

GPUMeshDeformation::GPUMeshDeformation(const PangaeaMeshData* _templateMesh, 
	const CameraInfo* _pCamera, 
	const std::string _terraEnergyFilePath)
{
	N = (unsigned int)_templateMesh->numVertices;
	E = (unsigned int)(_templateMesh->numVertices + _templateMesh->numFaces);

	img_width = (unsigned int)_pCamera->width;
	img_height = (unsigned int)_pCamera->height;

	allocateMemory();

	setIntrinsicParameters(_pCamera);

	copyTemplateToGPUMemory(_templateMesh);

	initUnknowns();

	m_optTrackingSolver = new TerraTrackingSolver(
		img_width, img_height, N, E, 
		d_neighbourIdx, d_neighbourOffset, _terraEnergyFilePath, 
		"gaussNewtonGPU");
}

void GPUMeshDeformation::allocateMemory()
{
	cutilSafeCall(cudaMalloc(&d_templateVertexPos, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_templateVertexGray, sizeof(float)*N));
	cutilSafeCall(cudaMalloc(&d_image, sizeof(float)*img_width*img_height));
	cutilSafeCall(cudaMalloc(&d_gradX_image, sizeof(float)*img_width*img_height));
	cutilSafeCall(cudaMalloc(&d_gradY_image, sizeof(float)*img_width*img_height));

	cutilSafeCall(cudaMalloc(&d_meshTrans, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_meshRot, sizeof(float3)*N));
	cutilSafeCall(cudaMalloc(&d_visibility, sizeof(int)*N));

	cutilSafeCall(cudaMalloc(&d_prevMeshTrans, sizeof(float3)*N));

	cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
	cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));

	h_meshTrans = new float3[N];
	h_meshRot = new float3[N];
	h_visibility = new int[N];
}

GPUMeshDeformation::~GPUMeshDeformation()
{
	cutilSafeCall(cudaFree(d_templateVertexPos));
	cutilSafeCall(cudaFree(d_templateVertexGray));
	cutilSafeCall(cudaFree(d_image));
	cutilSafeCall(cudaFree(d_gradX_image));
	cutilSafeCall(cudaFree(d_gradY_image));

	cutilSafeCall(cudaFree(d_meshTrans));
	cutilSafeCall(cudaFree(d_meshRot));
	cutilSafeCall(cudaFree(d_visibility));

	cutilSafeCall(cudaFree(d_prevMeshTrans));

	cutilSafeCall(cudaFree(d_numNeighbours));
	cutilSafeCall(cudaFree(d_neighbourIdx));
	cutilSafeCall(cudaFree(d_neighbourOffset));

	SAFE_DELETE(m_optTrackingSolver);

	SafeDeleteArray(h_meshTrans);
	SafeDeleteArray(h_meshRot);
	SafeDeleteArray(h_visibility);
}

void GPUMeshDeformation::setIntrinsicParameters(const CameraInfo* _pCamera)
{
	f_x = (float)_pCamera->KK[0][0];
	f_y = (float)_pCamera->KK[1][1];
	u_x = (float)_pCamera->KK[0][2];
	u_y = (float)_pCamera->KK[1][2];
}

void GPUMeshDeformation::copyTemplateToGPUMemory(
	const PangaeaMeshData* _templateMesh)
{
	float3 	*h_templateVertexPos = new float3[N];
	float 	*h_templateVertexGray= new float[N]; 

	int* 	h_numNeighbours   = new int[N];
	int* 	h_neighbourIdx	  = new int[2*E];
	int* 	h_neighbourOffset = new int[N+1];

	unsigned int count = 0;
	unsigned int offset = 0;
	h_neighbourOffset[0] = 0;

	std::cout << _templateMesh->numVertices << std::endl;

	const vector<vector<unsigned int> >& meshNeighbors = _templateMesh->adjVerticesInd;
	for (unsigned int i = 0; i < N; i++)
	{
		const vector<double> &v = _templateMesh->vertices[i];
		h_templateVertexPos[i] = make_float3(
			(float)v[0], (float)v[1], (float)v[2]);

		const double &gray = _templateMesh->grays[i];
		h_templateVertexGray[i] = (float)gray;

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
		sizeof(float3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_templateVertexGray, h_templateVertexGray, 
		sizeof(float)*N, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, 
		sizeof(int)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, 
		sizeof(int)* 2 * E, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, 
		sizeof(int)*(N + 1), cudaMemcpyHostToDevice));

	SafeDeleteArray(h_templateVertexPos);
	SafeDeleteArray(h_templateVertexGray);

	SafeDeleteArray(h_numNeighbours);
	SafeDeleteArray(h_neighbourIdx);
	SafeDeleteArray(h_neighbourOffset);
}

void GPUMeshDeformation::initUnknowns()
{
	float3 zero3f = make_float3( 0.0f, 0.0f, 0.0f);
	for (unsigned int i = 0; i < N; i++)
	{
		h_meshTrans[i] = zero3f;
		h_meshRot[i] = zero3f;
	}

	cutilSafeCall( cudaMemcpy(d_meshTrans, h_meshTrans, 
		sizeof(float3)*N, cudaMemcpyHostToDevice) );	
	cutilSafeCall( cudaMemcpy(d_meshRot, h_meshRot, 
		sizeof(float3)*N, cudaMemcpyHostToDevice) );
}

void GPUMeshDeformation::setEnergyWeights(
	double _w_photometric, 
	double _w_tv, 
	double _w_arap,
	double _w_tempdeform, 
	double _w_temptrans)
{
	w_photometric = (float)_w_photometric;
	w_tv = (float)_w_tv;
	w_arap = (float)_w_arap;
	w_tempdeform = (float)_w_tempdeform;
	w_temptrans = (float)_w_temptrans;
}

void GPUMeshDeformation::setNumIterations(
	unsigned int _num_iter, unsigned int _nonlinear_num_iter, 
	unsigned int _linear_num_iter)
{
	num_iter = _num_iter;
	nonlinear_num_iter = _nonlinear_num_iter;
	linear_num_iter = _linear_num_iter;
}

void GPUMeshDeformation::setFrame(ImageLevel* _pFrame)
{
	_pFrame->grayImage.convertTo(h_image, CV_32F);
	_pFrame->gradXImage.convertTo(h_gradX_image, CV_32F);
	_pFrame->gradYImage.convertTo(h_gradY_image, CV_32F);

	cutilSafeCall(cudaMemcpy(&d_image, &h_image, 
		sizeof(float) * img_width * img_height, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_gradX_image, &h_gradX_image, 
		sizeof(float) * img_width * img_height, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_gradY_image, &h_gradY_image, 
		sizeof(float) * img_width * img_height, cudaMemcpyHostToDevice));
}

void GPUMeshDeformation::setRigidRotTrans(const double* _camPose)
{
	camRot.x = (float)_camPose[0];
	camRot.y = (float)_camPose[1];
	camRot.z = (float)_camPose[2];
	camTrans.x = (float)_camPose[3];
	camTrans.y = (float)_camPose[4];
	camTrans.z = (float)_camPose[5];
} 

void GPUMeshDeformation::setVisibility(const vector<bool>* _visibility)
{
	for (unsigned int i = 0; i < N; i++)
	{
		h_visibility[i] = (int)(*_visibility)[i];
	}

	cutilSafeCall(cudaMemcpy(d_visibility, h_visibility, 
		sizeof(int)*N, cudaMemcpyHostToDevice));
}

void GPUMeshDeformation::solve()
{
	cutilSafeCall( cudaMemcpy(&d_prevMeshTrans, &d_meshTrans, 
		sizeof(float3), cudaMemcpyDeviceToDevice) );

	std::cout << "//////////// NON RIGID (OPT) ///////////////" << std::endl;
	for (unsigned int i = 0; i < num_iter; i++)
	{
		std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
		m_optTrackingSolver->solve(
			d_templateVertexPos, d_templateVertexGray,
			d_image, d_gradX_image, d_gradY_image,
			d_meshTrans, d_meshRot,
			camRot, camTrans,
			d_prevMeshTrans,
			d_visibility,
			f_x, f_y, u_x, u_y,
			nonlinear_num_iter,
			linear_num_iter,
			w_photometric, w_tv, w_arap, w_tempdeform);
	}
}

void GPUMeshDeformation::copyResultsFromDevice(MeshDeformation* _meshTrans, 
	MeshDeformation* _meshRot)
{
	cutilSafeCall(cudaMemcpy(h_meshTrans, d_meshTrans, 
		sizeof(float3)*N, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_meshRot, d_meshRot, 
		sizeof(float3)*N, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < N; i++)
	{
		(*_meshTrans)[i][0] = (double)h_meshTrans[i].x;
		(*_meshTrans)[i][1] = (double)h_meshTrans[i].y;
		(*_meshTrans)[i][2] = (double)h_meshTrans[i].z;

		(*_meshRot)[i][0] = (double)h_meshRot[i].x;
		(*_meshRot)[i][1] = (double)h_meshRot[i].y;
		(*_meshRot)[i][2] = (double)h_meshRot[i].z;
	}
}
