#include "main_engine/tracker/GPUMeshDeformation.h"

GPUMeshDeformation::GPUMeshDeformation()
{

}

GPUMeshDeformation::GPUMeshDeformation(PangaeaMeshData* _templateMesh, 
	double* _camPose, MeshDeformation* _meshTrans, MeshDeformation* _meshRot, 
	double* _prevCamPose, MeshDeformation* _prevMeshTrans,
	std::string _rigidEnergyFilePath, std::string _nonRigidEnergyFilePath) : 
	templateMesh(_templateMesh), camPose(_camPose), meshTrans(_meshTrans), 
	meshRot(_meshRot), prevCamPose(_prevCamPose), prevMeshTrans(_prevMeshTrans)
{
	unsigned int N = (unsigned int)templateMesh->numVertices;
	unsigned int E = (unsigned int)(templateMesh->numVertices + templateMesh->numFaces);

	cutilSafeCall(cudaMalloc(&d_templateVertexPos, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_meshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_meshRot, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_camRot, sizeof(double3)));
	cutilSafeCall(cudaMalloc(&d_camTrans, sizeof(double3)));

	cutilSafeCall(cudaMalloc(&d_prevMeshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_prevCamTrans, sizeof(double3)));

	cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
	cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));

	resetGPUMemory();			
   	m_optTrackingSolver_Rigid = new TerraTrackingSolver(N, 2 * E, d_neighbourIdx, 
   		d_neighbourOffset, _rigidEnergyFilePath, "gaussNewtonGPU");

	m_optTrackingSolver_NonRigid = new TerraTrackingSolver(N, 2 * E, d_neighbourIdx, 
   		d_neighbourOffset, _nonRigidEnergyFilePath, "gaussNewtonGPU");
} 

void GPUMeshDeformation::init(PangaeaMeshData* _templateMesh, 
	double* _camPose, MeshDeformation* _meshTrans, MeshDeformation* _meshRot, 
	double* _prevCamPose, MeshDeformation* _prevMeshTrans,
	std::string _rigidEnergyFilePath, std::string _nonRigidEnergyFilePath)
{
	templateMesh = _templateMesh;
	camPose = _camPose;
	meshTrans = _meshTrans;
	meshRot = _meshRot;
	prevCamPose = _prevCamPose;
	prevMeshTrans = _prevMeshTrans;

	unsigned int N = (unsigned int)templateMesh->numVertices;
	unsigned int E = (unsigned int)(templateMesh->numVertices + templateMesh->numFaces);

	cutilSafeCall(cudaMalloc(&d_templateVertexPos, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_meshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_meshRot, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_camRot, sizeof(double3)));
	cutilSafeCall(cudaMalloc(&d_camTrans, sizeof(double3)));

	cutilSafeCall(cudaMalloc(&d_prevMeshTrans, sizeof(double3)*N));
	cutilSafeCall(cudaMalloc(&d_prevCamTrans, sizeof(double3)));

	cutilSafeCall(cudaMalloc(&d_numNeighbours, sizeof(int)*N));
	cutilSafeCall(cudaMalloc(&d_neighbourIdx, sizeof(int)*2*E));
	cutilSafeCall(cudaMalloc(&d_neighbourOffset, sizeof(int)*(N+1)));

	h_templateVertexPos = new double3[N];
	h_meshTrans = new double3[N];
	h_meshRot = new double3[N];
	h_prevMeshTrans = new double3[N];

	h_numNeighbours   = new int[N];
	h_neighbourIdx	  = new int[2*E];
	h_neighbourOffset = new int[N+1];

	resetGPUMemory();			

   	m_optTrackingSolver_Rigid = new TerraTrackingSolver(N, 2 * E, d_neighbourIdx, 
   		d_neighbourOffset, _rigidEnergyFilePath, "gaussNewtonGPU");

	m_optTrackingSolver_NonRigid = new TerraTrackingSolver(N, 2 * E, d_neighbourIdx, 
   		d_neighbourOffset, _nonRigidEnergyFilePath, "gaussNewtonGPU");
} 

void GPUMeshDeformation::setIntrinsicMatrix(const CameraInfo* pCamera)
{
	f_x = pCamera->KK[0][0];
	f_y = pCamera->KK[1][1];
	u_x = pCamera->KK[0][2];
	u_y = pCamera->KK[1][2];
}

void GPUMeshDeformation::setEnergyWeights(double _w_photometric, double _w_tv, 
	double _w_arap, double _w_tempdeform, double _w_temptrans)
{
	w_photometric = _w_photometric;
	w_tv = _w_tv;
	w_arap = _w_arap;
	w_tempdeform = _w_tempdeform;
	w_temptrans = _w_temptrans;
}

void GPUMeshDeformation::setNumIterations(unsigned int _num_iter, 
	unsigned int _nonlinear_num_iter, unsigned int _linear_num_iter)
{
	num_iter = _num_iter;
	nonlinear_num_iter = _nonlinear_num_iter;
	linear_num_iter = _linear_num_iter;
}

void GPUMeshDeformation::resetGPUMemory()
{
	unsigned int N = (unsigned int)templateMesh->numVertices;
	unsigned int E = (unsigned int)(templateMesh->numVertices + templateMesh->numFaces);

	unsigned int count = 0;
	unsigned int offset = 0;
	h_neighbourOffset[0] = 0;

	vector<vector<unsigned int> >& meshNeighbors = templateMesh->adjVerticesInd[i];
	for (unsigned int i = 0; i < N; i++)
	{
		const vector<double> v = templateMesh->vertices[i];
		h_templateVertexPos[i] = make_double3(v[0], v[1], v[2]);

		const vector<double> v_trans = (*meshTrans)[i];
		h_meshTrans[i] = make_double3(v_trans[0], v_trans[1], v_trans[2]);

		const vector<double> v_rot = (*meshTrans)[i];
		h_meshRot[i] = make_double3(v_rot[0], v_rot[1], v_rot[2]);

		const vector<double> v_prev_trans = (*prevMeshTrans)[i];
		h_prevMeshTrans[i] = make_double3(v_prev_trans[0], v_prev_trans[1], v_prev_trans[2]);

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

	h_camRot.x = camPose[0];
	h_camRot.y = camPose[1];
	h_camRot.z = camPose[2];
	h_camTrans.x = camPose[3];
	h_camTrans.y = camPose[4];
	h_camTrans.z = camPose[5];
	h_prevCamTrans.x = prevCamPose[3];
	h_prevCamTrans.y = prevCamPose[4];
	h_prevCamTrans.z = prevCamPose[5];

	
	cutilSafeCall(cudaMemcpy(d_templateVertexPos, h_templateVertexPos, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_meshTrans, h_meshTrans, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));	
	cutilSafeCall(cudaMemcpy(d_meshRot, h_meshRot, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_prevMeshTrans, h_prevMeshTrans, 
		sizeof(double3)*N, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(&d_camRot, &h_camRot, 
		sizeof(double3), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_camTrans, &h_camTrans, 
		sizeof(double3), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(&d_prevCamTrans, &h_prevCamTrans, 
		sizeof(double3), cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy(d_numNeighbours, h_numNeighbours, 
		sizeof(int)*N, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourIdx, h_neighbourIdx, 
		sizeof(int)* 2 * E, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_neighbourOffset, h_neighbourOffset, 
		sizeof(int)*(N + 1), cudaMemcpyHostToDevice));
}

GPUMeshDeformation::~GPUMeshDeformation()
{
	cutilSafeCall(cudaFree(d_templateVertexPos));
	cutilSafeCall(cudaFree(d_meshTrans));
	cutilSafeCall(cudaFree(d_meshRot));
	cutilSafeCall(cudaFree(d_prevMeshTrans));
	cutilSafeCall(cudaFree(&d_camRot));
	cutilSafeCall(cudaFree(&d_camTrans));
	cutilSafeCall(cudaFree(&d_prevCamTrans));

	cutilSafeCall(cudaFree(d_numNeighbours));
	cutilSafeCall(cudaFree(d_neighbourIdx));
	cutilSafeCall(cudaFree(d_neighbourOffset));

	SAFE_DELETE(m_optTrackingSolver_Rigid);
	SAFE_DELETE(m_optTrackingSolver_NonRigid);

	delete[] h_templateVertexPos;
	delete[] h_meshTrans;
	delete[] h_meshRot;
	delete[] h_prevMeshTrans;

	delete [] h_numNeighbours;
	delete [] h_neighbourIdx;
	delete [] h_neighbourOffset;
}

void GPUMeshDeformation::solve()
{
	// copyResultsFromDevice();

	resetGPUMemory();
	for (unsigned int i = 1; i < num_iter; i++)
	{
		std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
		m_optTrackingSolver_Rigid->solveGN(d_vertexPosFloat3, d_anglesFloat3, 
			d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, nonlinear_num_iter, 
			linear_num_iter, weightFit, weightReg);
	}

	for (unsigned int i = 1; i < num_iter; i++)
	{
		std::cout << "//////////// ITERATION" << i << "  (OPT) ///////////////" << std::endl;
		m_optTrackingSolver_NonRigid->solveGN(d_vertexPosFloat3, d_anglesFloat3, 
			d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, nonlinear_num_iter, 
			linear_num_iter, weightFit, weightReg);

	}

	copyResultsFromDevice();
}

void GPUMeshDeformation::copyResultsFromDevice()
{
	unsigned int N = (unsigned int)templateMesh->numVertices;

	cutilSafeCall(cudaMemcpy(h_vertexPosFloat3, d_vertexPosFloat3, 
		sizeof(double3)*N, cudaMemcpyDeviceToHost));

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
		
		camPose[0] = h_camRot.x;
		camPose[1] = h_camRot.y;
		camPose[2] = h_camRot.z;
		camPose[3] = h_camTrans.x;
		camPose[4] = h_camTrans.y;
		camPose[5] = h_camTrans.z;
	}
}
