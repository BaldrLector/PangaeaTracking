#include "main_engine/tracker/TerraTrackingSolver.h"


TerraTrackingSolver::TerraTrackingSolver(unsigned int img_width,
	unsigned int img_height, unsigned int vertexCount, 
	unsigned int E, const int* d_xCoords, const int* d_offsets, 
	const std::string& terraFile, const std::string& optName) : 
	m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr)
{
	edgeCount = (int)E;
	m_optimizerState = Opt_NewState();
	m_problem = Opt_ProblemDefine(m_optimizerState, terraFile.c_str(), optName.c_str());


	std::vector<int> yCoords(edgeCount, 0);

	// for (int y = 0; y < (int)edgeCount; ++y) {
	// 	yCoords.push_back(0);
	// }

	d_headY = createDeviceBuffer(yCoords);
	d_tailY = createDeviceBuffer(yCoords);

	int* h_offsets = (int*)malloc(sizeof(int)*(vertexCount + 1));
	cutilSafeCall(cudaMemcpy(h_offsets, d_offsets, sizeof(int)*(vertexCount + 1), cudaMemcpyDeviceToHost));

	int* h_xCoords = (int*)malloc(sizeof(int)*(edgeCount + 1));
	cutilSafeCall(cudaMemcpy(h_xCoords, d_xCoords, sizeof(int)*(edgeCount), cudaMemcpyDeviceToHost));
	h_xCoords[edgeCount] = vertexCount;

	// Convert to our edge format
	std::vector<int> h_headX;
	std::vector<int> h_tailX;
	for (int headX = 0; headX < (int)vertexCount; ++headX) {
		for (int j = h_offsets[headX]; j < h_offsets[headX + 1]; ++j) {
			h_headX.push_back(headX);
			h_tailX.push_back(h_xCoords[j]);
		}
	}

	d_headX = createDeviceBuffer(h_headX);
	d_tailX = createDeviceBuffer(h_tailX);

	//uint32_t dims[] = { vertexCount, 1 };
	unsigned int dims[] = { img_width, img_height, vertexCount };

	m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

	assert(m_optimizerState);
	assert(m_problem);
	assert(m_plan);

	m_numUnknown = vertexCount;

	free(h_offsets);
	free(h_xCoords);
}

TerraTrackingSolver::~TerraTrackingSolver()
{
	cutilSafeCall(cudaFree(d_headX));
	cutilSafeCall(cudaFree(d_headY));
	cutilSafeCall(cudaFree(d_tailX));
	cutilSafeCall(cudaFree(d_tailY));

	if (m_plan) {
		Opt_PlanFree(m_optimizerState, m_plan);
	}

	if (m_problem) {
		Opt_ProblemDelete(m_optimizerState, m_problem);
	}
}


void TerraTrackingSolver::solve(
	float3* d_templateVertexPos,
	float* d_templateVertexColor,
	float* d_image, float* d_gradX_image, float* d_gradY_image,
	float3* d_meshTrans,
	float3* d_meshRot,
	float3 camRot,
	float3 camTrans,
	float3* d_prevMeshTrans,
	int* d_visibility,
	float f_x, float f_y, float u_x, float u_y,
	unsigned int nNonLinearIterations,
	unsigned int nLinearIterations,
	float w_photometric, float w_tv, float w_arap, float w_tempdeform)
{
	unsigned int nBlockIterations = 1;	//invalid just as a dummy;

	void* solverParams[] = { 
		&nNonLinearIterations, &nLinearIterations, &nBlockIterations 
	};

	float w_photometricSqrt = sqrt(w_photometric);
	float w_tvSqrt = sqrt(w_tv);
	float w_arapSqrt = sqrt(w_arap);
	float w_tempdeformSqrt = sqrt(w_tempdeform);

	// void* problemParams[] = { &f_x, &f_y, &u_x, &u_y, 
	// 	&w_photometricSqrt, &w_tvSqrt, &w_arapSqrt, &w_tempdeformSqrt,
	// 	d_meshTrans, d_meshRot, 
	// 	&camRot.x, &camRot.y, &camRot.z, &camTrans.x, &camTrans.y, &camTrans.z,  
	// 	d_image, d_gradX_image, d_gradY_image,
	// 	d_templateVertexColor, d_templateVertexPos,
	// 	d_prevMeshTrans, d_visibility,
	// 	&edgeCount, d_headX, d_headY, d_tailX, d_tailY };
	void* problemParams[] = {&w_photometricSqrt, d_meshTrans, d_templateVertexPos,
		&edgeCount, d_headX, d_headY, d_tailX, d_tailY };
	
	Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
}