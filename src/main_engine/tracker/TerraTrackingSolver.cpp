#include "main_engine/tracker/TerraTrackingSolver.h"


TerraTrackingSolver::TerraTrackingSolver(unsigned int vertexCount, 
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
	unsigned int dims[] = { vertexCount, 1 };

	m_plan = Opt_ProblemPlan(m_optimizerState, m_problem, dims);

	assert(m_optimizerState);
	assert(m_problem);
	assert(m_plan);


	m_numUnknown = vertexCount;
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

void TerraTrackingSolver::solveGN(
	float3* d_vertexPosFloat3,
	float3* d_anglesFloat3,
	float3* d_vertexPosFloat3Urshape,
	float3* d_vertexPosTargetFloat3,
	unsigned int nNonLinearIterations,
	unsigned int nLinearIterations,
	float weightFit,
	float weightReg)
{
	unsigned int nBlockIterations = 1;	//invalid just as a dummy;

	void* solverParams[] = { &nNonLinearIterations, &nLinearIterations, &nBlockIterations };

	float weightFitSqrt = sqrt(weightFit);
	float weightRegSqrt = sqrt(weightReg);
	
	int * d_zeros = d_headY;		
	void* problemParams[] = { &weightFitSqrt, &weightRegSqrt, d_vertexPosFloat3, d_anglesFloat3, d_vertexPosFloat3Urshape, d_vertexPosTargetFloat3, &edgeCount,  d_headX, d_headY, d_tailX, d_tailY };

	Opt_ProblemSolve(m_optimizerState, m_plan, problemParams, solverParams);
}
