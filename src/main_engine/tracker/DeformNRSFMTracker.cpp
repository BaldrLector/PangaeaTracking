#include "main_engine/tracker/DeformNRSFMTracker.h"

baType mapBA(std::string const& inString) {
  if (inString == "mot") return BA_MOT;
  if (inString == "str") return BA_STR;
  if (inString == "motstr") return BA_MOTSTR;
}

dataTermErrorType mapErrorType(std::string const& inString){
  if(inString == "gray") return PE_INTENSITY;
  if(inString == "color") return PE_COLOR;
  if(inString == "depth") return PE_DEPTH;
  if(inString == "depth_plane") return PE_DEPTH_PLANE;
  if(inString == "ncc") return PE_NCC;
  if(inString == "color_ncc") return PE_COLOR_NCC;
  if (inString == "intrinsic") return PE_INTRINSIC;
  if (inString == "intrinsic_color") return PE_INTRINSIC_COLOR;
}

// Linear Solver mapping
ceres::LinearSolverType mapLinearSolver(std::string const& inString){
  if(inString == "SNC") return ceres::SPARSE_NORMAL_CHOLESKY;
  if(inString == "DNC") return ceres::DENSE_NORMAL_CHOLESKY;
  if(inString == "DS") return  ceres::DENSE_SCHUR;
  if(inString == "SS") return  ceres::SPARSE_SCHUR;
  if(inString == "IS") return  ceres::ITERATIVE_SCHUR;
  if(inString == "CG") return  ceres::CGNR;
  if(inString == "DQR") return ceres::DENSE_QR;
}

ceres::MinimizerType mapMinimizerType(std::string const& inString){
  if(inString == "TRUST_REGION") return ceres::TRUST_REGION;
  if(inString == "LINE_SEARCH") return ceres::LINE_SEARCH;
}

ceres::LineSearchDirectionType mapLineSearchDirectionType(std::string const& inString){
  if(inString == "STEEPEST_DESCENT") return ceres::STEEPEST_DESCENT;
  if(inString == "NONLINEAR_CONJUGATE_GRADIENT") return ceres::NONLINEAR_CONJUGATE_GRADIENT;
  if(inString == "LBFGS") return ceres::LBFGS;
  if(inString == "BFGS") return ceres::BFGS;
}

ceres::LineSearchType mapLineSearchType(std::string const& inString){
  if(inString == "ARMIJO") return ceres::ARMIJO;
  if(inString == "WOLFE") return ceres::WOLFE;
}

ceres::LineSearchInterpolationType mapLineSearchInterpType(std::string const& inString){
  if(inString == "BISECTION") return ceres::BISECTION;
  if(inString == "QUADRATIC") return ceres::QUADRATIC;
  if(inString == "CUBIC") return ceres::CUBIC;
}

ceres::NonlinearConjugateGradientType mapNonLinearCGType(std::string const& inString){
  if(inString == "FLETCHER_REEVES") return ceres::FLETCHER_REEVES;
  if(inString == "POLAK_RIBIERE") return ceres::POLAK_RIBIERE;
  if(inString == "HESTENES_STIEFEL") return ceres::HESTENES_STIEFEL;
}

DeformNRSFMTracker::DeformNRSFMTracker(TrackerSettings& settings, int width, int height, double K[3][3],
                                       int startFrame, int numTrackingFrames):
  trackerInitialized(false),
  preProcessingThread(NULL),
  savingThread(NULL),
  dataInBuffer(false),
  useProblemWrapper(false),
  modeGT(false)
{
  BAType = mapBA(settings.baType);
  PEType = mapErrorType(settings.errorType);

  m_nWidth = width;
  m_nHeight = height;
  startFrameNo = startFrame;

  // setting up cameras
  setIntrinsicMatrix(K);
  initializeCamera();

  //create result folder
  cout << settings.savePath << endl;
  if(!bfs::exists( settings.savePath ))
    {
      if(!bfs::create_directories( settings.savePath ))
        {
          std::cerr << "Cannot create result directory: " << settings.savePath << std::endl;
        }
    }

  // debug info output
  std::stringstream ceresOutputPath;
  ceresOutputPath << settings.savePath << settings.ceresOutputFile;
  ceresOutput.open(ceresOutputPath.str().c_str(), std::ofstream::trunc);

  // cout << ceresOutputPath.str() << endl;

  pImagePyramid = new ImagePyramid;
  pFeaturePyramid = new FeaturePyramid;


  if(trackerSettings.printEnergy)
    {
      std::stringstream energyOutputPath;
      energyOutputPath << settings.savePath << settings.energyOutputFile;
      energyOutput.open(energyOutputPath.str().c_str(), std::ofstream::trunc);

      energyOutput << std::left << setw(15) << "Frame" << std::left << setw(15)  <<  "Level" <<
        std::left << setw(15)  << "GT" << std::left << setw(15)  << "DataTerm" <<
        std::left << setw(15)  << "FeatureTerm" << std::left << setw(15)  << "TVTerm" <<
        std::left << setw(15)  << "RotTVTerm" << std::left << setw(15) << "ARAPTerm" <<
        std::left << setw(15)  << "INEXTENTTerm" << std::left << setw(15)  << "DeformTerm" <<
		std::left << setw(15) << "TemporalTerm" << std::left << setw(15) << "SmoothingTerm" << 
		std::left << setw(15) << "SumCost" <<
        std::left << setw(15)  << "TotalCost" << endl;

      std::stringstream errorOutputPath;
      errorOutputPath << settings.savePath << "error_output.txt";
      errorOutput.open(errorOutputPath.str().c_str(), std::ofstream::trunc);

      errorOutput << std::left << setw(15) << "Frame," << std::left << setw(15) << "Level,"
                  << std::left << setw(15) << "Error" << endl;

      std::stringstream energyOutputForRPath;
      energyOutputForRPath << settings.savePath << "energy_output_for_R.txt";
      energyOutputForR.open(energyOutputForRPath.str().c_str(), std::ofstream::trunc);

      std::stringstream errorOutputForRPath;
      errorOutputForRPath << settings.savePath << "error_output_for_R.txt";
      errorOutputForR.open(errorOutputForRPath.str().c_str(), std::ofstream::trunc);

    }

  vector<std::string> temp({"DataTerm", "FeatureTerm", "TVTerm", "RotTVTerm", "ARAPTerm",
	  "INEXTENTTerm", "DeformTerm", "TermporalTerm", "SmoothingTerm", 
	  "TemporalSHCoeffTerm", "SpecularSmoothnessTerm", "SpecularMagnitudeTerm", 
	  "TemporalSpecularTerm", "SumCost", "TotalCost" });
  costNames = std::move(temp);

  meanError = 0;
  if(trackerSettings.hasGT && !settings.scoresPath.empty())
    {
      // we will record the scores of all runned experiments here
      std::stringstream scoresOutputPath;
      scoresOutputPath << settings.scoresPath;

      if( !bfs::exists( scoresOutputPath.str() ) )
        bfs::create_directories( scoresOutputPath.str() );

      scoresOutputPath << "scores_output.txt";

      // prints the file header
      if(!bfs::exists(scoresOutputPath.str()))
        {
          scoresOutput.open(scoresOutputPath.str().c_str(), std::ofstream::trunc);

          scoresOutput << std::left << setw(15) << "dataTermName" << ","
                       << std::left << setw(15) << "featureTermName" << ","
                       << std::left << setw(15) << "patchRadius" << ",";

          for(int i = 0; i < costNames.size() - 2 ; ++i)
            scoresOutput << std::left << setw(15) << costNames[i] << ",";

          scoresOutput << std::left << setw(15) << "error" << endl;

          scoresOutput.close();
        }

      scoresOutput.open(scoresOutputPath.str().c_str(), std::ofstream::app);

    }

  if (PEType == PE_INTRINSIC || PE_INTRINSIC_COLOR)
  {
	  qx_image = qx_allocu_3(m_nHeight, m_nWidth, 3);
	  qx_diffuse_image = qx_allocu_3(m_nHeight, m_nWidth, 3);
  }
  else
  {
	  qx_image = NULL;
	  qx_diffuse_image = NULL;
  }
}

DeformNRSFMTracker::~DeformNRSFMTracker()
{
  if(preProcessingThread != NULL)
    preProcessingThread->join();

  if(savingThread != NULL)
    savingThread->join();

  if(pImagePyramid) delete pImagePyramid;
  if(pFeaturePyramid) delete pFeaturePyramid;

  if(pStrategy) delete pStrategy;

  if (qx_image)
  {
	  qx_freeu_3(qx_image);
	  qx_image = NULL;
  }

  if (qx_diffuse_image)
  {
	  qx_freeu_3(qx_diffuse_image);
	  qx_diffuse_image = NULL;
  }
}

bool DeformNRSFMTracker::setCurrentFrame(int curFrame)
{
  //cannot really do this for a real tracker
  return true;
}

void DeformNRSFMTracker::setIntrinsicMatrix(double K[3][3])
{
  for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 3; ++j)
        {
          KK[i][j] = K[i][j];
        }
    }
}

void DeformNRSFMTracker::initializeCamera()
{
  camInfo.isOrthoCamera = trackerSettings.isOrthoCamera;
  memcpy(camInfo.KK, KK, 9*sizeof(double));
  memset(camInfo.invKK, 0, 9*sizeof(double));
  // set up invKK
  camInfo.invKK[0][0] = 1/KK[0][0];
  camInfo.invKK[1][1] = 1/KK[1][1];
  camInfo.invKK[0][2] = -KK[0][2] * camInfo.invKK[0][0];
  camInfo.invKK[1][2] = -KK[1][2] * camInfo.invKK[1][1];
  camInfo.invKK[2][2] = 1;

  camInfo.width = m_nWidth;
  camInfo.height = m_nHeight;

  memset(camPose, 0, 6*sizeof(double));
}

void DeformNRSFMTracker::setInitialMeshPyramid(PangaeaMeshPyramid& initMeshPyramid)
{

  templateMeshPyramid = std::move(initMeshPyramid);

  m_nMeshLevels = templateMeshPyramid.numLevels;

  // create the optimization strategy and setting up corresponding parameters
  pStrategy = new FreeNeighborStrategy(m_nMeshLevels);

  // set up the optimization strategy
  pStrategy->Initialize();

  // setting parameters
  WeightPara weightPara;
  weightPara.dataTermWeight     = trackerSettings.weightPhotometric;
  weightPara.dataIntensityTermWeight     = trackerSettings.weightPhotometricIntensity;
  weightPara.tvTermWeight       = trackerSettings.weightTV;
  weightPara.tvRotTermWeight    = trackerSettings.weightRotTV;
  weightPara.deformWeight       = trackerSettings.weightDeform;
  weightPara.arapTermWeight     = trackerSettings.weightARAP;
  weightPara.inextentTermWeight = trackerSettings.weightINEXTENT;
  weightPara.transWeight        = trackerSettings.weightTransPrior;
  weightPara.rotWeight = 0;
  weightPara.smoothingTermWeight = trackerSettings.weightSmoothing;
  weightPara.depthTermWeight = trackerSettings.weightDepth;

  //    weightPara.dataHuberWidth = trackerSettings.dataHuberWidth;
  weightPara.dataHuberWidth  = trackerSettings.photometricHuberWidth;
  weightPara.dataIntensityHuberWidth  = trackerSettings.photometricIntensityHuberWidth;
  weightPara.tvHuberWidth    = trackerSettings.tvHuberWidth;
  weightPara.tvRotHuberWidth = trackerSettings.tvRotHuberWidth;
  weightPara.depthHuberWidth = trackerSettings.depthHuberWidth;

  weightPara.featureTermWeight = featureSettings.featureTermWeight;
  weightPara.featureHuberWidth = featureSettings.featureHuberWidth;

  // setting weights of the 0th level of the pyramid
  pStrategy->setWeightParameters(weightPara);

  // setting weighting scales over different levels of the pyramid
  //pStrategy->setWeightScale(trackerSettings.meshVertexNum);
  pStrategy->setWeightScale(templateMeshPyramid.meshPyramidVertexNum);

  pStrategy->setWeightParametersVec();

  // setup propagation pyramid
  // check the way of converting distances to weights
  setupPropagation(
                   templateMeshPyramid,
                   meshPropagation,
                   trackerSettings.meshNeighborNum,
                   trackerSettings.meshNeighborRadius,
                   trackerSettings.meshPyramidUseRadius);

  // now we need to add those different pairs necessarily
  for(int i = 0; i < pStrategy->numOptimizationLevels; ++i)
    {
      vector<std::pair<int, int> >& dataTermPairs = pStrategy->optimizationSettings[i].dataTermPairs;
      vector<std::pair<int, int> >& regTermPairs = pStrategy->optimizationSettings[i].regTermPairs;

      AddMeshToMeshPropagation(
                               templateMeshPyramid,
                               dataTermPairs,
                               meshPropagation,
                               trackerSettings.meshNeighborNum,
                               trackerSettings.meshNeighborRadius);

      AddMeshToMeshPropagation(
                               templateMeshPyramid,
                               regTermPairs,
                               meshPropagation,
                               trackerSettings.meshNeighborNum,
                               trackerSettings.meshNeighborRadius);
    }

  // print everything used in meshPropagation
  for(std::map< pair<int, int>, int>::iterator it = meshPropagation.neighborMap.begin();
      it != meshPropagation.neighborMap.end(); ++it)
    {
      std::cout << "(" << it->first.first << "," << it->first.second << ")"
                << " => " << it->second << '\n';
    }

  // setup patch neighbors
  setupPatchNeighbor(
                     templateMeshPyramid,
                     meshPropagation,
                     trackerSettings.neighborPatchRadius);

  // imagePyramid will be created during the processing of the first image
  pImagePyramid->create(m_nWidth, m_nHeight);
  pImagePyramid->setupCameraPyramid(m_nMeshLevels, camInfo);

  // setup the feature pyramid
  if(trackerSettings.useFeatureImages)
    {
      pFeaturePyramid->create(m_nWidth / featureSettings.scalingFactor,
                              m_nHeight / featureSettings.scalingFactor,
                              featureSettings.channels,
                              m_nMeshLevels);
      pFeaturePyramid->setupCameraPyramid(m_nMeshLevels, camInfo);

      //need to update the number of channels for feature residuals
      PE_RESIDUAL_NUM_ARRAY[PE_FEATURE] = featureSettings.channels;
      PE_RESIDUAL_NUM_ARRAY[PE_FEATURE_NCC] = featureSettings.channels;
    }

  // setup visibilitymask pyramid
  visibilityMaskPyramid.resize(m_nMeshLevels);
  meshTransPyramid.resize(m_nMeshLevels);
  meshRotPyramid.resize(m_nMeshLevels);

  prevMeshTransPyramid.resize(m_nMeshLevels);
  prevMeshRotPyramid.resize(m_nMeshLevels);

  templateAlbedoPyramid.resize(m_nMeshLevels);
  meshSpecularPyramid.resize(m_nMeshLevels);
  prevMeshSpecularPyramid.resize(m_nMeshLevels);
  shCoeff = templateMeshPyramid.levels[0].sh_coefficients;
  prevSHCoeff = templateMeshPyramid.levels[0].sh_coefficients;

  outputInfoPyramid.resize(m_nMeshLevels);
  outputPropPyramid.resize(m_nMeshLevels);

  int imageSize = m_nWidth*m_nHeight;
  for(int i = 0; i < m_nMeshLevels; ++i)
    {

      int numVertices = templateMeshPyramid.levels[i].numVertices;
      visibilityMaskPyramid[i].resize(numVertices,true);

      meshTransPyramid[i].resize(numVertices);
      meshRotPyramid[i].resize(numVertices);

      vector<CoordinateType> zeros3D;
      zeros3D.resize(3); zeros3D[0] = 0; zeros3D[1] = 0; zeros3D[2] = 0;

      meshTransPyramid[i].resize(numVertices, zeros3D);
      meshRotPyramid[i].resize(numVertices, zeros3D);

      prevMeshTransPyramid[i].resize(numVertices, zeros3D);
      prevMeshRotPyramid[i].resize(numVertices, zeros3D);

	  for (int j = 0; j < numVertices; ++j)
	  {
		  meshTransPyramid[i][j].resize(3, 0);
		  meshRotPyramid[i][j].resize(3, 0);
		  prevMeshTransPyramid[i][j].resize(3, 0);
		  prevMeshRotPyramid[i][j].resize(3, 0);
	  }

    if (!trackerSettings.estimate_specularities)
    {
      templateMeshPyramid.levels[i].specular_colors.assign(numVertices, zeros3D);
	}

	meshSpecularPyramid[i] = templateMeshPyramid.levels[i].specular_colors;
	prevMeshSpecularPyramid[i] = templateMeshPyramid.levels[i].specular_colors;

	  templateAlbedoPyramid[i] = templateMeshPyramid.levels[i].colors;

      outputInfoPyramid[i].meshData = templateMeshPyramid.levels[i];
      outputInfoPyramid[i].meshDataColorDiff = templateMeshPyramid.levels[i];
      outputInfoPyramid[i].meshDataColorDiffGT = templateMeshPyramid.levels[i];

      outputInfoPyramid[i].nRenderLevel = i;

      vector<CoordinateType> proj2D;
      proj2D.resize(2); proj2D[0] = 0; proj2D[1] = 0;

      outputInfoPyramid[i].meshProj.resize(numVertices, proj2D);

      outputInfoPyramid[i].meshDataGT = outputInfoPyramid[i].meshData;
      outputInfoPyramid[i].meshProjGT = outputInfoPyramid[i].meshProj;

      outputInfoPyramid[i].visibilityMask.resize(numVertices, true);

      memset(outputInfoPyramid[i].camPose, 0, 6*sizeof(double));

      // get the outputInfo and visibilityMask of the first frame
      UpdateRenderingData(outputInfoPyramid[i], KK, camPose, templateMeshPyramid.levels[i]);

      // initialize outputPropPyramid as the same with outInfoPyramid
      outputPropPyramid[i] = outputInfoPyramid[i];

      // update the visibility of each vertex
      if(trackerSettings.useVisibilityMask)
        {

          long long int ii = i;

          TICK( "visibilityMask" + std::to_string(ii) );

          if(trackerSettings.useOpenGLMask)
            {
              double tempCamPose[6] = {0,0,0,0,0,0};
              UpdateVisibilityMaskGL(outputInfoPyramid[i], visibilityMaskPyramid[i], KK, tempCamPose, m_nWidth, m_nHeight);
            }
          else
            {
              UpdateVisibilityMask(outputInfoPyramid[i], visibilityMaskPyramid[i], m_nWidth, m_nHeight);
            }

          TOCK( "visibilityMask" + std::to_string(ii) );

        }

    }

  // initialize problemWrapper
  problemWrapper.Initialize( pStrategy->numOptimizationLevels );
  // problemWrapper.setOptimizationVariables(camPose,
  //                                         &templateMeshPyramid,
  //                                         &meshRotPyramid,
  //                                         &meshTransPyramid,
  //                                         &prevMeshRotPyramid,
  //                                         &prevMeshTransPyramid);

  //dummy ground truth variables
  meshRotPyramidGT = meshRotPyramid;
  meshTransPyramidGT = meshTransPyramid;
  prevMeshRotPyramidGT = prevMeshRotPyramid;
  prevMeshTransPyramidGT = prevMeshTransPyramid;
  templateMeshPyramidGT = templateMeshPyramid;
  visibilityMaskPyramidGT = visibilityMaskPyramid;

  // load ground truth if there is any
  if(trackerSettings.hasGT)
    initializeGT();
  
  initNeighboursWeightsFineToCoarse();

  trackerInitialized = true;

}

void DeformNRSFMTracker::initializeGT()
{
  loadGTMeshFromFile(startFrameNo);

  meshRotPyramidGT = meshRotPyramid;
  meshTransPyramidGT = meshTransPyramid;
  prevMeshRotPyramidGT = prevMeshRotPyramid;
  prevMeshTransPyramidGT = prevMeshTransPyramid;

  //  problemWrapperGT.Initialize( pStrategy->numOptimizationLevels );
  problemWrapperGT.Initialize( trackerSettings.meshLevelListGT.size() );
  // problemWrapperGT.setOptimizationVariables(camPoseGT,
  //                                           &templateMeshPyramidGT,
  //                                           &meshRotPyramidGT,
  //                                           &meshTransPyramidGT,
  //                                           &prevMeshRotPyramidGT,
  //                                           &prevMeshTransPyramidGT
  //                                           );
  memset(camPoseGT, 0, 6*sizeof(double));
}

void DeformNRSFMTracker::updateGT()
{
  // update previous value
  prevMeshRotPyramidGT = meshRotPyramidGT;
  prevMeshTransPyramidGT = meshTransPyramidGT;

  memcpy(prevCamPoseGT, camPoseGT, 6*sizeof(double));

  // update current frame
  loadGTMeshFromFile(currentFrameNo);

  // have to do rigid registeration to get rigid transformation first
  // e.g. icp with known correspondences
  // just use the top level is good enough

  PangaeaMeshData& templateMeshGT = templateMeshPyramidGT.levels[0];
  PangaeaMeshData& currentMeshGT = currentMeshPyramidGT.levels[0];

  // camPoseGT is composed of rotation + translation
  KnownCorrespondencesICP(templateMeshGT, currentMeshGT, camPoseGT);

  // update meshTransPyramidGT to the correct values
  // and set meshRotPyramidGT to 0

  // update ground truth
  char buffer[BUFFER_SIZE];

  for(int i = 0; i < currentMeshPyramidGT.levels.size(); ++i)
    {
      int numVertices = meshTransPyramidGT[i].size();

      PangaeaMeshData& templateMesh = templateMeshPyramidGT.levels[i];
      PangaeaMeshData& currentMesh = currentMeshPyramidGT.levels[i];
      MeshDeformation& meshTrans = meshTransPyramidGT[i];
      MeshDeformation& meshRot = meshRotPyramidGT[i];

      GetDeformation(templateMesh, currentMesh, camPoseGT, meshTrans, meshRot);

      std::stringstream meshFileGT;
      sprintf(buffer, trackerSettings.meshLevelFormatGT.c_str(), currentFrameNo,
              trackerSettings.meshLevelListGT[i]);

      meshFileGT << trackerSettings.meshPathGT << buffer;

      PangaeaMeshIO::loadfromFile(meshFileGT.str(),
                                  outputInfoPyramid[i].meshDataGT,
                                  trackerSettings.clockwise);

      UpdateRenderingDataFast(outputInfoPyramid[i], KK, outputInfoPyramid[i].meshDataGT, true);

      vector<bool>& visibilityMask = visibilityMaskPyramidGT[i];
      if(trackerSettings.useVisibilityMask)
        {
          if(trackerSettings.useOpenGLMask)
            {
              double tempCamPose[6] = {0,0,0,0,0,0};
              cout << "opengl visibility test" << endl;
              UpdateVisibilityMaskGL(currentMesh, visibilityMask, KK, tempCamPose, m_nWidth, m_nHeight);
            }
          else
            {
              UpdateVisibilityMask(outputInfoPyramid[i], visibilityMask, m_nWidth, m_nHeight, true);
            }
        }

      InternalIntensityImageType* color_image_split = pImagePyramid->getColorImageSplit(i);
      UpdateColorDiffGT(outputInfoPyramid[i], visibilityMask, color_image_split);

    }

}

void DeformNRSFMTracker::loadGTMeshFromFile(int nFrame)
{
  currentMeshPyramidGT = std::move(
                                   PangaeaMeshPyramid(trackerSettings.meshPathGT,
                                                      trackerSettings.meshLevelFormatGT,
                                                      nFrame + trackerSettings.firstFrameGT,
                                                      trackerSettings.meshLevelListGT,
                                                      trackerSettings.clockwise));
}

bool DeformNRSFMTracker::trackFrame(int nFrame, unsigned char* pColorImageRGB,
	TrackerOutputInfo** pOutputInfoRendering,
  bool use_depth, CoordinateType* pDepthImage)
{
  if(!trackerInitialized)
    cout << "this tracker has not been initialized with a template mesh" << endl;

  currentFrameNo = nFrame;

  TICK("imagePreprocessing");

  // prepare data in buffer
  pImagePyramid->setupPyramid(pColorImageRGB, m_nMeshLevels, use_depth, pDepthImage);
  // get new data from buffer
  pImagePyramid->updateData();

  TOCK("imagePreprocessing");

  if(trackerSettings.hasGT)
    updateGT();

  // update camPose of previous frame
  memcpy(prevCamPose, camPose, 6*sizeof(double));

  if(trackerSettings.useFeatureImages && featureSettings.featureTermWeight > 0)
    {

      TICK("featurePreprocessing");

      char buffer[BUFFER_SIZE];
      sprintf(buffer, featureSettings.keyNameFormat.c_str(), currentFrameNo);

      // prepare data in buffer
      pFeaturePyramid->setupPyramid(string(buffer));
      // get new data from buffer
      pFeaturePyramid->updateData();

      AttachFeaturesToMeshPyramid();
      templateMeshPyramid.swapFeatures();
      if(trackerSettings.hasGT)
        templateMeshPyramidGT.swapFeatures();

      TOCK("featurePreprocessing");
    }

  int numOptimizationLevels = pStrategy->numOptimizationLevels;

  // how many levels to do optimization on ?
  for(int i = numOptimizationLevels - 1; i >= 0; --i)
    {

      long long int ii = i;

      TICK( "trackingTimeLevel" + std::to_string(ii) );

      currLevel = i;
      // start tracking
      // create the optimization problem we are trying to solve
      // should make problem a member of DeformNRSFMTracker to
      // avoid the same memory allocation every frame, could take
      // seconds to allocate memory for each frame
      //ceres::Problem problem;
      useProblemWrapper = true;
      ceres::Problem& problem = problemWrapper.getProblem(i);

      // // check if ground truth has been changed after optimization
      // PangaeaMeshData& templateMesh = templateMeshPyramidGT.levels[currLevel];
      // PangaeaMeshData& currentMesh = currentMeshPyramidGT.levels[currLevel];
      // MeshDeformation& meshTrans = meshTransPyramidGT[currLevel];

      // double temp[3];
      // double first_point[3];

      // for(int k = 0; k < 3; ++k)
      //   first_point[k] = templateMesh.vertices[0][k] + meshTrans[0][k];

      // ceres::AngleAxisRotatePoint(camPoseGT, first_point, temp);

      // for(int k = 0; k < 3; ++k)
      //   first_point[k] = temp[k] + camPoseGT[3+k];

      // // check the first point
      // cout << "difference before optimization ";

      // for(int k = 0; k < 3; ++k)
      //   cout << first_point[k] - currentMesh.vertices[0][k] << " ";

      // cout << endl;


      modeGT = false;

      TICK( "trackingTimeLevel" + std::to_string(ii)  + "::ProblemSetup");
      EnergySetup(problem);
      TOCK( "trackingTimeLevel" + std::to_string(ii)  + "::ProblemSetup");

      TICK( "trackingTimeLevel" + std::to_string(ii)  + "::ProblemMinimization");
      EnergyMinimization(problem);
      TOCK( "trackingTimeLevel" + std::to_string(ii)  + "::ProblemMinimization");

      // for(int k = 0; k < 3; ++k)
      //   first_point[k] = templateMesh.vertices[0][k] + meshTrans[0][k];

      // ceres::AngleAxisRotatePoint(camPoseGT, first_point, temp);

      // for(int k = 0; k < 3; ++k)
      //   first_point[k] = temp[k] + camPoseGT[3+k];

      // // check the first point
      // cout << "difference after optimization ";

      // for(int k = 0; k < 3; ++k)
      //   cout << first_point[k] - currentMesh.vertices[0][k] << " ";

      // cout << endl;


      if(useProblemWrapper && trackerSettings.useRGBImages && trackerSettings.weightPhotometric > 0)
        {

          ceresOutput << "number of tracking data terms " << endl
                      << "levels " << currLevel << endl
                      << problemWrapper.getDataTermNum(currLevel) << endl;

          TICK( "trackingTimeLevel" + std::to_string(ii)  + "::RemoveDataTermResidual");

          problemWrapper.clearDataTerm(currLevel);
          problemWrapper.clearDataTermCost(currLevel);
          problemWrapper.clearDataTermLoss(currLevel);

          TOCK( "trackingTimeLevel" + std::to_string(ii)  + "::RemoveDataTermResidual");
        }

      if(useProblemWrapper && trackerSettings.useFeatureImages && featureSettings.featureTermWeight > 0)
        {

          ceresOutput << "number of tracking feature terms " << endl
                      << "levels " << currLevel << endl
                      << problemWrapper.getFeatureTermNum(currLevel) << endl;

          TICK( "trackingTimeLevel" + std::to_string(ii)  + "::RemoveFeatureTermResidual");

          // remove featureTermResidualBlocks from previous frame
          // be careful if we want to use multi-threading
          problemWrapper.clearFeatureTerm(currLevel);
          problemWrapper.clearFeatureTermCost(currLevel);
          problemWrapper.clearFeatureTermLoss(currLevel);

          TOCK( "trackingTimeLevel" + std::to_string(ii)  + "::RemoveFeatureTermResidual");
        }

	  TICK("intrinsicPostprocessing");

	  // estimate intrinsic properties
	  if ((PEType == PE_INTRINSIC || PEType == PE_INTRINSIC_COLOR) && !trackerSettings.estimate_all_together)
	  {
		  if (!trackerSettings.update_intrinsics_finest_only || (trackerSettings.update_intrinsics_finest_only && currLevel == 0))
		  {
			  //Specularities and SH coeff may have changed during minimization because of the regularisation. 
			  //We reset these values
			  if (trackerSettings.refine_all_together)
			  {
				  resetIntrinsics();
			  }

			  updateIntrinsics(pColorImageRGB);

			  if (trackerSettings.refine_all_together)
			  {
				  TICK("trackingTimeLevel" + std::to_string(ii) + "::ProblemSetup");
				  EnergySetup(problem, true);
				  TOCK("trackingTimeLevel" + std::to_string(ii) + "::ProblemSetup");

				  TICK("trackingTimeLevel" + std::to_string(ii) + "::ProblemMinimization");
				  EnergyMinimization(problem, false);
				  TOCK("trackingTimeLevel" + std::to_string(ii) + "::ProblemMinimization");

				  if (useProblemWrapper && trackerSettings.useRGBImages && trackerSettings.weightPhotometric > 0)
				  {

					  ceresOutput << "number of tracking data terms " << endl
						  << "levels " << currLevel << endl
						  << problemWrapper.getDataTermNum(currLevel) << endl;

					  TICK("trackingTimeLevel" + std::to_string(ii) + "::RemoveDataTermResidual");

					  problemWrapper.clearDataTerm(currLevel);
					  problemWrapper.clearDataTermCost(currLevel);
					  problemWrapper.clearDataTermLoss(currLevel);

					  TOCK("trackingTimeLevel" + std::to_string(ii) + "::RemoveDataTermResidual");
				  }
			  }
		  }
	  }

	  if (trackerSettings.use_white_specularities)
	  {
		  fixWhiteSpecularities();
	  }

	  TOCK("intrinsicPostprocessing");

	  // at this point we've finished the optimization on level i
	  // now we need to update all the results and propagate the optimization
	  // results to next level if necessary
	  // the first step is update the current results

	  TICK("trackingTimeLevel" + std::to_string(ii) + "::UpdateResults");
	  UpdateResults();
	  TOCK("trackingTimeLevel" + std::to_string(ii) + "::UpdateResults");

      TICK( "trackingTimeLevel" + std::to_string(ii)  + "::PropagateMesh");
      PropagateMesh();
      TOCK( "trackingTimeLevel" + std::to_string(ii)  + "::PropagateMesh");

      TOCK( "trackingTimeLevel" + std::to_string(ii) );

      if(trackerSettings.hasGT && problemWrapperGT.getLevelsNum() > i)
        {
          //          ceres::Problem problemGT;
          ceres::Problem& problemGT = problemWrapperGT.getProblem(i);

          modeGT = true;

          EnergySetup(problemGT);
          EnergyMinimizationGT(problemGT);

          if(useProblemWrapper && trackerSettings.useRGBImages && trackerSettings.weightPhotometric > 0)
            {
              ceresOutput << "number of ground truth data terms " << endl
                          << "levels " << currLevel << endl
                          << problemWrapperGT.getDataTermNum(currLevel) << endl;
              problemWrapperGT.clearDataTerm(currLevel);
              problemWrapperGT.clearDataTermCost(currLevel);
              problemWrapperGT.clearDataTermLoss(currLevel);
            }

          if(useProblemWrapper && trackerSettings.useFeatureImages && featureSettings.featureTermWeight > 0)
            {
              ceresOutput << "number of ground truth feature terms " << endl
                          << "levels " << currLevel << endl
                          << problemWrapperGT.getFeatureTermNum(currLevel) << endl;
              problemWrapperGT.clearFeatureTerm(currLevel);
              problemWrapperGT.clearFeatureTermCost(currLevel);
              problemWrapperGT.clearFeatureTermLoss(currLevel);
            }

        }


    }

  if(trackerSettings.hasGT)
    {
      for(int i = 0; i < problemWrapperGT.getLevelsNum(); ++i)
        {
          double error = ComputeRMSError(outputInfoPyramid[i].meshData, currentMeshPyramidGT.levels[i]);

          // print error to errorOutputForR
          if(trackerSettings.hasGT)
            {
              errorOutputForR << std::left << setw(15) << currentFrameNo << std::left << setw(15) << i
                              << std::left << setw(15) << error << endl;
              errorOutput << std::left << setw(15) << currentFrameNo << "," << std::left << setw(15) << i
                          << "," << std::left << setw(15) << error << endl;
            }

          if(i == 0)
            meanError += error;
        }
    }

  TICK("updateProp");
  // update the results
  updateRenderingLevel(pOutputInfoRendering, 0);

  // update the top level of propagation result
  outputPropPyramid[m_nMeshLevels-1] = outputInfoPyramid[m_nMeshLevels-1];
  TOCK("updateProp");

  // compute the energy before updating visibilityMasks
  for(int level = 0; level < numOptimizationLevels; ++level)
    {

      long long int ii = level;
      vector<bool>& visibility_mask = visibilityMaskPyramid[level];

      TrackerOutputInfo& output_info = outputInfoPyramid[level];

      // update visibility mask if necessary
      if(trackerSettings.useVisibilityMask)
        {

          TICK( "updateVisbilityMaskLevel" + std::to_string( ii ) );

          if(trackerSettings.useOpenGLMask)
            {
              double tempCamPose[6] = {0,0,0,0,0,0};
              cout << "opengl visibility test" << endl;
              UpdateVisibilityMaskGL(output_info, visibility_mask, KK, tempCamPose, m_nWidth, m_nHeight);
            }
          else
            {
              UpdateVisibilityMask(output_info, visibility_mask, m_nWidth, m_nHeight);
            }

          TOCK( "updateVisbilityMaskLevel" + std::to_string( ii ) );
        }

    }

  //save data
  TICK("SavingTime");

  SaveThread(pOutputInfoRendering);

  TOCK("SavingTime");

  // update previous feature channels to current one
  if(trackerSettings.useFeatureImages && featureSettings.featureTermWeight > 0)
    pFeaturePyramid->updatePrev();

  // if this is the last frame, print out the scores to the file
  if(trackerSettings.hasGT && !trackerSettings.scoresPath.empty() && nFrame == imageSourceSettings.numFrames)
    {
      std::string dataTermName = "not_used";
      std::string featureTermName = "not_used";
      if(trackerSettings.useRGBImages)
        dataTermName = trackerSettings.errorType;
      if(trackerSettings.useFeatureImages)
        featureTermName = featureSettings.useNCC ? "feature_ncc" : "feature";
      scoresOutput << std::left << setw(15) << dataTermName << ","
                   << std::left << setw(15) << featureTermName << ",";

      char buffer[BUFFER_SIZE];
      sprintf(buffer, "patch_radius%02d", trackerSettings.neighborPatchRadius);
      scoresOutput << std::left << setw(15) << buffer << ",";

      // print the all the weights of the terms
      scoresOutput << std::left << setw(15) << trackerSettings.weightPhotometric << ",";
      scoresOutput << std::left << setw(15) << featureSettings.featureTermWeight << ",";

      scoresOutput << std::left << setw(15) << trackerSettings.weightTV << ",";
      scoresOutput << std::left << setw(15) << trackerSettings.weightRotTV << ",";
      scoresOutput << std::left << setw(15) << trackerSettings.weightARAP << ",";
      scoresOutput << std::left << setw(15) << trackerSettings.weightINEXTENT << ",";
      scoresOutput << std::left << setw(15) << trackerSettings.weightDeform << ",";
      scoresOutput << std::left << setw(15) << trackerSettings.weightTransPrior << ",";

      meanError = meanError / (nFrame - imageSourceSettings.startFrame + 1);
      scoresOutput << std::left << setw(15) << meanError << endl;

      scoresOutput.close();
    }

  //Update prev sh coefficients and propagate to coarser levels
  updateSHCoeff();


  // print out memory usage
  double vm, rss;
  process_mem_usage(vm, rss);

  ceresOutput << "--------------- " << endl;
  ceresOutput << "memory usage " << endl;
  ceresOutput << "VM: " << vm << "; RSS: " << rss << endl;
  ceresOutput << "--------------- " << endl;

  return true;
}

void DeformNRSFMTracker::AddVariableMask(ceres::Problem& problem, baType BA)
{
  switch(BA)
    {
    case BA_MOT:
      {
        problem.SetParameterBlockVariable(&camPose[0]);
        problem.SetParameterBlockVariable(&camPose[3]);
        break;
      }
    case BA_STR:
      {
        vector<double*> parameter_blocks;
        problem.GetParameterBlocks(&parameter_blocks);
        // loop over all the parameter blocks
        // and we set all parameter blocks except camPose[0] and camPose[3] constant
        for(int i = 0; i < parameter_blocks.size(); ++i)
          {
            if(parameter_blocks[i] != &camPose[0]
               && parameter_blocks[i] != &camPose[3])
              problem.SetParameterBlockVariable( parameter_blocks[i] );
          }
        break;
      }
    }
}


void DeformNRSFMTracker::AddConstantMask(ceres::Problem& problem, baType BA)
{
  switch(BA)
    {
    case BA_MOT:
      {
        problem.SetParameterBlockConstant(&camPose[0]);
        problem.SetParameterBlockConstant(&camPose[3]);
        break;
      }
    case BA_STR:
      {
        vector<double*> parameter_blocks;
        problem.GetParameterBlocks(&parameter_blocks);
        // loop over all the parameter blocks
        // and we set all parameter blocks except camPose[0] and camPose[3] constant
        for(int i = 0; i < parameter_blocks.size(); ++i)
          {
            if(parameter_blocks[i] != &camPose[0]
               && parameter_blocks[i] != &camPose[3])
              problem.SetParameterBlockConstant( parameter_blocks[i] );
          }
        break;
      }
    }
}

void DeformNRSFMTracker::KnownCorrespondencesICP(PangaeaMeshData& templateMesh,
                                                PangaeaMeshData& currentMesh,
                                                double pose[6])
{

  int numVertices = templateMesh.numVertices;

  ceres::Problem problemICP;

  // ceresOutput << "initial pose " << endl;
  // for(int i = 0; i < 6; ++i)
  //   ceresOutput << pose[i] << " ";
  // ceresOutput << endl;

  for(int i = 0; i < numVertices; ++i)
    {
      ResidualKnownICP* pResidual = new ResidualKnownICP(1,
                                                         &templateMesh.vertices[i][0],
                                                         &currentMesh.vertices[i][0]);

      ceres::AutoDiffCostFunction<ResidualKnownICP, 3, 3, 3>* cost_function =
        new ceres::AutoDiffCostFunction<ResidualKnownICP, 3, 3, 3>(pResidual);

      problemICP.AddResidualBlock(
                                  cost_function,
                                  NULL,
                                  &pose[0],
                                  &pose[3]);

      // ceresOutput << " point " << i << " difference " <<
      //   templateMesh.vertices[i][0] - currentMesh.vertices[i][0] << " " <<
      //   templateMesh.vertices[i][1] - currentMesh.vertices[i][1] << " " <<
      //   templateMesh.vertices[i][2] - currentMesh.vertices[i][2] << endl;

    }

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  options.num_threads = trackerSettings.numThreads;
  options.max_num_iterations = 10;

  ceres::Solve(options, &problemICP, &summary);

  //
  ceresOutput << "+++++++++++++++++" << endl;
  ceresOutput << "ICP Optimization" << endl;
  ceresOutput << "Frame" << " " << currentFrameNo << std::endl;
  ceresOutput << summary.FullReport() << std::endl;
  ceresOutput << "+++++++++++++++++" << endl;

}


void DeformNRSFMTracker::GetDeformation(PangaeaMeshData& templateMesh,
                                        PangaeaMeshData& currentMesh,
                                        double camPose[6],
                                        MeshDeformation& meshTrans,
                                        MeshDeformation& meshRot)
{
  // compute the ground truth deformations
  // deform back the currentMesh to templateMesh
  // and substract template Mesh
  int numVertices = templateMesh.numVertices;

  double rotTemp[3];
  double transTemp[3];
  double result[3];

  double check_rot[3];

  double check_result[3];
  double check_temp[3];

  for(int j = 0; j < numVertices; ++j)
    {
      for(int k = 0; k < 3; ++k)
        {
          meshRot[j][k] = 0;
          rotTemp[k] = 0-camPose[k];
          transTemp[k] = currentMesh.vertices[j][k] - camPose[3+k];
        }
      ceres::AngleAxisRotatePoint(rotTemp, transTemp, result);
      // ceres::AngleAxisRotatePoint(camPose, result, check_rot);
      // cout << "check ceres rotations: " << endl;
      // cout << "before ceres rotation " << transTemp[0]
      //      << " " << transTemp[1] << " " << transTemp[2] << endl;
      // cout << "after ceres rotation " << check_rot[0]
      //      << " " << check_rot[1] << " " << check_rot[2] << endl;

      for(int k = 0; k < 3; ++k)
        {
          meshTrans[j][k] = result[k] - templateMesh.vertices[j][k];
        }

      // for(int k = 0; k < 3; ++k)
      //   check_result[k] = meshTrans[j][k] + templateMesh.vertices[j][k];

      // cout << "check results after deformation";
      // cout << "result "
      //      << result[0] << " "
      //      << result[1] << " "
      //      << result[2] << endl;
      // cout << "check_result "
      //      << check_result[0] << " "
      //      << check_result[1] << " "
      //      << check_result[2] << endl;

      // ceres::AngleAxisRotatePoint(camPose, check_result, check_temp);
      // for(int k = 0; k < 3; ++k)
      //   check_result[k] = check_temp[k] + camPose[3+k] - currentMesh.vertices[j][k];

      // cout << "point difference of " << j << " " << check_result[0] << " "
      //      << check_result[1] << " " << check_result[2] << endl;

    }
}

void DeformNRSFMTracker::UpdateResultsLevel(int level)
{
  PangaeaMeshData& template_mesh = templateMeshPyramid.levels[level];

  MeshDeformation& mesh_trans = meshTransPyramid[level];
  MeshDeformation& prev_mesh_trans = prevMeshTransPyramid[level];

  MeshDeformation& mesh_rot = meshRotPyramid[level];
  MeshDeformation& prev_mesh_rot = prevMeshRotPyramid[level];

  MeshDeformation& prev_mesh_specular = prevMeshSpecularPyramid[level];

  // update output results
  TrackerOutputInfo& output_info = outputInfoPyramid[level];

  long long int ii = level;

  // output result for rendering
  TICK( "updateRenderingLevel" + std::to_string( ii ) );

  UpdateRenderingData(output_info, KK, camPose, template_mesh, mesh_trans);

  // compute normals
  if(trackerSettings.loadMesh)
    output_info.meshData.computeNormalsNeil();
  else
    output_info.meshData.computeNormals();

  TOCK( "updateRenderingLevel" + std::to_string( ii ) );

  vector<bool>& visibility_mask = visibilityMaskPyramid[level];

  // need to update the color diff
  InternalIntensityImageType* color_image_split = pImagePyramid->getColorImageSplit(level);

  UpdateColorDiff(output_info, visibility_mask, color_image_split);

  // update previous deformation
  for(int i = 0; i < mesh_trans.size(); ++i)
    {
      prev_mesh_trans[i][0] = mesh_trans[i][0];
      prev_mesh_trans[i][1] = mesh_trans[i][1];
      prev_mesh_trans[i][2] = mesh_trans[i][2];

      prev_mesh_rot[i][0] = mesh_rot[i][0];
      prev_mesh_rot[i][1] = mesh_rot[i][1];
      prev_mesh_rot[i][2] = mesh_rot[i][2];

	  prev_mesh_specular[i][0] = template_mesh.specular_colors[i][0];
	  prev_mesh_specular[i][1] = template_mesh.specular_colors[i][1];
	  prev_mesh_specular[i][2] = template_mesh.specular_colors[i][2];
    }

}

void DeformNRSFMTracker::UpdateResults()
{
  // a few things need to update
  // update the mesh if we've used any interpolation in the data term
  // update the translation field
  // update the normals of optimized mesh
  // update rendering results

  vector<std::pair<int,int> >& data_pairs =
    pStrategy->optimizationSettings[currLevel].dataTermPairs;
  int num_data_pairs = data_pairs.size();

  for(int i = 0; i < num_data_pairs; ++i)
    {
      std::pair<int, int>& data_pair = data_pairs[i];
      UpdateResultsLevel(data_pair.second);

      cout << "updateResultsLevel " << data_pair.second << endl;

      // if(data_pair.second != data_pair.first)
      // {
      //     UpdateResultsLevel(data_pair.first);
      //     cout << "updateResultsLevel " << data_pair.first << endl;
      // }
    }
}

void DeformNRSFMTracker::PropagateMeshCoarseToFine(int coarse_level, int fine_level)
{

  if(coarse_level == fine_level)
    return;

  MeshDeformation& mesh_rot = meshRotPyramid[coarse_level];
  MeshDeformation& mesh_trans = meshTransPyramid[coarse_level];

  MeshDeformation& mesh_rot_fine = meshRotPyramid[fine_level];
  MeshDeformation& mesh_trans_fine = meshTransPyramid[fine_level];

  pair<int, int> meshPair( fine_level, coarse_level );
  MeshNeighbors&  neighbors = meshPropagation.getNeighbors( meshPair );
  MeshWeights& weights = meshPropagation.getWeights( meshPair );

  PangaeaMeshData& template_coarse_mesh = templateMeshPyramid.levels[coarse_level];
  PangaeaMeshData& template_fine_mesh = templateMeshPyramid.levels[fine_level];

  vector<vector<double>> &local_lighting_coarse = template_coarse_mesh.specular_colors;
  vector<vector<double>> &local_lighting_fine = template_fine_mesh.specular_colors;

  // For Siggraph14 case, we to do propagation from coarse level to next
  // fine level, if rotations are among the optimization variables
  // (arap coeffcient is not zero or rotation is used in data term) we use
  // rotaton to do propagation otherwise do interpolation
  // check if arap coeffcient is zero
  // notice that for dynamicFusion case, this is not consistent with dual quarternion interpolation
  // anyway just for an initialization
  WeightPara& weight_para = pStrategy->weightPara;
  WeightScale& weight_scale = pStrategy->weightScale;

  WeightPara weight_para_level;

  weight_para_level.arapTermWeight = weight_para.arapTermWeight * weight_scale.arapTermScale[currLevel];

  // to be updated, check if rotation is used in data term as well
  if(weight_para_level.arapTermWeight > 0)
    {
      double temp_vertex[3];
      double diff_vertex[3];
      double rot_diff_vertex[3];

      double diff_vertex_bk[3];

      //#pragma omp parallel for
      for(int i = 0; i < template_fine_mesh.numVertices; ++i)
        {
          temp_vertex[0] = 0;
          temp_vertex[1] = 0;
          temp_vertex[2] = 0;
          // find its neighbors in coarse_mesh
          int num_neighbors = neighbors[i].size();
          for(int j = 0; j < num_neighbors; ++j)
            {
              for(int index = 0; index < 3; ++index)
                {
                  diff_vertex[index] = template_fine_mesh.vertices[i][index] - template_coarse_mesh.vertices[ neighbors[i][j] ][index];
                  diff_vertex_bk[index] = diff_vertex[index];
                }

              ceres::AngleAxisRotatePoint(&mesh_rot[ neighbors[i][j] ][0], diff_vertex, rot_diff_vertex);

              for(int index = 0; index < 3; ++index)
                {
                  temp_vertex[index] += weights[i][j] *
                    (rot_diff_vertex[index] + template_coarse_mesh.vertices[ neighbors[i][j] ][index] + mesh_trans[ neighbors[i][j] ][index]);
                }

              // if(std::isnan(weights[i][j]))
              //   {
              //     ceresOutput << "difference " << diff_vertex_bk[0] << " " << diff_vertex_bk[1] << " " << diff_vertex_bk[2] << endl;

              //     ceresOutput << "weights is nan " << j << " " << i << endl;
              //     ceresOutput << "fine_level " << fine_level << " point " << i << " " <<
              //       "coarse_level " << coarse_level << " point " << neighbors[i][j] << endl;
              //     ceresOutput << "fine_level point " << template_fine_mesh.vertices[i][0] << " "
              //                 << template_fine_mesh.vertices[i][1] << " "
              //                 << template_fine_mesh.vertices[i][2] << endl;
              //     ceresOutput << "coarse_level point " << template_coarse_mesh.vertices[ neighbors[i][j]  ][0] << " "
              //                 << template_coarse_mesh.vertices[ neighbors[i][j] ][1] << " "
              //                 << template_coarse_mesh.vertices[ neighbors[i][j] ][2] << endl;

              //   }

            }

          for(int j = 0; j < 3; ++j)
            mesh_trans_fine[i][j] = temp_vertex[j] - template_fine_mesh.vertices[i][j];

          // need to update the rotations of the fine mesh as well
          // compute the rigid transformation between two sets of
          // neighboring points(both are defined on the fine mesh)
          // notice that for siggraph14 optimization, the arap edges
          // are defined on the same level
          //
          vector<double> arap_weights;
          arap_weights.resize( template_fine_mesh.adjVerticesInd[i].size(), 1 );

          computeRot(
                     template_fine_mesh.vertices[i],
                     mesh_trans_fine[i],
                     template_fine_mesh.vertices,
                     mesh_trans_fine,
                     template_fine_mesh.adjVerticesInd[i],
                     arap_weights,
                     mesh_rot_fine[i],
                     true);

        }

    }
  else
    {
	  bool update_specular = trackerSettings.estimate_all_together 
		  || trackerSettings.estimate_sh_coeff_specular_together || trackerSettings.estimate_specularities 
		  || trackerSettings.refine_all_together;

	  bool update_sh_coeff = trackerSettings.estimate_all_together 
		  || trackerSettings.estimate_sh_coeff_specular_together || trackerSettings.estimate_with_sh_coeff 
		  || trackerSettings.refine_all_together;

      // just do interpolation
      for(int i = 0; i < template_fine_mesh.numVertices; ++i)
        {
          mesh_trans_fine[i][0] = 0;
          mesh_trans_fine[i][1] = 0;
          mesh_trans_fine[i][2] = 0;

          if (update_specular)
          {
            local_lighting_fine[i][0] = 0;
            local_lighting_fine[i][1] = 0;
            local_lighting_fine[i][2] = 0;
          }

          // find its neighbors in coarse_mesh
          int num_neighbors = neighbors[i].size();
          for(int j = 0; j < num_neighbors; ++j)
            {
              for(int index = 0; index < 3; ++index)
                {
                  mesh_trans_fine[i][index] += weights[i][j] * mesh_trans[ neighbors[i][j] ][index];

                  if (update_specular)
                  {
                    local_lighting_fine[i][index] += weights[i][j] * local_lighting_coarse[ neighbors[i][j] ][index];
                  }
                }
            }
        }

	  meshSpecularPyramid[fine_level] = local_lighting_fine;

        if (update_sh_coeff)
        {
          template_fine_mesh.sh_coefficients = template_coarse_mesh.sh_coefficients;
		  shCoeff = template_fine_mesh.sh_coefficients;
        }

    }
}

void DeformNRSFMTracker::PropagateMesh()
{

  // propagate the mesh from coarse level to fine level,
  // nothing needs to be done if this is the lowest level
  vector<std::pair<int, int> >& prop_pairs =
    pStrategy->optimizationSettings[currLevel].propPairs;
  int num_prop_pairs = prop_pairs.size();

  for(int k = 0; k < num_prop_pairs; ++k)
    {
      std::pair<int, int>& prop_pair = prop_pairs[k];
      int coarse_level = prop_pair.first;
      int fine_level = prop_pair.second;

      //
      // ceresOutput << "before propagation" << endl;
      // CheckNaN();

      PropagateMeshCoarseToFine(coarse_level, fine_level);

      // after propagation, we update the correponding mesh
      UpdateResultsLevel(fine_level);

      // ceresOutput << "after propagation" << endl;
      // CheckNaN();

      // copy over the mesh right after propagation to outputPropPyramid
      outputPropPyramid[fine_level] = outputInfoPyramid[fine_level];

      cout << "prop pairs" << endl;
      cout << coarse_level << "->" << fine_level << endl;

      cout << "updateResultsLevel " << fine_level << endl;

    }

  // if this is level zero, we propagate the mesh to all levels below
  if( currLevel == 0)
    {
      vector<std::pair<int, int> >& final_pairs = pStrategy->propPairsFinal;
      cout << "final prop" << endl;
      for(int k = 0; k < final_pairs.size(); ++k)
        {
          PropagateMeshCoarseToFine(final_pairs[k].first, final_pairs[k].second);
          UpdateResultsLevel(final_pairs[k].second);
          cout << final_pairs[k].first << "->" << final_pairs[k].second << endl;
        }
    }

}

void DeformNRSFMTracker::AddCostImageProjection(ceres::Problem& problem,
                                                ceres::LossFunction* loss_function,
                                                dataTermErrorType errorType,
                                                PangaeaMeshData& templateMesh,
                                                MeshDeformation& meshTrans,
                                                vector<bool>& visibilityMask,
                                                CameraInfo* pCamera,
                                                Level* pFrame)
{

  for(int i = 0; i < templateMesh.numVertices; ++i){

    if(visibilityMask[i])
      {

        switch(errorType)
          {
          case PE_INTENSITY:

            {
              ResidualImageProjection* pResidual = new ResidualImageProjection(1,
                                                                               &templateMesh.grays[i],
                                                                               &templateMesh.vertices[i][0],
                                                                               pCamera,
                                                                               pFrame,
                                                                               errorType);

              ceres::AutoDiffCostFunction<ResidualImageProjection, 1, 3, 3, 3>* cost_function =
                new ceres::AutoDiffCostFunction<ResidualImageProjection, 1, 3, 3, 3>( pResidual );

              // add the residualBlockId to problemWrapper
              ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                                cost_function,
                                                                                loss_function,
                                                                                modeGT ? &camPoseGT[0] : &camPose[0],
                                                                                modeGT ? &camPoseGT[3] : &camPose[3],
                                                                                &meshTrans[i][0]);

              if(useProblemWrapper)
                {
                  if(modeGT)
                    {
                      problemWrapperGT.addDataTerm(currLevel, residualBlockId);
                      problemWrapperGT.addDataTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addDataTerm(currLevel, residualBlockId);
                      problemWrapper.addDataTermCost(currLevel, cost_function);
                    }

                }

            }

            break;

          case PE_COLOR:
          case PE_DEPTH:
          case PE_DEPTH_PLANE:

            {
              ResidualImageProjection* pResidual = new ResidualImageProjection(1,
                                                                               &templateMesh.colors[i][0],
                                                                               &templateMesh.vertices[i][0],
                                                                               pCamera,
                                                                               pFrame,
                                                                               errorType);

              ceres::AutoDiffCostFunction<ResidualImageProjection, 3, 3, 3, 3>* cost_function =
                new ceres::AutoDiffCostFunction<ResidualImageProjection, 3, 3, 3, 3>( pResidual );

              ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                                cost_function,
                                                                                loss_function,
                                                                                modeGT ? &camPoseGT[0] : &camPose[0],
                                                                                modeGT ? &camPoseGT[3] : &camPose[3],
                                                                                &meshTrans[i][0]);

              if(useProblemWrapper)
                {
                  if(modeGT)
                    {
                      problemWrapperGT.addDataTerm(currLevel, residualBlockId);
                      problemWrapperGT.addDataTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addDataTerm(currLevel, residualBlockId);
                      problemWrapper.addDataTermCost(currLevel, cost_function);
                    }

                }

            }

            break;

          case PE_FEATURE:

            {

              ResidualImageProjectionDynamic* pResidual = new ResidualImageProjectionDynamic(1,
                                                                                             &templateMesh.features[i][0],
                                                                                             &templateMesh.vertices[i][0],
                                                                                             pCamera,
                                                                                             pFrame,
                                                                                             errorType);

              ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionDynamic, 5>* cost_function =
                new ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionDynamic, 5>( pResidual );

              vector<double*> parameter_blocks;
              parameter_blocks.push_back( modeGT ? &camPoseGT[0] : &camPose[0]);
              parameter_blocks.push_back( modeGT ? &camPoseGT[3] : &camPose[3]);
              parameter_blocks.push_back( &meshTrans[i][0] );

              // add rotation, translation and vertice parameter block
              cost_function->AddParameterBlock(3);
              cost_function->AddParameterBlock(3);
              cost_function->AddParameterBlock(3);

              cost_function->SetNumResiduals( PE_RESIDUAL_NUM_ARRAY[errorType] );

              ceres::ResidualBlockId residualBlockId =  problem.AddResidualBlock(
                                                                                 cost_function,
                                                                                 loss_function,
                                                                                 parameter_blocks);
              if(useProblemWrapper)
                {
                  if(modeGT)
                    {
                      problemWrapperGT.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapperGT.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapper.addFeatureTermCost(currLevel, cost_function);
                    }

                }

              break;

            }
          }


      }

  }

}

void DeformNRSFMTracker::AddCostImageProjection(ceres::Problem& problem,
	ceres::LossFunction* loss_function,
	dataTermErrorType errorType,
	PangaeaMeshData& templateMesh,
	MeshDeformation& meshTrans,
	vector<bool>& visibilityMask,
	CameraInfo* pCamera,
	Level* pFrame,
	MeshDeformation& local_lighting,
	bool refinement)
{
	vector<double> &sh_coeff = templateMesh.sh_coefficients;
	int sh_order = templateMesh.sh_order;

	for (int i = 0; i < templateMesh.numVertices; ++i){

		if (visibilityMask[i])
		{
			if (trackerSettings.estimate_all_together || refinement)
			{
				// Dynamic photometric cost function
				ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionIntrinsic, 5>* dyn_cost_function
					= new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionIntrinsic, 5 >(
					new ResidualImageProjectionIntrinsic(1,
					errorType == PE_INTRINSIC ? &templateMesh.grays[i] : &templateMesh.colors[i][0],
					&templateMesh.vertices[i][0], pCamera, pFrame, templateMesh.vertices,
					templateMesh.adjVerticesInd[i], templateMesh.adjFacesInd[i].size(),
					errorType, trackerSettings.clockwise, sh_order, trackerSettings.use_white_specularities)
					);

				// List of pointers to translations per vertex
				vector<double*> v_parameter_blocks;

				// SH Coefficients
				dyn_cost_function->AddParameterBlock(sh_coeff.size());
				v_parameter_blocks.push_back(&sh_coeff[0]);

				// Rigid rotation
				dyn_cost_function->AddParameterBlock(3);
				v_parameter_blocks.push_back(&camPose[0]);
				// Rigid translation
				dyn_cost_function->AddParameterBlock(3);
				v_parameter_blocks.push_back(&camPose[3]);

				// Local lighting
				unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;
				dyn_cost_function->AddParameterBlock(specular_n_channels);
				v_parameter_blocks.push_back(&local_lighting[i][0]);

				// Local translations
				v_parameter_blocks.push_back(&meshTrans[i][0]);
				dyn_cost_function->AddParameterBlock(3);
				for (int j = 0; j < templateMesh.adjVerticesInd[i].size(); j++)
				{
					int v_idx = templateMesh.adjVerticesInd[i][j];
					v_parameter_blocks.push_back(&meshTrans[v_idx][0]);
					dyn_cost_function->AddParameterBlock(3);
				}

				dyn_cost_function->SetNumResiduals(PE_RESIDUAL_NUM_ARRAY[errorType]);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					dyn_cost_function,
					loss_function,
					v_parameter_blocks);

				if (useProblemWrapper)
				{
					if (modeGT)
					{
						problemWrapperGT.addDataTerm(currLevel, residualBlockId);
						problemWrapperGT.addDataTermCost(currLevel, dyn_cost_function);
					}
					else
					{
						problemWrapper.addDataTerm(currLevel, residualBlockId);
						problemWrapper.addDataTermCost(currLevel, dyn_cost_function);
					}
				}
			}
			else
			{
				if (trackerSettings.estimate_with_sh_coeff)
				{
					// Dynamic photometric cost function
					ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionIntrinsicSpec, 5>* dyn_cost_function
						= new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionIntrinsicSpec, 5 >(
						new ResidualImageProjectionIntrinsicSpec(1,
						errorType == PE_INTRINSIC ? &templateMesh.grays[i] : &templateMesh.colors[i][0],
						&templateMesh.vertices[i][0], pCamera, pFrame, templateMesh.vertices,
						templateMesh.adjVerticesInd[i], templateMesh.adjFacesInd[i].size(),
						errorType, local_lighting[i], trackerSettings.clockwise, sh_order, 
							trackerSettings.use_white_specularities)
						);

					// List of pointers to translations per vertex
					vector<double*> v_parameter_blocks;

					// SH Coefficients
					dyn_cost_function->AddParameterBlock(sh_coeff.size());
					v_parameter_blocks.push_back(&sh_coeff[0]);

					// Rigid rotation
					dyn_cost_function->AddParameterBlock(3);
					v_parameter_blocks.push_back(&camPose[0]);
					// Rigid translation
					dyn_cost_function->AddParameterBlock(3);
					v_parameter_blocks.push_back(&camPose[3]);

					// Local translations
					v_parameter_blocks.push_back(&meshTrans[i][0]);
					dyn_cost_function->AddParameterBlock(3);
					for (int j = 0; j < templateMesh.adjVerticesInd[i].size(); j++)
					{
						int v_idx = templateMesh.adjVerticesInd[i][j];
						v_parameter_blocks.push_back(&meshTrans[v_idx][0]);
						dyn_cost_function->AddParameterBlock(3);
					}

					dyn_cost_function->SetNumResiduals(PE_RESIDUAL_NUM_ARRAY[errorType]);

					ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
						dyn_cost_function,
						loss_function,
						v_parameter_blocks);

					if (useProblemWrapper)
					{
						if (modeGT)
						{
							problemWrapperGT.addDataTerm(currLevel, residualBlockId);
							problemWrapperGT.addDataTermCost(currLevel, dyn_cost_function);
						}
						else
						{
							problemWrapper.addDataTerm(currLevel, residualBlockId);
							problemWrapper.addDataTermCost(currLevel, dyn_cost_function);
						}
					}
				}
				else // Known sh coeff and specularities
				{
					// Dynamic photometric cost function
					ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionIntrinsicSHSpec, 5>* dyn_cost_function
						= new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionIntrinsicSHSpec, 5 >(
						new ResidualImageProjectionIntrinsicSHSpec(1,
						errorType == PE_INTRINSIC ? &templateMesh.grays[i] : &templateMesh.colors[i][0],
						&templateMesh.vertices[i][0], pCamera, pFrame, templateMesh.vertices,
						templateMesh.adjVerticesInd[i], templateMesh.adjFacesInd[i].size(),
						errorType, sh_coeff, local_lighting[i],
						trackerSettings.clockwise, sh_order, trackerSettings.use_white_specularities)
						);

					// List of pointers to translations per vertex
					vector<double*> v_parameter_blocks;

					// Rigid rotation
					dyn_cost_function->AddParameterBlock(3);
					v_parameter_blocks.push_back(&camPose[0]);
					// Rigid translation
					dyn_cost_function->AddParameterBlock(3);
					v_parameter_blocks.push_back(&camPose[3]);

					// Local translations
					v_parameter_blocks.push_back(&meshTrans[i][0]);
					dyn_cost_function->AddParameterBlock(3);
					for (int j = 0; j < templateMesh.adjVerticesInd[i].size(); j++)
					{
						int v_idx = templateMesh.adjVerticesInd[i][j];
						v_parameter_blocks.push_back(&meshTrans[v_idx][0]);
						dyn_cost_function->AddParameterBlock(3);
					}

					dyn_cost_function->SetNumResiduals(PE_RESIDUAL_NUM_ARRAY[errorType]);

					ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
						dyn_cost_function,
						loss_function,
						v_parameter_blocks);

					if (useProblemWrapper)
					{
						if (modeGT)
						{
							problemWrapperGT.addDataTerm(currLevel, residualBlockId);
							problemWrapperGT.addDataTermCost(currLevel, dyn_cost_function);
						}
						else
						{
							problemWrapper.addDataTerm(currLevel, residualBlockId);
							problemWrapper.addDataTermCost(currLevel, dyn_cost_function);
						}
					}
				}
			}    
		}
	}
}

void DeformNRSFMTracker::AddCostImageProjectionPatch(ceres::Problem& problem,
                                                     ceres::LossFunction* loss_function,
                                                     dataTermErrorType errorType,
                                                     PangaeaMeshData& templateMesh,
                                                     MeshDeformation& meshTrans,
                                                     vector<bool>& visibilityMask,
                                                     MeshNeighbors& patchNeighbors,
                                                     MeshNeighbors& patchRadii,
                                                     MeshWeights& patchWeights,
                                                     CameraInfo* pCamera,
                                                     Level* pFrame)
{

  vector<double> patchWeightsI;
  vector<unsigned int> patchRadiiI;
  vector<unsigned int> patchNeighborsI;

  for(int i = 0; i < templateMesh.numVertices; ++i)
    {
      if(visibilityMask[i])
        {

          patchWeightsI.clear();
          patchRadiiI.clear();
          patchNeighborsI.clear();

          for(int k = 0; k < patchNeighbors[i].size(); ++k)
            {
              if(visibilityMask[ patchNeighbors[i][k] ])
                {
                  patchWeightsI.push_back( patchWeights[i][k] );
                  patchRadiiI.push_back( patchRadii[i][k] );
                  patchNeighborsI.push_back( patchNeighbors[i][k] );
                }
            }

          // int numNeighbors = patchNeighbors[i].size();

          // vector<double*> parameter_blocks;
          // for(int j = 0; j < numNeighbors; ++j)
          //   parameter_blocks.push_back( &(meshTrans[ patchNeighbors[i][j] ][0]) );

          // parameter_blocks.push_back( modeGT ? &camPoseGT[0] : &camPose[0] );
          // parameter_blocks.push_back( modeGT ? &camPoseGT[3] : &camPose[3] );

          // ResidualImageProjectionPatch* pResidualPatch = new ResidualImageProjectionPatch(1,
          //                                                                                 &templateMesh,
          //                                                                                 pCamera,
          //                                                                                 pFrame,
          //                                                                                 numNeighbors,
          //                                                                                 patchWeights[i],
          //                                                                                 patchRadii[i],
          //                                                                                 patchNeighbors[i],
          //                                                                                 errorType );

          int numNeighbors = patchNeighborsI.size();

          vector<double*> parameter_blocks;
          for(int j = 0; j < numNeighbors; ++j)
            parameter_blocks.push_back( &(meshTrans[ patchNeighborsI[j] ][0]) );

          parameter_blocks.push_back( modeGT ? &camPoseGT[0] : &camPose[0] );
          parameter_blocks.push_back( modeGT ? &camPoseGT[3] : &camPose[3] );

          ResidualImageProjectionPatch* pResidualPatch = new ResidualImageProjectionPatch(1,
                                                                                          &templateMesh,
                                                                                          pCamera,
                                                                                          pFrame,
                                                                                          numNeighbors,
                                                                                          patchWeightsI,
                                                                                          patchRadiiI,
                                                                                          patchNeighborsI,
                                                                                          errorType );

          ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionPatch, 5>* cost_function =
            new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionPatch, 5>( pResidualPatch );

          for(int j = 0; j < numNeighbors; ++j)
            cost_function->AddParameterBlock(3);

          cost_function->AddParameterBlock(3);
          cost_function->AddParameterBlock(3);

          cost_function->SetNumResiduals( PE_RESIDUAL_NUM_ARRAY[errorType] );

          ceres::ResidualBlockId residualBlockId =  problem.AddResidualBlock(
                                                                             cost_function,
                                                                             loss_function,
                                                                             parameter_blocks);
          if(useProblemWrapper)
            {
              if(modeGT)
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapperGT.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapperGT.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapperGT.addDataTerm(currLevel, residualBlockId);
                      problemWrapperGT.addDataTermCost(currLevel, cost_function);
                    }
                }
              else
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapper.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapper.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addDataTerm(currLevel, residualBlockId);
                      problemWrapper.addDataTermCost(currLevel, cost_function);
                    }
                }
            }

        }

    }

}

void DeformNRSFMTracker::AddCostImageProjectionCoarse(ceres::Problem& problem,
                                                      ceres::LossFunction* loss_function,
                                                      dataTermErrorType errorType,
                                                      PangaeaMeshData& templateMesh,
                                                      vector<bool>& visibilityMask,
                                                      PangaeaMeshData& templateNeighborMesh,
                                                      MeshDeformation& neighborMeshTrans,
                                                      MeshDeformation& neighborMeshRot,
                                                      MeshNeighbors& neighbors,
                                                      MeshWeights& weights,
                                                      CameraInfo* pCamera,
                                                      Level* pFrame)
{
  double* pValue = NULL;

  for(int i = 0; i < templateMesh.numVertices; ++i)
    {
      if(visibilityMask[i])
        {
          vector<double*> neighborVertices;
          vector<double> neighborWeights;
          vector<double*> parameter_blocks;

          int numNeighbors = neighbors[i].size();
          for(int j = 0; j < numNeighbors; ++j )
            {
              neighborWeights.push_back( weights[i][j] );
              neighborVertices.push_back( &( templateNeighborMesh.vertices[ neighbors[i][j] ][ 0 ] ) );
              parameter_blocks.push_back( &( neighborMeshTrans[ neighbors[i][j] ][0] ) );
            }
          for(int j = 0; j < numNeighbors; ++j )
            parameter_blocks.push_back( &( neighborMeshRot[ neighbors[i][j] ][0] ) );

          parameter_blocks.push_back( modeGT ? &camPoseGT[0] : &camPose[0] );
          parameter_blocks.push_back( modeGT ? &camPoseGT[3] : &camPose[3] );

          getValueFromMesh(&templateMesh, errorType, i, &pValue);

          ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionCoarse, 5>* cost_function =
            new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionCoarse, 5 >(
                                                                                       new ResidualImageProjectionCoarse(
                                                                                                                         1,
                                                                                                                         pValue,
                                                                                                                         &templateMesh.vertices[i][0],
                                                                                                                         pCamera,
                                                                                                                         pFrame,
                                                                                                                         numNeighbors,
                                                                                                                         neighborWeights,
                                                                                                                         neighborVertices,
                                                                                                                         errorType ) );

          for(int j = 0; j < 2*numNeighbors; ++j)
            cost_function->AddParameterBlock(3);

          cost_function->AddParameterBlock(3);
          cost_function->AddParameterBlock(3);

          cost_function->SetNumResiduals( PE_RESIDUAL_NUM_ARRAY[errorType] );

          ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                            cost_function,
                                                                            loss_function,
                                                                            parameter_blocks);

          if(useProblemWrapper)
            {
              if(modeGT)
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapperGT.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapperGT.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapperGT.addDataTerm(currLevel, residualBlockId);
                      problemWrapperGT.addDataTermCost(currLevel, cost_function);
                    }
                }
              else
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapper.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapper.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addDataTerm(currLevel, residualBlockId);
                      problemWrapper.addDataTermCost(currLevel, cost_function);
                    }
                }
            }

        }
    }

}

void DeformNRSFMTracker::AddCostImageProjectionPatchCoarse(ceres::Problem& problem,
                                                           ceres::LossFunction* loss_function,
                                                           dataTermErrorType& errorType,
                                                           PangaeaMeshData& templateMesh,
                                                           vector<bool>& visibilityMask,
                                                           MeshNeighbors& patchNeighbors,
                                                           MeshNeighbors& patchRadii,
                                                           MeshWeights& patchWeights,
                                                           PangaeaMeshData& templateNeighborMesh,
                                                           MeshDeformation& neighborMeshTrans,
                                                           MeshDeformation& neighborMeshRot,
                                                           MeshNeighbors& neighbors,
                                                           MeshWeights& weights,
                                                           CameraInfo* pCamera,
                                                           Level* pFrame)
{

  vector<double> patchWeightsI;
  vector<unsigned int> patchRadiiI;
  vector<unsigned int> patchNeighborsI;

  for(int i = 0; i < templateMesh.numVertices; ++i)
    {
      if(visibilityMask[i])
        {
          patchWeightsI.clear();
          patchRadiiI.clear();
          patchNeighborsI.clear();

          for(int k = 0; k < patchNeighbors[i].size(); ++k)
            {
              if(visibilityMask[ patchNeighbors[i][k] ])
                {
                  patchWeightsI.push_back( patchWeights[i][k] );
                  patchRadiiI.push_back( patchRadii[i][k] );
                  patchNeighborsI.push_back( patchNeighbors[i][k] );
                }
            }

          // patch neighbors
          //          int numNeighbors = patchNeighbors[i].size();
          int numNeighbors = patchNeighborsI.size();

          //coarse neighbors;
          int numCoarseNeighbors;
          vector<unsigned int> parameterIndices;
          vector<unsigned int> coarseNeighborIndices;
          vector<unsigned int> coarseNeighborBiases;
          vector<double> coarseNeighborWeights;

          vector<double*> parameter_blocks;
          vector<double*> parameter_blocks_rot;

          int bias = 0;
          vector<double*>::iterator iter;
          for(int j = 0; j < numNeighbors; ++j)
            {
              //              int m = patchNeighbors[i][j];
              int m = patchNeighborsI[j];
              int coarseNum = neighbors[ m ].size();

              bias += coarseNum;
              coarseNeighborBiases.push_back( bias );

              // cout << "coarse neighbors of point ";
              // for(int k = 0; k < coarseNum; ++k)
              //   cout << neighbors[m][k] << " ";
              // cout << endl;

              for(int k = 0; k < coarseNum; ++k)
                {
                  coarseNeighborIndices.push_back( neighbors[m][k] );
                  coarseNeighborWeights.push_back( weights[m][k] );

                  // if not in parameter_blocks yet
                  double* block = &( neighborMeshTrans[ neighbors[m][k] ][0] );
                  double* block_rot = &( neighborMeshRot[ neighbors[m][k] ][0] );

                  iter = std::find(parameter_blocks.begin(), parameter_blocks.end(), block);
                  if(iter == parameter_blocks.end())
                    {
                      parameter_blocks.push_back( block );
                      parameter_blocks_rot.push_back( block_rot );
                      parameterIndices.push_back( parameter_blocks.size() - 1 );
                    }
                  else
                    parameterIndices.push_back( iter - parameter_blocks.begin() );

                }
            }

          numCoarseNeighbors = parameter_blocks.size();

          for(int k = 0; k < parameter_blocks_rot.size(); ++k)
            parameter_blocks.push_back( parameter_blocks_rot[k] );

          parameter_blocks.push_back( modeGT ? &camPoseGT[0] : &camPose[0] );
          parameter_blocks.push_back( modeGT ? &camPoseGT[3] : &camPose[3] );

          // ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionPatchCoarse, 5>* cost_function =
          //   new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionPatchCoarse, 5 >(
          //                                                                                   new ResidualImageProjectionPatchCoarse(
          //                                                                                                                          1,
          //                                                                                                                          &templateMesh,
          //                                                                                                                          &templateNeighborMesh,
          //                                                                                                                          pCamera,
          //                                                                                                                          pFrame,
          //                                                                                                                          numNeighbors,
          //                                                                                                                          numCoarseNeighbors,
          //                                                                                                                          patchWeights[i],
          //                                                                                                                          patchRadii[i],
          //                                                                                                                          patchNeighbors[i],
          //                                                                                                                          parameterIndices,
          //                                                                                                                          coarseNeighborIndices,
          //                                                                                                                          coarseNeighborBiases,
          //                                                                                                                          coarseNeighborWeights,
          //                                                                                                                          errorType ) );

          ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionPatchCoarse, 5>* cost_function =
            new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionPatchCoarse, 5 >(
                                                                                            new ResidualImageProjectionPatchCoarse(
                                                                                                                                   1,
                                                                                                                                   &templateMesh,
                                                                                                                                   &templateNeighborMesh,
                                                                                                                                   pCamera,
                                                                                                                                   pFrame,
                                                                                                                                   numNeighbors,
                                                                                                                                   numCoarseNeighbors,
                                                                                                                                   patchWeightsI,
                                                                                                                                   patchRadiiI,
                                                                                                                                   patchNeighborsI,
                                                                                                                                   parameterIndices,
                                                                                                                                   coarseNeighborIndices,
                                                                                                                                   coarseNeighborBiases,
                                                                                                                                   coarseNeighborWeights,
                                                                                                                                   errorType ) );

          for(int j = 0; j < 2*numCoarseNeighbors; ++j)
            cost_function->AddParameterBlock(3);

          cost_function->AddParameterBlock(3);
          cost_function->AddParameterBlock(3);

          cost_function->SetNumResiduals( PE_RESIDUAL_NUM_ARRAY[errorType] );

          ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                            cost_function,
                                                                            loss_function,
                                                                            parameter_blocks);

          if(useProblemWrapper)
            {
              if(modeGT)
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapperGT.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapperGT.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapperGT.addDataTerm(currLevel, residualBlockId);
                      problemWrapperGT.addDataTermCost(currLevel, cost_function);
                    }
                }
              else
                {
                  if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC)
                    {
                      problemWrapper.addFeatureTerm(currLevel, residualBlockId);
                      problemWrapper.addFeatureTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addDataTerm(currLevel, residualBlockId);
                      problemWrapper.addDataTermCost(currLevel, cost_function);
                    }
                }
            }

        }

    }

}

void DeformNRSFMTracker::AddPhotometricCostNew(ceres::Problem& problem,
                                               ceres::LossFunction* loss_function,
                                               dataTermErrorType errorType,
											   bool refinement)
{
  // add photometric cost
  // there are two different cases
  vector<std::pair<int,int> >& data_pairs =
    pStrategy->optimizationSettings[currLevel].dataTermPairs;

  int num_data_pairs = data_pairs.size();

  CameraInfo* pCamera;
  Level* pFrame;

  for(int k = 0; k < num_data_pairs; ++k)
    {
      std::pair<int, int>& data_pair = data_pairs[k];

      if(errorType == PE_FEATURE || errorType == PE_FEATURE_NCC){
        pCamera = &pFeaturePyramid->getCameraInfo(data_pair.first);
        pFrame = &pFeaturePyramid->getCurrFeatureLevel(data_pair.first);
      }else{
        pCamera = &pImagePyramid->getCameraInfo(data_pair.first);
        pFrame = &pImagePyramid->getImageLevel(data_pair.first);
      }

      cout << "camera width and height " << pCamera->width << " " << pCamera->height << endl;

      cout << "dataTerm pair" << endl;
      cout << data_pair.first << "->" << data_pair.second << endl;

      PangaeaMeshData& templateMesh = modeGT ?
        templateMeshPyramidGT.levels[ data_pair.first ] :
        templateMeshPyramid.levels[ data_pair.first ];
      MeshDeformation& meshTrans = modeGT ?
        meshTransPyramidGT[ data_pair.first ] :
        meshTransPyramid[ data_pair.first ];

      vector<bool>& visibilityMask = modeGT ?
        visibilityMaskPyramidGT[ data_pair.first ]:
        visibilityMaskPyramid[ data_pair.first ];

      // compare the ground truth mask and tracking result mask
      // int visibleNum = 0;
      // int visibleNumGT = 0;
      // for(int num = 0; num < visibilityMaskPyramidGT[ data_pair.first ].size(); ++num)
      //   {
      //     if(visibilityMaskPyramid[ data_pair.first ][ num ])
      //       visibleNum++;
      //     if(visibilityMaskPyramidGT[ data_pair.first ][ num ])
      //       visibleNumGT++;

      //     if(visibilityMaskPyramidGT[ data_pair.first ][num] !=
      //        visibilityMaskPyramid[ data_pair.first ][num])
      //       ceresOutput << "ground truth and tracking results have different mask" << endl;
      //   }

      // ceresOutput << "tracking result visibility number " << visibleNum << endl;
      // ceresOutput << "ground truth visibility number " << visibleNumGT << endl;

      MeshNeighbors& patchNeighbors = meshPropagation.getPatchNeighbors( data_pair.first );
      MeshWeights& patchWeights = meshPropagation.getPatchWeights( data_pair.first );
      MeshNeighbors& patchRadii = meshPropagation.getPatchRadii( data_pair.first );

      if(data_pair.first == data_pair.second)
        {
          switch(errorType)
            {
            case PE_INTENSITY:
            case PE_COLOR:
            case PE_FEATURE:
            case PE_DEPTH:
			{
				if (trackerSettings.use_intensity_pyramid)
				{
					PangaeaMeshData& templateMeshIntensity =
						templateIntensityPyramid.levels[data_pair.first];

					AddCostImageProjection(problem,
						loss_function,
						errorType,
						templateMeshIntensity,
						meshTrans,
						visibilityMask,
						pCamera,
						pFrame);
				}
				else
				{
					AddCostImageProjection(problem,
						loss_function,
						errorType,
						templateMesh,
						meshTrans,
						visibilityMask,
						pCamera,
						pFrame);
				}
			}
              break;
			case PE_INTRINSIC:
			case PE_INTRINSIC_COLOR:
			{
				MeshDeformation &local_lighting = templateMeshPyramid.levels[data_pair.first].specular_colors;
				AddCostImageProjection(problem,
					loss_function,
					errorType,
					templateMesh,
					meshTrans,
					visibilityMask,
					pCamera,
					pFrame,
					local_lighting,
					refinement);
			}
			break;
            case PE_NCC:
            case PE_COLOR_NCC:
            case PE_FEATURE_NCC:
              AddCostImageProjectionPatch(problem,
                                          loss_function,
                                          errorType,
                                          templateMesh,
                                          meshTrans,
                                          visibilityMask,
                                          patchNeighbors,
                                          patchRadii,
                                          patchWeights,
                                          pCamera,
                                          pFrame);
              break;
            }
        }
      else{

        PangaeaMeshData& templateNeighborMesh = modeGT ?
          templateMeshPyramidGT.levels[ data_pair.second ] :
          templateMeshPyramid.levels[ data_pair.second ];
        MeshDeformation& neighborMeshTrans = modeGT ?
          meshTransPyramidGT[ data_pair.second ] :
          meshTransPyramid[ data_pair.second ];
        MeshDeformation& neighborMeshRot = modeGT ?
          meshRotPyramid[ data_pair.second ] :
          meshRotPyramid[ data_pair.second ];

        pair<int, int> meshPair( data_pair.first, data_pair.second );
        MeshNeighbors&  neighbors = meshPropagation.getNeighbors( meshPair );
        MeshWeights& weights = meshPropagation.getWeights( meshPair );

        switch(errorType)
          {

          case PE_INTENSITY:
          case PE_COLOR:
          case PE_FEATURE:

            AddCostImageProjectionCoarse(problem,
                                         loss_function,
                                         errorType,
                                         templateMesh,
                                         visibilityMask,
                                         templateNeighborMesh,
                                         neighborMeshTrans,
                                         neighborMeshRot,
                                         neighbors,
                                         weights,
                                         pCamera,
                                         pFrame);

            break;

          case PE_NCC:
          case PE_COLOR_NCC:
          case PE_FEATURE_NCC:

            AddCostImageProjectionPatchCoarse(problem,
                                              loss_function,
                                              errorType,
                                              templateMesh,
                                              visibilityMask,
                                              patchNeighbors,
                                              patchRadii,
                                              patchWeights,
                                              templateNeighborMesh,
                                              neighborMeshTrans,
                                              neighborMeshRot,
                                              neighbors,
                                              weights,
                                              pCamera,
                                              pFrame);
            break;

          }

      }
    }

}

void DeformNRSFMTracker::AddPhotometricCost(ceres::Problem& problem,
                                            ceres::LossFunction* loss_function,
                                            dataTermErrorType errorType)
{
  // add photometric cost
  // there are two different cases
  vector<std::pair<int,int> >& data_pairs =
    pStrategy->optimizationSettings[currLevel].dataTermPairs;

  int num_data_pairs = data_pairs.size();

  //TICK("SetupPhotometricCost" + std::to_string( currLevel ) );

  for(int k = 0; k < num_data_pairs; ++k)
    {
      std::pair<int, int>& data_pair = data_pairs[k];

      CameraInfo* pCamera = &pImagePyramid->getCameraInfo(data_pair.first);
      ImageLevel* pFrame = &pImagePyramid->getImageLevel(data_pair.first);

      cout << "camera width and height " << pCamera->width << " " << pCamera->height << endl;

      cout << "dataTerm pair" << endl;
      cout << data_pair.first << "->" << data_pair.second << endl;

      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[ data_pair.first ];
      MeshDeformation& meshTrans = meshTransPyramid[ data_pair.first ];

      vector<bool>& visibilityMask = visibilityMaskPyramid[ data_pair.first ];

      MeshNeighbors& patchNeighbors = meshPropagation.getPatchNeighbors( data_pair.first );
      MeshWeights& patchWeights = meshPropagation.getPatchWeights( data_pair.first );
      MeshNeighbors& patchRadii = meshPropagation.getPatchRadii( data_pair.first );

      //
      int per_residual_num = PE_RESIDUAL_NUM_ARRAY[ errorType ];

      vector<double> Residuals;
      Residuals.resize( per_residual_num * templateMesh.numVertices );

      if(data_pair.first == data_pair.second )
        {

          for(int i = 0; i < templateMesh.numVertices; ++i)
            {
              if(visibilityMask[i])
                {
                  switch(errorType)
                    {
                    case PE_INTENSITY:
                      {
                        ResidualImageProjection* pResidual = new ResidualImageProjection(1,
                                                                                         &templateMesh.grays[i],
                                                                                         &templateMesh.vertices[i][0],
                                                                                         pCamera,
                                                                                         pFrame,
                                                                                         errorType);

                        ceres::AutoDiffCostFunction<ResidualImageProjection, 1, 3, 3, 3>* cost_function =
                          new ceres::AutoDiffCostFunction<ResidualImageProjection, 1, 3, 3, 3>( pResidual );

                      }

                      break;

                    case PE_COLOR:
                      {
                        ResidualImageProjection* pResidual = new ResidualImageProjection(1,
                                                                                         &templateMesh.colors[i][0],
                                                                                         &templateMesh.vertices[i][0],
                                                                                         pCamera,
                                                                                         pFrame,
                                                                                         errorType);

                        ceres::AutoDiffCostFunction<ResidualImageProjection, 3, 3, 3, 3>* cost_function =
                          new ceres::AutoDiffCostFunction<ResidualImageProjection, 3, 3, 3, 3>( pResidual );

                      }

                      break;

                    case PE_NCC:
                    case PE_COLOR_NCC:
                      {
                        // in patchNeighbors, point itself is counted as a neighbor of itself
                        // patch based photometric error
                        int numNeighbors = patchNeighbors[i].size();

                        vector<double*> parameter_blocks;
                        for(int j = 0; j < numNeighbors; ++j)
                          parameter_blocks.push_back( &(meshTrans[ patchNeighbors[i][j] ][0]) );

                        parameter_blocks.push_back( &camPose[0] );
                        parameter_blocks.push_back( &camPose[3] );

                        ResidualImageProjectionPatch* pResidualPatch = new ResidualImageProjectionPatch(1,
                                                                                                        &templateMesh,
                                                                                                        pCamera,
                                                                                                        pFrame,
                                                                                                        numNeighbors,
                                                                                                        patchWeights[i],
                                                                                                        patchRadii[i],
                                                                                                        patchNeighbors[i],
                                                                                                        errorType );

                        ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionPatch, 5>* cost_function =
                          new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionPatch, 5>( pResidualPatch );

                        for(int j = 0; j < numNeighbors; ++j)
                          cost_function->AddParameterBlock(3);

                        cost_function->AddParameterBlock(3);
                        cost_function->AddParameterBlock(3);

                        cost_function->SetNumResiduals( per_residual_num  );

                        // I would like to compute residuals myself
                        (*pResidualPatch)(&(parameter_blocks[0]),  &Residuals[ per_residual_num * i ] );

                      }

                      break;

                    }
                }
            }

          // output the residuals please
          double data_term_residual_sum = 0;

          for(int i = 0; i < templateMesh.numVertices; ++i)
            {
              if(visibilityMask[i])
                {
                  data_term_residual_sum += Residuals[i] * Residuals[i];
                }
            }

          cout << "ncc data term cost test: " << data_term_residual_sum << endl;

        }

      else
        {

          // if(errorType == PE_NCC )
          // cerr << " NCC error measure not supported in sparse deformation node case yet! "<< endl;

          // just try propagation strategy first
          PangaeaMeshData& templateNeighborMesh = templateMeshPyramid.levels[ data_pair.second ];
          MeshDeformation& neighborMeshTrans = meshTransPyramid[ data_pair.second ];
          MeshDeformation& neighborMeshRot = meshRotPyramid[ data_pair.second ];

          pair<int, int> meshPair( data_pair.first, data_pair.second );
          MeshNeighbors&  neighbors = meshPropagation.getNeighbors( meshPair );
          MeshWeights& weights = meshPropagation.getWeights( meshPair );

          for(int i = 0; i < templateMesh.numVertices; ++i)
            {
              if(visibilityMask[i])
                {
                  switch(errorType)
                    {
                    case PE_INTENSITY:
                    case PE_COLOR:
                      {
                        // collect neighbors info
                        vector<double*> neighborVertices;
                        vector<double> neighborWeights;
                        vector<double*> parameter_blocks;

                        int numNeighbors = neighbors[i].size();
                        for(int j = 0; j < numNeighbors; ++j )
                          {
                            neighborWeights.push_back( weights[i][j] );
                            neighborVertices.push_back( &( templateNeighborMesh.vertices[ neighbors[i][j] ][ 0 ] ) );
                            parameter_blocks.push_back( &( neighborMeshTrans[ neighbors[i][j] ][0] ) );
                          }
                        for(int j = 0; j < numNeighbors; ++j )
                          parameter_blocks.push_back( &( neighborMeshRot[ neighbors[i][j] ][0] ) );

                        parameter_blocks.push_back( &camPose[0] );
                        parameter_blocks.push_back( &camPose[3] );

                        ResidualImageProjectionCoarse* pResidual = new ResidualImageProjectionCoarse(1,
                                                                                                     errorType == PE_INTENSITY ?
                                                                                                     &templateMesh.grays[i] :
                                                                                                     &templateMesh.colors[i][0],
                                                                                                     &templateMesh.vertices[i][0],
                                                                                                     pCamera,
                                                                                                     pFrame,
                                                                                                     numNeighbors,
                                                                                                     neighborWeights,
                                                                                                     neighborVertices,
                                                                                                     errorType );


                        ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionCoarse, 5>* cost_function =
                          new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionCoarse, 5>(pResidual);

                        for(int j = 0; j < 2*numNeighbors; ++j)
                          cost_function->AddParameterBlock(3);

                        cost_function->AddParameterBlock(3);
                        cost_function->AddParameterBlock(3);

                        cost_function->SetNumResiduals( per_residual_num );

                      }

                      break;

                    case PE_NCC:
                    case PE_COLOR_NCC:
                      {
                        // patch neighbors
                        int numNeighbors;

                        // coarse neighbors
                        int numCoarseNeighbors;
                        vector<unsigned int> parameterIndices;
                        vector<unsigned int> coarseNeighborIndices;
                        vector<unsigned int> coarseNeighborBiases;
                        vector<double> coarseNeighborWeights;

                        vector<double*> parameter_blocks;
                        vector<double*> parameter_blocks_rot;

                        numNeighbors = patchNeighbors[i].size();

                        int bias = 0;
                        vector<double*>::iterator iter;
                        for(int j = 0; j < numNeighbors; ++j)
                          {
                            int m = patchNeighbors[i][j];
                            int coarseNum = neighbors[ m ].size();

                            bias += coarseNum;
                            coarseNeighborBiases.push_back( bias );

                            for(int k = 0; k < coarseNum; ++k)
                              {
                                coarseNeighborIndices.push_back( neighbors[m][k] );
                                coarseNeighborWeights.push_back( weights[m][k] );

                                // if not in parameter_blocks yet
                                double* block = &( neighborMeshTrans[ neighbors[m][k] ][0] );
                                double* block_rot = &( neighborMeshRot[ neighbors[m][k] ][0] );

                                iter = std::find(parameter_blocks.begin(), parameter_blocks.end(), block);
                                if(iter == parameter_blocks.end())
                                  {
                                    parameter_blocks.push_back( block );
                                    parameter_blocks_rot.push_back( block_rot );
                                    parameterIndices.push_back( parameter_blocks.size() - 1 );
                                  }
                                else
                                  parameterIndices.push_back( iter - parameter_blocks.begin() );

                              }
                          }

                        numCoarseNeighbors = parameter_blocks.size();

                        for(int k = 0; k < parameter_blocks_rot.size(); ++k)
                          parameter_blocks.push_back( parameter_blocks_rot[k] );

                        parameter_blocks.push_back( &camPose[0] );
                        parameter_blocks.push_back( &camPose[3] );

                        ResidualImageProjectionPatchCoarse* pResidual = new ResidualImageProjectionPatchCoarse(1,
                                                                                                               &templateMesh,
                                                                                                               &templateNeighborMesh,
                                                                                                               pCamera,
                                                                                                               pFrame,
                                                                                                               numNeighbors,
                                                                                                               numCoarseNeighbors,
                                                                                                               patchWeights[i],
                                                                                                               patchRadii[i],
                                                                                                               patchNeighbors[i],
                                                                                                               parameterIndices,
                                                                                                               coarseNeighborIndices,
                                                                                                               coarseNeighborBiases,
                                                                                                               coarseNeighborWeights,
                                                                                                               errorType );

                        ceres::DynamicAutoDiffCostFunction<ResidualImageProjectionPatchCoarse, 5>* cost_function =
                          new ceres::DynamicAutoDiffCostFunction< ResidualImageProjectionPatchCoarse, 5 >(pResidual);

                        for(int j = 0; j < 2*numCoarseNeighbors; ++j)
                          cost_function->AddParameterBlock(3);

                        cost_function->AddParameterBlock(3);
                        cost_function->AddParameterBlock(3);

                        cost_function->SetNumResiduals( per_residual_num );

                      }

                      break;

                    }
                }
            }

        }
    }

  //TOCK("SetupPhotometricCost" + std::to_string( currLevel ) );
}

void DeformNRSFMTracker::AddTotalVariationCost(ceres::Problem& problem,
                                               ceres::LossFunction* loss_function)
{

  vector<std::pair<int,int> >& tv_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_tv_pairs = tv_pairs.size();

  for(int k = 0; k < num_tv_pairs; ++k)
    {
      std::pair<int, int>& tv_pair = tv_pairs[k];

      bool same_level = tv_pair.first == tv_pair.second;

      cout << "tv pair" << endl;
      cout << tv_pair.first << "->" << tv_pair.second << endl;

      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[tv_pair.first];

      MeshDeformation& meshTrans = trackerSettings.usePrevForTemplateInTV ?
        prevMeshTransPyramid[tv_pair.first] : meshTransPyramid[tv_pair.first];
      MeshDeformation& neighborMeshTrans = trackerSettings.usePrevForTemplateInTV ?
        prevMeshTransPyramid[tv_pair.second] : meshTransPyramid[tv_pair.second];

      MeshDeformation& meshTransGT = trackerSettings.usePrevForTemplateInTV ?
        prevMeshTransPyramidGT[tv_pair.first] : meshTransPyramidGT[tv_pair.first];
      MeshDeformation& neighborMeshTransGT = trackerSettings.usePrevForTemplateInTV ?
        prevMeshTransPyramidGT[tv_pair.second] : meshTransPyramidGT[tv_pair.second];

      vector<vector<unsigned int> >& meshNeighbors = same_level ?
        templateMesh.adjVerticesInd : meshPropagation.getNeighbors( tv_pair );
      vector<vector<double> >& meshWeights = meshPropagation.getWeights( tv_pair );

      for(int vertex = 0; vertex < templateMesh.numVertices; ++vertex)
        {
          for(int neighbor = 0; neighbor < meshNeighbors[vertex].size(); ++neighbor)
            {
              double weight = same_level ? 1 : meshWeights[vertex][neighbor];
              if(!same_level || vertex < meshNeighbors[vertex][neighbor])
                {
                  ResidualTV* pResidual = new ResidualTV( weight );

                  ceres::AutoDiffCostFunction<ResidualTV, 3, 3, 3>* cost_function =
                    new ceres::AutoDiffCostFunction<ResidualTV, 3, 3, 3>(pResidual);

                  ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(cost_function,
                                                                                    loss_function,
                                                                                    modeGT ? &meshTransGT[ vertex  ][0] :
                                                                                    &meshTrans[ vertex  ][0],
                                                                                    modeGT ? &neighborMeshTransGT[ meshNeighbors[vertex][neighbor] ][0] :
                                                                                    &neighborMeshTrans[ meshNeighbors[vertex][neighbor] ][0]
                                                                                    );
                  if(useProblemWrapper)
                    {
                      if(modeGT)
                        {
                          problemWrapperGT.addTVTerm(currLevel, residualBlockId);
                          problemWrapperGT.addRegTermCost(currLevel, cost_function);
                        }
                      else
                        {
                          problemWrapper.addTVTerm(currLevel, residualBlockId);
                          problemWrapper.addRegTermCost(currLevel, cost_function);
                        }
                    }

                }

            }
        }
    }

}

void DeformNRSFMTracker::AddRotTotalVariationCost(ceres::Problem& problem,
                                                  ceres::LossFunction* loss_function)
{
  vector<std::pair<int,int> >& tv_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;
  int num_tv_pairs = tv_pairs.size();

  for(int k = 0; k < num_tv_pairs; ++k)
    {
      std::pair<int, int>& tv_pair = tv_pairs[k];

      bool same_level = tv_pair.first == tv_pair.second;

      cout << "rot_tv pair" << endl;
      cout << tv_pair.first << "->" << tv_pair.second << endl;

      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[tv_pair.first];

      vector<vector<double> >& meshRot = trackerSettings.usePrevForTemplateInTV ?
        prevMeshRotPyramid[tv_pair.first] : meshRotPyramid[tv_pair.first];
      vector<vector<double> >& neighborMeshRot = trackerSettings.usePrevForTemplateInTV ?
        prevMeshRotPyramid[tv_pair.second] : meshRotPyramid[tv_pair.second];

      vector<vector<double> >& meshRotGT = trackerSettings.usePrevForTemplateInTV ?
        prevMeshRotPyramidGT[tv_pair.first] : meshRotPyramidGT[tv_pair.first];
      vector<vector<double> >& neighborMeshRotGT = trackerSettings.usePrevForTemplateInTV ?
        prevMeshRotPyramidGT[tv_pair.second] : meshRotPyramidGT[tv_pair.second];

      vector<vector<unsigned int> >& meshNeighbors = same_level ?
        templateMesh.adjVerticesInd : meshPropagation.getNeighbors( tv_pair );
      vector<vector<double> >& meshWeights = meshPropagation.getWeights( tv_pair );

      for(int vertex = 0; vertex < templateMesh.numVertices; ++vertex)
        {
          for(int neighbor = 0; neighbor < meshNeighbors[vertex].size(); ++neighbor)
            {
              double weight = same_level ? 1 : meshWeights[vertex][neighbor];
              if(!same_level || vertex < meshNeighbors[vertex][neighbor])
                {
                  ResidualRotTV* pResidual = new ResidualRotTV( weight );
                  ceres::AutoDiffCostFunction<ResidualRotTV, 3, 3, 3>* cost_function =
                    new ceres::AutoDiffCostFunction<ResidualRotTV, 3, 3, 3>(pResidual);

                  ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                                    cost_function,
                                                                                    loss_function,
                                                                                    modeGT ? &meshRotGT[ vertex  ][0] :
                                                                                    &meshRot[ vertex  ][0],
                                                                                    modeGT ? &neighborMeshRotGT[ meshNeighbors[vertex][neighbor] ][0] :
                                                                                    &neighborMeshRot[ meshNeighbors[vertex][neighbor] ][0]
                                                                                    );

                  if(useProblemWrapper)
                    {
                      if(modeGT)
                        {
                          problemWrapperGT.addRotTVTerm(currLevel, residualBlockId);
                          problemWrapperGT.addRegTermCost(currLevel, cost_function);
                        }
                      else
                        {
                          problemWrapper.addRotTVTerm(currLevel, residualBlockId);
                          problemWrapper.addRegTermCost(currLevel, cost_function);
                        }
                    }

                }

            }
        }
    }

}

void DeformNRSFMTracker::AddARAPCost(ceres::Problem& problem,
                                     ceres::LossFunction* loss_function)
{
  vector<std::pair<int,int> >& arap_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;
  int num_arap_pairs = arap_pairs.size();

  for(int k = 0; k < num_arap_pairs; ++k)
    {
      std::pair<int, int>& arap_pair = arap_pairs[k];

      bool same_level = arap_pair.first == arap_pair.second;

      cout << "arap pair" << endl;
      cout << arap_pair.first << "->" << arap_pair.second << endl;

      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[arap_pair.first];
      PangaeaMeshData& templateNeighborMesh = templateMeshPyramid.levels[arap_pair.second];

      MeshDeformation& meshTrans = meshTransPyramid[arap_pair.first];
      MeshDeformation& neighborMeshTrans = meshTransPyramid[arap_pair.second];
      vector<vector<double> >& meshRot = meshRotPyramid[arap_pair.first];

      PangaeaMeshData& templateMeshGT = templateMeshPyramidGT.levels[arap_pair.first];
      PangaeaMeshData& templateNeighborMeshGT = templateMeshPyramidGT.levels[arap_pair.second];

      MeshDeformation& meshTransGT = meshTransPyramidGT[arap_pair.first];
      MeshDeformation& neighborMeshTransGT = meshTransPyramidGT[arap_pair.second];
      vector<vector<double> >& meshRotGT = meshRotPyramidGT[arap_pair.first];

      vector<vector<unsigned int> >& meshNeighbors = same_level ?
        templateMesh.adjVerticesInd : meshPropagation.getNeighbors( arap_pair );
      vector<vector<double> >& meshWeights = meshPropagation.getWeights( arap_pair );


      for(int vertex = 0; vertex < templateMesh.numVertices; ++vertex)
        {
          //#pragma omp parallel for ordered schedule(dynamic)
          //#pragma omp parallel for
          for(int neighbor = 0; neighbor < meshNeighbors[vertex].size(); ++neighbor)
            {
              double weight = same_level ? 1 : meshWeights[vertex][neighbor];

              // ResidualARAP* pResidualARAP = new ResidualARAP( weight, &templateMesh.vertices[vertex][0],
              //     &templateNeighborMesh.vertices[ meshNeighbors[vertex][neighbor] ][0], true);
              // ceres::AutoDiffCostFunction<ResidualARAP, 3, 3, 3, 3>* pAutoDiffCostFunction =
              //     new ceres::AutoDiffCostFunction<ResidualARAP, 3, 3, 3, 3>(pResidualARAP);

              // //#pragma omp ordered
              // problem.AddResidualBlock(pAutoDiffCostFunction,
              // loss_function,
              // &meshTrans[ vertex  ][0],
              // &neighborMeshTrans[ meshNeighbors[vertex][neighbor] ][0],
              // &meshRot[ vertex ][0]);


              ResidualARAP* pResidual = new ResidualARAP(weight,
                                                         modeGT ? &templateMeshGT.vertices[vertex][0] :
                                                         &templateMesh.vertices[vertex][0],
                                                         modeGT ? &templateNeighborMeshGT.vertices[ meshNeighbors[vertex][neighbor] ][0] :
                                                         &templateNeighborMesh.vertices[ meshNeighbors[vertex][neighbor] ][0],
                                                         true);

              ceres::AutoDiffCostFunction<ResidualARAP, 3, 3, 3, 3>* cost_function =
                new ceres::AutoDiffCostFunction<ResidualARAP, 3, 3, 3, 3>(pResidual);

              ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                                cost_function,
                                                                                loss_function,
                                                                                modeGT ? &meshTransGT[ vertex  ][0] :
                                                                                &meshTrans[ vertex  ][0],
                                                                                modeGT ? &neighborMeshTransGT[ meshNeighbors[vertex][neighbor] ][0] :
                                                                                &neighborMeshTrans[ meshNeighbors[vertex][neighbor] ][0],
                                                                                modeGT ? &meshRotGT[ vertex ][0] :
                                                                                &meshRot[ vertex ][0]);
              if(useProblemWrapper)
                {
                  if(modeGT)
                    {
                      problemWrapperGT.addARAPTerm(currLevel, residualBlockId);
                      problemWrapperGT.addRegTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addARAPTerm(currLevel, residualBlockId);
                      problemWrapper.addRegTermCost(currLevel, cost_function);
                    }
                }

            }
        }

    }

}

void DeformNRSFMTracker::AddInextentCost(ceres::Problem& problem,
                                         ceres::LossFunction* loss_function)
{
  vector<std::pair<int,int> >& inextent_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_inextent_pairs = inextent_pairs.size();

  for(int k = 0; k < num_inextent_pairs; ++k)
    {
      std::pair<int, int>& inextent_pair = inextent_pairs[k];

      bool same_level = inextent_pair.first == inextent_pair.second;

      cout << "inextent pair" << endl;
      cout << inextent_pair.first << "->" << inextent_pair.second << endl;

      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[inextent_pair.first];
      PangaeaMeshData& templateNeighborMesh = templateMeshPyramid.levels[inextent_pair.second];

      MeshDeformation& meshTrans = meshTransPyramid[inextent_pair.first];
      MeshDeformation& neighborMeshTrans = meshTransPyramid[inextent_pair.second];

      MeshDeformation& meshTransGT = meshTransPyramidGT[inextent_pair.first];
      MeshDeformation& neighborMeshTransGT = meshTransPyramidGT[inextent_pair.second];

      vector<vector<unsigned int> >& meshNeighbors = same_level ?
        templateMesh.adjVerticesInd : meshPropagation.getNeighbors( inextent_pair );
      vector<vector<double> >& meshWeights = meshPropagation.getWeights( inextent_pair );

      for(int vertex = 0; vertex < templateMesh.numVertices; ++vertex)
        {
          for(int neighbor = 0; neighbor < meshNeighbors[vertex].size(); ++neighbor)
            {
              double weight = same_level ? 1 : meshWeights[vertex][neighbor];

              ResidualINEXTENT* pResidual = new ResidualINEXTENT( weight );
              ceres::AutoDiffCostFunction<ResidualINEXTENT, 1, 3, 3>* cost_function =
                new ceres::AutoDiffCostFunction<ResidualINEXTENT, 1, 3, 3>(pResidual);

              ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                                cost_function,
                                                                                loss_function,
                                                                                modeGT ? &meshTransGT[ vertex  ][0] :
                                                                                &meshTrans[ vertex  ][0],
                                                                                modeGT ? &neighborMeshTransGT[ meshNeighbors[vertex][neighbor] ][0] :
                                                                                &neighborMeshTrans[ meshNeighbors[vertex][neighbor] ][0]
                                                                                );
              if(useProblemWrapper)
                {
                  if(modeGT)
                    {
                      problemWrapperGT.addINEXTENTTerm(currLevel, residualBlockId);
                      problemWrapperGT.addRegTermCost(currLevel, cost_function);
                    }
                  else
                    {
                      problemWrapper.addINEXTENTTerm(currLevel, residualBlockId);
                      problemWrapper.addRegTermCost(currLevel, cost_function);
                    }
                }
            }
        }
    }
}

void DeformNRSFMTracker::AddDeformationCost(ceres::Problem& problem,
                                            ceres::LossFunction* loss_function)
{
  vector<int>& deform_level_vec =
    pStrategy->optimizationSettings[currLevel].deformTermLevelIDVec;

  int num_deform_levels = deform_level_vec.size();

  for(int k = 0; k < num_deform_levels; ++k)
    {
      int deform_level = deform_level_vec[k];

      cout << "deformation level " << deform_level << endl;

      MeshDeformation& meshTrans = meshTransPyramid[deform_level];
      MeshDeformation& prevMeshTrans = prevMeshTransPyramid[deform_level];

      MeshDeformation& meshTransGT = meshTransPyramidGT[deform_level];
      MeshDeformation& prevMeshTransGT = prevMeshTransPyramidGT[deform_level];

      for(int vertex = 0; vertex < meshTrans.size(); ++vertex)
        {
          ResidualDeform* pResidual = new ResidualDeform(1,
                                                         modeGT ? &prevMeshTransGT[ vertex ][ 0 ] :
                                                         &prevMeshTrans[ vertex ][ 0 ]);

          ceres::AutoDiffCostFunction<ResidualDeform, 3, 3>* cost_function =
            new ceres::AutoDiffCostFunction<ResidualDeform, 3, 3>(pResidual);

          ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                            cost_function,
                                                                            loss_function,
                                                                            modeGT ? &meshTransGT[ vertex ][0] :
                                                                            &meshTrans[ vertex ][0]
                                                                            );

          if(useProblemWrapper)
            {
              if(modeGT)
                {
                  problemWrapperGT.addDeformTerm(currLevel, residualBlockId);
                  problemWrapperGT.addRegTermCost(currLevel, cost_function);
                }
              else
                {
                  problemWrapper.addDeformTerm(currLevel, residualBlockId);
                  problemWrapper.addRegTermCost(currLevel, cost_function);
                }
            }
        }
    }
}

void DeformNRSFMTracker::AddTemporalMotionCost(ceres::Problem& problem,
                                               double rotWeight, double transWeight)
{
  // cout << "prev motion started:" << endl;

  // for(int i = 0; i < 6; ++i)
  // {
  //     prevCamPose[i] = camPose[i];
  //     cout << camPose[i] << endl;
  // }

  ResidualTemporalMotion* pResidual = new ResidualTemporalMotion(modeGT ? prevCamPoseGT : prevCamPose,
                                                                 modeGT ? prevCamPoseGT + 3 :  prevCamPose+3,
                                                                 rotWeight,
                                                                 transWeight);

  ceres::AutoDiffCostFunction<ResidualTemporalMotion, 6, 3, 3>* cost_function =
    new ceres::AutoDiffCostFunction<ResidualTemporalMotion, 6, 3, 3>( pResidual );

  ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
                                                                    cost_function,
                                                                    NULL,
                                                                    modeGT ? &camPoseGT[0] : &camPose[0],
                                                                    modeGT ? &camPoseGT[3] : &camPose[3]);

  if(useProblemWrapper)
    {
      if(modeGT)
        {
          problemWrapperGT.addTemporalTerm(currLevel, residualBlockId);
          problemWrapperGT.addRegTermCost(currLevel, cost_function);
        }
      else
        {
          problemWrapper.addTemporalTerm(currLevel, residualBlockId);
          problemWrapper.addRegTermCost(currLevel, cost_function);
        }
    }

}

void DeformNRSFMTracker::AddSmoothingCost(ceres::Problem& problem,
	ceres::LossFunction* loss_function)
{
	vector<std::pair<int, int> >& smoothing_pairs =
		pStrategy->optimizationSettings[currLevel].regTermPairs;

	int num_smoothing_pairs = smoothing_pairs.size();

	for (int k = 0; k < num_smoothing_pairs; ++k)
	{
		std::pair<int, int>& smoothing_pair = smoothing_pairs[k];

		bool same_level = smoothing_pair.first == smoothing_pair.second;

		cout << "smoothing pair" << endl;
		cout << smoothing_pair.first << "->" << smoothing_pair.second << endl;

		PangaeaMeshData& templateMesh = templateMeshPyramid.levels[smoothing_pair.first];

		MeshDeformation& meshTrans = meshTransPyramid[smoothing_pair.first];

		vector<vector<double> >& meshRotGT = meshRotPyramidGT[smoothing_pair.first];

		MeshDeformation& meshTransGT = meshTransPyramidGT[smoothing_pair.first];

		vector<vector<unsigned int> >& meshNeighbors = same_level ?
			templateMesh.adjVerticesInd : meshPropagation.getNeighbors(smoothing_pair);

		for (int vertex = 0; vertex < templateMesh.numVertices; ++vertex)
		{
			ResidualLaplacianSmoothing* pResidual =
				new ResidualLaplacianSmoothing(&templateMesh.vertices[vertex][0],
				templateMesh.vertices, meshNeighbors[vertex],
				templateMesh.adjFacesInd[vertex].size(), templateMesh.clockwise);

			ceres::DynamicAutoDiffCostFunction<ResidualLaplacianSmoothing, 5> *cost_function =
				new ceres::DynamicAutoDiffCostFunction< ResidualLaplacianSmoothing, 5 >(pResidual);

			// List of pointers to translations per vertex
			vector<double*> v_parameter_blocks;

			// Local translations
			v_parameter_blocks.push_back(&meshTrans[vertex][0]);
			cost_function->AddParameterBlock(3);
			for (int j = 0; j < meshNeighbors[vertex].size(); j++)
			{
				int v_idx = meshNeighbors[vertex][j];
				v_parameter_blocks.push_back(&meshTrans[v_idx][0]);
				cost_function->AddParameterBlock(3);
			}

			cost_function->SetNumResiduals(1);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				cost_function,
				loss_function,
				v_parameter_blocks);

			if (useProblemWrapper)
			{
				if (modeGT)
				{
					problemWrapperGT.addSmoothingTerm(currLevel, residualBlockId);
					problemWrapperGT.addRegTermCost(currLevel, cost_function);
				}
				else
				{
					problemWrapper.addSmoothingTerm(currLevel, residualBlockId);
					problemWrapper.addRegTermCost(currLevel, cost_function);
				}
			}
		}
	}
}

void DeformNRSFMTracker::AddTemporalSHCoeffCost(ceres::Problem& problem,
  ceres::LossFunction* loss_function)
{
  vector<std::pair<int, int> >& temp_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_temp_pairs = temp_pairs.size();

  for (int k = 0; k < num_temp_pairs; ++k)
  {
    std::pair<int, int>& temp_pair = temp_pairs[k];

    bool same_level = temp_pair.first == temp_pair.second;

    cout << "temporal sh coeff pair" << endl;
    cout << temp_pair.first << "->" << temp_pair.second << endl;

    PangaeaMeshData& templateMesh = templateMeshPyramid.levels[temp_pair.first];

    vector<double> &sh_coeff = templateMesh.sh_coefficients;

    ResidualTemporalSHCoeff *residual =
      new ResidualTemporalSHCoeff(prevSHCoeff);

    ceres::DynamicAutoDiffCostFunction<ResidualTemporalSHCoeff, 5>* dyn_cost_function
      = new ceres::DynamicAutoDiffCostFunction< ResidualTemporalSHCoeff, 5 >(residual);

    // List of pointers to sh coefficients
    vector<double*> v_parameter_blocks;

    // SH Coeff
    dyn_cost_function->AddParameterBlock(sh_coeff.size());
    v_parameter_blocks.push_back(&sh_coeff[0]);

    dyn_cost_function->SetNumResiduals(sh_coeff.size());

    ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
      dyn_cost_function,
      loss_function,
      v_parameter_blocks);

    if (useProblemWrapper)
    {
      if (modeGT)
      {
        problemWrapperGT.addTemporalSHCoeffTerm(currLevel, residualBlockId);
        problemWrapperGT.addRegTermCost(currLevel, dyn_cost_function);
      }
      else
      {
        problemWrapper.addTemporalSHCoeffTerm(currLevel, residualBlockId);
        problemWrapper.addRegTermCost(currLevel, dyn_cost_function);
      }
    }
  }
}

void DeformNRSFMTracker::AddSpecularSmoothnessCost(ceres::Problem& problem,
  ceres::LossFunction* loss_function)
{
  vector<std::pair<int, int> >& temp_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_temp_pairs = temp_pairs.size();

  for (int k = 0; k < num_temp_pairs; ++k)
  {
    std::pair<int, int>& temp_pair = temp_pairs[k];

    bool same_level = temp_pair.first == temp_pair.second;

    cout << "specular smoothness pair" << endl;
    cout << temp_pair.first << "->" << temp_pair.second << endl;

    PangaeaMeshData& templateMesh = templateMeshPyramid.levels[temp_pair.first];

    vector<vector<double>> &local_lightings = templateMesh.specular_colors;

    for (size_t i = 0; i < templateMesh.numVertices; i++)
    {
      for (size_t j = 0; j < templateMesh.adjVerticesInd[i].size(); j++)
      {
        unsigned int adj_v_idx = templateMesh.adjVerticesInd[i][j];

        double weight = 1;// diff_weights[i][j];

        if (weight > 0.0)
        {
			if (trackerSettings.use_white_specularities)
			{
				ResidualWeightedDifference *residual = new ResidualWeightedDifference(
					weight, 1);

				ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>* cost_function =
					new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>(residual);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					cost_function,
					loss_function,
					&local_lightings[i][0],
					&local_lightings[adj_v_idx][0]);

				if (useProblemWrapper)
				{
					if (modeGT)
					{
						problemWrapperGT.addTemporalSpecularTerm(currLevel, residualBlockId);
						problemWrapperGT.addRegTermCost(currLevel, cost_function);
					}
					else
					{
						problemWrapper.addTemporalSpecularTerm(currLevel, residualBlockId);
						problemWrapper.addRegTermCost(currLevel, cost_function);
					}
				}
			}
			else
			{
				ResidualWeightedDifference *residual = new ResidualWeightedDifference(
					weight, 3);

				ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>* cost_function =
					new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>(residual);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					cost_function,
					loss_function,
					&local_lightings[i][0],
					&local_lightings[adj_v_idx][0]);

				if (useProblemWrapper)
				{
					if (modeGT)
					{
						problemWrapperGT.addTemporalSpecularTerm(currLevel, residualBlockId);
						problemWrapperGT.addRegTermCost(currLevel, cost_function);
					}
					else
					{
						problemWrapper.addTemporalSpecularTerm(currLevel, residualBlockId);
						problemWrapper.addRegTermCost(currLevel, cost_function);
					}
				}
			}
        }
      }
    }
  }
}

void DeformNRSFMTracker::AddSpecularMagnitudeCost(ceres::Problem& problem,
  ceres::LossFunction* loss_function)
{
  vector<std::pair<int, int> >& temp_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_temp_pairs = temp_pairs.size();

  for (int k = 0; k < num_temp_pairs; ++k)
  {
    std::pair<int, int>& temp_pair = temp_pairs[k];

    bool same_level = temp_pair.first == temp_pair.second;

    cout << "specular magnitude pair" << endl;
    cout << temp_pair.first << "->" << temp_pair.second << endl;

    PangaeaMeshData& templateMesh = templateMeshPyramid.levels[temp_pair.first];

    vector<vector<double>> &local_lightings = templateMesh.specular_colors;

	bool use_lower_bound = true;
	bool use_upper_bound = true;

    for (size_t i = 0; i < templateMesh.numVertices; i++)
    {
		if (trackerSettings.use_white_specularities)
		{
			ResidualValueMagnitude *residual = new ResidualValueMagnitude(1);

			ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>* cost_function =
				new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>(residual);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				cost_function,
				loss_function,
				&local_lightings[i][0]);

			if (useProblemWrapper)
			{
				if (modeGT)
				{
					problemWrapperGT.addTemporalSpecularTerm(currLevel, residualBlockId);
					problemWrapperGT.addRegTermCost(currLevel, cost_function);
				}
				else
				{
					problemWrapper.addTemporalSpecularTerm(currLevel, residualBlockId);
					problemWrapper.addRegTermCost(currLevel, cost_function);
				}
			}
		}
		else
		{
			ResidualValueMagnitude *residual = new ResidualValueMagnitude(3);

			ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>* cost_function =
				new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>(residual);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				cost_function,
				loss_function,
				&local_lightings[i][0]);

			if (useProblemWrapper)
			{
				if (modeGT)
				{
					problemWrapperGT.addTemporalSpecularTerm(currLevel, residualBlockId);
					problemWrapperGT.addRegTermCost(currLevel, cost_function);
				}
				else
				{
					problemWrapper.addTemporalSpecularTerm(currLevel, residualBlockId);
					problemWrapper.addRegTermCost(currLevel, cost_function);
				}
			}
		}

	  if (use_lower_bound)
	  {
		  problem.SetParameterLowerBound(&local_lightings[i][0], 0, 0.0);
		  if (!trackerSettings.use_white_specularities)
		  {
			  problem.SetParameterLowerBound(&local_lightings[i][0], 1, 0.0);
			  problem.SetParameterLowerBound(&local_lightings[i][0], 2, 0.0);
		  }
	  }

	  if (use_upper_bound)
	  {
		  problem.SetParameterUpperBound(&local_lightings[i][0], 0, 1.0);

		  if (!trackerSettings.use_white_specularities)
		  {
			  problem.SetParameterUpperBound(&local_lightings[i][0], 1, 1.0);
			  problem.SetParameterUpperBound(&local_lightings[i][0], 2, 1.0);
		  }
	  }
    }
  }
}

void DeformNRSFMTracker::AddTemporalSpecularCost(ceres::Problem& problem,
  ceres::LossFunction* loss_function)
{
  vector<std::pair<int, int> >& temp_pairs =
    pStrategy->optimizationSettings[currLevel].regTermPairs;

  int num_temp_pairs = temp_pairs.size();

  unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;

  for (int k = 0; k < num_temp_pairs; ++k)
  {
    std::pair<int, int>& temp_pair = temp_pairs[k];

    bool same_level = temp_pair.first == temp_pair.second;

    cout << "temporal specular pair" << endl;
    cout << temp_pair.first << "->" << temp_pair.second << endl;

    PangaeaMeshData& templateMesh = templateMeshPyramid.levels[temp_pair.first];

	vector<vector<double>> &local_lightings = templateMesh.specular_colors;
	vector<vector<double>> &prev_local_lightings = prevMeshSpecularPyramid[temp_pair.first];

    for (size_t i = 0; i < templateMesh.numVertices; i++)
    {


      ResidualTemporalLocalLighting *residual = 
        new ResidualTemporalLocalLighting(prev_local_lightings[i], specular_n_channels);

      ceres::DynamicAutoDiffCostFunction<ResidualTemporalLocalLighting, 5>* dyn_cost_function
        = new ceres::DynamicAutoDiffCostFunction< ResidualTemporalLocalLighting, 5 >(residual);

      // List of pointers to translations per vertex
      vector<double*> v_parameter_blocks;

      dyn_cost_function->AddParameterBlock(specular_n_channels);
      v_parameter_blocks.push_back(&local_lightings[i][0]);

      dyn_cost_function->SetNumResiduals(specular_n_channels);

      ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
        dyn_cost_function,
        loss_function,
        v_parameter_blocks);

      if (useProblemWrapper)
      {
        if (modeGT)
        {
          problemWrapperGT.addTemporalSpecularTerm(currLevel, residualBlockId);
          problemWrapperGT.addRegTermCost(currLevel, dyn_cost_function);
        }
        else
        {
          problemWrapper.addTemporalSpecularTerm(currLevel, residualBlockId);
          problemWrapper.addRegTermCost(currLevel, dyn_cost_function);
        }
      }
    }
  }
}

void DeformNRSFMTracker::EnergySetup(ceres::Problem& problem, bool refinement)
{
  // now we are already to construct the energy
  // photometric term
  WeightPara& weightParaLevel = pStrategy->weightParaVec[currLevel];

  // get parameter weightings
  // cout << "data term weight" << " : " << weightParaLevel.dataTermWeight << endl;
  // cout << "data huber width" << " : " << weightParaLevel.dataHuberWidth << endl;

  long long int ii = currLevel;

  // cout << "Data Term Weights " << trackerSettings.weightPhotometric
  //      << " Levels " << currLevel << endl;
  // cout << "Feature Term Weights " << featureSettings.featureTermWeight
  //      << " Levels " << currLevel << endl;

  if(trackerSettings.useRGBImages && trackerSettings.weightPhotometric > 0)
    {
      TICK( "SetupDataTermCost" + std::to_string(ii) );

      ceres::LossFunction* pPhotometricLossFunction = NULL;
      if(weightParaLevel.dataHuberWidth)
        {
          pPhotometricLossFunction = new ceres::HuberLoss(
                                                          weightParaLevel.dataHuberWidth);
        }
      ceres::ScaledLoss* photometricScaledLoss = new ceres::ScaledLoss(
                                                                       pPhotometricLossFunction,
                                                                       weightParaLevel.dataTermWeight,
                                                                       ceres::TAKE_OWNERSHIP);

      AddPhotometricCostNew(problem, photometricScaledLoss, PEType, refinement);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addDataTermLoss(currLevel, photometricScaledLoss);
          else
            problemWrapper.addDataTermLoss(currLevel, photometricScaledLoss);
        }

	  if ((PEType == PE_INTRINSIC || PEType == PE_INTRINSIC_COLOR) 
		  && trackerSettings.use_intensity_pyramid)
	  {
		  ceres::LossFunction* pPhotometricLossFunction = NULL;
		  if (weightParaLevel.dataIntensityHuberWidth)
		  {
			  pPhotometricLossFunction = new ceres::HuberLoss(
				  weightParaLevel.dataIntensityHuberWidth);
		  }
		  ceres::ScaledLoss* photometricScaledLoss = new ceres::ScaledLoss(
			  pPhotometricLossFunction,
			  weightParaLevel.dataIntensityTermWeight,
			  ceres::TAKE_OWNERSHIP);

		  dataTermErrorType PEIntensityType = mapErrorType(trackerSettings.errorIntensityType);

		  AddPhotometricCostNew(problem, photometricScaledLoss, PEIntensityType, refinement);

		  if (useProblemWrapper)
		  {
			  if (modeGT)
				  problemWrapperGT.addDataTermLoss(currLevel, photometricScaledLoss);
			  else
				  problemWrapper.addDataTermLoss(currLevel, photometricScaledLoss);
		  }
	  }

      TOCK( "SetupDataTermCost" + std::to_string(ii) );
    }

  if(trackerSettings.useFeatureImages && featureSettings.featureTermWeight > 0)
    {
      TICK("SetupFeatureTermCost" + std::to_string(ii));

      ceres::LossFunction* pFeatureLossFunction = NULL;
      if(weightParaLevel.featureHuberWidth)
        {
          pFeatureLossFunction = new ceres::HuberLoss(
                                                      weightParaLevel.featureHuberWidth);

        }
      ceres::ScaledLoss* featureScaledLoss = new ceres::ScaledLoss(
                                                                   pFeatureLossFunction,
                                                                   weightParaLevel.featureTermWeight,
                                                                   ceres::TAKE_OWNERSHIP);
      AddPhotometricCostNew(problem, featureScaledLoss, featureSettings.useNCC ? PE_FEATURE_NCC : PE_FEATURE );

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addFeatureTermLoss(currLevel, featureScaledLoss);
          else
            problemWrapper.addFeatureTermLoss(currLevel, featureScaledLoss);
        }

      TOCK("SetupFeatureTermCost" + std::to_string(ii));

    }

  if(trackerSettings.weightDepth > 0)
    {
      TICK( "SetupDepthDataTermCost" + std::to_string(ii) );

      ceres::LossFunction* pPhotometricLossFunction = NULL;
      if(weightParaLevel.depthHuberWidth)
        {
          pPhotometricLossFunction = new ceres::HuberLoss(weightParaLevel.depthHuberWidth);
        }
      ceres::ScaledLoss* photometricScaledLoss = new ceres::ScaledLoss(
                                                                       pPhotometricLossFunction,
                                                                       weightParaLevel.depthTermWeight,
                                                                       ceres::TAKE_OWNERSHIP);

      AddPhotometricCostNew(problem, photometricScaledLoss, PE_DEPTH);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addDataTermLoss(currLevel, photometricScaledLoss);
          else
            problemWrapper.addDataTermLoss(currLevel, photometricScaledLoss);
        }

      TOCK( "SetupDepthDataTermCost" + std::to_string(ii) );
    }

  TICK( "SetupRegTermCost" + std::to_string(ii) );

  if(modeGT)
    {
      if(!useProblemWrapper || !problemWrapperGT.getLevelFlag( currLevel ) )
        {
          RegTermsSetup( problem, weightParaLevel );
          problemWrapperGT.setLevelFlag( currLevel );
        }
    }
  else
    {
      if(!useProblemWrapper || !problemWrapper.getLevelFlag( currLevel ) )
        {
          RegTermsSetup( problem, weightParaLevel );
          problemWrapper.setLevelFlag( currLevel );
        }
    }

  TOCK( "SetupRegTermCost" + std::to_string(ii) );


  // check whether the parameter blocks has nan values
  // CheckNaN();

}

void DeformNRSFMTracker::RegTermsSetup(ceres::Problem& problem, WeightPara& weightParaLevel)
{

  // cout << "TV Term Weights " << weightParaLevel.tvTermWeight
  //      << " Levels " << currLevel << endl;
  // cout << "ARAP Term Weights " << weightParaLevel.arapTermWeight
  //      << " Levels " << currLevel << endl;

  // totatl variation term
  if(weightParaLevel.tvTermWeight)
    {
      //TICK("SetupTVCost"  + std::to_string( currLevel ) );

      ceres::LossFunction* pTVLossFunction = NULL;

      if(trackerSettings.tvTukeyWidth)
        {
          pTVLossFunction = new ceres::TukeyLoss(trackerSettings.tvTukeyWidth);
        }else if(trackerSettings.tvHuberWidth)
        {
          pTVLossFunction = new ceres::HuberLoss(trackerSettings.tvHuberWidth);
        }
      ceres::ScaledLoss* tvScaledLoss = new ceres::ScaledLoss(
                                                              pTVLossFunction,
                                                              weightParaLevel.tvTermWeight,
                                                              ceres::TAKE_OWNERSHIP);

      AddTotalVariationCost(problem, tvScaledLoss);
      //AddTotalVariationCost(problem, NULL);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addRegTermLoss(currLevel, tvScaledLoss);
          else
            problemWrapper.addRegTermLoss(currLevel, tvScaledLoss);
        }

      //TOCK("SetupTVCost"  + std::to_string( currLevel ) );
    }

  // rotation total variation term
  // arap has to be turned on, otherwise there is no rotation variable
  if(weightParaLevel.arapTermWeight &&  weightParaLevel.tvRotTermWeight)
    {
      //TICK("SetupRotTVCost"  + std::to_string( currLevel ) );

      ceres::LossFunction* pRotTVLossFunction = NULL;

      if(trackerSettings.tvRotHuberWidth)
        {
          pRotTVLossFunction = new ceres::HuberLoss(trackerSettings.tvRotHuberWidth);
        }
      ceres::ScaledLoss* tvRotScaledLoss = new ceres::ScaledLoss(
                                                                 pRotTVLossFunction,
                                                                 weightParaLevel.tvRotTermWeight,
                                                                 ceres::TAKE_OWNERSHIP);

      AddRotTotalVariationCost(problem, tvRotScaledLoss);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addRegTermLoss(currLevel, tvRotScaledLoss);
          else
            problemWrapper.addRegTermLoss(currLevel, tvRotScaledLoss);
        }

      //TOCK("SetupRotTVCost"  + std::to_string( currLevel ) );

    }

  // arap term
  if(weightParaLevel.arapTermWeight)
    {
      //TICK("SetupARAPCost"  + std::to_string( currLevel ) );

	  ceres::LossFunction* pArapLossFunction = NULL;
	  if (trackerSettings.arapHuberWidth)
	  {
		  pArapLossFunction = new ceres::HuberLoss(trackerSettings.arapHuberWidth);
	  }

      ceres::ScaledLoss* arapScaledLoss = new ceres::ScaledLoss(
																pArapLossFunction,
                                                                weightParaLevel.arapTermWeight,
                                                                ceres::TAKE_OWNERSHIP);
      AddARAPCost(problem, arapScaledLoss);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addRegTermLoss(currLevel, arapScaledLoss);
          else
            problemWrapper.addRegTermLoss(currLevel, arapScaledLoss);
        }

      //TOCK("SetupARAPCost"  + std::to_string( currLevel ) );
    }

  // inextensibility term
  //    cout << "inextent weight: " << weightParaLevel.inextentTermWeight << endl;
  if(weightParaLevel.inextentTermWeight)
    {
      //TICK("SetupInextentCost"  + std::to_string( currLevel ) );

      ceres::ScaledLoss* inextentScaledLoss = new ceres::ScaledLoss(
                                                                    NULL,
                                                                    weightParaLevel.inextentTermWeight,
                                                                    ceres::TAKE_OWNERSHIP);
      AddInextentCost(problem, inextentScaledLoss);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addRegTermLoss(currLevel, inextentScaledLoss);
          else
            problemWrapper.addRegTermLoss(currLevel, inextentScaledLoss);
        }

      //TOCK("SetupInextentCost"  + std::to_string( currLevel ) );
    }

  // deformation term
  //cout << "deform weight: " << weightParaLevel.deformWeight << endl;
  if(weightParaLevel.deformWeight)
    {
      //TICK("SetupDeformationCost"  + std::to_string( currLevel ) );

      ceres::ScaledLoss* deformScaledLoss = new ceres::ScaledLoss(
                                                                  NULL,
                                                                  weightParaLevel.deformWeight,
                                                                  ceres::TAKE_OWNERSHIP);
      AddDeformationCost(problem, deformScaledLoss);

      if(useProblemWrapper)
        {
          if(modeGT)
            problemWrapperGT.addRegTermLoss(currLevel, deformScaledLoss);
          else
            problemWrapper.addRegTermLoss(currLevel, deformScaledLoss);
        }

      //TOCK("SetupDeformationCost" + std::to_string( currLevel ) );
    }

  // temporal term
  // cout << "translation and rotation parameter: " << weightParaLevel.transWeight
  //      << " " << weightParaLevel.rotWeight << endl;
  if(weightParaLevel.transWeight || weightParaLevel.rotWeight)
    AddTemporalMotionCost(problem, sqrt(weightParaLevel.rotWeight),
                          sqrt(weightParaLevel.transWeight));

  // smoothing term
  if (weightParaLevel.smoothingTermWeight)
  {
	  //TICK("SetupARAPCost"  + std::to_string( currLevel ) );

	  ceres::LossFunction* loss_function = NULL;
	  if (trackerSettings.smoothingHuberWidth)
	  {
		  loss_function = new ceres::HuberLoss(weightParaLevel.smoothingHuberWidth);
	  }

	  ceres::ScaledLoss* scaled_loss = new ceres::ScaledLoss(
		  loss_function,
		  weightParaLevel.smoothingTermWeight,
		  ceres::TAKE_OWNERSHIP);
	  AddSmoothingCost(problem, scaled_loss);

	  if (useProblemWrapper)
	  {
		  if (modeGT)
			  problemWrapperGT.addRegTermLoss(currLevel, scaled_loss);
		  else
			  problemWrapper.addRegTermLoss(currLevel, scaled_loss);
	  }

	  //TOCK("SetupARAPCost"  + std::to_string( currLevel ) );
  }

  bool add_sh_coeff_spec_terms = trackerSettings.estimate_all_together || ( trackerSettings.refine_all_together 
	  && (!trackerSettings.update_intrinsics_finest_only || (trackerSettings.update_intrinsics_finest_only && currLevel == 0)) );

  // If we estimate shape, sh coefficients and specular highlights together, 
  // we need to add the temporal regularisation terms also
  if (trackerSettings.estimate_with_sh_coeff || add_sh_coeff_spec_terms)
  {
    // Temporal SH Coefficient
    if (trackerSettings.sh_coeff_temporal_weight > 0)
    {
      ceres::HuberLoss* huber_temporal_loss = NULL;

      if (trackerSettings.sh_coeff_temporal_huber_width > 0.0)
      {
        huber_temporal_loss =
          new ceres::HuberLoss(trackerSettings.sh_coeff_temporal_huber_width);
      }

      ceres::ScaledLoss* scaled_temporal_loss =
        new ceres::ScaledLoss(huber_temporal_loss,
        trackerSettings.sh_coeff_temporal_weight, 
        ceres::TAKE_OWNERSHIP);

      AddTemporalSHCoeffCost(problem, scaled_temporal_loss);

      if (useProblemWrapper)
      {
        if (modeGT)
          problemWrapperGT.addRegTermLoss(currLevel, scaled_temporal_loss);
        else
          problemWrapper.addRegTermLoss(currLevel, scaled_temporal_loss);
      }
    }

	if (add_sh_coeff_spec_terms)
	{
		// Specular smoothness
		if (trackerSettings.local_lighting_smoothness_weight > 0.0)
		{
			ceres::LossFunction* huber_loss = NULL;

			if (trackerSettings.local_lighting_smoothness_huber_width > 0)
			{
				huber_loss =
					new ceres::HuberLoss(trackerSettings.local_lighting_smoothness_huber_width);
			}

			ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
				huber_loss,
				trackerSettings.local_lighting_smoothness_weight,
				ceres::TAKE_OWNERSHIP);

			AddSpecularSmoothnessCost(problem, loss_function);

			if (useProblemWrapper)
			{
				if (modeGT)
					problemWrapperGT.addRegTermLoss(currLevel, loss_function);
				else
					problemWrapper.addRegTermLoss(currLevel, loss_function);
			}
		}

		// Specular magnitude
		if (trackerSettings.local_lighting_magnitude_weight > 0.0)
		{
			ceres::LossFunction* huber_loss = NULL;

			if (trackerSettings.local_lighting_magnitude_huber_width > 0)
			{
				huber_loss =
					new ceres::HuberLoss(trackerSettings.local_lighting_magnitude_huber_width);
			}

			ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
				huber_loss,
				trackerSettings.local_lighting_magnitude_weight,
				ceres::TAKE_OWNERSHIP);

			AddSpecularMagnitudeCost(problem, loss_function);

			if (useProblemWrapper)
			{
				if (modeGT)
					problemWrapperGT.addRegTermLoss(currLevel, loss_function);
				else
					problemWrapper.addRegTermLoss(currLevel, loss_function);
			}
		}

		// Temporal specular highlight
		if (trackerSettings.local_lighting_temporal_weight > 0.0)
		{
			ceres::LossFunction* huber_loss = NULL;

			if (trackerSettings.local_lighting_temporal_huber_width > 0)
			{
				huber_loss =
					new ceres::HuberLoss(trackerSettings.local_lighting_temporal_huber_width);
			}

			ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
				huber_loss,
				trackerSettings.local_lighting_temporal_weight,
				ceres::TAKE_OWNERSHIP);

			AddTemporalSpecularCost(problem, loss_function);

			if (useProblemWrapper)
			{
				if (modeGT)
					problemWrapperGT.addRegTermLoss(currLevel, loss_function);
				else
					problemWrapper.addRegTermLoss(currLevel, loss_function);
			}
		}
	}

  }
}

void DeformNRSFMTracker::EnergyMinimization(ceres::Problem& problem, bool estimate_motion)
{
  ceres::Solver::Options options;

  // solve the term and get solution
  options.max_num_iterations = trackerSettings.maxNumIterations[currLevel];
  options.linear_solver_type = mapLinearSolver(trackerSettings.linearSolver);
  options.minimizer_progress_to_stdout = trackerSettings.isMinimizerProgressToStdout;

  options.function_tolerance = trackerSettings.functionTolerances[currLevel];
  options.gradient_tolerance = trackerSettings.gradientTolerances[currLevel];
  options.parameter_tolerance = trackerSettings.parameterTolerances[currLevel];
  options.min_relative_decrease = trackerSettings.minRelativeDecreases[currLevel];

  options.initial_trust_region_radius = trackerSettings.initialTrustRegionRadiuses[0];
  options.max_trust_region_radius = trackerSettings.maxTrustRegionRadiuses[currLevel];
  options.min_trust_region_radius = trackerSettings.minTrustRegionRadiuses[currLevel];

  options.num_linear_solver_threads = trackerSettings.numLinearSolverThreads;
  options.num_threads = trackerSettings.numThreads;
  options.max_num_consecutive_invalid_steps = 0;

  options.minimizer_type = mapMinimizerType(trackerSettings.minimizerType);
  options.line_search_direction_type = mapLineSearchDirectionType(trackerSettings.lineSearchDirectionType);
  options.line_search_type = mapLineSearchType(trackerSettings.lineSearchType);
  options.nonlinear_conjugate_gradient_type = mapNonLinearCGType(trackerSettings.nonlinearConjugateGradientType);
  options.line_search_interpolation_type = mapLineSearchInterpType(trackerSettings.lineSearchInterpolationType);

  options.max_linear_solver_iterations = 100;

  EnergyCallback energy_callback = EnergyCallback();

  // set this to true if you want to have access to updated parameters inside callback
  options.update_state_every_iteration = true;
  options.callbacks.push_back(&energy_callback);

  ceres::Solver::Summary summary;

  ceresOutput << "*********************" << std::endl;
  ceresOutput << "Frame" << " " << currentFrameNo << "  Level" << " " << currLevel << std::endl;

  if(BAType == BA_MOTSTR && trackerSettings.doAlternation)
    {
	  if (estimate_motion)
	  {
		  // fix the structure and optimize the motion
		  AddConstantMask(problem, BA_STR); // make structure parameters constant

		  options.minimizer_type = ceres::TRUST_REGION;
		  ceres::Solve(options, &problem, &summary);

		  if (trackerSettings.saveResults)
		  {
			  ceresOutput << summary.FullReport() << std::endl;
			  ceresOutput << "Optimize Motion" << std::endl;
			  energy_callback.PrintEnergy(ceresOutput);
		  }
		  else if (trackerSettings.isMinimizerProgressToStdout)
		  {
			  std::cout << summary.FullReport() << std::endl;
			  std::cout << "Optimize Motion" << std::endl;
		  }

		  energy_callback.Reset();

		  // make the structure variable again after optimization
		  AddVariableMask(problem, BA_STR);

		  // cout << "printing after motion optimization started" << endl;
		  // cout << camPose[0] << " " << camPose[1] << " " << camPose[2] << endl;
		  // cout << camPose[3] << " " << camPose[4] << " " << camPose[5] << endl;
		  // cout << "printing after motion optimization finished" << endl;
	  }

      // optimize the shape
      // fix the motion
      AddConstantMask(problem, BA_MOT);

      options.minimizer_type = mapMinimizerType(trackerSettings.minimizerType);
      ceres::Solve(options, &problem, &summary);

      if(trackerSettings.saveResults)
        {
          ceresOutput << summary.FullReport() << std::endl;
          ceresOutput << "Optimize Shape" << std::endl;
          energy_callback.PrintEnergy(ceresOutput);
        }
      else if(trackerSettings.isMinimizerProgressToStdout)
        {
          std::cout << summary.FullReport() << std::endl;
          std::cout << "Optimize Shape" << std::endl;
        }

      // make the motion parameters variable again after optimization
      AddVariableMask(problem, BA_MOT);
    }
  else
    {
      // add const mask if necessary,
      // if we want to optimie motion or structure only
      AddConstantMask(problem, BAType == BA_MOT ? BA_STR : BA_MOTSTR);
      AddConstantMask(problem, BAType == BA_STR ? BA_MOT : BA_MOTSTR);

      ceres::Solve(options, &problem, &summary);
      if(trackerSettings.saveResults)
        {
          ceresOutput << summary.FullReport() << std::endl;
          ceresOutput << "Optimize Motion and Shape" << std::endl;
          energy_callback.PrintEnergy(ceresOutput);
        }
      else if(trackerSettings.isMinimizerProgressToStdout)
        {
          std::cout << summary.FullReport() << std::endl;
          std::cout << "Optimize Motion and Shape" << std::endl;
        }

      AddVariableMask(problem, BAType == BA_MOT ? BA_STR : BA_MOTSTR);
      AddVariableMask(problem, BAType == BA_STR ? BA_MOT : BA_MOTSTR);

    }

  // // print energy
  // if(trackerSettings.printEnergy)
  //   {
  //     double cost[8];
  //     double total_cost, sum_cost;

  //     problemWrapper.getAllCost(currLevel, cost, &total_cost, &sum_cost);

  //     energyOutput << currentFrameNo << '\t' << currLevel << '\t';

  //     for(int i = 0; i < 8; ++i)
  //       energyOutput << cost[i] << '\t';

  //     energyOutput << sum_cost << '\t' << total_cost << endl;

  //   }

  // print energy
  if(useProblemWrapper && trackerSettings.printEnergy)
    {
      double cost[NUM_PRINT_COSTS];

	  problemWrapper.getAllCost(currLevel, cost, &cost[NUM_PRINT_COSTS - 1], &cost[NUM_PRINT_COSTS - 2]);

      energyOutput << std::left << setw(15) << currentFrameNo << std::left << setw(15) << currLevel
                   << std::left << setw(15) << 0;

	  for (int i = 0; i < NUM_PRINT_COSTS; ++i)
        energyOutput << std::left << setw(15) << cost[i];

      energyOutput << endl;

	  for (int i = 0; i < NUM_PRINT_COSTS; ++i)
        energyOutputForR << std::left << setw(15) << currentFrameNo << std::left << setw(15) << currLevel
                         << std::left << setw(15) << cost[i] << std::left << setw(15) << "NotGT"
                         << std::left << setw(15) << costNames[i] << endl;
    }

}

// do optimization on ground truth shapes
void DeformNRSFMTracker::EnergyMinimizationGT(ceres::Problem& problem)
{

  ceres::Solver::Options options;

  // solve the term and get solution
  options.max_num_iterations = trackerSettings.maxNumIterations[currLevel];
  options.linear_solver_type = mapLinearSolver(trackerSettings.linearSolver);
  options.minimizer_progress_to_stdout = trackerSettings.isMinimizerProgressToStdout;

  options.function_tolerance = trackerSettings.functionTolerances[currLevel];
  options.gradient_tolerance = trackerSettings.gradientTolerances[currLevel];
  options.parameter_tolerance = trackerSettings.parameterTolerances[currLevel];
  options.min_relative_decrease = trackerSettings.minRelativeDecreases[currLevel];

  options.initial_trust_region_radius = trackerSettings.initialTrustRegionRadiuses[0];
  options.max_trust_region_radius = trackerSettings.maxTrustRegionRadiuses[currLevel];
  options.min_trust_region_radius = trackerSettings.minTrustRegionRadiuses[currLevel];

  options.num_linear_solver_threads = trackerSettings.numLinearSolverThreads;
  options.num_threads = trackerSettings.numThreads;
  options.max_num_consecutive_invalid_steps = 0;

  options.minimizer_type = mapMinimizerType(trackerSettings.minimizerType);
  options.line_search_direction_type = mapLineSearchDirectionType(trackerSettings.lineSearchDirectionType);
  options.line_search_type = mapLineSearchType(trackerSettings.lineSearchType);
  options.nonlinear_conjugate_gradient_type = mapNonLinearCGType(trackerSettings.nonlinearConjugateGradientType);
  options.line_search_interpolation_type = mapLineSearchInterpType(trackerSettings.lineSearchInterpolationType);

  EnergyCallback energy_callback = EnergyCallback();

  options.update_state_every_iteration = true;
  options.callbacks.push_back(&energy_callback);

  ceres::Solver::Summary summary;

  ceresOutput << "*********************" << std::endl;
  ceresOutput << "Ground Truth Optimization" << std::endl;
  ceresOutput << "Frame" << " " << currentFrameNo << "  Level" << " " << currLevel << std::endl;

  AddGroundTruthConstantMask(problem);

  // // check if ground truth has been changed after optimization
  // PangaeaMeshData& templateMesh = templateMeshPyramidGT.levels[currLevel];
  // PangaeaMeshData& currentMesh = currentMeshPyramidGT.levels[currLevel];
  // MeshDeformation& meshTrans = meshTransPyramidGT[currLevel];

  // double temp[3];
  // double first_point[3];

  // for(int k = 0; k < 3; ++k)
  //   first_point[k] = templateMesh.vertices[0][k] + meshTrans[0][k];

  // ceres::AngleAxisRotatePoint(camPoseGT, first_point, temp);

  // for(int k = 0; k < 3; ++k)
  //   first_point[k] = temp[k] + camPoseGT[k+3];

  // // check the first point
  // cout << "difference before optimization ";

  // for(int k = 0; k < 3; ++k)
  //   cout << first_point[k] - currentMesh.vertices[0][k] << " ";

  // cout << endl;

  ceres::Solve(options, &problem, &summary);
  ceresOutput << summary.FullReport() << endl;

  // print energy
  if(useProblemWrapper && trackerSettings.printEnergy)
    {
      double cost[10];

      problemWrapperGT.getAllCost(currLevel, cost, &cost[9], &cost[8]);

      energyOutput << std::left << setw(15) << currentFrameNo << std::left << setw(15) << currLevel
                   << std::left << setw(15) << 1;

      for(int i = 0; i < 10; ++i)
        energyOutput << std::left << setw(15) << cost[i];

      energyOutput << endl;

      for(int i = 0; i < 10; ++i)
        energyOutputForR << std::left << setw(15) << currentFrameNo << std::left << setw(15) << currLevel
                         << std::left << setw(15) << cost[i] << std::left << setw(15) << "GT"
                         << std::left << setw(15) << costNames[i] << endl;

    }

  AddGroundTruthVariableMask(problem);

  // for(int k = 0; k < 3; ++k)
  //   first_point[k] = templateMesh.vertices[0][k] + meshTrans[0][k];

  // ceres::AngleAxisRotatePoint(camPoseGT, first_point, temp);

  // for(int k = 0; k < 3; ++k)
  //   first_point[k] = temp[k] + camPoseGT[k+3];

  // // check the first point
  // cout << "difference after optimization ";

  // for(int k = 0; k < 3; ++k)
  //   cout << first_point[k] - currentMesh.vertices[0][k] << " ";

  // cout << endl;

}

void DeformNRSFMTracker::AddGroundTruthConstantMask(ceres::Problem& problem)
{
  // set all the stuff to constant except for arap local rotations
  // loop over all the parameter blocks

  // count the number of constant blocks
  int numConstantBlocks = 0;

  // set all rigid transformation variables to constant
  problem.SetParameterBlockConstant(&camPoseGT[0]);
  problem.SetParameterBlockConstant(&camPoseGT[3]);

  numConstantBlocks = 2;

  // set all translation variables to constant
  for(int i = 0; i < m_nMeshLevels; ++i)
    {
      int numVertices = meshTransPyramidGT[i].size();
      for(int j = 0; j < numVertices; ++j)
        {
          if(problem.HasParameterBlock( &meshTransPyramidGT[i][j][0] ))
            {
              problem.SetParameterBlockConstant( &meshTransPyramidGT[i][j][0] );
              numConstantBlocks++;
            }
        }
    }

  ceresOutput << "number of constant parameter blocks for ground truth optimization " <<
    numConstantBlocks << endl;

}

void DeformNRSFMTracker::AddGroundTruthVariableMask(ceres::Problem& problem)
{

  // set all the stuff to constant except for arap local rotations
  // loop over all the parameter blocks

  // set all rigid transformation variables to constant
  problem.SetParameterBlockVariable(&camPoseGT[0]);
  problem.SetParameterBlockVariable(&camPoseGT[3]);

  // set all translation variables to constant
  for(int i = 0; i < m_nMeshLevels; ++i)
    {
      int numVertices = meshTransPyramidGT[i].size();
      for(int j = 0; j < numVertices; ++j)
        {
          if(problem.HasParameterBlock( &meshTransPyramidGT[i][j][0] ))
            problem.SetParameterBlockVariable( &meshTransPyramidGT[i][j][0] );
        }
    }

}

double DeformNRSFMTracker::ComputeRMSError(PangaeaMeshData& results,
                                           PangaeaMeshData& resultsGT)
{
  int numVertices = results.numVertices;
  double gt_norm = 0;
  double diff_norm = 0;
  double diff;
  for(int i = 0; i < numVertices; ++i)
    {
      for(int k = 0; k < 3; ++k)
        {
          diff = resultsGT.vertices[i][k] - results.vertices[i][k];
          diff_norm += diff * diff;

          gt_norm += resultsGT.vertices[i][k] * resultsGT.vertices[i][k];
        }
    }

  return sqrt(diff_norm) / sqrt(gt_norm);

}

void DeformNRSFMTracker::CheckNaN()
{

  ceresOutput << "starting checking optimization variables" << endl;
  ceresOutput << "Frame" << " " << currentFrameNo << endl;

  if(modeGT)
    {


    }
  else
    {
      // check rigid transformation values
      for(int i = 0; i < 6; ++i)
        if(std::isnan(camPose[i]))
          ceresOutput << "camPose " << i << " is NaN " << endl;

      // check all other variables
      for(int i = 0; i < m_nMeshLevels; ++i)
        {
          int numVertices = meshTransPyramid[i].size();
          for(int j = 0; j < numVertices; ++j)
            {
              for(int k = 0; k < 3; ++k)
                {
                  if(std::isnan(meshTransPyramid[i][j][k]))
                    ceresOutput << "meshTransPyramid " << i << " " << j << " " << k << " is NaN " << endl;

                  if(std::isnan(prevMeshTransPyramid[i][j][k]))
                    ceresOutput << "previous meshTransPyramid " << i << " " << j << " " << k << " is NaN " << endl;

                  if(std::isnan(meshRotPyramid[i][j][k]))
                    ceresOutput << "meshRotPyramid " << i << " " << j << " " << k << " is NaN " << endl;

                  if(std::isnan(prevMeshRotPyramid[i][j][k]))
                    ceresOutput << "previous meshRotPyramid " << i << " " << j << " " << k << " is NaN " << endl;

                  if(std::isnan(templateMeshPyramid.levels[i].vertices[j][k]))
                    ceresOutput << "templateMeshPyramid " << i << " " << j << " " << k << " is NaN " << endl;

                }

            }

        }

    }

}

bool DeformNRSFMTracker::SaveData()
{
  // save shape
  char buffer[BUFFER_SIZE];
  std::ofstream shapeFile;
  std::stringstream shapeFilePath;
  sprintf(buffer,"shape_%04d.txt",currentFrameNo);
  shapeFilePath << trackerSettings.savePath << buffer;

  shapeFile.open(shapeFilePath.str().c_str(),std::ofstream::trunc);

  PangaeaMeshData& currentMesh = outputInfoPyramid[0].meshData;
  for(int i =0; i < currentMesh.numVertices; ++i)
    {
      shapeFile << currentMesh.vertices[i][0] << " "
                << currentMesh.vertices[i][1] << " "
                << currentMesh.vertices[i][2] << " "
                << std::endl;
    }
  shapeFile.close();

  return true;
}

bool DeformNRSFMTracker::SaveMeshToFile(TrackerOutputInfo& outputInfo)
{
  // create the directory if there isn't one
  if(!bfs::exists(trackerSettings.savePath.c_str()))
    {
      bfs::create_directories(trackerSettings.savePath.c_str());
    }

  // save current mesh to file for results checking afterwards
  char buffer[BUFFER_SIZE];
  std::stringstream meshFile;
  sprintf(buffer, trackerSettings.meshFormat.c_str(), currentFrameNo);
  meshFile << trackerSettings.savePath << buffer;

  PangaeaMeshIO::writeToFile(meshFile.str(), outputInfo.meshData,
	  trackerSettings.save_binary_mesh);

  return true;
}

bool DeformNRSFMTracker::SaveMeshPyramid()
{
  // meshPyramid directory
  char buffer[BUFFER_SIZE];
  std::stringstream mesh_pyramid_path;
  std::stringstream mesh_file;

  if(!bfs::exists(trackerSettings.savePath.c_str()))
    {
      bfs::create_directories(trackerSettings.savePath.c_str());
      cout << "creating dir " << trackerSettings.savePath << endl;
    }

  mesh_pyramid_path << trackerSettings.savePath << "/mesh_pyramid/";

  cout << mesh_pyramid_path.str() << endl;

  if(!bfs::exists(mesh_pyramid_path.str().c_str()))
    {
      bfs::create_directory(mesh_pyramid_path.str().c_str());
      cout << "creating dir " << mesh_pyramid_path.str() << endl;
    }

  // save mesh pyramid
  for(int i = 0; i < trackerSettings.levelsMeshPyramidSave.size(); ++i)
    {
      int mesh_level = trackerSettings.levelsMeshPyramidSave[i];

      sprintf(buffer, trackerSettings.meshPyramidFormat.c_str(), currentFrameNo, mesh_level);
      mesh_file << mesh_pyramid_path.str() << buffer;

      PangaeaMeshIO::writeToFile(mesh_file.str(), 
		  outputInfoPyramid[mesh_level].meshData, 
		  trackerSettings.save_binary_mesh);

      mesh_file.str("");
      //memset(&buffer[0], 0, sizeof(buffer));
    }

  if(trackerSettings.savePropPyramid)
    {
      // save mesh pyramid
      for(int i = 0; i < m_nMeshLevels; ++i)
        {
          sprintf(buffer, trackerSettings.propPyramidFormat.c_str(), currentFrameNo, i);
          mesh_file << mesh_pyramid_path.str() << buffer;

          PangaeaMeshIO::writeToFile(mesh_file.str(), 
			  outputPropPyramid[i].meshData, 
			  trackerSettings.save_binary_mesh);

          mesh_file.str("");
          //memset(&buffer[0], 0, sizeof(buffer));
        }
    }

  return true;
}

void DeformNRSFMTracker::updateRenderingLevel(TrackerOutputInfo** pOutputInfoRendering,
                                              int nRenderLevel, bool renderType)
{
  if(renderType)
    *pOutputInfoRendering = &(outputPropPyramid[nRenderLevel]);
  else
    *pOutputInfoRendering = &(outputInfoPyramid[nRenderLevel]);
  // // show camera center
  // cout << "camera center of level " << nRenderLevel << " is " << endl;
  // cout << outputInfoPyramid[nRenderLevel].meshData.center[0] << " "
  //      << outputInfoPyramid[nRenderLevel].meshData.center[1] << " "
  //      << outputInfoPyramid[nRenderLevel].meshData.center[2] << " "
  //      << endl;
}

void DeformNRSFMTracker::SaveThread(TrackerOutputInfo** pOutputInfoRendering)
{
  if(trackerSettings.saveResults)
    {
      if(trackerSettings.saveMesh)
        {
          if(trackerSettings.saveMeshPyramid)
            {
              // the current mesh pyramid is outputInfoPyramid
              cout << "save mesh pyramid started " << endl;
              SaveMeshPyramid();
              cout << "save mesh pyramid finished " << endl;
            }
          else
            {
              cout << "save mesh started " << endl;
              // the output mesh has been rotated and translated
              // to align with images
              SaveMeshToFile(**pOutputInfoRendering);
              cout << "save mesh finished " << endl;
            }
        }
      else
        SaveData();
    }
}

void DeformNRSFMTracker::AttachFeaturesToMeshPyramid()
{
  // attach features based on previous frame to the mesh
  // project those visible points based on visibilityMask
  // to the feature image
  for(int i = 0; i < m_nMeshLevels; ++i)
    {

      // mesh from previous frame
      // be careful when trying to use multi-threading, as outputInfoPyramid during
      // the process of tracking, or copy over the previous rotation and translation
      // and use the templateMesh
      PangaeaMeshData& prevMesh = outputInfoPyramid[i].meshData;
      FeatureLevel& featureLevel = pFeaturePyramid->getPrevFeatureLevel(i);
      CameraInfo& camInfo = pFeaturePyramid->getCameraInfo(i);
      vector<bool>& visibilityMask = visibilityMaskPyramid[i];

      // update the features in template mesh
      PangaeaMeshData& templateMesh = templateMeshPyramid.levels[i];
      AttachFeatureToMesh(&prevMesh, &featureLevel, &camInfo, visibilityMask, &templateMesh);

      // update the features in ground truth template mesh
      if( trackerSettings.hasGT &&  i < currentMeshPyramidGT.levels.size() )
        {
          vector<bool>& visibilityMaskGT = visibilityMaskPyramidGT[i];
          PangaeaMeshData& templateMeshGT =  templateMeshPyramidGT.levels[i];
          PangaeaMeshData& prevMeshGT = currentMeshPyramidGT.levels[i];
          AttachFeatureToMesh(&prevMeshGT, &featureLevel, &camInfo, visibilityMaskGT, &templateMeshGT);
        }

    }
}

void DeformNRSFMTracker::AttachFeatureToMesh(PangaeaMeshData* pMesh,
                                             FeatureLevel* pFeatureLevel,
                                             CameraInfo* pCamera,
                                             vector<bool>& visibilityMask,
                                             PangaeaMeshData* pOutputMesh)
{


  int numVertices = visibilityMask.size();
  pOutputMesh->featuresBuffer.resize(numVertices);

  int numChannels = pFeatureLevel->featureImageVec.size();

  for(int i = 0; i < numVertices; ++i)
    {
      pOutputMesh->featuresBuffer[i].resize( numChannels, 0 );

      if(visibilityMask[i])
        {
          getValue(pCamera,
                   pFeatureLevel,
                   &(pMesh->vertices[i][0]),
                   &(pOutputMesh->featuresBuffer[i][0]),
                   PE_FEATURE);
        }

    }

}

dataTermErrorType DeformNRSFMTracker::getPEType()
{
	return PEType;
}

void DeformNRSFMTracker::setMeshIntensityPyramid(PangaeaMeshPyramid &_templateIntensityPyramid)
{
	templateIntensityPyramid = std::move(_templateIntensityPyramid);
}

void DeformNRSFMTracker::updateIntrinsics(unsigned char* pColorImageRGB)
{
	// Create color image
	cv::Mat color_image_uchar(m_nHeight, m_nWidth, CV_8UC3, pColorImageRGB);

	// Convert color image to double
	InternalColorImageType color_image;
	color_image_uchar.convertTo(color_image,
		cv::DataType<Vec3d>::type, 1. / 255);

  // Pointers to mesh, projections, visibility, albedo and local lighting
  TrackerOutputInfo &outputInfo = outputInfoPyramid[currLevel];
  PangaeaMeshData &mesh = outputInfo.meshData;
  const vector<vector<CoordinateType> > &meshProj = outputInfo.meshProj;
  const vector<bool> &visibility = visibilityMaskPyramid[currLevel];
  MeshDeformation &albedos = mesh.colors;
  MeshDeformation &local_lightings = templateMeshPyramid.levels[currLevel].specular_colors;
  vector<double> &sh_coeff = templateMeshPyramid.levels[currLevel].sh_coefficients;
  int sh_order = templateMeshPyramid.levels[currLevel].sh_order;
  const MeshDeformation &template_albedos = templateAlbedoPyramid[currLevel];

  mesh.computeNormalsNeil();

  // Intensities
  MeshDeformation intensities;

  if (trackerSettings.estimate_diffuse)
  {
    // Estimate brightness
    std::vector<cv::Mat> planes(3);
    cv::split(color_image, planes);
    InternalIntensityImageType brightnessImage = cv::Mat(
      cv::max(planes[2],
      cv::max(planes[1], planes[0]))
      );

    // Brightness
    vector<double> brightness;

	// Estimate diffuse image
	cv::Mat diffuse_image_uchar;
	estimateDiffuse(color_image_uchar, diffuse_image_uchar);

	// Estimate diffuse brightness
	std::vector<cv::Mat> diffuse_planes(3);
	cv::split(diffuse_image_uchar, diffuse_planes);
	cv::Mat diffuseBrightnessImage_uchar = cv::Mat(
		cv::max(diffuse_planes[2],
		cv::max(diffuse_planes[1], diffuse_planes[0]))
		);

	// Convert diffuse brightness to double
	InternalIntensityImageType diffuseBrightnessImage;
	diffuseBrightnessImage_uchar.convertTo(diffuseBrightnessImage,
		cv::DataType<double>::type, 1. / 255);

	// Compute specular brightness as the difference between full brightness 
	// and diffuse brightness
	InternalIntensityImageType specularImage;
	cv::subtract(brightnessImage, diffuseBrightnessImage, specularImage);

	// Get projected values from images
	initProjectedValues(meshProj, visibility, color_image, brightnessImage,
		specularImage, intensities, brightness, local_lightings);

	// Compute weight based on specularities
	vector< double > specular_weights;
	specular_weights.reserve(mesh.numVertices);
	for (size_t i = 0; i < mesh.numVertices; i++)
	{
		double weight = exp(
			-local_lightings[i][0] / (2 * trackerSettings.specular_weight_var)
			);
		specular_weights.push_back(weight);
	}

    //vector<double> max_albedos;
    //max_albedos.reserve(albedos.size());
    //for (size_t i = 0; i < albedos.size(); i++)
    //{
    //  max_albedos.push_back(*max_element(albedos[i].begin(), albedos[i].end()));
    //}

    // Estimate spherical harmonic coefficients
    cout << "Estimating spherical harmonic coefficients..." << endl;
    estimateSHCoeff(mesh, visibility, intensities, albedos,
      specular_weights, sh_order, sh_coeff);

    // Update shading values
    vector<double> shadings;
    updateShading(mesh, sh_coeff, sh_order, shadings);

    // Estimate albedo map assuming zero local lighting
    if (trackerSettings.update_albedo)
    {
      cout << "Estimating albedo map..." << endl;
      estimateAlbedo(mesh, visibility, intensities, shadings, local_lightings,
        specular_weights, template_albedos, albedos);
    }

    // Estimate local lighting variations
    cout << "Estimating specularities..." << endl;
    estimateLocalLighting(mesh, visibility, intensities, albedos, shadings, 
      local_lightings);
  }
  else
  {
		projectValues(meshProj, visibility, color_image, intensities);

		if (trackerSettings.estimate_sh_coeff_specular_together)
		{
		  // Estimate sh coeff and local lighting variations together
		  cout << "Estimating sh coeff and specularities..." << endl;
		  estimateSHCoeffLocalLighting(mesh, visibility, intensities, albedos, 
			sh_order, sh_coeff, local_lightings);
		}
		else
		{
			if (!trackerSettings.estimate_with_sh_coeff)
			{
				// Estimate spherical harmonic coefficients
				cout << "Estimating spherical harmonic coefficients..." << endl;
				estimateSHCoeff(mesh, visibility, intensities, albedos, local_lightings,
					sh_order, sh_coeff);
			}

		  // Update shading values
		  vector<double> shadings;
		  updateShading(mesh, sh_coeff, sh_order, shadings);

		  if (trackerSettings.estimate_specularities)
		  {
			  // Estimate local lighting variations
			  cout << "Estimating specularities..." << endl;
			  estimateLocalLighting(mesh, visibility, intensities, albedos, shadings,
				  local_lightings);
		  }
		}
  }

	//// Propagate albedo and local lighting from finest level to coarser levels
	//cout << "Propagating albedo and local lighting variations to coarser levels..." << endl;
	//propagateAlbedoLocalLightingFineToCoarse();
}

void DeformNRSFMTracker::initProjectedValues(
	const vector<vector<CoordinateType> > &meshProj,
	const vector<bool> &visibility, const InternalColorImageType &colorImage,
	const InternalIntensityImageType &brightnessImage,
	const InternalIntensityImageType &specularImage,
	vector< vector<double> > &intensities,
	vector<double> &brightness,
	vector< vector<double> > &local_lightings)
{
	vector<double> zeros = { 0, 0, 0 };

	size_t n_vertices = meshProj.size();

	intensities.resize(n_vertices, zeros);
	brightness.resize(n_vertices, 0);
	local_lightings.resize(n_vertices, zeros);

	InternalIntensityImageType colorImageSplit[3];

	cv::split(colorImage, colorImageSplit);

	for (size_t i = 0; i < meshProj.size(); i++)
	{
		if (visibility[i])
		{
			CoordinateType u = meshProj[i][0];
			CoordinateType v = meshProj[i][1];

			for (size_t j = 0; j < 3; j++)
			{
				CoordinateType intensity_value;
				SampleLinear(colorImageSplit[j], v, u, &intensity_value);

				intensities[i][j] = intensity_value;
			}

			CoordinateType brightness_value;
			SampleLinear(brightnessImage, v, u, &brightness_value);

			brightness[i] = brightness_value;

			CoordinateType specular_value;
			SampleLinear(specularImage, v, u, &specular_value);

			local_lightings[i][0] = specular_value;
			local_lightings[i][1] = specular_value;
			local_lightings[i][2] = specular_value;
		}
	}
}

void DeformNRSFMTracker::projectValues(
  const vector<vector<CoordinateType> > &meshProj,
  const vector<bool> &visibility, const InternalColorImageType &colorImage,
  vector< vector<double> > &intensities)
{
  vector<double> zeros = { 0, 0, 0 };

  size_t n_vertices = meshProj.size();

  intensities.resize(n_vertices, zeros);

  InternalIntensityImageType colorImageSplit[3];

  cv::split(colorImage, colorImageSplit);

  for (size_t i = 0; i < meshProj.size(); i++)
  {
    if (visibility[i])
    {
      CoordinateType u = meshProj[i][0];
      CoordinateType v = meshProj[i][1];

      for (size_t j = 0; j < 3; j++)
      {
        CoordinateType intensity_value;
        SampleLinear(colorImageSplit[j], v, u, &intensity_value);

        intensities[i][j] = intensity_value;
      }
    }
  }
}

void DeformNRSFMTracker::estimateSHCoeff(
	const PangaeaMeshData &mesh, const vector<bool> &visibility,
	const vector<vector<double>> &intensities, vector<vector<double>> &albedos,
	const vector<double> &specular_weights,
	const int sh_order, vector<double> &sh_coeff)
{
	vector<double> black = { 0.0, 0.0, 0.0 };

	// Initialize with zeros

	//std::fill(sh_coeff.begin(), sh_coeff.end(), 0);

	ceres::Problem problem;

	ceres::LossFunction *huber_data_loss = NULL;

	if (trackerSettings.sh_coeff_data_huber_width > 0.0)
	{
		huber_data_loss = 
			new ceres::HuberLoss(trackerSettings.sh_coeff_data_huber_width);
	}

	bool use_lower_bound_shading = false;
	bool use_upper_bound_shading = false;

	const unsigned int N_CHANNELS = 3;

	const int n_sh_coeff = pow(sh_order + 1, 2);

	for (size_t i = 0; i < mesh.numVertices; i++)
	{
		if (visibility[i])
		{
			ceres::ScaledLoss* scaled_data_loss = new ceres::ScaledLoss(
				huber_data_loss,
				specular_weights[i],
				ceres::TAKE_OWNERSHIP);

			ResidualPhotometricErrorWithNormal *residual =
				new ResidualPhotometricErrorWithNormal(
				&intensities[i][0], (double *)&mesh.normals[i][0], N_CHANNELS, 
				sh_order, use_lower_bound_shading, use_upper_bound_shading, trackerSettings.use_white_specularities);

			ceres::DynamicAutoDiffCostFunction<ResidualPhotometricErrorWithNormal, 5>* dyn_cost_function
				= new ceres::DynamicAutoDiffCostFunction< ResidualPhotometricErrorWithNormal, 5 >(residual);

			// List of pointers to translations per vertex
			vector<double*> v_parameter_blocks;

			// Albedo
			dyn_cost_function->AddParameterBlock(N_CHANNELS);
			v_parameter_blocks.push_back(&albedos[i][0]);
			// SH Coeff
			dyn_cost_function->AddParameterBlock(n_sh_coeff);
			v_parameter_blocks.push_back(&sh_coeff[0]);
			// Local lighting variations
			unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;
			dyn_cost_function->AddParameterBlock(specular_n_channels);
			v_parameter_blocks.push_back(&black[0]);

			dyn_cost_function->SetNumResiduals(N_CHANNELS);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				dyn_cost_function,
				scaled_data_loss,
				v_parameter_blocks);
		}
	}

	if (trackerSettings.sh_coeff_temporal_weight > 0)
	{
		ceres::HuberLoss* huber_temporal_loss = NULL;

		if (trackerSettings.sh_coeff_temporal_huber_width > 0.0)
		{
			huber_temporal_loss =
				new ceres::HuberLoss(trackerSettings.sh_coeff_temporal_huber_width);
		}

		ceres::ScaledLoss* scaled_temporal_loss =
			new ceres::ScaledLoss(huber_temporal_loss,
			trackerSettings.sh_coeff_temporal_weight, 
			ceres::TAKE_OWNERSHIP);

		ResidualTemporalSHCoeff *residual =
			new ResidualTemporalSHCoeff(prevSHCoeff);

		ceres::DynamicAutoDiffCostFunction<ResidualTemporalSHCoeff, 5>* dyn_cost_function
			= new ceres::DynamicAutoDiffCostFunction< ResidualTemporalSHCoeff, 5 >(residual);

		// List of pointers to sh coefficients
		vector<double*> v_parameter_blocks;

		// SH Coeff
		dyn_cost_function->AddParameterBlock(sh_coeff.size());
		v_parameter_blocks.push_back(&sh_coeff[0]);

		dyn_cost_function->SetNumResiduals(sh_coeff.size());

		ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
			dyn_cost_function,
			scaled_temporal_loss,
			v_parameter_blocks);
	}

	for (size_t i = 0; i < albedos.size(); i++)
	{
		if (visibility[i])
		{
			problem.SetParameterBlockConstant(&albedos[i][0]);
		}
	}
	problem.SetParameterBlockConstant(&black[0]);

	ceres::Solver::Options options;
	options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
	options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
	options.line_search_type = ceres::LineSearchType::WOLFE;
	options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
	options.line_search_interpolation_type = ceres::LineSearchInterpolationType::CUBIC;
	options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
	options.max_num_iterations = 10;
	options.num_threads = 8;
	options.initial_trust_region_radius = 1e4;
	options.max_trust_region_radius = 1e8;
	options.min_trust_region_radius = 1e-32;
	options.min_relative_decrease = 1e-3;
	options.max_num_consecutive_invalid_steps = 5;
	options.function_tolerance = 1e-4;
	options.gradient_tolerance = 1e-10;
	options.parameter_tolerance = 1e-10;
	options.linear_solver_type = ceres::LinearSolverType::CGNR;
	options.preconditioner_type = ceres::PreconditionerType::JACOBI;
	options.num_linear_solver_threads = 8;
	options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
	options.update_state_every_iteration = 0;

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
}

void DeformNRSFMTracker::estimateSHCoeff(
  const PangaeaMeshData &mesh, const vector<bool> &visibility,
  const vector<vector<double>> &intensities, vector<vector<double>> &albedos,
  vector<vector<double>> &local_lightings, const int sh_order,  
  vector<double> &sh_coeff)
{
  ceres::Problem problem;

  ceres::LossFunction *huber_data_loss = NULL;

  if (trackerSettings.sh_coeff_data_huber_width > 0.0)
  {
    huber_data_loss = 
      new ceres::HuberLoss(trackerSettings.sh_coeff_data_huber_width);
  }

  ceres::ScaledLoss* scaled_data_loss = new ceres::ScaledLoss(
    huber_data_loss,
    trackerSettings.sh_coeff_data_weight,
    ceres::TAKE_OWNERSHIP);

  bool use_lower_bound_shading = false;
  bool use_upper_bound_shading = false;

  const unsigned int N_CHANNELS = 3;

  const int n_sh_coeff = pow(sh_order + 1, 2);

  for (size_t i = 0; i < mesh.numVertices; i++)
  {
    if (visibility[i])
    {
      ResidualPhotometricErrorWithNormal *residual =
        new ResidualPhotometricErrorWithNormal(
        &intensities[i][0], (double *)&mesh.normals[i][0], N_CHANNELS, 
        sh_order, use_lower_bound_shading, use_upper_bound_shading, trackerSettings.use_white_specularities);

      ceres::DynamicAutoDiffCostFunction<ResidualPhotometricErrorWithNormal, 5>* dyn_cost_function
        = new ceres::DynamicAutoDiffCostFunction< ResidualPhotometricErrorWithNormal, 5 >(residual);

      // List of pointers to translations per vertex
      vector<double*> v_parameter_blocks;

      // Albedo
      dyn_cost_function->AddParameterBlock(N_CHANNELS);
      v_parameter_blocks.push_back(&albedos[i][0]);
      // SH Coeff
      dyn_cost_function->AddParameterBlock(n_sh_coeff);
      v_parameter_blocks.push_back(&sh_coeff[0]);
      // Local lighting variations
	  unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;
      dyn_cost_function->AddParameterBlock(specular_n_channels);
      v_parameter_blocks.push_back(&local_lightings[i][0]);

      dyn_cost_function->SetNumResiduals(N_CHANNELS);

      ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
        dyn_cost_function,
        scaled_data_loss,
        v_parameter_blocks);
    }
  }

  if (trackerSettings.sh_coeff_temporal_weight > 0)
  {
    ceres::HuberLoss* huber_temporal_loss = NULL;

    if (trackerSettings.sh_coeff_temporal_huber_width > 0.0)
    {
      huber_temporal_loss =
        new ceres::HuberLoss(trackerSettings.sh_coeff_temporal_huber_width);
    }

    ceres::ScaledLoss* scaled_temporal_loss =
      new ceres::ScaledLoss(huber_temporal_loss,
      trackerSettings.sh_coeff_temporal_weight, 
      ceres::TAKE_OWNERSHIP);

    ResidualTemporalSHCoeff *residual =
      new ResidualTemporalSHCoeff(prevSHCoeff);

    ceres::DynamicAutoDiffCostFunction<ResidualTemporalSHCoeff, 5>* dyn_cost_function
      = new ceres::DynamicAutoDiffCostFunction< ResidualTemporalSHCoeff, 5 >(residual);

    // List of pointers to sh coefficients
    vector<double*> v_parameter_blocks;

    // SH Coeff
    dyn_cost_function->AddParameterBlock(sh_coeff.size());
    v_parameter_blocks.push_back(&sh_coeff[0]);

    dyn_cost_function->SetNumResiduals(sh_coeff.size());

    ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
      dyn_cost_function,
      scaled_temporal_loss,
      v_parameter_blocks);
  }

  for (size_t i = 0; i < albedos.size(); i++)
  {
    if (visibility[i])
    {
      problem.SetParameterBlockConstant(&albedos[i][0]);
      problem.SetParameterBlockConstant(&local_lightings[i][0]);
    }
  }

  ceres::Solver::Options options;
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
  options.line_search_type = ceres::LineSearchType::WOLFE;
  options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
  options.line_search_interpolation_type = ceres::LineSearchInterpolationType::CUBIC;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10;
  options.num_threads = 8;
  options.initial_trust_region_radius = 1e4;
  options.max_trust_region_radius = 1e8;
  options.min_trust_region_radius = 1e-32;
  options.min_relative_decrease = 1e-3;
  options.max_num_consecutive_invalid_steps = 5;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-10;
  options.linear_solver_type = ceres::LinearSolverType::CGNR;
  options.preconditioner_type = ceres::PreconditionerType::JACOBI;
  options.num_linear_solver_threads = 8;
  options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
  options.update_state_every_iteration = 0;

  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  cout << summary.FullReport() << endl;
}

void DeformNRSFMTracker::updateShading(const PangaeaMeshData &mesh,
	const vector<double> &sh_coeff, const int sh_order,
	vector<double> &shadings)
{
	shadings.reserve(mesh.numVertices);
	for (size_t i = 0; i < mesh.numVertices; i++)
	{
		double shading = computeShading(&mesh.normals[i][0], &sh_coeff[0], 
			sh_order);
		shading = max(shading, std::numeric_limits<double>::epsilon());
		shadings.push_back(shading);
	}
}

void DeformNRSFMTracker::estimateAlbedo(const PangaeaMeshData &mesh,
	const vector<bool> &visibility,
	const MeshDeformation &intensities,
	const vector<double> &shadings,
	const MeshDeformation &local_lightings,
	const vector<double> &specular_weights,
	const MeshDeformation &template_albedos,
	MeshDeformation &albedos)
{
	vector<double> black = { 0, 0, 0 };

	//for (size_t i = 0; i < albedos.size(); i++)
	//{
	//	albedos[i][0] = intensities[i][0] / shadings[i];
	//	albedos[i][1] = intensities[i][1] / shadings[i];
	//	albedos[i][2] = intensities[i][2] / shadings[i];

	//	//fill(albedos[i].begin(), albedos[i].end(), 0);
	//}

	ceres::Problem problem;

	ceres::LossFunction *huber_data_loss = NULL;
	if (trackerSettings.albedo_data_huber_width > 0.0)
	{
		huber_data_loss =
			new ceres::HuberLoss(trackerSettings.albedo_data_huber_width);
	}

	for (size_t i = 0; i < mesh.numVertices; i++)
	{
		if (visibility[i])
		{
			ceres::ScaledLoss* lighting_scaled_loss = new ceres::ScaledLoss(
				huber_data_loss,
				min(shadings[i], specular_weights[i]),
				ceres::TAKE_OWNERSHIP);

			ceres::ScaledLoss* scaled_data_loss = new ceres::ScaledLoss(
				lighting_scaled_loss,
				trackerSettings.albedo_data_weight,
				ceres::TAKE_OWNERSHIP);

			ResidualPhotometricErrorWithShading *residual =
				new ResidualPhotometricErrorWithShading(
				&intensities[i][0], shadings[i], 3);

			ceres::DynamicAutoDiffCostFunction<ResidualPhotometricErrorWithShading, 5>* dyn_cost_function
				= new ceres::DynamicAutoDiffCostFunction< ResidualPhotometricErrorWithShading, 5 >(residual);

			// List of pointers to translations per vertex
			vector<double*> v_parameter_blocks;

			// Albedo
			dyn_cost_function->AddParameterBlock(3);
			v_parameter_blocks.push_back(&albedos[i][0]);
			// No local lighting variations
			dyn_cost_function->AddParameterBlock(3);
			v_parameter_blocks.push_back(&black[0]);

			dyn_cost_function->SetNumResiduals(3);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				dyn_cost_function,
				scaled_data_loss,
				v_parameter_blocks);
		}
	}

	if (trackerSettings.albedo_smoothness_weight > 0.0)
	{
		ceres::LossFunction* huber_loss = NULL;

		if (trackerSettings.albedo_smoothness_huber_width > 0)
		{
			huber_loss = 
				new ceres::HuberLoss(trackerSettings.albedo_smoothness_huber_width);
		}

		ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
			huber_loss,
			trackerSettings.albedo_smoothness_weight,
			ceres::TAKE_OWNERSHIP);

		for (size_t i = 0; i < mesh.numVertices; i++)
		{
			//if (visibility[i])
			//{
				for (size_t j = 0; j < mesh.adjVerticesInd[i].size(); j++)
				{
					unsigned int adj_v_idx = mesh.adjVerticesInd[i][j];

					//if (visibility[adj_v_idx])
					//{
						double weight = computeDiffWeight(local_lightings, intensities, i, j);

						if (weight > 0.0)
						{
							ResidualWeightedDifference *residual =
								new ResidualWeightedDifference(weight, 3);

							ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>* cost_function =
								new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>(residual);

							ceres::ResidualBlockId residualBlockId =
								problem.AddResidualBlock(
								cost_function,
								loss_function,
								&albedos[i][0],
								&albedos[adj_v_idx][0]);
						}
					//}
				}
			//}
		}
	}

	if (trackerSettings.albedo_difference_weight > 0)
	{
		ceres::HuberLoss* huber_temporal_loss = NULL;

		if (trackerSettings.albedo_difference_huber_width > 0.0)
		{
			huber_temporal_loss =
				new ceres::HuberLoss(trackerSettings.albedo_difference_huber_width);
		}

		ceres::ScaledLoss* scaled_temporal_loss =
			new ceres::ScaledLoss(huber_temporal_loss,
			trackerSettings.albedo_difference_weight,
			ceres::TAKE_OWNERSHIP);
		
		for (size_t i = 0; i < mesh.numVertices; i++)
		{
			//if (visibility[i])
			//{
				ResidualTemplateAlbedo *residual =
					new ResidualTemplateAlbedo(&template_albedos[i][0], false);

				ceres::DynamicAutoDiffCostFunction<ResidualTemplateAlbedo, 5>* dyn_cost_function
					= new ceres::DynamicAutoDiffCostFunction< ResidualTemplateAlbedo, 5 >(residual);

				// List of pointers to translations per vertex
				vector<double*> v_parameter_blocks;

				// SH Coeff
				dyn_cost_function->AddParameterBlock(3);
				v_parameter_blocks.push_back(&albedos[i][0]);

				dyn_cost_function->SetNumResiduals(3);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					dyn_cost_function,
					scaled_temporal_loss,
					v_parameter_blocks);
			//}
		}
	}

	problem.SetParameterBlockConstant(&black[0]);

	bool use_lower_bound_albedo = true;
	bool use_upper_bound_albedo = true;

	for (size_t i = 0; i < mesh.numVertices; ++i)
	{
		//if (visibility[i])
		//{
			if (use_lower_bound_albedo)
			{
				problem.SetParameterLowerBound(&albedos[i][0], 0, 0.0);
				problem.SetParameterLowerBound(&albedos[i][0], 1, 0.0);
				problem.SetParameterLowerBound(&albedos[i][0], 2, 0.0);
			}

			if (use_upper_bound_albedo)
			{
				problem.SetParameterUpperBound(&albedos[i][0], 0, 1.0);
				problem.SetParameterUpperBound(&albedos[i][0], 1, 1.0);
				problem.SetParameterUpperBound(&albedos[i][0], 2, 1.0);
			}
		//}
	}

	ceres::Solver::Options options;
	options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
	options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
	options.line_search_type = ceres::LineSearchType::WOLFE;
	options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
	options.line_search_interpolation_type = ceres::LineSearchInterpolationType::CUBIC;
	options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
	options.max_num_iterations = 10;
	options.num_threads = 8;
	options.initial_trust_region_radius = 1e4;
	options.max_trust_region_radius = 1e8;
	options.min_trust_region_radius = 1e-32;
	options.min_relative_decrease = 1e-3;
	options.max_num_consecutive_invalid_steps = 5;
	options.function_tolerance = 1e-4;
	options.gradient_tolerance = 1e-10;
	options.parameter_tolerance = 1e-10;
	options.linear_solver_type = ceres::LinearSolverType::CGNR;
	options.preconditioner_type = ceres::PreconditionerType::JACOBI;
	options.num_linear_solver_threads = 8;
	options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
	options.update_state_every_iteration = 0;

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
}

double DeformNRSFMTracker::computeDiffWeight(
	const MeshDeformation &local_lightings, const MeshDeformation &colors, 
	const unsigned int v_idx1, const unsigned int v_idx2)
{
	double weight = 0.0;

	weight = trackerSettings.smoothness_specular_weight 
		* max(local_lightings[v_idx1][0], local_lightings[v_idx2][0]);

	if (trackerSettings.smoothness_color_diff_var > 0)
	{
		const vector<double> &color1 = colors[v_idx1];
		const vector<double> &color2 = colors[v_idx2];
		vector<double> color_diff = {
			color1[0] - color2[0],
			color1[1] - color2[1],
			color1[2] - color2[2]
		};

		double color_diff_norm2 = pow(color_diff[0], 2)
			+ pow(color_diff[1], 2)
			+ pow(color_diff[2], 2);

		if (color_diff_norm2 <= trackerSettings.smoothness_color_diff_threshold)
		{
			weight += exp(
				-color_diff_norm2 / (2 * trackerSettings.smoothness_color_diff_var)
				);
		}
	}

	return weight;
}

void DeformNRSFMTracker::estimateLocalLighting(const PangaeaMeshData &mesh,
	const vector<bool> &visibility,
	const MeshDeformation &intensities,
	const MeshDeformation &albedos,
	const vector<double> &shadings,
	MeshDeformation &local_lightings)
{
	//for (size_t i = 0; i < local_lightings.size(); i++)
	//{
	//	fill(local_lightings[i].begin(), local_lightings[i].end(), 0);
	//}

	ceres::Problem problem;

	ceres::LossFunction *data_loss_function = NULL;
	if (trackerSettings.local_lighting_data_huber_width > 0.0)
	{
		data_loss_function = 
			new ceres::HuberLoss(trackerSettings.local_lighting_data_huber_width);
	}

	ceres::ScaledLoss* scaled_data_loss = new ceres::ScaledLoss(
		data_loss_function,
		trackerSettings.local_lighting_data_weight,
		ceres::TAKE_OWNERSHIP);

	unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;

	for (size_t i = 0; i < mesh.numVertices; i++)
	{
		if (visibility[i])
		{
			for (size_t j = 0; j < 3; j++)
			{
				local_lightings[i][j] = 
					intensities[i][j] - albedos[i][j] * shadings[i];
			}

			if (trackerSettings.use_white_specularities)
			{
				local_lightings[i][0] = (local_lightings[i][0] + local_lightings[i][1] + local_lightings[i][2]) / 3;
			}

			ResidualPhotometricErrorWithAlbedoShading *residual = 
				new ResidualPhotometricErrorWithAlbedoShading(
				&intensities[i][0], &albedos[i][0], shadings[i], 3, trackerSettings.use_white_specularities);

			ceres::DynamicAutoDiffCostFunction<ResidualPhotometricErrorWithAlbedoShading, 5>* dyn_cost_function
				= new ceres::DynamicAutoDiffCostFunction< ResidualPhotometricErrorWithAlbedoShading, 5 >(residual);

			// List of pointers to translations per vertex
			vector<double*> v_parameter_blocks;

			// Local lighting variations
			dyn_cost_function->AddParameterBlock(specular_n_channels);

			v_parameter_blocks.push_back(&local_lightings[i][0]);

			dyn_cost_function->SetNumResiduals(3);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				dyn_cost_function,
				scaled_data_loss,
				v_parameter_blocks);
		}
	}

	if (trackerSettings.local_lighting_smoothness_weight > 0.0)
	{
		ceres::LossFunction* huber_loss = NULL;

		if (trackerSettings.local_lighting_smoothness_huber_width > 0)
		{
			huber_loss = 
				new ceres::HuberLoss(trackerSettings.local_lighting_smoothness_huber_width);
		}

		ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
			huber_loss,
			trackerSettings.local_lighting_smoothness_weight,
			ceres::TAKE_OWNERSHIP);

		for (size_t i = 0; i < mesh.numVertices; i++)
		{
			//if (visibility[i])
			//{
				for (size_t j = 0; j < mesh.adjVerticesInd[i].size(); j++)
				{
					unsigned int adj_v_idx = mesh.adjVerticesInd[i][j];

					//if (visibility[adj_v_idx])
					//{
						double weight = 1;// diff_weights[i][j];

						if (weight > 0.0)
						{
							if (trackerSettings.use_white_specularities)
							{
								ResidualWeightedDifference *residual = new ResidualWeightedDifference(
									weight, 1);

								ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>* cost_function =
									new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>(residual);

								ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
									cost_function,
									loss_function,
									&local_lightings[i][0],
									&local_lightings[adj_v_idx][0]);
							}
							else
							{
								ResidualWeightedDifference *residual = new ResidualWeightedDifference(
									weight, 3);

								ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>* cost_function =
									new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>(residual);

								ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
									cost_function,
									loss_function,
									&local_lightings[i][0],
									&local_lightings[adj_v_idx][0]);
							}
						}
					//}
				}
			//}
		}
	}

	if (trackerSettings.local_lighting_magnitude_weight > 0.0)
	{
		ceres::LossFunction* huber_loss = NULL;

		if (trackerSettings.local_lighting_magnitude_huber_width > 0)
		{
			huber_loss = 
				new ceres::HuberLoss(trackerSettings.local_lighting_magnitude_huber_width);
		}

		ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
			huber_loss,
			trackerSettings.local_lighting_magnitude_weight,
			ceres::TAKE_OWNERSHIP);

		for (size_t i = 0; i < mesh.numVertices; i++)
		{
			//if (visibility[i])
			//{

			if (trackerSettings.use_white_specularities)
			{
				ResidualValueMagnitude *residual = new ResidualValueMagnitude(1);

				ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>* cost_function =
					new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>(residual);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					cost_function,
					loss_function,
					&local_lightings[i][0]);
			}
			else
			{
				ResidualValueMagnitude *residual = new ResidualValueMagnitude(3);

				ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>* cost_function =
					new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>(residual);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					cost_function,
					loss_function,
					&local_lightings[i][0]);
			}

			//}
		}
	}

	if (trackerSettings.local_lighting_temporal_weight > 0.0)
	{
		ceres::LossFunction* huber_loss = NULL;

		if (trackerSettings.local_lighting_temporal_huber_width > 0)
		{
			huber_loss =
				new ceres::HuberLoss(trackerSettings.local_lighting_temporal_huber_width);
		}

		ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
			huber_loss,
			trackerSettings.local_lighting_temporal_weight,
			ceres::TAKE_OWNERSHIP);

		vector<vector<double>> &prev_local_lightings = prevMeshSpecularPyramid[currLevel];

		for (size_t i = 0; i < mesh.numVertices; i++)
		{
			//if (visibility[i])
			//{
				ResidualTemporalLocalLighting *residual = 
					new ResidualTemporalLocalLighting(prev_local_lightings[i], specular_n_channels);

				ceres::DynamicAutoDiffCostFunction<ResidualTemporalLocalLighting, 5>* dyn_cost_function
					= new ceres::DynamicAutoDiffCostFunction< ResidualTemporalLocalLighting, 5 >(residual);

				// List of pointers to translations per vertex
				vector<double*> v_parameter_blocks;

				// SH Coeff

				dyn_cost_function->AddParameterBlock(specular_n_channels);

				v_parameter_blocks.push_back(&local_lightings[i][0]);

				dyn_cost_function->SetNumResiduals(specular_n_channels);

				ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
					dyn_cost_function,
					loss_function,
					v_parameter_blocks);
			//}
		}
	}

	bool use_lower_bound = true;
	bool use_upper_bound = true;

	for (size_t i = 0; i < mesh.numVertices; ++i)
	{
		//if (visibility[i])
		//{
		if (use_lower_bound)
		{
			problem.SetParameterLowerBound(&local_lightings[i][0], 0, 0.0);

			if (!trackerSettings.use_white_specularities)
			{
				problem.SetParameterLowerBound(&local_lightings[i][0], 1, 0.0);
				problem.SetParameterLowerBound(&local_lightings[i][0], 2, 0.0);
			}
		}

		if (use_upper_bound)
		{
			problem.SetParameterUpperBound(&local_lightings[i][0], 0, 1.0);

			if (!trackerSettings.use_white_specularities)
			{
				problem.SetParameterUpperBound(&local_lightings[i][0], 1, 1.0);
				problem.SetParameterUpperBound(&local_lightings[i][0], 2, 1.0);
			}
		}
		//}
	}

	ceres::Solver::Options options;
	options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
	options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
	options.line_search_type = ceres::LineSearchType::WOLFE;
	options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
	options.line_search_interpolation_type = ceres::LineSearchInterpolationType::CUBIC;
	options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
	options.max_num_iterations = 10;
	options.num_threads = 8;
	options.initial_trust_region_radius = 1e4;
	options.max_trust_region_radius = 1e8;
	options.min_trust_region_radius = 1e-32;
	options.min_relative_decrease = 1e-3;
	options.max_num_consecutive_invalid_steps = 5;
	options.function_tolerance = 1e-4;
	options.gradient_tolerance = 1e-10;
	options.parameter_tolerance = 1e-10;
	options.linear_solver_type = ceres::LinearSolverType::CGNR;
	options.preconditioner_type = ceres::PreconditionerType::JACOBI;
	options.num_linear_solver_threads = 8;
	options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
	options.update_state_every_iteration = 0;

	ceres::Solver::Summary summary;

	ceres::Solve(options, &problem, &summary);

	cout << summary.FullReport() << endl;
}

void DeformNRSFMTracker::estimateSHCoeffLocalLighting(const PangaeaMeshData &mesh,
  const vector<bool> &visibility,
  const MeshDeformation &intensities,
  MeshDeformation &albedos,
  const int sh_order, vector<double> &sh_coeff,
  MeshDeformation &local_lightings)
{
  //for (size_t i = 0; i < local_lightings.size(); i++)
  //{
  //  fill(local_lightings[i].begin(), local_lightings[i].end(), 0);
  //}

  ceres::Problem problem;

  ceres::LossFunction *data_loss_function = NULL;
  if (trackerSettings.local_lighting_data_huber_width > 0.0)
  {
    data_loss_function = 
      new ceres::HuberLoss(trackerSettings.local_lighting_data_huber_width);
  }

  ceres::ScaledLoss* scaled_data_loss = new ceres::ScaledLoss(
    data_loss_function,
    trackerSettings.local_lighting_data_weight,
    ceres::TAKE_OWNERSHIP);

    bool use_lower_bound_shading = false;
  bool use_upper_bound_shading = false;

  const unsigned int N_CHANNELS = 3;

  const int n_sh_coeff = pow(sh_order + 1, 2);

  unsigned int specular_n_channels = trackerSettings.use_white_specularities ? 1 : 3;

  for (size_t i = 0; i < mesh.numVertices; i++)
  {
    if (visibility[i])
    {
      ResidualPhotometricErrorWithNormal *residual =
        new ResidualPhotometricErrorWithNormal(
        &intensities[i][0], (double *)&mesh.normals[i][0], N_CHANNELS, 
        sh_order, use_lower_bound_shading, use_upper_bound_shading, trackerSettings.use_white_specularities);

      ceres::DynamicAutoDiffCostFunction<ResidualPhotometricErrorWithNormal, 5>* dyn_cost_function
        = new ceres::DynamicAutoDiffCostFunction< ResidualPhotometricErrorWithNormal, 5 >(residual);

      // List of pointers to translations per vertex
      vector<double*> v_parameter_blocks;

      // Albedo
      dyn_cost_function->AddParameterBlock(N_CHANNELS);
      v_parameter_blocks.push_back(&albedos[i][0]);
      // SH Coeff
      dyn_cost_function->AddParameterBlock(n_sh_coeff);
      v_parameter_blocks.push_back(&sh_coeff[0]);
      // Local lighting variations
      dyn_cost_function->AddParameterBlock(specular_n_channels);
      v_parameter_blocks.push_back(&local_lightings[i][0]);

      dyn_cost_function->SetNumResiduals(N_CHANNELS);

      ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
        dyn_cost_function,
        scaled_data_loss,
        v_parameter_blocks);
    }
  }

  if (trackerSettings.sh_coeff_temporal_weight > 0)
  {
    ceres::HuberLoss* huber_temporal_loss = NULL;

    if (trackerSettings.sh_coeff_temporal_huber_width > 0.0)
    {
      huber_temporal_loss =
        new ceres::HuberLoss(trackerSettings.sh_coeff_temporal_huber_width);
    }

    ceres::ScaledLoss* scaled_temporal_loss =
      new ceres::ScaledLoss(huber_temporal_loss,
      trackerSettings.sh_coeff_temporal_weight, 
      ceres::TAKE_OWNERSHIP);

    ResidualTemporalSHCoeff *residual =
      new ResidualTemporalSHCoeff(prevSHCoeff);

    ceres::DynamicAutoDiffCostFunction<ResidualTemporalSHCoeff, 5>* dyn_cost_function
      = new ceres::DynamicAutoDiffCostFunction< ResidualTemporalSHCoeff, 5 >(residual);

    // List of pointers to sh coefficients
    vector<double*> v_parameter_blocks;

    // SH Coeff
    dyn_cost_function->AddParameterBlock(sh_coeff.size());
    v_parameter_blocks.push_back(&sh_coeff[0]);

    dyn_cost_function->SetNumResiduals(sh_coeff.size());

    ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
      dyn_cost_function,
      scaled_temporal_loss,
      v_parameter_blocks);
  }

  if (trackerSettings.local_lighting_smoothness_weight > 0.0)
  {
    ceres::LossFunction* huber_loss = NULL;

    if (trackerSettings.local_lighting_smoothness_huber_width > 0)
    {
      huber_loss = 
        new ceres::HuberLoss(trackerSettings.local_lighting_smoothness_huber_width);
    }

    ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
      huber_loss,
      trackerSettings.local_lighting_smoothness_weight,
      ceres::TAKE_OWNERSHIP);

    for (size_t i = 0; i < mesh.numVertices; i++)
    {
      //if (visibility[i])
      //{
        for (size_t j = 0; j < mesh.adjVerticesInd[i].size(); j++)
        {
          unsigned int adj_v_idx = mesh.adjVerticesInd[i][j];

          //if (visibility[adj_v_idx])
          //{
            double weight = 1;// diff_weights[i][j];

            if (weight > 0.0)
            {
				if (trackerSettings.use_white_specularities)
				{
					ResidualWeightedDifference *residual = new ResidualWeightedDifference(
						weight, 1);

					ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>* cost_function =
						new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 1, 1, 1>(residual);

					ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
						cost_function,
						loss_function,
						&local_lightings[i][0],
						&local_lightings[adj_v_idx][0]);
				}
				else
				{
					ResidualWeightedDifference *residual = new ResidualWeightedDifference(
						weight, 3);

					ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>* cost_function =
						new ceres::AutoDiffCostFunction<ResidualWeightedDifference, 3, 3, 3>(residual);

					ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
						cost_function,
						loss_function,
						&local_lightings[i][0],
						&local_lightings[adj_v_idx][0]);
				}
            }
          //}
        }
      //}
    }
  }

  if (trackerSettings.local_lighting_magnitude_weight > 0.0)
  {
    ceres::LossFunction* huber_loss = NULL;

    if (trackerSettings.local_lighting_magnitude_huber_width > 0)
    {
      huber_loss = 
        new ceres::HuberLoss(trackerSettings.local_lighting_magnitude_huber_width);
    }

    ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
      huber_loss,
      trackerSettings.local_lighting_magnitude_weight,
      ceres::TAKE_OWNERSHIP);

    for (size_t i = 0; i < mesh.numVertices; i++)
    {
      //if (visibility[i])
      //{
		if (trackerSettings.use_white_specularities)
		{
			ResidualValueMagnitude *residual = new ResidualValueMagnitude(1);

			ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>* cost_function =
				new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 1, 1>(residual);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				cost_function,
				loss_function,
				&local_lightings[i][0]);
		}
		else
		{
			ResidualValueMagnitude *residual = new ResidualValueMagnitude(3);

			ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>* cost_function =
				new ceres::AutoDiffCostFunction<ResidualValueMagnitude, 3, 3>(residual);

			ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
				cost_function,
				loss_function,
				&local_lightings[i][0]);
		}
      //}
    }
  }

  if (trackerSettings.local_lighting_temporal_weight > 0.0)
  {
    ceres::LossFunction* huber_loss = NULL;

    if (trackerSettings.local_lighting_temporal_huber_width > 0)
    {
      huber_loss =
        new ceres::HuberLoss(trackerSettings.local_lighting_temporal_huber_width);
    }

    ceres::ScaledLoss* loss_function = new ceres::ScaledLoss(
      huber_loss,
      trackerSettings.local_lighting_temporal_weight,
      ceres::TAKE_OWNERSHIP);

	vector<vector<double>> &prev_local_lightings = prevMeshSpecularPyramid[currLevel];

    for (size_t i = 0; i < mesh.numVertices; i++)
    {
      //if (visibility[i])
      //{
        ResidualTemporalLocalLighting *residual = 
          new ResidualTemporalLocalLighting(prev_local_lightings[i], specular_n_channels);

        ceres::DynamicAutoDiffCostFunction<ResidualTemporalLocalLighting, 5>* dyn_cost_function
          = new ceres::DynamicAutoDiffCostFunction< ResidualTemporalLocalLighting, 5 >(residual);

        // List of pointers to translations per vertex
        vector<double*> v_parameter_blocks;

        // SH Coeff
        dyn_cost_function->AddParameterBlock(specular_n_channels);
        v_parameter_blocks.push_back(&local_lightings[i][0]);

        dyn_cost_function->SetNumResiduals(specular_n_channels);

        ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(
          dyn_cost_function,
          loss_function,
          v_parameter_blocks);
      //}
    }
  }

  bool use_lower_bound = true;
  bool use_upper_bound = true;

  for (size_t i = 0; i < mesh.numVertices; ++i)
  {
    //if (visibility[i])
    //{
    if (use_lower_bound)
    {
      problem.SetParameterLowerBound(&local_lightings[i][0], 0, 0.0);
	  if (!trackerSettings.use_white_specularities)
	  {
		  problem.SetParameterLowerBound(&local_lightings[i][0], 1, 0.0);
		  problem.SetParameterLowerBound(&local_lightings[i][0], 2, 0.0);
	  }
    }

    if (use_upper_bound)
    {
      problem.SetParameterUpperBound(&local_lightings[i][0], 0, 1.0);
	  if (!trackerSettings.use_white_specularities)
	  {
		  problem.SetParameterUpperBound(&local_lightings[i][0], 1, 1.0);
		  problem.SetParameterUpperBound(&local_lightings[i][0], 2, 1.0);
	  }
    }

    if (visibility[i])
    {
      problem.SetParameterBlockConstant(&albedos[i][0]);
    }
    //}
  }

  ceres::Solver::Options options;
  options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
  options.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
  options.line_search_type = ceres::LineSearchType::WOLFE;
  options.nonlinear_conjugate_gradient_type = ceres::NonlinearConjugateGradientType::FLETCHER_REEVES;
  options.line_search_interpolation_type = ceres::LineSearchInterpolationType::CUBIC;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 10;
  options.num_threads = 8;
  options.initial_trust_region_radius = 1e4;
  options.max_trust_region_radius = 1e8;
  options.min_trust_region_radius = 1e-32;
  options.min_relative_decrease = 1e-3;
  options.max_num_consecutive_invalid_steps = 5;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-10;
  options.linear_solver_type = ceres::LinearSolverType::CGNR;
  options.preconditioner_type = ceres::PreconditionerType::JACOBI;
  options.num_linear_solver_threads = 8;
  options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
  options.update_state_every_iteration = 0;

  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  cout << summary.FullReport() << endl;
}

void DeformNRSFMTracker::initNeighboursWeightsFineToCoarse()
{
	typedef vector<CoordinateType> MeshSigmas;
	typedef vector<vector<CoordinateType> > MeshDistances;

	if (templateMeshPyramid.numLevels > 1)
	{
		fineToCoarseNeighbours.reserve(templateMeshPyramid.numLevels - 1);
		fineToCoarseWeights.reserve(templateMeshPyramid.numLevels - 1);
	}

	for (size_t i = 0; i < templateMeshPyramid.numLevels - 1; i++)
	{
		MeshWeights weights;
		MeshNeighbors neighbors;

		MeshSigmas sigmas;
		MeshDistances distances;
		MeshNeighborsNano neighborsNano;

		PangaeaMeshData& fineMesh = templateMeshPyramid.levels[i];
		PangaeaMeshData& coarseMesh = templateMeshPyramid.levels[i + 1];

		calcNeighborsAndWeights(coarseMesh.vertices, fineMesh.vertices,
			neighborsNano, distances, 3);

		int numVertices = coarseMesh.vertices.size();
		sigmas.reserve(numVertices);
		for (int j = 0; j < numVertices; ++j)
		{
			sigmas.push_back(distances[j].back());
		}

		distToWeights(distances, weights, sigmas);

		fineToCoarseNeighbours.push_back(std::move(neighborsNano));
		fineToCoarseWeights.push_back(std::move(weights));
	}
}

void DeformNRSFMTracker::propagateAlbedoLocalLightingFineToCoarse()
{
	for (size_t i = 0; i < templateMeshPyramid.numLevels - 1; i++)
	{
		PangaeaMeshData &fineMesh = templateMeshPyramid.levels[i];
		PangaeaMeshData &coarseMesh = templateMeshPyramid.levels[i + 1];
		MeshNeighborsNano &neighbours = fineToCoarseNeighbours[i];
		MeshWeights &weights = fineToCoarseWeights[i];

		for (size_t j = 0; j < coarseMesh.numVertices; j++)
		{
			vector<size_t> &curr_neigh = neighbours[j];
			vector<double> &curr_weight = weights[j];

			vector<double> &albedo = coarseMesh.colors[j];
			albedo[0] = 0;
			albedo[1] = 0;
			albedo[2] = 0;

			vector<double> &local_light = coarseMesh.specular_colors[j];
			local_light[0] = 0;
			local_light[1] = 0;
			local_light[2] = 0;
			for (int k = 0; k < neighbours[i].size(); k++)
			{
				size_t v_idx = curr_neigh[k];

				albedo[0] += curr_weight[k] * fineMesh.colors[v_idx][0];
				albedo[1] += curr_weight[k] * fineMesh.colors[v_idx][1];
				albedo[2] += curr_weight[k] * fineMesh.colors[v_idx][2];

				local_light[0] += curr_weight[k] * fineMesh.specular_colors[v_idx][0];
				local_light[1] += curr_weight[k] * fineMesh.specular_colors[v_idx][1];
				local_light[2] += curr_weight[k] * fineMesh.specular_colors[v_idx][2];
			}
		}

		coarseMesh.sh_coefficients = fineMesh.sh_coefficients;
	}
}

void DeformNRSFMTracker::estimateDiffuse(const cv::Mat src_image, cv::Mat &diffuse_image)
{
	qx_timer timer;

	int h = src_image.rows;
	int w = src_image.cols;
	int c = src_image.channels();

	const uchar* p;
	for (size_t i = 0; i < h; i++)
	{
		p = src_image.ptr<uchar>(i);
		for (size_t j = 0; j < w; j++)
		{
			for (size_t k = 0; k < c; k++)
			{
				qx_image[i][j][k] = p[j * c + k];
			}
		}
	}

	qx_highlight_removal_bf m_highlight;
	m_highlight.init(h, w);/*initialization*/
	timer.start();

	int nr_iter_converge = m_highlight.diffuse(qx_diffuse_image, qx_image);/*extracting diffuse reflection*/

	timer.time_display("Highlight Removal");
	printf("# of iterations before convergence: [%02d]\n", nr_iter_converge);

	diffuse_image = cv::Mat_<Vec3b>(h, w);
	uchar* p2;
	for (size_t i = 0; i < h; i++)
	{
		p2 = diffuse_image.ptr<uchar>(i);
		for (size_t j = 0; j < w; j++)
		{
			for (size_t k = 0; k < c; k++)
			{
				p2[j * c + k] = qx_diffuse_image[i][j][k];
			}
		}
	}
}

void DeformNRSFMTracker::updateSHCoeff()
{
	prevSHCoeff = templateMeshPyramid.levels[0].sh_coefficients;

	for (size_t i = 1; i < templateMeshPyramid.numLevels; i++)
	{
		PangaeaMeshData &mesh = templateMeshPyramid.levels[i];
		mesh.sh_coefficients = prevSHCoeff;
	}
}

void DeformNRSFMTracker::resetIntrinsics()
{
	PangaeaMeshData& template_mesh = templateMeshPyramid.levels[currLevel];
	MeshDeformation& mesh_specular = prevMeshSpecularPyramid[currLevel];

	for (int i = 0; i < template_mesh.numVertices; ++i)
	{
		template_mesh.specular_colors[i][0] = mesh_specular[i][0];
		template_mesh.specular_colors[i][1] = mesh_specular[i][1];
		template_mesh.specular_colors[i][2] = mesh_specular[i][2];
	}

	if (!trackerSettings.estimate_with_sh_coeff)
	{
		template_mesh.sh_coefficients = shCoeff;
	}
}

void DeformNRSFMTracker::fixWhiteSpecularities()
{
	PangaeaMeshData& template_mesh = templateMeshPyramid.levels[currLevel];
	MeshDeformation& mesh_specular = prevMeshSpecularPyramid[currLevel];

	for (int i = 0; i < template_mesh.numVertices; ++i)
	{
		template_mesh.specular_colors[i][1] = template_mesh.specular_colors[i][0];
		template_mesh.specular_colors[i][2] = template_mesh.specular_colors[i][0];
	}
}