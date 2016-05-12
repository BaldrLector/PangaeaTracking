// settings

#ifndef _SETTINGS_H
#define _SETTINGS_H

#include "global.h"

class ImageSourceSettings
{
public:

  ImageSourceSettings();
  void read(const cv::FileNode& node);

  std::string dataPath;
  std::string imageFormat;
  std::string intrinsicsFile;

  int width;
  int height;
  int startFrame;
  int numFrames;

  double KK[3][3];
  bool isOrthoCamera;
  int frameStep;

  bool useMultiImages;
  std::string dataPathLevelRoot;
  std::string dataPathLevelFormat;
  IntegerContainerType dataPathLevelList;
  std::string imageLevelFormat;

};

class ShapeLoadingSettings
{
public:

  ShapeLoadingSettings();
  void read(const cv::FileNode& node);

  bool hasGT;
  std::string resultsPath;
  std::string shapeFormat;
  std::string shapeFormatGT;

  std::string gtModelFile;
  std::string solModelFile;
  std::string shapeMaskFile;
  std::string labelColorFile;

  bool loadShapeMask;
  bool shapeColMajor;
  float shapeSamplingScale;

  int modelNum;

};

class MeshLoadingSettings
{
public:

  MeshLoadingSettings();
  void read(const cv::FileNode& node);

  std::string meshPath;
  std::string meshFormat;
  bool visibilityMask;

  // added for pyramid
  std::string meshLevelFormat;
  std::string propLevelFormat;
  IntegerContainerType meshLevelList;

  bool hasGT;
  bool loadProp;
  bool fastLoading;

  std::string meshPathGT;
  std::string meshLevelFormatGT;
  IntegerContainerType meshLevelListGT;

  // Faces of the mesh are defined clockwise or not (anti-clockwise)
	bool clockwise;
};

class TrackerSettings
{
public:

  TrackerSettings();
  void read(const cv::FileNode& node);

  // DeformNRSFM para
  std::string errorType;
  std::string baType;
  std::string meshFile;
  std::string optimizationType;

  bool isRigid;
  bool doAlternation;
  bool updateColor;
  bool useVisibilityMask;
  bool useOpenGLMask;
  bool fastARAP;
  bool onlyDeformDepthPrior;
  bool useXYZ;
  bool isOrthoCamera;
  bool loadMesh;
  double depth2MeshScale;

  double weightPhotometric;
  double weightPhotometricIntensity;
  double weightTV;
  double weightRotTV;

  double weightDeform;
  double weightGradient;
  double weightARAP;
  double weightINEXTENT;
  double weightTransPrior;
  double weightSmoothing;
  double photometricHuberWidth;
  double photometricIntensityHuberWidth;
  double tvHuberWidth;
  double tvRotHuberWidth;
  double arapHuberWidth;
  double smoothingHuberWidth;

  bool use_cotangent;

  double meshScaleUpFactor;

	// Faces of the mesh are defined clockwise or not (anti-clockwise)
	bool clockwise;

	// Path to Spherical Harmonic Coefficients for Illumination
	std::string sh_coeff_file;

	double specular_weight_var;
	double brightness_percentile;

	double sh_coeff_data_weight;
	double sh_coeff_data_huber_width;
	double sh_coeff_temporal_weight;
	double sh_coeff_temporal_huber_width;

  bool update_albedo;
	double albedo_data_weight;
	double albedo_data_huber_width;
	double albedo_smoothness_weight;
	double albedo_smoothness_huber_width;
	double albedo_difference_weight;
	double albedo_difference_huber_width;

	double smoothness_specular_weight;
	double smoothness_color_diff_var;
	double smoothness_color_diff_threshold;

	double local_lighting_data_weight;
	double local_lighting_data_huber_width;
	double local_lighting_smoothness_weight;
	double local_lighting_smoothness_huber_width;
	double local_lighting_magnitude_weight;
	double local_lighting_magnitude_huber_width;
	double local_lighting_temporal_weight;
	double local_lighting_temporal_huber_width;

	bool estimate_all_together;
	bool estimate_with_sh_coeff;

  bool estimate_diffuse;
	bool estimate_sh_coeff_specular_together;

  // ceres parameter
  std::string linearSolver;
  int numOptimizationLevels;
  int numOptimizationLevelsToDo;
  IntegerContainerType blurFilterSizes;
  CoordinateContainerType blurSigmaSizes;

  IntegerContainerType imagePyramidSamplingFactors;
  CoordinateContainerType imageGradientScalingFactors;

  IntegerContainerType maxNumIterations;
  CoordinateContainerType functionTolerances;
  CoordinateContainerType gradientTolerances;
  CoordinateContainerType parameterTolerances;
  CoordinateContainerType initialTrustRegionRadiuses;
  CoordinateContainerType maxTrustRegionRadiuses;
  CoordinateContainerType minTrustRegionRadiuses;
  CoordinateContainerType minRelativeDecreases;
  int numLinearSolverThreads;
  int numThreads;
  bool isMinimizerProgressToStdout;

  // debugging
  bool save_binary_mesh;

  bool saveResults;
  std::string ceresOutputFile;
  std::string diffFileFormat;
  std::string savePath;
  std::string scoresPath;

  std::string energyOutputFile;

  bool saveMesh;
  std::string meshFormat;

  bool saveMeshPyramid;
  std::string meshPyramidFormat;

  bool savePropPyramid;
  std::string propPyramidFormat;

  bool saveColorDiff;
  std::string meshColorDiffFormat;

  // either show the window
  // or hide the window and run over all frames
  bool showWindow;

  // mesh pyramid stuff
  std::string meshLevelFormat;
  IntegerContainerType meshVertexNum;
  IntegerContainerType meshNeighborNum;
  CoordinateContainerType meshNeighborRadius;
  bool meshPyramidUseRadius;

  bool use_intensity_pyramid;
  std::string meshIntensityLevelFormat;

  // create a mesh pyramid from depth image
  bool useDepthPyramid;
  bool usePrevForTemplateInTV;

  double tvTukeyWidth;  // the real threshold is square

  // arbitary neighbor mesh for data term and regularization term
  // if data term pairs are not defined
  // we use siggraph strategy(all neighbors
  // are defined on the same mesh and for all levles)
  // if reg terms are not defined, we use the same as data term
  VecVecPairType dataTermPair;
  VecVecPairType regTermPair;

  // minimizer typpe(if we switch to line search)
  std::string minimizerType;
  std::string lineSearchDirectionType;
  std::string lineSearchType;
  std::string nonlinearConjugateGradientType;
  std::string lineSearchInterpolationType;

  // patch based optimization
  int neighborPatchRadius;

  // use featureImages
  bool useFeatureImages;
  bool useRGBImages;

  // ground truth
  bool hasGT;
  std::string meshPathGT;
  std::string meshLevelFormatGT;
  IntegerContainerType meshLevelListGT;
  int firstFrameGT;

  // print energy and error compared with ground truth
  bool printEnergy;
  bool printEnergyGT;
  bool computeError;

  // save the mesh as ply files
  // bool savePLY;

  // List of pyramid levels to save
  IntegerContainerType levelsMeshPyramidSave; 
};

class FeatureSettings
{
public:

  FeatureSettings();
  void read(const cv::FileNode& node);

  // parameters
  int channels;

  bool useNCC;

  // 0th level feature image could be smaller than input image
  // downsampling scale factor
  double scalingFactor;

  bool hasGradient;

  string dataType;
  string gradType;

  int dataElemSize;
  int dataTypeINT;
  int gradTypeINT;

  // input file, we use lmdb at the moment
  string dbPath;
  string keyNameFormat;

  // feature image stuff
  IntegerContainerType blurFeatureFilterSizes;
  CoordinateContainerType blurFeatureSigmaSizes;
  IntegerContainerType featurePyramidSamplingFactors;
  CoordinateContainerType featureGradientScalingFactors;

  //
  double featureTermWeight;
  double featureHuberWidth;

};

void read(const cv::FileNode& node, std::string& value, const char* default_value);

void read(const cv::FileNode& node, ImageSourceSettings& settings,
          const ImageSourceSettings& default_settings = ImageSourceSettings());

void read(const cv::FileNode& node, ShapeLoadingSettings& settings,
          const ShapeLoadingSettings& default_settings = ShapeLoadingSettings());

void read(const cv::FileNode& node, MeshLoadingSettings& settings,
          const MeshLoadingSettings& default_settings = MeshLoadingSettings());

void read(const cv::FileNode& node, TrackerSettings& settings,
          const TrackerSettings& default_settings = TrackerSettings());

void read(const cv::FileNode& node, FeatureSettings& settings,
          const FeatureSettings& default_settings = FeatureSettings());

extern ImageSourceType imageSourceType;
extern TrackingType trackingType;

extern ImageSourceSettings imageSourceSettings;
extern ShapeLoadingSettings shapeLoadingSettings;
extern MeshLoadingSettings meshLoadingSettings;
extern TrackerSettings trackerSettings;
extern FeatureSettings featureSettings;

#endif
