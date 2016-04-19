#ifndef _MAINEngine_H
#define _MAINEngine_H

#include "./utils/settings.h"

#include "./image_source/ImageSourceEngine.h"

#include "./tracker/Mesh.h"
#include "./tracker/ImagePyramid.h"

#include "./tracker/TrackingEngine.h"

#ifndef VIS_ONLY
#include "./tracker/DeformNRSFMTracker.h"
#endif

class MainEngine
{

public:

    MainEngine();
    virtual ~MainEngine();

    void LoadInitialMeshUVD();
    void LoadInitialMeshFromFile();
    void SetIntrinsicMatrix(double K[3][3]);
    void SetupInputAndTracker();

    // void ReadConfigurationFile(int argc, wxChar* argv[]);
    void ReadConfigurationFile(int argc, char* argv[]);

    void GetInput(int nFrame);

    void Run();
    bool ProcessNextFrame();
    virtual bool ProcessOneFrame(int nFrame);

    ImageSourceEngine* m_pImageSourceEngine;
    TrackingEngine* m_pTrackingEngine;

	ImageSourceEngine* m_pSpecularImageSourceEngine;

    unsigned char* m_pColorImageRGB;
    unsigned char* m_pColorImageRGBBuffer;

	unsigned char* m_pSpecularGrayImage;
	unsigned char* m_pSpecularGrayImageBuffer;

    int m_nWidth;
    int m_nHeight;
    int m_nStartFrame;
    int m_nCurrentFrame;
    int m_NumTrackingFrames;

    int m_nFrameStep;

    double center[3];
    double KK[3][3];

    TrackerOutputInfo outputInfo;
    TrackerOutputInfo* pOutputInfo;
    PangaeaMeshData templateMesh;
    PangaeaMeshPyramid templateMeshPyramid;

    int m_nNumMeshLevels;

    boost::mutex mutex;
    //boost::thread_group inputThreadGroup;
    boost::thread* pInputThread;
    bool inputFlag;

};

#endif
