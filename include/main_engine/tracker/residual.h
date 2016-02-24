#pragma once

#include "./Mesh.h"
#include "./ImagePyramid.h"

#include "sample.h"
#include "jet_extras.h"
#include "ceres/rotation.h"

enum baType{
  BA_MOT,
  BA_STR,
  BA_MOTSTR
};

enum dataTermErrorType{
  PE_INTENSITY = 0,
  PE_COLOR,
  PE_DEPTH,
  PE_DEPTH_PLANE,
  PE_NCC,
  PE_COLOR_NCC,
  PE_FEATURE, // number of residuals depend on the number of channels
  PE_FEATURE_NCC,
  PE_INTRINSIC,
  PE_INTRINSIC_COLOR,
  COST_TYPE_NUM
};

static int PE_RESIDUAL_NUM_ARRAY[COST_TYPE_NUM] = {1,3,3,1,1,3,-1,-1,1,3};

template<typename T>
void getValueFromMesh(const PangaeaMeshData* pMeshData, dataTermErrorType errorType, int pntInd, T* pValue)
{
  switch(errorType)
    {
    case PE_INTENSITY:
    case PE_NCC:
      pValue = (T*)(&pMeshData->grays[ pntInd ]);
      break;
    case PE_COLOR:
    case PE_COLOR_NCC:
      pValue = (T*)(&pMeshData->colors[ pntInd ][0]);
      break;
    case PE_FEATURE:
    case PE_FEATURE_NCC:
      pValue = (T*)(&pMeshData->features[ pntInd ][0]);
      break;
    }
}

template<typename T>
void IntrinsicProjection(const CameraInfo* pCamera, T* p, T* u, T* v)
{
  if(pCamera->isOrthoCamera)
    {
      u[0] = p[0]; //transformed x (2D)
      v[0] = p[1]; //transformed y (2D)
    }
  else
    {
      u[0] = ( ( p[0] * T( pCamera->KK[0][0] ) ) / p[2] ) + T( pCamera->KK[0][2] ); //transformed x (2D)
      v[0] = ( ( p[1] * T( pCamera->KK[1][1] ) ) / p[2] ) + T( pCamera->KK[1][2] ); //transformed y (2D)
    }
}

// backprojection
template<typename T>
void BackProjection(const CameraInfo* pCamera, const Level* pFrame, T* u, T* v, T* backProj)
{

  ImageLevel* pImageLevel = (ImageLevel*)pFrame;

  T currentValue;
  currentValue = SampleWithDerivative< T, InternalIntensityImageType > (pImageLevel->depthImage,
                                                                        pImageLevel->depthGradXImage,
                                                                        pImageLevel->depthGradYImage, u[0], v[0]);

  backProj[2] = currentValue;

  if(pCamera->isOrthoCamera)
    {
      backProj[0] = u[0]; backProj[1] = v[0];
    }else
    {
      backProj[0] = backProj[2] * (pCamera->invKK[0][0]*u[0] + pCamera->invKK[0][2]);
      backProj[1] = backProj[2] * (pCamera->invKK[1][1]*v[0] + pCamera->invKK[1][2]);
    }

}

// patch based score estimation
template<typename T>
void nccScore(vector<T>& neighborValues, vector<T>& projValues, int k1, int k2, T* pScore)
{

  T neighborMean, projMean, neighborSTD, projSTD;
  neighborMean = T(0.0); projMean = T(0.0);
  neighborSTD = T(0.0); projSTD = T(0.0);

  T my_epsilon = T(0.00001);

  pScore[0] = T(0.0);

  int num = neighborValues.size();
  for(int i = 0; i < num; ++i)
    {
      neighborMean += neighborValues[ k1*i + k2 ];
      projMean += projValues[ k1*i+k2 ];
    }

  neighborMean = neighborMean / T(num);
  projMean = projMean / T(num);


  //// pScore[0] = neighborMean - projMean;

  for(int i = 0; i < num; ++i)
    {
      neighborSTD += (neighborValues[k1*i + k2] - neighborMean) *
        (neighborValues[k1*i + k2] - neighborMean);
      projSTD += (projValues[k1*i + k2] - projMean) *
        (projValues[k1*i + k2] - projMean);
    }

  //pScore[0] = neighborSTD - projSTD;

  neighborSTD = sqrt(neighborSTD + my_epsilon);
  projSTD = sqrt(projSTD + my_epsilon);

  //////  pScore[0] = neighborSTD - projSTD;

  for(int i = 0; i < num; ++i)
    {
      pScore[0] += (neighborValues[k1*i + k2] - neighborMean) / neighborSTD *
        (projValues[k1*i + k2] - projMean) / projSTD;
    }

  // pScore[0] = T(0.0);

}

template<typename T>
void getPatchScore(vector<T>& neighborValues, vector<T>& projValues,
                   T* pScore, int numChannels)
{

  for(int i = 0; i < numChannels; ++i)
    nccScore(neighborValues, projValues, numChannels, i, pScore+i);

}

template<typename T>
void getValue(const CameraInfo* pCamera, const Level* pFrame,
              T* p, T* value, const dataTermErrorType& PE_TYPE)
{

  T transformed_r, transformed_c;

  IntrinsicProjection(pCamera, p, &transformed_c, &transformed_r);

  T templateValue, currentValue;

  if( transformed_r >= T(0.0) && transformed_r < T(pCamera->height) &&
      transformed_c >= T(0.0) && transformed_c < T(pCamera->width))
    {
      switch(PE_TYPE)
        {
        case PE_INTENSITY:
          {

            ImageLevel* pImageLevel = (ImageLevel*)pFrame;
            value[0] = SampleWithDerivative< T, InternalIntensityImageType > (pImageLevel->grayImage,
                                                                              pImageLevel->gradXImage,
                                                                              pImageLevel->gradYImage,
                                                                              transformed_c,
                                                                              transformed_r );
            break;
          }
        case PE_COLOR:
          {
            ImageLevel* pImageLevel = (ImageLevel*)pFrame;
            for(int i = 0; i < 3; ++i)
              {
                value[i] = SampleWithDerivative< T, InternalIntensityImageType >( pImageLevel->colorImageSplit[i],
                                                                                  pImageLevel->colorImageGradXSplit[i],
                                                                                  pImageLevel->colorImageGradYSplit[i],
                                                                                  transformed_c,
                                                                                  transformed_r );
              }
            break;
          }
        case PE_DEPTH:   // point-to-point error
          {
            // depth value of the point
            ImageLevel* pImageLevel = (ImageLevel*)pFrame;
            BackProjection(pCamera, pImageLevel, &transformed_c, &transformed_r, value);

            for(int i = 0; i < 3; ++i)
              value[i] = value[i] - p[i];

            break;
          }
        case PE_DEPTH_PLANE: // point-to-plane error
          {
            //
            ImageLevel* pImageLevel = (ImageLevel*)pFrame;
            BackProjection(pCamera, pImageLevel, &transformed_c, &transformed_r, value);

            // normals at back projection point
            T normals_at_bp[3];
            for(int i = 0; i < 3; ++i)
              {
                normals_at_bp[i] = SampleWithDerivative< T, InternalIntensityImageType > (pImageLevel->depthNormalImageSplit[i],
                                                                                          pImageLevel->depthNormalImageGradXSplit[i],
                                                                                          pImageLevel->depthNormalImageGradYSplit[i],
                                                                                          transformed_c,
                                                                                          transformed_r );
                value[i] = normals_at_bp[i] * (value[i] - p[i]);
              }
            break;
          }
        case PE_FEATURE:
          {
            //
            FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
            int numChannels = pFeatureLevel->featureImageVec.size();
            for(int i = 0; i < numChannels; ++i)
              {
                value[i] = SampleWithDerivative<T, FeatureImageType>(pFeatureLevel->featureImageVec[i],
                                                                     pFeatureLevel->featureImageGradXVec[i],
                                                                     pFeatureLevel->featureImageGradYVec[i],
                                                                     transformed_c,
                                                                     transformed_r);
              }
          }
        }
    }

}

template<typename T>
void getResidual(double weight, const CameraInfo* pCamera, const Level* pFrame,
                 double* pValue, T* p, T* residuals, const dataTermErrorType& PE_TYPE)
{
  vector<T> projValues;
  int numChannels;

  numChannels = PE_RESIDUAL_NUM_ARRAY[ PE_TYPE ];

  projValues.resize( numChannels );

  getValue(pCamera, pFrame, p, &projValues[0], PE_TYPE);

  switch(PE_TYPE)
    {
    case PE_INTENSITY:
    case PE_COLOR:
    case PE_FEATURE:
      {
        for(int i = 0; i < numChannels; ++i)
          residuals[i] = T(weight) * (pValue[i] - projValues[i]);
      }
      break;
    case PE_DEPTH:
    case PE_DEPTH_PLANE:
      {
        for(int i = 0; i < numChannels; ++i)
          residuals[i] = T(weight) * projValues[i];
      }
      break;
    }
}

template<typename T>
void getPatchResidual(double weight, const CameraInfo* pCamera, const Level* pFrame,
                      const PangaeaMeshData* pMesh, vector<T>& neighborVertices,
                      int numNeighbors, const vector<unsigned int>& neighbors, T* residuals,
                      const dataTermErrorType& PE_TYPE=PE_NCC)
{
  vector<T> neighborValues;
  vector<T> projValues;
  vector<T> score;

  T my_epsilon = T(0.00001);
  int numChannels;

  numChannels = PE_RESIDUAL_NUM_ARRAY[ PE_TYPE ];
  T* pValue = NULL;

  neighborValues.resize( numChannels * numNeighbors );
  projValues.resize( numChannels * numNeighbors );

  for(int i = 0; i < numNeighbors; ++i)
    {
      getValue(pCamera, pFrame, &neighborVertices[3*i], &projValues[numChannels*i], PE_TYPE);
      getValueFromMesh( pMesh, PE_TYPE, i, pValue );
      for(int k = 0; k < numChannels; ++k)
        neighborValues[ numChannels*i + k] = T( pValue[k] );
    }

  // get the residual for two different cases, with or without NCC
  // without NCC: just sum the square differences between residual or of all the pixels in the patch
  // with NCC: estimate the normalized cross-correlation
  if(PE_TYPE == PE_INTENSITY ||
     PE_TYPE == PE_COLOR ||
     PE_TYPE == PE_FEATURE)
    {
      for(int i = 0; i < numChannels; ++i){
        residuals[i] = T(0.0);
        for(int j = 0; j < numNeighbors; ++j)
          {
            residuals[i] += T(weight) * (projValues[ numChannels*j + i ] - neighborValues[ numChannels*j + i ] ) *
              T(weight) * (projValues[ numChannels*j + i ] - neighborValues[ numChannels*j + i ] );
          }
        residuals[i] = sqrt( residuals[i] + my_epsilon );
      }
    }
  else{
    score.resize(numChannels);
    getPatchScore(neighborValues, projValues, &score[0], numChannels);
    for(int i = 0; i < numChannels; ++i)
      residuals[i] = T(weight) * (T(1.0) - score[i]);
  }

}

// need to write a new ResidualImageProjection(ResidualImageInterpolationProjection)
// for a data term similar to dynamicFusion
// optional parameters: vertex value(will be needed if we are using photometric)
// paramters: vertex position, neighboring weights, pCamera and pFrame
// optimization: rotation, translation and neighboring transformations
// (split up rotation and translation)
// several things we could try here:
// fix the number of neighbors, if we use knn nearest neighbors
// or use variable number of neighbors, if we use neighbors in a certain radius range
// arap term of dynamicFusion can be implemented as usual

// Image Projection residual covers all the possible projection cases
// Including gray, rgb, point-to-point and point-to-plane error
// For different pyramid level, just change the pCamera and pFrame accordingly,
// make sure that pCamera and pFrame are consistent

class ResidualImageProjection
{
public:

  ResidualImageProjection(double weight, double* pValue, double* pVertex,
                          const CameraInfo* pCamera, const Level* pFrame,
                          dataTermErrorType PE_TYPE=PE_INTENSITY):
    weight(weight),
    pValue(pValue),
    pVertex(pVertex),
    pCamera(pCamera),
    pFrame(pFrame),
    PE_TYPE(PE_TYPE),
    optimizeDeformation(true)
  {
    // check the consistency between camera and images

    if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
      {
        FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
        assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
        assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
      }
    else
      {
        ImageLevel* pImageLevel = (ImageLevel*)pFrame;
        assert(pCamera->width == pImageLevel->grayImage.cols);
        assert(pCamera->height == pImageLevel->grayImage.rows);
      }


  }

  // ResidualImageProjection(double weight, double* pTemplateVertex,
  //                         const CameraInfo* pCamera, const Level* pFrame,
  //                         dataTermErrorType PE_TYPE=PE_INTENSITY):
  //   weight(weight),
  //   pVertex(pVertex),
  //   pCamera(pCamera),
  //   pFrame(pFrame),
  //   PE_TYPE(PE_TYPE),
  //   optimizeDeformation(true)
  // {
  //   // check the consistency between camera and images
  //   // assert(pCamera->width == pFrame->grayImage.cols);
  //   // assert(pCamera->height == pFrame->grayImage.rows);

  //   if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
  //     {
  //       FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
  //       assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
  //       assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
  //     }
  //   else
  //     {
  //       ImageLevel* pImageLevel = (ImageLevel*)pFrame;
  //       assert(pCamera->width == pImageLevel->grayImage.cols);
  //       assert(pCamera->height == pImageLevel->grayImage.rows);
  //     }

  // }

  template<typename T>
  bool operator()(const T* const rotation,
                  const T* const translation,
                  const T* const xyz,
                  T* residuals) const
  {
    int residual_num = PE_RESIDUAL_NUM_ARRAY[PE_TYPE];
    for(int i = 0; i < residual_num; ++i)
      residuals[i] = T(0.0);

    T p[3], afterTrans[3];

    // if we are doing optimization on the transformation,
    // we need to add up the template position first
    if(optimizeDeformation)
      {
        afterTrans[0] = xyz[0] + pVertex[0];
        afterTrans[1] = xyz[1] + pVertex[1];
        afterTrans[2] = xyz[2] + pVertex[2];
      }
    else
      {
        afterTrans[0] = xyz[0];
        afterTrans[1] = xyz[1];
        afterTrans[2] = xyz[2];
      }

    ceres::AngleAxisRotatePoint( rotation, afterTrans, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];

    //debug
    // double xyz_[3],rotation_[3],translation_[3];
    // double p_[3];
    // for(int i = 0; i < 3; ++i)
    // {
    //     xyz_[i] = ceres::JetOps<T>::GetScalar(xyz[i]);
    //     rotation_[i] = ceres::JetOps<T>::GetScalar(rotation[i]);
    //     translation_[i] = ceres::JetOps<T>::GetScalar(translation[i]);
    //     p_[i] = ceres::JetOps<T>::GetScalar(p[i]);
    // }

    getResidual(weight, pCamera, pFrame, pValue, p, residuals, PE_TYPE);

    return true;
  }

protected:
  bool optimizeDeformation;  // whether optimize deformation directly
  double* pVertex;
  double weight;
  // this will only be useful if we are using gray or rgb value
  // give a dummy value in other cases
  double* pValue;
  const CameraInfo* pCamera;
  const Level* pFrame;
  dataTermErrorType PE_TYPE;
};

class ResidualImageProjectionDynamic
{
public:

  ResidualImageProjectionDynamic(double weight, double* pValue, double* pVertex,
                          const CameraInfo* pCamera, const Level* pFrame,
                          dataTermErrorType PE_TYPE=PE_INTENSITY):
    weight(weight),
    pValue(pValue),
    pVertex(pVertex),
    pCamera(pCamera),
    pFrame(pFrame),
    PE_TYPE(PE_TYPE),
    optimizeDeformation(true)
  {
    // check the consistency between camera and images
    // assert(pCamera->width == pFrame->grayImage.cols);
    // assert(pCamera->height == pFrame->grayImage.rows);

    if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
      {
        FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
        assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
        assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
      }
    else
      {
        ImageLevel* pImageLevel = (ImageLevel*)pFrame;
        assert(pCamera->width == pImageLevel->grayImage.cols);
        assert(pCamera->height == pImageLevel->grayImage.rows);
      }

  }

  // ResidualImageProjectionDynamic(double weight, double* pTemplateVertex,
  //                         const CameraInfo* pCamera, const Level* pFrame,
  //                         dataTermErrorType PE_TYPE=PE_INTENSITY):
  //   weight(weight),
  //   pVertex(pVertex),
  //   pCamera(pCamera),
  //   pFrame(pFrame),
  //   PE_TYPE(PE_TYPE),
  //   optimizeDeformation(true)
  // {
  //   // check the consistency between camera and images
  //   // assert(pCamera->width == pFrame->grayImage.cols);
  //   // assert(pCamera->height == pFrame->grayImage.rows);

  //   if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
  //     {
  //       FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
  //       assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
  //       assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
  //     }
  //   else
  //     {
  //       ImageLevel* pImageLevel = (ImageLevel*)pFrame;
  //       assert(pCamera->width == pImageLevel->grayImage.cols);
  //       assert(pCamera->height == pImageLevel->grayImage.rows);
  //     }
  // }

  template<typename T>
  bool operator()(const T* const* const parameters, T* residuals) const
  {

    FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
    int numChannels = pFeatureLevel->featureImageVec.size();

    for(int i = 0; i < numChannels; ++i)
      residuals[i] = T(0.0);

    const T* const rotation = parameters[ 0 ];
    const T* const translation = parameters[ 1 ];
    const T* const xyz = parameters[ 2 ];

    T p[3], afterTrans[3];

    // if we are doing optimization on the transformation,
    // we need to add up the template position first
    if(optimizeDeformation)
      {
        afterTrans[0] = xyz[0] + pVertex[0];
        afterTrans[1] = xyz[1] + pVertex[1];
        afterTrans[2] = xyz[2] + pVertex[2];
      }
    else
      {
        afterTrans[0] = xyz[0];
        afterTrans[1] = xyz[1];
        afterTrans[2] = xyz[2];
      }

    ceres::AngleAxisRotatePoint( rotation, afterTrans, p);
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];

    getResidual(weight, pCamera, pFrame, pValue, p, residuals, PE_TYPE);

    return true;

  }


private:
  bool optimizeDeformation;  // whether optimize deformation directly
  double* pVertex;
  double weight;
  // this will only be useful if we are using gray or rgb value
  // give a dummy value in other cases
  double* pValue;
  const CameraInfo* pCamera;
  const Level* pFrame;
  dataTermErrorType PE_TYPE;
};


// ResidualImageProjection from coarse level deformation,
class ResidualImageProjectionCoarse
{
public:
  ResidualImageProjectionCoarse(double weight, double* pValue, double* pVertex,
                                const CameraInfo* pCamera, const Level* pFrame, int numNeighbors,
                                vector<double> neighborWeights, vector<double*> neighborVertices,
                                dataTermErrorType PE_TYPE=PE_INTENSITY):
    weight(weight),
    pValue(pValue),
    pVertex(pVertex),
    pCamera(pCamera),
    pFrame(pFrame),
    numNeighbors(numNeighbors),
    neighborWeights(neighborWeights),
    neighborVertices(neighborVertices),
    PE_TYPE(PE_TYPE)
  {
    // check the consistency between camera and images
    // assert(pCamera->width == pFrame->grayImage.cols);
    // assert(pCamera->height == pFrame->grayImage.rows);

    if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
      {
        FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
        assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
        assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
      }
    else
      {
        ImageLevel* pImageLevel = (ImageLevel*)pFrame;
        assert(pCamera->width == pImageLevel->grayImage.cols);
        assert(pCamera->height == pImageLevel->grayImage.rows);
      }

  }

  template<typename T>
  bool operator()(const T* const* const parameters, T* residuals) const
  {
    int residual_num = PE_RESIDUAL_NUM_ARRAY[PE_TYPE];
    for(int i = 0; i < residual_num; ++i)
      residuals[i] = T(0.0);

    // v is the position after applying deformation only
    // p is the position after rigid transformation on v
    T v[3], p[3], diff_vertex[3], rot_diff_vertex[3];
    v[0] = T(0.0); v[1] = T(0.0); v[2] = T(0.0);
    p[0] = T(0.0); p[1] = T(0.0); p[2] = T(0.0);
    T transformed_r, transformed_c;

    const T* const* const trans = parameters;
    const T* const* const rotations = &(parameters[ numNeighbors ]);

    const T* const rigid_rot = parameters[ 2*numNeighbors ];
    const T* const rigid_trans = parameters[ 2*numNeighbors+1 ];

    // compute the position from coarse neighbors nodes first
    for(int i = 0; i < numNeighbors; ++i)
      {
        // get template difference
        for(int index = 0; index < 3; ++index)
          diff_vertex[index] = T(pVertex[index]) - neighborVertices[ i ][ index ];

        ceres::AngleAxisRotatePoint( &(rotations[ i ][ 0 ]), diff_vertex, rot_diff_vertex );

        for(int index = 0; index < 3; ++index)
          v[index] += neighborWeights[i] * (rot_diff_vertex[ index ]
                                            + neighborVertices[ i ][ index ] + trans[ i ][ index] );
      }

    ceres::AngleAxisRotatePoint( rigid_rot, v , p);
    p[0] += rigid_trans[0];
    p[1] += rigid_trans[1];
    p[2] += rigid_trans[2];

    getResidual(weight, pCamera, pFrame, pValue, p, residuals, PE_TYPE);

    return true;

  }

private:

  double* pVertex;
  double weight;
  double* pValue;
  const CameraInfo* pCamera;
  const Level* pFrame;
  dataTermErrorType PE_TYPE;

  int numNeighbors;
  vector<double> neighborWeights;
  vector<double*> neighborVertices;

};

// ResidualImageProjectionPatch
class ResidualImageProjectionPatch
{
public:

  ResidualImageProjectionPatch(double weight, const PangaeaMeshData* pMesh,
                               const CameraInfo* pCamera, const Level* pFrame, int numNeighbors,
                               vector<double> neighborWeights, vector<unsigned int> neighborRadii,
                               vector<unsigned int> neighbors, dataTermErrorType PE_TYPE=PE_NCC):
    weight(weight),
    pMesh(pMesh),
    pCamera(pCamera),
    pFrame(pFrame),
    numNeighbors(numNeighbors),
    neighborWeights(neighborWeights),
    neighborRadii(neighborRadii),
    neighbors(neighbors),
    PE_TYPE(PE_TYPE)
  {
    // check the consistency between camera and images
    // assert(pCamera->width == pFrame->grayImage.cols);
    // assert(pCamera->height == pFrame->grayImage.rows);

    if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
      {
        FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
        assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
        assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
      }
    else
      {
        ImageLevel* pImageLevel = (ImageLevel*)pFrame;
        assert(pCamera->width == pImageLevel->grayImage.cols);
        assert(pCamera->height == pImageLevel->grayImage.rows);
      }
  }

  template<typename T>
  bool operator()(const T* const* const parameters, T* residuals) const
  {
    int residual_num = PE_RESIDUAL_NUM_ARRAY[PE_TYPE];
    for(int i = 0; i < residual_num; ++i)
      residuals[i] = T(0.0);

    const T* const* const trans = parameters;

    const T* const rigid_rot = parameters[ numNeighbors ];
    const T* const rigid_trans = parameters[ numNeighbors + 1 ];

    vector<T> neighborVertices;
    neighborVertices.resize( 3*numNeighbors );

    T p[3];
    for(int i = 0; i < numNeighbors; ++i)
      {
        for( int k = 0; k < 3; ++k)
          p[k] = trans[i][k] + T(pMesh->vertices[ neighbors[i] ][k]);

        ceres::AngleAxisRotatePoint( rigid_rot, p, &neighborVertices[3*i] );

        for( int k = 0; k < 3; ++k)
          neighborVertices[3*i+k] += rigid_trans[k];
      }

    getPatchResidual(weight, pCamera, pFrame, pMesh, neighborVertices, numNeighbors, neighbors, residuals, PE_TYPE);

    //residuals[0] = T(0.0);

    return true;

  }


private:

  double weight;

  const PangaeaMeshData* pMesh;
  const CameraInfo* pCamera;
  const Level* pFrame;

  int numNeighbors;

  vector<unsigned int> neighborRadii;
  vector<unsigned int> neighbors;
  vector<double> neighborWeights;

  vector<double> neighborVertexPositions;

  dataTermErrorType PE_TYPE;

};

// ResidualImageProjectionPatch from coarse level deformation
class ResidualImageProjectionPatchCoarse
{
public:

  ResidualImageProjectionPatchCoarse(double weight, const PangaeaMeshData* pMesh,
                                     const PangaeaMeshData* pNeighborMesh, const CameraInfo* pCamera,
                                     const Level* pFrame, int numNeighbors, int numCoarseNeighbors,
                                     vector<double> neighborWeights, vector<unsigned int> neighborRadii,
                                     vector<unsigned int> neighbors, vector<unsigned int> parameterIndices,
                                     vector<unsigned int> coarseNeighborIndices, vector<unsigned int> coarseNeighborBiases,
                                     vector<double> coarseNeighborWeights, dataTermErrorType PE_TYPE=PE_NCC):
    weight(weight),
    pMesh(pMesh),
    pNeighborMesh(pNeighborMesh),
    pCamera(pCamera),
    pFrame(pFrame),
    numNeighbors(numNeighbors),
    numCoarseNeighbors(numCoarseNeighbors),
    neighborWeights(neighborWeights),
    neighborRadii(neighborRadii),
    neighbors(neighbors),
    parameterIndices(parameterIndices),
    coarseNeighborIndices(coarseNeighborIndices),
    coarseNeighborBiases(coarseNeighborBiases),
    coarseNeighborWeights(coarseNeighborWeights),
    PE_TYPE(PE_TYPE)
  {
    // check the consistency between camera and images
    // assert(pCamera->width == pFrame->grayImage.cols);
    // assert(pCamera->height == pFrame->grayImage.rows);

    if(PE_TYPE == PE_FEATURE || PE_TYPE == PE_FEATURE_NCC)
      {
        FeatureLevel* pFeatureLevel = (FeatureLevel*)pFrame;
        assert(pCamera->width == pFeatureLevel->featureImageVec[0].cols);
        assert(pCamera->height == pFeatureLevel->featureImageVec[0].rows);
      }
    else
      {
        ImageLevel* pImageLevel = (ImageLevel*)pFrame;
        assert(pCamera->width == pImageLevel->grayImage.cols);
        assert(pCamera->height == pImageLevel->grayImage.rows);
      }

  }

  template<typename T>
  bool operator()(const T* const* const parameters, T* residuals) const
  {

    int residual_num = PE_RESIDUAL_NUM_ARRAY[PE_TYPE];
    for(int i = 0; i < residual_num; ++i)
      residuals[i] = T(0.0);

    const T* const* const trans = parameters;
    const T* const* const rotations = &(parameters[ numCoarseNeighbors ]  );

    const T* const rigid_rot = parameters[ 2*numCoarseNeighbors ];
    const T* const rigid_trans = parameters[ 2*numCoarseNeighbors + 1 ];

    vector<T> neighborVertices;
    neighborVertices.resize( 3*numNeighbors );

    // computer the position from coarse neighbors nodes first

    T p[3], diff_vertex[3], rot_diff_vertex[3];

    int startPos = 0; // starting position for neighbors of i
    for(int i = 0; i < numNeighbors; ++i)
      {

        p[0] = T(0.0); p[1] = T(0.0); p[2] = T(0.0);

        int endPos = coarseNeighborBiases[i];  // ending position for neighbors of i

        for(int j = startPos; j < endPos; ++j)
          {
            for(int k = 0; k < 3; ++k)
              diff_vertex[k] = pMesh->vertices[ neighbors[i] ][k] - T(pNeighborMesh->vertices[ coarseNeighborIndices[ j ]  ][k] );

            ceres::AngleAxisRotatePoint( &(rotations[ parameterIndices[j] ][ 0 ]), diff_vertex, rot_diff_vertex );

            for(int index = 0; index < 3; ++index)
              p[index] += coarseNeighborWeights[j] * (rot_diff_vertex[ index ]
                                                      + pNeighborMesh->vertices[ coarseNeighborIndices[j] ][ index ]
                                                      + trans[ parameterIndices[j] ][ index] );
          }

        startPos = endPos;

        ceres::AngleAxisRotatePoint( rigid_rot, p, &neighborVertices[3*i] );

        for( int k = 0; k < 3; ++k)
          neighborVertices[3*i+k] += rigid_trans[k];

      }

    getPatchResidual(weight, pCamera, pFrame, pMesh, neighborVertices, numNeighbors, neighbors, residuals, PE_TYPE);

    return true;

  }

private:

  double weight;

  const PangaeaMeshData* pMesh;
  const PangaeaMeshData* pNeighborMesh;
  const CameraInfo* pCamera;
  const Level* pFrame;

  int numNeighbors;
  int numCoarseNeighbors;

  vector<unsigned int> neighborRadii;
  vector<unsigned int> neighbors;
  vector<double> neighborWeights;

  vector<unsigned int> parameterIndices;
  vector<unsigned int> coarseNeighborIndices;
  vector<unsigned int> coarseNeighborBiases;
  vector<double> coarseNeighborWeights;

  dataTermErrorType PE_TYPE;

};

class ResidualTV
{
public:

  ResidualTV(double weight):
    weight(weight), optimizeDeformation(true) {}

  ResidualTV(double weight, double* pVertex, double* pNeighbor):
    weight(weight), pVertex(pVertex), pNeighbor(pNeighbor), optimizeDeformation(false) {}

  template <typename T>
  bool operator()(const T* const pCurrentVertex,
                  const T* const pCurrentNeighbor,
                  T* residuals) const
  {

    if(optimizeDeformation){
      // in this case, pCurrentVertex and pCurrentNeighbor refer to translation
      for(int i = 0; i <3; ++i)
        residuals[i] = T(weight) * ( pCurrentVertex[i] - pCurrentNeighbor[i]);
    }
    else{
      for(int i = 0; i <3; ++i)
        residuals[i] = T(weight) * ( T(pVertex[i]  - pNeighbor[i]) -
                                     ( pCurrentVertex[i] - pCurrentNeighbor[i]) );
    }
    return true;
  }

private:
  bool optimizeDeformation;
  double weight;
  const double* pVertex;
  const double* pNeighbor;

};

// total variation on top of the local rotations
class ResidualRotTV
{
public:

  ResidualRotTV(double weight):weight(weight){}

  template<typename T>
  bool operator()(const T* const pCurrentRot,
                  const T* const pCurrentNeighbor,
                  T* residuals) const
  {
    for(int i= 0; i < 3; ++i)
      residuals[i] = T(weight) * ( pCurrentRot[i] - pCurrentNeighbor[i] );

    return true;
  }

private:

  double weight;
};

class ResidualINEXTENT
{
public:

  ResidualINEXTENT(double weight):
    weight(weight), optimizeDeformation(true)
  {}

  ResidualINEXTENT(double weight, double* pVertex, double* pNeighbor):
    weight(weight),pVertex(pVertex),pNeighbor(pNeighbor), optimizeDeformation(false)
  {}

  template <typename T>
  bool operator()(const T* const pCurrentVertex,
                  const T* const pCurrentNeighbor,
                  T* residuals) const
  {
    T diff[3];
    T diffref[3];

    if(optimizeDeformation){
      for(int i = 0; i < 3; i++){
        diff[i] = pCurrentNeighbor[i];
        diffref[i] = T(pNeighbor[i]);
      }
    }
    else{
      for(int i = 0; i < 3; ++i){
        diff[i] = pCurrentVertex[i] - pCurrentNeighbor[i];
        diffref[i] = T(pVertex[i] - pNeighbor[i]);
      }
    }

    T length;
    T lengthref;

    length = sqrt(diff[0] * diff[0] +
                  diff[1] * diff[1] +
                  diff[2] * diff[2]);

    lengthref = sqrt(diffref[0] * diffref[0] +
                     diffref[1] * diffref[1] +
                     diffref[2] * diffref[2]);

    residuals[0] = T(weight) * (lengthref - length);

    return true;

  }

private:

  bool optimizeDeformation;
  double weight;
  const double* pVertex;
  const double* pNeighbor;

};

// the rotation to be optimized is from template mesh to current mesh
class ResidualARAP
{
public:

  ResidualARAP(double weight, double* pVertex, double* pNeighbor, bool optDeform=false):
    weight(weight), pVertex(pVertex), pNeighbor(pNeighbor), optimizeDeformation(optDeform) {}

  template <typename T>
  bool operator()(const T* const pCurrentVertex,
                  const T* const pCurrentNeighbor,
                  const T* const pRotVertex,
                  T* residuals) const
  {
    T templateDiff[3];
    T rotTemplateDiff[3];
    T currentDiff[3];

    if(optimizeDeformation){
      // deformation optimization
      for(int i = 0; i < 3; ++i){
        templateDiff[i] =  T(pVertex[i] - pNeighbor[i]);
        currentDiff[i] = templateDiff[i] + (pCurrentVertex[i] - pCurrentNeighbor[i]);
      }
    }
    else{
      for(int i = 0; i <3; ++i){
        templateDiff[i] =  T(pVertex[i] - pNeighbor[i]);
        currentDiff[i]  = pCurrentVertex[i] - pCurrentNeighbor[i];
      }
    }

    ceres::AngleAxisRotatePoint(pRotVertex, templateDiff, rotTemplateDiff);

    residuals[0] = T(weight) * ( currentDiff[0] - rotTemplateDiff[0] );
    residuals[1] = T(weight) * ( currentDiff[1] - rotTemplateDiff[1] );
    residuals[2] = T(weight) * ( currentDiff[2] - rotTemplateDiff[2] );

    return true;

  }

private:

  bool optimizeDeformation;
  double weight;
  const double* pVertex;
  const double* pNeighbor;

};

// temporal shape
class ResidualDeform
{
public:

  ResidualDeform(double weight, double* pVertex):
    weight(weight), pVertex(pVertex) {}

  template <typename T>
  bool operator()(const T* const pCurrentVertex,
                  T* residuals) const
  {
    for(int i = 0; i < 3; ++i)
      residuals[i] = T(weight) * (pCurrentVertex[i] - T(pVertex[i]));

    return true;
  }

private:

  bool optimizeDeformation;
  double weight;
  const double* pVertex;

};

class ResidualTemporalMotion
{
public:
  ResidualTemporalMotion(double* pPrevRot, double* pPrevTrans,
                         double rotWeight, double transWeight):
    pPrevRot(pPrevRot), pPrevTrans(pPrevTrans),
    rotWeight(rotWeight), transWeight(transWeight)
  {}

  template <typename T>
  bool operator()(const T* const pRot,
                  const T* const pTrans,
                  T* residuals) const
  {
    residuals[0] = rotWeight * (pRot[0] -     pPrevRot[0]);
    residuals[1] = rotWeight * (pRot[1] -     pPrevRot[1]);
    residuals[2] = rotWeight * (pRot[2] -     pPrevRot[2]);
    residuals[3] = transWeight * (pTrans[0] - pPrevTrans[0]);
    residuals[4] = transWeight * (pTrans[1] - pPrevTrans[1]);
    residuals[5] = transWeight * (pTrans[2] - pPrevTrans[2]);

    return true;
  }

  double rotWeight;
  double transWeight;
  const double* pPrevRot;
  const double* pPrevTrans;

};


// ICP known correspondences residual
class ResidualKnownICP
{
public:

  ResidualKnownICP(double weight, double* pTemplate, double* pTarget):
    weight(weight),pTemplate(pTemplate),pTarget(pTarget){}

  template<typename T>
  bool operator()(const T* const pRot,
                  const T* const pTrans,
                  T* residuals) const
  {
    T pRotTemplate[3];

    T pTemp[3];
    for(int i = 0;  i < 3; ++i)
      pTemp[i] = T(pTemplate[i]);

    ceres::AngleAxisRotatePoint(pRot, pTemp, pRotTemplate);

    for(int i = 0; i < 3; ++i)
      residuals[i] = T(weight)*(pRotTemplate[i] + pTrans[i]- T(pTarget[i]));

    return true;

  }

private:

  double weight;
  double* pTemplate;
  double* pTarget;

};

class EnergyCallback: public ceres::IterationCallback
{

private:
  std::vector<double> m_EnergyRecord;

public:
  EnergyCallback(){}
  virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary & summary)
  {
    m_EnergyRecord.push_back(summary.cost);
    return ceres::SOLVER_CONTINUE;
  }
  void PrintEnergy(std::ostream& output)
  {
    output << "Energy Started" << std::endl;
    for(int i=0; i< m_EnergyRecord.size(); ++i)
      output<<(i+1)<<" "<< m_EnergyRecord[i]<<std::endl;
    output << "Energy Ended" << std::endl;
  }
  void Reset()
  {
    m_EnergyRecord.clear();
  }

};

// class ResidualPhotometricCallback: public ceres::IterationCallback
// {


// }

// Applies deformation, rotation and translation to a vertex
template <typename T>
void getRotTransP(const T* const rotation, const T* const translation,
	const T* const xyz, const double* const pVertex, bool optimizeDeformation, T* p)
{
	T afterTrans[3];

	// if we are doing optimization on the transformation,
	// we need to add up the template position first
	if (optimizeDeformation)
	{
		afterTrans[0] = xyz[0] + pVertex[0];
		afterTrans[1] = xyz[1] + pVertex[1];
		afterTrans[2] = xyz[2] + pVertex[2];
	}
	else
	{
		afterTrans[0] = xyz[0];
		afterTrans[1] = xyz[1];
		afterTrans[2] = xyz[2];
	}

	ceres::AngleAxisRotatePoint(rotation, afterTrans, p);
	p[0] += translation[0];
	p[1] += translation[1];
	p[2] += translation[2];
}

// Computes vertex normal direction given its position, its one-ring neighbours 
// and the corresponding face indexes. Can handle clockwise and counter-clockwise
template <typename T>
void computeNormal(const T* p, const vector<T*> &adjP, const vector<unsigned int> &face_vIdxs,
	const bool clockwise, T* normal)
{
	normal[0] = T(0.0);
	normal[1] = T(0.0);
	normal[2] = T(0.0);

	for (int i = 0; i < face_vIdxs.size() / 2; i++)
	{
		unsigned int vIdx1 = face_vIdxs[2 * i];
		unsigned int vIdx2 = face_vIdxs[2 * i + 1];

		T face_normal[3];
		//compnorm(p, adjP[vIdx1], adjP[vIdx2], face_normal, false);

		// WORKAROUND
		// Problems with ambiguity with compnorm. This should be solved.
		// For now, the function has just been copied and pasted here

		const T* ver1 = p;
		const T* ver2 = adjP[vIdx1];
		const T* ver3 = adjP[vIdx2];

		T a[3];
		T b[3];

		if (clockwise)
		{
			a[0] = ver1[0] - ver3[0];
			a[1] = ver1[1] - ver3[1];
			a[2] = ver1[2] - ver3[2];

			b[0] = ver1[0] - ver2[0];
			b[1] = ver1[1] - ver2[1];
			b[2] = ver1[2] - ver2[2];
		}
		else	// Anti-clockwsie
		{
			a[0] = ver1[0] - ver2[0];
			a[1] = ver1[1] - ver2[1];
			a[2] = ver1[2] - ver2[2];

			b[0] = ver1[0] - ver3[0];
			b[1] = ver1[1] - ver3[1];
			b[2] = ver1[2] - ver3[2];
		}

		face_normal[0] = a[1] * b[2] - a[2] * b[1];
		face_normal[1] = a[2] * b[0] - a[0] * b[2];
		face_normal[2] = a[0] * b[1] - a[1] * b[0];

		if (face_normal[1] * face_normal[1]
			+ face_normal[2] * face_normal[2]
			+ face_normal[0] * face_normal[0] != T(0))
		{
			T temp = T(1.0f) /
				sqrt(face_normal[1] * face_normal[1]
				+ face_normal[2] * face_normal[2]
				+ face_normal[0] * face_normal[0]);
			face_normal[0] *= temp;
			face_normal[1] *= temp;
			face_normal[2] *= temp;
		}

		normal[0] += face_normal[0];
		normal[1] += face_normal[1];
		normal[2] += face_normal[2];
	}

	T norm = sqrt(normal[1] * normal[1] + normal[2] * normal[2] + normal[0] * normal[0]);
	if (norm != T(0.0))
	{
		normal[0] /= norm;
		normal[1] /= norm;
		normal[2] /= norm;
	}
}

// Computes shading value given normal direction, spherical harmonic coefficients 
// and the SH order
template <typename T>
T computeShading(const T* _normal, const double* _sh_coeff, int _sh_order)
{
	T n_x = _normal[0];
	T n_y = _normal[1];
	T n_z = _normal[2];

	T n_x2 = n_x * n_x;
	T n_y2 = n_y * n_y;
	T n_z2 = n_z * n_z;
	T n_xy = n_x * n_y;
	T n_xz = n_x * n_z;
	T n_yz = n_y * n_z;
	T n_x2_y2 = n_x2 - n_y2;

	T shading = T(_sh_coeff[0]);

	if (_sh_order > 0)
		shading = shading
		+ T(_sh_coeff[1]) * n_x					// x
		+ T(_sh_coeff[2]) * n_y					// y
		+ T(_sh_coeff[3]) * n_z;				// z

	if (_sh_order > 1)
		shading = shading
		+ T(_sh_coeff[4]) * n_xy						// x * y
		+ T(_sh_coeff[5]) * n_xz						// x * z
		+ T(_sh_coeff[6]) * n_yz						// y * z
		+ T(_sh_coeff[7]) * n_x2_y2						// x^2 - y^2
		+ T(_sh_coeff[8]) * (T(3.0) * n_z2 - T(1.0));	// 3 * z^2 - 1

	if (_sh_order > 2)
		shading = shading
		+ T(_sh_coeff[9]) * (T(3.0) * n_x2 - n_y2) * n_y		// (3 * x^2 - y^2) * y 
		+ T(_sh_coeff[10]) * n_x * n_y * n_z					// x * y * z
		+ T(_sh_coeff[11]) * (T(5.0) * n_z2 - T(1.0)) * n_y		// (5 * z^2 - 1) * y
		+ T(_sh_coeff[12]) * (T(5.0) * n_z2 - T(3.0)) * n_z		// (5 * z^2 - 3) * z
		+ T(_sh_coeff[13]) * (T(5.0) * n_z2 - T(1.0)) * n_x		// (5 * z^2 - 1) * x
		+ T(_sh_coeff[14]) * n_x2_y2 * n_z						// (x^2 - y^2) * z
		+ T(_sh_coeff[15]) * (n_x2 - T(3.0) * n_y2) * n_x;		// (x^2 - 3 * y^2) * x

	if (_sh_order > 3)
		shading = shading
		+ T(_sh_coeff[16]) * n_x2_y2 * n_x * n_y								// (x^2 - y^2) * x * y
		+ T(_sh_coeff[17]) * (T(3.0) * n_x2 - n_y2) * n_yz						// (3 * x^2 - y^2) * yz
		+ T(_sh_coeff[18]) * (T(7.0) * n_z2 - T(1.0)) * n_xy					// (7 * z^2 - 1) * x * y
		+ T(_sh_coeff[19]) * (T(7.0) * n_z2 - T(3.0)) * n_yz					// (7 * z^2 - 3) * y * z
		+ T(_sh_coeff[20]) * (T(3.0) - T(30.0) * n_z2 + T(35.0) * n_z2 * n_z2)	// 3 - 30 * z^2 + 35 * z^4
		+ T(_sh_coeff[21]) * (T(7.0) * n_z - T(3.0)) * n_xz						// (7 * z^2 - 3) * x * z
		+ T(_sh_coeff[22]) * (T(7.0) * n_z - T(1.0)) * n_x2_y2					// (7 * z^2 - 1) * (x^2 - y^2)
		+ T(_sh_coeff[23]) * (n_x2 - T(3.0) * n_y2) * n_xz						// (x^2 - 3 * y^2) * x * z
		+ T(_sh_coeff[24]) * ((n_x2 - T(3.0) * n_y2) * n_x2					// (x^2 - 3 * y^2) * x^2 - (3 * x^2 - y^2) * y^2 
		- (T(3.0) * n_x2 - n_y2) * n_y2);

	return shading;
}

template<typename T>
void getResidualIntrinsic(double weight, const CameraInfo* pCamera, const Level* pFrame,
	double* pValue, T* shading, T* p, T* residuals, const dataTermErrorType& PE_TYPE)
{
	T transformed_r, transformed_c;

	IntrinsicProjection(pCamera, p, &transformed_c, &transformed_r);

	T templateValue, currentValue;

	if (transformed_r >= T(0.0) && transformed_r < T(pCamera->height) &&
		transformed_c >= T(0.0) && transformed_c < T(pCamera->width))
	{
		ImageLevel* pImageLevel = (ImageLevel*)pFrame;
		switch (PE_TYPE)
		{
		case PE_INTRINSIC:
			templateValue = T(pValue[0]) * shading[0];
			currentValue = SampleWithDerivative< T, InternalIntensityImageType >(pImageLevel->grayImage,
				pImageLevel->gradXImage,
				pImageLevel->gradYImage,
				transformed_c,
				transformed_r);
			residuals[0] = T(weight) * (currentValue - templateValue);
			break;

		case PE_INTRINSIC_COLOR:
			for (int i = 0; i < 3; ++i)
			{
				templateValue = T(pValue[i]) * shading[0];
				currentValue = SampleWithDerivative< T, InternalIntensityImageType >(pImageLevel->colorImageSplit[i],
					pImageLevel->colorImageGradXSplit[i],
					pImageLevel->colorImageGradYSplit[i],
					transformed_c,
					transformed_r);
				residuals[i] = T(weight) * (currentValue - templateValue);
			}
			break;

		default:
			break;
		}
	}

}

// Photometric cost for the case of known albedo and illumination (sh representation)
class ResidualImageProjectionIntrinsic : public ResidualImageProjection
{
public:
	ResidualImageProjectionIntrinsic(double weight, double* pValue, double* pVertex,
		const CameraInfo* pCamera, const Level* pFrame, const int _num_neighbours,
		const vector<double*> &_adjPVertex, const vector<unsigned int> &_face_vIdxs,
		dataTermErrorType PE_TYPE = PE_INTRINSIC, const bool _clockwise = true,
		const int _sh_order = 0, const double* _sh_coeff = 0) :
		ResidualImageProjection(weight, pValue, pVertex,
		pCamera, pFrame, PE_TYPE),
		num_neighbours(_num_neighbours),
		adjPVertex(_adjPVertex),
		face_vIdxs(_face_vIdxs),
		clockwise(_clockwise),
		sh_order(_sh_order),
		sh_coeff(_sh_coeff)
	{

	}

	template<typename T>
	bool operator()(const T* const* const parameters, T* residuals) const
	{
		// Parameters:
		// 0 - Rotation
		// 1 - Translation
		// 2 - Current vertex position or translation
		// >2 - Neighbour vertices positions or translations

		const T* rotation = parameters[0];
		const T* translation = parameters[1];

		T p[3];
		getRotTransP(rotation, translation, parameters[2], pVertex,
			optimizeDeformation, p);

		vector<T*> adjP;
		T* p_neighbour;
		for (int i = 0; i < num_neighbours; i++)
		{
			p_neighbour = new T[3];
			getRotTransP(rotation, translation, parameters[3 + i], adjPVertex[i],
				optimizeDeformation, p_neighbour);
			adjP.push_back(p_neighbour);
		}

		T normal[3];
		computeNormal(p, adjP, face_vIdxs, clockwise, normal);

		for (int i = 0; i < num_neighbours; i++)
		{
			delete[] adjP[i];
		}

		T shading = computeShading(normal, sh_coeff, sh_order);

		getResidualIntrinsic(weight, pCamera, pFrame, pValue, &shading, p, residuals,
			PE_TYPE);

		return true;

	}

private:
	// Number of one-ring neighbours
	const int num_neighbours;
	// Adjacent faces indexes
	const vector<unsigned int> face_vIdxs;
	// Adjacent vertices coordinates
	const vector<double*> adjPVertex;

	// Identifies if faces are defined clockwise
	const bool clockwise;
	// SH order
	const int sh_order;
	//SH coefficients
	const double* sh_coeff;
};

// Residual of the difference between the previous SH coefficients and the current ones
class ResidualSHCoeff
{
public:
	ResidualSHCoeff(const int _n_sh_coeff) : n_sh_coeff(_n_sh_coeff)
	{

	}

	template<typename T>
	bool operator()(const T* const diff_sh_coeff, T* residuals) const
	{
		for (int i = 0; i < n_sh_coeff; i++)
		{
			residuals[i] = diff_sh_coeff[i];
		}

		return true;
	}

private:
	// Number of SH coeff
	const int n_sh_coeff;
};

// Residual of the difference between the previous albedo values and the current ones
class ResidualAlbedo
{
public:
	ResidualAlbedo(const bool _is_grayscale) : n_channels(_is_grayscale ? 1 : 3)
	{

	}

	template<typename T>
	bool operator()(const T* const diff_albedo, T* residuals) const
	{
		for (int i = 0; i < n_channels; i++)
		{
			residuals[i] = diff_albedo[i];
		}

		return true;
	}

private:
	// Number of albedo channels
	const int n_channels;
<<<<<<< HEAD
};
=======
};
>>>>>>> 66ea330734cce9125dc558af2742d6d33ba7774c
