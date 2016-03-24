#include "main_engine/tracker/FeatureReader.h"

#include "main_engine/utils/settings.h"

//// LMDB DataBase

LMDBReader::LMDBReader(string folder)
{
  db_path = folder;
}

LMDBReader::~LMDBReader( )
{
  ShutDownDB();
}

void LMDBReader::InitializeDB(int height, int width, int numChannels)
{

  m_nWidth = width;
  m_nHeight = height;
  m_nNumChannels = numChannels;

  // check the status of the database
  std::cout << "Openning lmdb " << db_path << endl;
  // check the folder already exists

  // bool valid;

  // valid = (mdb_env_create(&mdb_env) ==  MDB_SUCCESS) &&
  //   (mdb_env_set_mapsize(mdb_env, 10485760000000) == MDB_SUCCESS) &&
  //   (mdb_env_open(mdb_env, db_path, 0, 0664) == MDB_SUCCESS) &&
  //   (mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn) == MDB_SUCCESS) &&
  //   (mdb_open(mdb_txn, NULL, 0, &mdb_dbi) == MDB_SUCCESS);

  // assert(mdb_cursor_open(mdb_txn, mdb_dbi, &mdb_cursor) == MDB_SUCCESS);
  // int mdb_status = mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_data, MDB_NEXT);
  // if(mdb_status == MDB_NOTFOUND)
  //   valid = false;
  // else{
  //   MDB_CHECK(mdb_status);
  //   valid = true;
  // }

  int test1 = mdb_env_create(&mdb_env);
  std::cout << "return value of environment creation " << test1 << endl;

  //  int test2 = mdb_env_set_mapsize(mdb_env, 10485760000000);
  int test2 = mdb_env_set_mapsize(mdb_env, 10485760000);
  std::cout << "return value of setting environment mapsize " << test2 << endl;

  // int test3 = mdb_env_open(mdb_env, db_path.c_str(), 0, 0664);
  int test3 = mdb_env_open(mdb_env, db_path.c_str(), MDB_NOLOCK, 0664);
  std::cout << "return value of environment openning " << test3 << endl;

  int test4 = mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn);
  std::cout << "return value of context beginning " << test4 << endl;

  int test5 = mdb_open(mdb_txn, NULL, 0, &mdb_dbi);
  std::cout << "return value of mdb openning " << test5 << endl;

  // if(!valid)
  //   {
  //     std::cout << "Open lmdb error" << std::endl;
  //     throw 20;
  //   };

}

void LMDBReader::ShutDownDB()
{
  mdb_close(mdb_env, mdb_dbi);
  mdb_env_close(mdb_env);
}

void LMDBReader::getFeatureLevel(string key, int channel,
                            FeatureImageType& featureBufferImage)
{

  // setup current pyramid
  mdb_key.mv_size = key.length();
  mdb_key.mv_data = reinterpret_cast<void*>(&key[0]);

  mdb_get(mdb_txn, mdb_dbi, &mdb_key, &mdb_data);

  // setup the 0th level of current feature image pyramid

  int data_size = mdb_data.mv_size;
  int channel_num = m_nHeight * m_nWidth;
  int channel_size = channel_num * featureSettings.dataElemSize;
  int real_size =  m_nNumChannels * channel_size;
  int shift = (data_size - real_size) / featureSettings.dataElemSize;

  switch(featureSettings.dataElemSize)
    {
    case 1:
      {
        unsigned char* data_pointer = reinterpret_cast<unsigned char*>(mdb_data.mv_data);
        featureBufferImage = cv::Mat(m_nHeight,
                                     m_nWidth,
                                     featureSettings.dataTypeINT,
                                     data_pointer + shift + channel * channel_num);
      }
      break;
    case 4:
      {
        float* data_pointer = reinterpret_cast<float*>(mdb_data.mv_data);
        featureBufferImage = cv::Mat(m_nHeight,
                                     m_nWidth,
                                     featureSettings.dataTypeINT,
                                     data_pointer + shift + channel * channel_num);
      }
      break;
    case 8:
      {
        double* data_pointer = reinterpret_cast<double*>(mdb_data.mv_data);
        featureBufferImage = cv::Mat(m_nHeight,
                                     m_nWidth,
                                     featureSettings.dataTypeINT,
                                     data_pointer + shift + channel * channel_num);
        // cv::namedWindow("featureBufferImage", cv::WINDOW_AUTOSIZE);
        // cv::imshow("featureBufferImage", featureBufferImage);
        // cv::waitKey(0);

      }
      break;
    }

}

//// HDF5 DataBase

HDF5Reader::HDF5Reader(string file)
  :pFeatureImages(NULL)
{
  hdf5file = file;
}

HDF5Reader::~HDF5Reader()
{
  if(pFeatureImages)
    delete[] pFeatureImages;
}

void HDF5Reader::InitializeDB(int height, int width, int numChannels)
{
  m_nWidth = width;
  m_nHeight = height;
  m_nNumChannels = numChannels;

  pFeatureImages = new double[height*width*numChannels];

}

void HDF5Reader::ShutDownDB(){}

void HDF5Reader::getFeatureLevel(string key, int channel,
                            FeatureImageType& featureBufferImage)
{

  cout << "feature " << key << endl
       << "channel " << channel << endl;

  featureBufferImage = cv::Mat(m_nHeight, m_nWidth, featureSettings.dataTypeINT);

  if(currentKey == key)
    {
      memcpy(featureBufferImage.data, (double*)pFeatureImages + channel*m_nHeight*m_nWidth, 8*m_nHeight*m_nWidth);
      return;
    }

  file_id = H5Fopen(hdf5file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  dataset_id = H5Dopen2(file_id, key.c_str(), H5P_DEFAULT);

  H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pFeatureImages);
  memcpy(featureBufferImage.data, (double*)pFeatureImages + channel*m_nHeight*m_nWidth, 8*m_nHeight*m_nWidth);

  currentKey = key;

  H5Dclose(dataset_id);

  H5Fclose(file_id);

}
