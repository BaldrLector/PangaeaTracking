/*************************************************************************************************
\Author:	Qingxiong Yang
\Function:	This class implements the O(1) bilateral filtering method presented in the reference.
\reference: Qingxiong Yang, Kar-Han Tan and Narendra Ahuja, Real-time O(1) Bilateral Filtering, 
               IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2009.
**************************************************************************************************/
#ifndef QX_CTBF_SS_H
#define QX_CTBF_SS_H
#define QX_DEF_CTBF_GAUSSIAN_BILATERAL_FILTER		0
#define QX_DEF_CTBF_BOX_BILATERAL_FILTER			1
#define QX_DEF_CTBF_INTENSITY_RANGE					256
#define QX_DEF_CTBF_SIGMA_SPATIAL_DEFAULT			0.03
#define QX_DEF_CTBF_SIGMA_RANGE_DEFAULT				0.08
#define QX_DEF_CTBF_H_MIN							24
#define QX_DEF_CTBF_W_MIN							32

#include "qx_basic.h"
//#include "qx_ppm.h"
               
class qx_ctbf_ss
{
public:
    qx_ctbf_ss();
    ~qx_ctbf_ss();
    void clean();
	int init(int h_original,int w_original,
		int spatial_filter=QX_DEF_CTBF_GAUSSIAN_BILATERAL_FILTER,//default: Gaussian BF
		double sigma_spatial=QX_DEF_CTBF_SIGMA_SPATIAL_DEFAULT,//0.03~16/512
		double sigma_range=QX_DEF_CTBF_SIGMA_RANGE_DEFAULT);//0.08~20/255
	int joint_bilateral_filter(unsigned char**image_filtered,unsigned char**image,unsigned char**texture,unsigned char**mask,int nr_scale,double sigma_spatial,double sigma_range);
	int joint_bilateral_filter(float**image_filtered,float**image,unsigned char**texture,unsigned char**mask,int nr_scale=8,double sigma_spatial=0,double sigma_range=0);
private:
	char m_str[300];
	int m_h,m_w,m_h_original,m_w_original,m_nr_shift,m_radius; double m_sigma_range,m_sigma_spatial; int m_nr_scale; int m_spatial_filter;
	double ***m_jk,**m_wk; double **m_box; 
	double *m_grayscale;
	double *m_table;
	unsigned char **m_image_y,**m_image_y_downsampled,**m_image_y_downsampled_texture;
	float**m_image_y_downsampled_f;
	void get_down_sampled_y_component(unsigned char **out,unsigned char ***in,int h,int w,int scale_exp);
	void rgb_2_yuv(unsigned char **out_y,unsigned char **out_u,unsigned char **out_v,unsigned char ***in,int h,int w);
	//void yuv_2_rgb_upsample(unsigned char ***out,unsigned char **in_y,unsigned char **in_u,unsigned char **in_v,int h,int w);
	void yuv_2_rgb(unsigned char ***out,unsigned char **in_y,unsigned char ***in_rgb,int h,int w);
};
#endif