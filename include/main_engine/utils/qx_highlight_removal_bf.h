/***************************************************************************************************
\Author:	Qingxiong Yang
\Function:	Highlight removal using bilateral filter.
\reference: Qingxiong Yang, Shengnan Wang and Narendra Ahuja, Real-time Specular Highlight Removal 
			Using Bilateral Filtering, European Conference on Computer Vision (ECCV) 2010.
****************************************************************************************************/
#ifndef QX_HIGHLIGHT_REMOVAL_BF_H
#define QX_HIGHLIGHT_REMOVAL_BF_H

#define QX_DEF_DARK_PIXEL								20
#define QX_DEF_THRESHOLD_SIGMA_CHANGE					0.03f

#include "qx_ctbf_ss.h"

class qx_highlight_removal_bf
{
public:
	qx_highlight_removal_bf();
	~qx_highlight_removal_bf();
	void clean();
	int init(int h,int w,
		unsigned char threshold_dark_pixel=QX_DEF_DARK_PIXEL,
		float threshold_sigma_change=QX_DEF_THRESHOLD_SIGMA_CHANGE);
	int diffuse(unsigned char***image_diffuse,unsigned char***image,int nr_iter=0);
private:
	qx_ctbf_ss m_bf;
	int m_h,m_w; unsigned char m_threshold_dark_pixel; float m_threshold_sigma_change; int m_nr_iteration;
	unsigned char ***m_image_diffuse,**m_image_sf,***m_image_backup,**m_mask,**m_mask_dark_pixel,**m_temp;
	float ***m_max_chrom,**m_max_chrom_backup;
	void compute_approximated_maximum_diffuse_chromaticity(unsigned char **image_approximated_max_diffuse_chromaticity,
		unsigned char ***image_normalized,float**image_max_chrom,unsigned char**mask,unsigned char threshold_dark_pixel,int h,int w);
	void compute_diffuse_reflection_from_maximum_diffuse_chromaticity(unsigned char ***image_approximated_max_diffuse_chromaticity,unsigned char ***image_normalized,
		float **max_diffuse_chromaticity,unsigned char**mask,int h,int w);
};

#endif
