/*
Mex function for normal computation
mex normals.cpp -I/usr/include/pcl-1.5/ -I/usr/local/include/eigen3/ -L/usr/lib/ -lpcl_search -lpcl_kdtree -lpcl_common -lpcl_features

Use like this:
norms = normals(xyz, 'knn', 30);
norms = normals(xyz, 'radius', 0.05);
*/

#include "mex.h"
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>
#include <string.h>

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{

	// input checks
	if (nrhs != 4)
    mexErrMsgTxt("Expected 4 input arguments");
    
	// create required objects
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

	// get cloud from mex input arguments...
	int rows = mxGetM(prhs[0]);
  int cols = mxGetN(prhs[0]);
  if (cols != 3)
    mexErrMsgTxt("Wrong number of columns - expected 3.");
  if (rows != 480*640)
    mexErrMsgTxt("Wrong number of rows - expected 480*640.");

  mexPrintf("Computing normals...\t");
  double *data = mxGetPr(prhs[0]);
  const float bad_point = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < rows; ++i)
 	{
    if (((float)data[i] == 0) && ((float)data[i + rows] == 0) && ((float)data[i + 2 * rows] == 0))
    {
     pcl::PointXYZ point( NAN, NAN, NAN );
     cloud->push_back( point );
    }
    else
    {
 		 pcl::PointXYZ point( (float)data[i], (float)data[i + rows], (float)data[i + 2 * rows] );
 		 cloud->push_back( point );
    }
	}
  cloud->height = 640;
  cloud->width = 480;
  cloud->is_dense = false;

  // Getting string referring to search type
	const mxArray *yData = prhs[1];
	int yLength = mxGetN(yData)+1;;
	char *search_type = (char*)mxCalloc(yLength, sizeof(char));
	mxGetString(yData,search_type,yLength);

  // setting inputs
  ne.setInputCloud (cloud);
  double *depth_change_factor = mxGetPr(prhs[2]);
  double *smoothing_size = mxGetPr(prhs[3]);
  ne.setMaxDepthChangeFactor(*depth_change_factor);
  ne.setNormalSmoothingSize(*smoothing_size);

  // setting the type of integral image to do
  if (strncmp(search_type,"cov",3)==0)
    ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  else if  (strncmp(search_type,"grad",4)==0)
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
  else if   (strncmp(search_type,"depth",5)==0)
    ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);
  else
   mexErrMsgTxt("Unknown search type - expected 'cov' or 'grad' or 'depth'.");

  // computing the normals
  ne.compute (*cloud_normals);

	// return the normals...
  plhs[0] = mxCreateDoubleMatrix( (mwSize)rows, (mwSize)cols, mxREAL);
  double *normals_out_ptr = mxGetPr(plhs[0]);

  plhs[1] = mxCreateDoubleMatrix( (mwSize)rows, 1, mxREAL);
  double *curve_out_ptr = mxGetPr(plhs[1]);  
  
  for (int i = 0; i < rows; ++i)
 	{
  	normals_out_ptr[i] = cloud_normals->at(i).normal_x;
  	normals_out_ptr[i + rows] =  cloud_normals->at(i).normal_y;
  	normals_out_ptr[i + 2 * rows] = cloud_normals->at(i).normal_z;
    curve_out_ptr[i] = cloud_normals->at(i).curvature;
	}

  mexPrintf("Done\n");
}
