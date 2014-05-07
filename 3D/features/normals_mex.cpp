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
#include <pcl/features/normal_3d.h>
#include <string.h>

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{

	// input checks
	if (nrhs != 3)
    mexErrMsgTxt("Expected 3 input arguments");
    
	// create required objects
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

	// get cloud from mex input arguments...
	int rows = mxGetM(prhs[0]);
  int cols = mxGetN(prhs[0]);
  if (cols != 3)
    mexErrMsgTxt("Wrong number of columns - expected 3.");
  
  mexPrintf("Computing normals...\t");
  double *data = mxGetPr(prhs[0]);
  for (int i = 0; i < rows; ++i)
 	{
 		pcl::PointXYZ point( (float)data[i], (float)data[i + rows], (float)data[i + 2 * rows] );
 		cloud->push_back( point );
	}
  
  // Getting string referring to search type
	const mxArray *yData = prhs[1];
	int yLength = mxGetN(yData)+1;;
	char *search_type = (char*)mxCalloc(yLength, sizeof(char));
	mxGetString(yData,search_type,yLength);

  // compute normals
  ne.setInputCloud (cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);

  if (strncmp(search_type,"knn",3)==0)
  {
    double *nn = mxGetPr(prhs[2]);
    int nn_int = (int)(*nn);
    ne.setKSearch (nn_int);
  }
  else if  (strncmp(search_type,"radius",6)==0)
  {
    double *r = mxGetPr(prhs[2]);
    ne.setRadiusSearch (*r);
  }
  else
   mexErrMsgTxt("Unknown search type - expected 'knn' or 'radius'.");

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
  //cloud->clear();
  //cloud_normals->clear();
  
  //delete &cloud, &cloud_normals, &ne;

  
}
