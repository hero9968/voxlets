/*
Mex function for segmentation
mex segment_mex.cpp -I/usr/include/pcl-1.5/ -I/usr/local/include/eigen3/ -L/usr/lib/ -lpcl_search -lpcl_kdtree -lpcl_common -lpcl_features

Use like this:

*/

#include "mex.h"
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <string.h>

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{

  // input checks
  if (nrhs != 8)
    mexErrMsgTxt("Expected 7 input arguments");
    
  // create required objects
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  
  // get cloud and normals from mex input arguments...
  int rows = mxGetM(prhs[0]);
  int cols = mxGetN(prhs[0]);
  if (cols != 3)
    mexErrMsgTxt("Wrong number of columns - expected 3.");

  if (rows != mxGetM(prhs[1]))
    mexErrMsgTxt("XYZ and normals must have same number of rows.");

  double *data = mxGetPr(prhs[0]);
  for (int i = 0; i < rows; ++i)
  {
    pcl::PointXYZ point( (float)data[i], (float)data[i + rows], (float)data[i + 2 * rows] );
    cloud->push_back( point );
  }

  double *normals_data = mxGetPr(prhs[1]);
  double *curve_data = mxGetPr(prhs[2]);
  mexPrintf("There are %d rows and %d cols\n", rows, cols);
  mexPrintf("1\n");
  for (int i = 0; i < rows; ++i)
  {
    pcl::Normal normal( (float)normals_data[i], (float)normals_data[i + rows], (float)normals_data[i + 2 * rows] );
    normal.curvature = (float)curve_data[i];
    cloud_normals->push_back( normal );
  }

  // creating kdtree
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);

  // extracting parameters
  int *minsize = (int *)mxGetData(prhs[3]);
  int *maxsize = (int *)mxGetData(prhs[4]);
  int *num_neighbours = (int *)mxGetData(prhs[5]);
  double *smoothness_threshold = mxGetPr(prhs[6]);
  double *curvature_threshold = mxGetPr(prhs[7]);

  mexPrintf("Minx size %d and max size %d\n", *minsize, *maxsize);
  mexPrintf("Neighbours %d\n", *num_neighbours);
  mexPrintf("Smooth %f and curve %f\n", *smoothness_threshold, *curvature_threshold);

  // doing segmentation
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (*minsize);
  reg.setMaxClusterSize (*maxsize);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (*num_neighbours);
  reg.setInputCloud (cloud);
  reg.setInputNormals (cloud_normals);
  reg.setSmoothnessThreshold (*smoothness_threshold);
  reg.setCurvatureThreshold (*curvature_threshold);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

  
  // create output vector and initialise with -1
  plhs[0] = mxCreateNumericMatrix( (mwSize)rows, 1, mxINT16_CLASS, mxREAL);
  uint16_t *out_ptr = (uint16_t *)mxGetData(plhs[0]);
  for (size_t i = 0; i < rows; ++i) { out_ptr[i] = -1; }

  // return the indices of the segments found
  for (size_t clust_idx = 0; clust_idx < clusters.size (); ++clust_idx)
  {
    for (size_t i = 0; i < clusters[clust_idx].indices.size(); ++i)
    {
      size_t this_point_idx = clusters[clust_idx].indices[i];
      out_ptr[this_point_idx] = clust_idx;
    }
  }

}
