/*
 * test_training.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <boost/make_shared.hpp>
#include <pcl/pcl_macros.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/apps/3d_rec_framework/pipeline/global_nn_classifier.h>
 #include <pcl/apps/3d_rec_framework/pc_source/source.h>
#include <pcl/apps/3d_rec_framework/pc_source/mesh_source.h>
//#include <pcl/apps/3d_rec_framework/feature_wrapper/global/vfh_estimator.h>
//#include <pcl/apps/3d_rec_framework/feature_wrapper/global/esf_estimator.h>
//#include <pcl/apps/3d_rec_framework/feature_wrapper/global/cvfh_estimator.h>
//#include <pcl/apps/3d_rec_framework/feature_wrapper/global/crh_estimator.h>
#include "my_crh_estimator.h"
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_crh.h>
#include <pcl/apps/3d_rec_framework/tools/openni_frame_source.h>
#include <pcl/apps/3d_rec_framework/utils/metrics.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


typedef pcl::PointXYZ PointT;
typedef pcl::rec_3d_framework::Model<PointT> ModelT;
typedef pcl::VFHSignature308 VFH_T;

std::pair<std::vector <pcl::PointCloud<PointT>::Ptr>, 
          std::vector <pcl::PointIndices> >
myPrismSegmenter(pcl::PointCloud<PointT>::Ptr cloud, 
              pcl::PointCloud<pcl::Normal>::Ptr cloud_normals);

int
main (int argc, char ** argv)
{

  // Setting up input arguments
  std::string path = "../../data/ply/";
  std::string training_dir = "trained_models/";
  int NN = 1;

  pcl::console::parse_argument (argc, argv, "-models_dir", path);
  pcl::console::parse_argument (argc, argv, "-training_dir", training_dir);
  pcl::console::parse_argument (argc, argv, "-nn", NN);

  // Sorting out the training bit
  std::cout << "Setting up mesh source" << std::endl;
  boost::shared_ptr<pcl::rec_3d_framework::MeshSource<PointT> > mesh_source (new pcl::rec_3d_framework::MeshSource<PointT>);
  mesh_source->setPath (path);
  mesh_source->setResolution (150);
  mesh_source->setTesselationLevel (1);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (1.f);
  mesh_source->generate (training_dir);

  // Probably all of this will be removed...
  boost::shared_ptr<pcl::rec_3d_framework::Source<PointT> > cast_source;
  cast_source = boost::static_pointer_cast<pcl::rec_3d_framework::MeshSource<PointT> > (mesh_source);

  //boost::shared_ptr<pcl::rec_3d_framework::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, PointT, VFH_T> > cvfh_estimator;

  boost::shared_ptr<pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal> > normal_estimator;
  normal_estimator.reset (new pcl::rec_3d_framework::PreProcessorAndNormalEstimator<PointT, pcl::Normal>);
  normal_estimator->setCMR (true);
  normal_estimator->setDoVoxelGrid (true);
  normal_estimator->setRemoveOutliers (true);
  normal_estimator->setFactorsForCMR (3, 7);



  // setting up the CRH estimator - mainly for the purposes of training the model...
  std::cout << "Setting up CRH estimator" << std::endl;

  pcl::rec_3d_framework::CRHEstimation<PointT, VFH_T> crh_internal;
  crh_internal.setNormalEstimator(normal_estimator);

  //pcl::rec_3d_framework::GlobalEstimator<PointT, VFH_T> glob_est;
  
  boost::shared_ptr<pcl::rec_3d_framework::CRHEstimation<PointT, VFH_T> > crh_cast_estimator = 
    boost::make_shared<pcl::rec_3d_framework::CRHEstimation<PointT, VFH_T> >(crh_internal);

  boost::shared_ptr<pcl::rec_3d_framework::CRHEstimation<PointT, VFH_T> > crh_cast_to_global = 
    boost::make_shared<pcl::rec_3d_framework::CRHEstimation<PointT, VFH_T> >(crh_internal);


    //crh_cast_estimator->setFeatureEstimator(crh_cast_estimator);
    //boost::dynamic_pointer_cast< pcl::rec_3d_framework::GlobalEstimator<PointT, VFH_T> > (crh_internal);


  pcl::rec_3d_framework::GlobalNNCRHRecognizer<Metrics::HistIntersectionUnionDistance, PointT, VFH_T> crh_estimator;
  std::string descriptor = "crh";
  crh_estimator.setDataSource (cast_source);
  crh_estimator.setTrainingDir (training_dir);
  crh_estimator.setDescriptorName (descriptor);
  crh_estimator.setFeatureEstimator (crh_cast_estimator);
  crh_estimator.setNN (NN);
  crh_estimator.initialize (true);


  //boost::shared_ptr<pcl::rec_3d_framework::OURCVFHEstimator<PointT, VFH_T> > cast_estimator;
  //cast_estimator = boost::dynamic_pointer_cast<pcl::rec_3d_framework::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, PointT, VFH_T> > (ourcvfh_estimator);
  std::cout << "Setting up cvfh estimator" << std::endl;

  pcl::rec_3d_framework::OURCVFHEstimator<PointT, VFH_T> ourcvfh_estimator;
  ourcvfh_estimator.setNormalEstimator(normal_estimator);
  boost::shared_ptr<pcl::rec_3d_framework::OURCVFHEstimator<PointT, VFH_T> > cast_estimator = 
    boost::make_shared<pcl::rec_3d_framework::OURCVFHEstimator<PointT, VFH_T> >(ourcvfh_estimator);

  // // this is global_nn_classifer.h(pp)
  // pcl::rec_3d_framework::GlobalNNPipeline<Metrics::HistIntersectionUnionDistance, PointT, VFH_T> global;
  pcl::rec_3d_framework::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, PointT, VFH_T> cvfh_estimator;
  //cvfh_estimator.reset (new pcl::rec_3d_framework::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, PointT, VFH_T>);
  cvfh_estimator.setComputeScale(true);
  descriptor = "cvfh";
  cvfh_estimator.setDataSource (cast_source);
  cvfh_estimator.setTrainingDir (training_dir);
  cvfh_estimator.setDescriptorName (descriptor);
  cvfh_estimator.setFeatureEstimator (cast_estimator);
  cvfh_estimator.setNN (NN);
  cvfh_estimator.initialize (false);

  // visualiser
  pcl::visualization::PCLVisualizer vis ("kinect");

  // setting up some bits for the classification step    
  size_t previous_categories_size = 0;
  float Z_DIST_ = 1.25f;
  float text_scale = 0.015f;

  std::cout << "Loading point cloud" << std::endl;

  // load point cloud
  pcl::PointCloud<PointT>::Ptr xyz_points (new pcl::PointCloud<PointT>);
  pcl::io::loadPCDFile<PointT> ("../../data/scenes/scene2.pcd", *xyz_points);
  std::cout << "Size is " << xyz_points->size() << std::endl;
  
//  std::vector< int > idx;
//  pcl::removeNaNFromPointCloud (*xyz_points, *xyz_points, idx);
  vis.addPointCloud(xyz_points, "points");


  // estimating normals
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  ne.setInputCloud (xyz_points);
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  ne.setSearchMethod (tree);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);

  //Step 1 -> Segment
  std::pair<std::vector <pcl::PointCloud<PointT>::Ptr>, 
          std::vector <pcl::PointIndices> > cluster_out;
  cluster_out = myPrismSegmenter(xyz_points, cloud_normals);

  std::vector<pcl::PointCloud<PointT>::Ptr> clusters = cluster_out.first;
  std::vector <pcl::PointIndices> indices = cluster_out.second;

  std::cout << ">> Done the segmentation - found " << clusters.size() << " clusters " << std::endl;

  float dist_ = 0.03f;  // offset to position the 3d text

  // classifying each segment in turn
  for (size_t i = 0; i < clusters.size (); i++)
  {

    std::cout << std::endl;
    std::cout << "Segmented region number " << i << std::endl;
    std::cout << "*****************************" << std::endl;

    std::stringstream cluster_name;
    cluster_name << "cluster_" << i;
    pcl::visualization::PointCloudColorHandlerRandom<PointT> random_handler (clusters[i]);
    vis.addPointCloud<PointT> (clusters[i], random_handler, cluster_name.str ());
//    std::cout << ">> Cluster name is " << cluster_name.str() << std::endl;

    cvfh_estimator.setInputCloud (xyz_points);
    cvfh_estimator.setIndices (indices[i].indices);
    cvfh_estimator.recognize ();

    std::vector < std::string > categories;
    std::vector<float> conf;

    //cvfh_estimator.getCategory (categories);  // getting the categories after classifiercaiont. As we have set just NN=1, then there should only be one category
    //cvfh_estimator.getConfidence (conf);
    boost::shared_ptr<std::vector<ModelT> > models;
    boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > > poses;

    models = cvfh_estimator.getModels();
    poses = cvfh_estimator.getTransforms();

    std::cout << "Found " << models->size() << " matches" << std::endl;

    for (size_t kk = 0; kk < models->size (); kk++)
    {
      std::cout << "Match [" << kk << "] : " << models->at(kk).id_ << std::endl;
    }
    

    //cvfh_estimator.

    // std::string category = categories[0];
    // Eigen::Vector4f centroid;
    // pcl::compute3DCentroid (*xyz_points, indices[i].indices, centroid);

    // std::cout << ">> There are  " << categories.size() << " categories " << std::endl;
    // for (size_t kk = 0; kk < categories.size (); kk++)
    // {

    //   // deciding where to add the text string...
    //   // PointT pos;
    //   // pos.x = centroid[0] + normal_plane_[0] * static_cast<float> (kk + 1) * dist_;
    //   // pos.y = centroid[1] + normal_plane_[1] * static_cast<float> (kk + 1) * dist_;
    //   // pos.z = centroid[2] + normal_plane_[2] * static_cast<float> (kk + 1) * dist_;

    //   std::ostringstream prob_str;
    //   prob_str.precision (1);
    //   prob_str << categories[kk] << " [" << conf[kk] << "]";

    //   std::stringstream cluster_text;
    //   cluster_text << "cluster_" << previous_categories_size << "_text";
    //   std::cout << "Found " << prob_str.str () << std::endl;
    //   previous_categories_size++;
    // }
  }


  while (!vis.wasStopped ())
  {
    vis.spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}