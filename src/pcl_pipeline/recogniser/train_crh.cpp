// copied from global_nn_recognizer_crh
#include <iostream>
#include <boost/make_shared.hpp>
//#include <pcl/pcl_macros.h>
//#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/apps/3d_rec_framework/pc_source/source.h>

typedef pcl::PointXYZ PointT;
typedef pcl::Normal PointNT;
typedef pcl::VFHSignature308 FeatureT;
typedef pcl::PointCloud<PointT>::Ptr PointTPtr;
typedef pcl::rec_3d_framework::Model<PointT> ModelT;
//typedef pcl::PointCloud<pcl::Histogram<90> > MyCRHPointCloud;

void train_models (bool force_retrain, boost::shared_ptr<pcl::rec_3d_framework::Source<PointT> > source_);

void write_histogram (std::string, pcl::Histogram<90>);

std::string training_dir_ = "trained_models/";
std::string path = "../../data/ply/";
std::string descr_name_ = "crh";

#include <fstream>
#include <pcl/apps/3d_rec_framework/pc_source/mesh_source.h>
//#include <pcl/apps/3d_rec_framework/utils/metrics.h>
#include <pcl/filters/filter.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/console/parse.h>
#include <pcl/common/centroid.h>
#include <pcl/features/crh.h>
#include <pcl/features/normal_3d.h>


int main()
{
  
  std::cout << "Setting up mesh source" << std::endl;
  boost::shared_ptr<pcl::rec_3d_framework::MeshSource<PointT> > mesh_source (new pcl::rec_3d_framework::MeshSource<PointT>);
  mesh_source->setPath (path);
  mesh_source->setResolution (150);
  mesh_source->setTesselationLevel (1);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (1.f);
  mesh_source->generate (training_dir_);

  // Probably all of this will be removed...
  boost::shared_ptr<pcl::rec_3d_framework::Source<PointT> > cast_source;
  cast_source = boost::static_pointer_cast<pcl::rec_3d_framework::MeshSource<PointT> > (mesh_source);

  std::cout << "Doing the model training" << std::endl;
  train_models(false, cast_source);

}

void write_histogram(std::string savepath, pcl::Histogram<90> hist_in)
{
  std::ofstream fileout;
  fileout.open(savepath.c_str());
  fileout << "# .PCD v0.7 - Point Cloud Data file format" << std::endl;
  fileout << "VERSION 0.7" << std::endl;
  fileout << "FIELDS crh" << std::endl;
  fileout << "SIZE 4" << std::endl;
  fileout << "TYPE F" << std::endl;
  fileout << "COUNT 90" << std::endl;
  fileout << "WIDTH 1" << std::endl;
  fileout << "HEIGHT 1" << std::endl;
  fileout << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
  fileout << "POINTS 1" << std::endl;
  fileout << "DATA ascii" << std::endl;
  for (size_t i = 0; i < 90; ++i)
    fileout << hist_in << " ";
  fileout.close();
}


void train_models (bool force_retrain, boost::shared_ptr<pcl::rec_3d_framework::Source<PointT> > source_)
{

  //use the source to know what has to be trained and what not, checking if the descr_name directory exists
  //unless force_retrain is true, then train everything
  boost::shared_ptr < std::vector<ModelT> > models = source_->getModels ();
  std::cout << "Models size:" << models->size () << std::endl;

  if (force_retrain)
  {
    for (size_t i = 0; i < models->size (); i++)
    {
      source_->removeDescDirectory (models->at (i), training_dir_, descr_name_);
    }
  }

  for (size_t i = 0; i < models->size (); i++)
  {
    if (!source_->modelAlreadyTrained (models->at (i), training_dir_, descr_name_))
    {
      for (size_t v = 0; v < models->at (i).views_->size (); v++)
      {
        PointTPtr processed (new pcl::PointCloud<PointT>);
        PointTPtr view = models->at (i).views_->at (v);

        std::cout << "Size of view is " << view->points.size() << std::endl;

        //pro view, compute signatures and CRH (MF - I have changed these to not be vectors...)
        //crh_estimator_->estimate (view, processed, signatures, centroids);

        std::string path = source_->getModelDescriptorDir (models->at (i), training_dir_, descr_name_);

        bf::path desc_dir = path;
        if (!bf::exists (desc_dir))
          bf::create_directory (desc_dir);

        // std::stringstream path_view;
        // path_view << path << "/view_" << v << ".pcd";
        // pcl::io::savePCDFileBinary (path_view.str (), *processed);

        // std::stringstream path_pose;
        // path_pose << path << "/pose_" << v << ".txt";
        // pcl::rec_3d_framework::PersistenceUtils::writeMatrixToFile (path_pose.str (), models->at (i).poses_->at (v));

        // std::stringstream path_entropy;
        // path_entropy << path << "/entropy_" << v << ".txt";
        // pcl::rec_3d_framework::PersistenceUtils::writeFloatToFile (path_entropy.str (), models->at (i).self_occlusions_->at (v));

        pcl::PointCloud<pcl::Histogram<90> >::Ptr crh_histogram (new pcl::PointCloud<pcl::Histogram<90> >());
        //std::vector<pcl::PointCloud<FeatureT>, Eigen::aligned_allocator<pcl::PointCloud<FeatureT> > > signatures;
        Eigen::Vector4f centroid4;
        pcl::compute3DCentroid(*view, centroid4);
        Eigen::Vector3f centroid(centroid4[0], centroid4[1], centroid4[2]);

        // Create the normal estimation class, and pass the input dataset to it
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud (view);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        ne.setKSearch(50);  //setSearch (0.03);
        ne.compute (*cloud_normals);

        pcl::CRHEstimation<PointT, PointNT, pcl::Histogram<90> > crh_engine;
        crh_engine.setCentroid(centroid4);
        crh_engine.setInputCloud(view);
        crh_engine.setInputNormals(cloud_normals);
        crh_engine.compute(*crh_histogram);
        //crh_engine.compue()
        
        //save signature and centroids to disk
        size_t j = 0;
        
        std::stringstream path_centroid;
        path_centroid << path << "/centroid_" << v << "_" << j << ".txt";
        pcl::rec_3d_framework::PersistenceUtils::writeCentroidToFile (path_centroid.str (), centroid);

        //std::stringstream path_descriptor;
        //path_descriptor << path << "/descriptor_" << v << "_" << j << ".pcd";
        //pcl::io::savePCDFileBinary (path_descriptor.str (), signatures);

        std::stringstream path_roll;
        path_roll << path << "/crh_" << v << "_" << j << ".pcd";
        //std::vector<pcl::PointCloud<pcl::Histogram<90> >::Ptr> crh_histograms;
        //pcl::io::savePCDFileBinary (path_roll.str (), *crh_histograms[0]);
        //pcl::io::savePCDFileBinary (path_roll.str (), *crh_histograms[0]);
        write_histogram (path_roll.str(), crh_histogram->points[0]);
      }

    }
    else
    {
      //else skip model
      std::cout << "The model has already been trained..." << std::endl;
    }
  }

}
