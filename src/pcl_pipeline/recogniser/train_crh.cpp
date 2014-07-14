// copied from global_nn_recognizer_crh
#include <iostream>
#include <boost/make_shared.hpp>
#include <pcl/pcl_macros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/apps/3d_rec_framework/pc_source/source.h>
#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_crh.h>
#include <pcl/apps/3d_rec_framework/pc_source/mesh_source.h>
#include <pcl/apps/3d_rec_framework/utils/metrics.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


typedef pcl::PointXYZ PointInT;
typedef pcl::VFHSignature308 FeatureT;
typedef pcl::PointCloud<PointInT>::Ptr PointInTPtr;
typedef pcl::rec_3d_framework::Model<PointInT> ModelT;

void train_models (bool force_retrain, boost::shared_ptr<pcl::rec_3d_framework::Source<PointInT> > source_);

std::string training_dir_ = "trained_models/";
std::string path = "../../data/ply/";
std::string descr_name_ = "crh";

int main()
{
  
  std::cout << "Setting up mesh source" << std::endl;
  boost::shared_ptr<pcl::rec_3d_framework::MeshSource<PointInT> > mesh_source (new pcl::rec_3d_framework::MeshSource<PointInT>);
  mesh_source->setPath (path);
  mesh_source->setResolution (150);
  mesh_source->setTesselationLevel (1);
  mesh_source->setViewAngle (57.f);
  mesh_source->setRadiusSphere (1.5f);
  mesh_source->setModelScale (1.f);
  mesh_source->generate (training_dir_);

  // Probably all of this will be removed...
  boost::shared_ptr<pcl::rec_3d_framework::Source<PointInT> > cast_source;
  cast_source = boost::static_pointer_cast<pcl::rec_3d_framework::MeshSource<PointInT> > (mesh_source);

  std::cout << "Doing the model training" << std::endl;
  train_models(false, cast_source);

}


void train_models (bool force_retrain, boost::shared_ptr<pcl::rec_3d_framework::Source<PointInT> > source_)
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
        PointInTPtr processed (new pcl::PointCloud<PointInT>);
        PointInTPtr view = models->at (i).views_->at (v);

        std::cout << "Size of view is " << view->points.size() << std::endl;

        //pro view, compute signatures and CRH
        std::vector<pcl::PointCloud<FeatureT>, Eigen::aligned_allocator<pcl::PointCloud<FeatureT> > > signatures;
        std::vector < Eigen::Vector3f > centroids;
        crh_estimator_->estimate (view, processed, signatures, centroids);

        std::string path = source_->getModelDescriptorDir (models->at (i), training_dir_, descr_name_);

        bf::path desc_dir = path;
        if (!bf::exists (desc_dir))
          bf::create_directory (desc_dir);

        std::stringstream path_view;
        path_view << path << "/view_" << v << ".pcd";
        pcl::io::savePCDFileBinary (path_view.str (), *processed);

        std::stringstream path_pose;
        path_pose << path << "/pose_" << v << ".txt";
        pcl::rec_3d_framework::PersistenceUtils::writeMatrixToFile (path_pose.str (), models->at (i).poses_->at (v));

        std::stringstream path_entropy;
        path_entropy << path << "/entropy_" << v << ".txt";
        pcl::rec_3d_framework::PersistenceUtils::writeFloatToFile (path_entropy.str (), models->at (i).self_occlusions_->at (v));

        std::vector<CRHPointCloud::Ptr> crh_histograms;
        crh_estimator_->getCRHHistograms (crh_histograms);

        //save signatures and centroids to disk
        for (size_t j = 0; j < signatures.size (); j++)
        {
          std::stringstream path_centroid;
          path_centroid << path << "/centroid_" << v << "_" << j << ".txt";
          Eigen::Vector3f centroid (centroids[j][0], centroids[j][1], centroids[j][2]);
          pcl::rec_3d_framework::PersistenceUtils::writeCentroidToFile (path_centroid.str (), centroid);

          std::stringstream path_descriptor;
          path_descriptor << path << "/descriptor_" << v << "_" << j << ".pcd";
          pcl::io::savePCDFileBinary (path_descriptor.str (), signatures[j]);

          std::stringstream path_roll;
          path_roll << path << "/crh_" << v << "_" << j << ".pcd";
          pcl::io::savePCDFileBinary (path_roll.str (), *crh_histograms[j]);
        }
      }

    }
    else
    {
      //else skip model
      std::cout << "The model has already been trained..." << std::endl;
    }
  }

}