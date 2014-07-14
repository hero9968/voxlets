/*
 * vfh_estimator.h
 *
 *  Created on: Mar 22, 2012
 *      Author: aitor
 */

#ifndef REC_FRAMEWORK_CRH_ESTIMATOR_H_
#define REC_FRAMEWORK_CRH_ESTIMATOR_H_

#include <pcl/apps/3d_rec_framework/feature_wrapper/global/global_estimator.h>
#include <pcl/apps/3d_rec_framework/feature_wrapper/normal_estimator.h>
#include <pcl/features/crh.h>
#include <pcl/common/centroid.h>


using std::endl;
using std::cerr;

namespace pcl
{
  namespace rec_3d_framework
  {
    template<typename PointInT, typename FeatureT>
    class CRHEstimation : public GlobalEstimator<PointInT, FeatureT>
    {

      typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
      using GlobalEstimator<PointInT, FeatureT>::normal_estimator_;
      using GlobalEstimator<PointInT, FeatureT>::normals_;

      //typename boost::shared_ptr<GlobalEstimator<PointInT, FeatureT> > feature_estimator_;
      typedef pcl::PointCloud<pcl::Histogram<90> > CRHPointCloud;
      std::vector< CRHPointCloud::Ptr > crh_histograms_;

    public:

      CRHEstimation ()
      {

      }

      //void
      //setFeatureEstimator(typename boost::shared_ptr<GlobalEstimator<PointInT, FeatureT> > & feature_estimator) {
      //  feature_estimator_ = feature_estimator;
      //}

      void
      estimate (PointInTPtr & in, PointInTPtr & processed,
                std::vector<pcl::PointCloud<FeatureT>, Eigen::aligned_allocator<pcl::PointCloud<FeatureT> > > & signatures,
                std::vector<Eigen::Vector3f> & centroids)
      {

        if (processed->size() == 0)
        {
          PCL_ERROR("No points in the input cloud...\n");
          throw;
        }

        // classicyl, this is where the estimation goes on... instead, I will be rewriting this code...
        //feature_estimator_->estimate(in, processed, signatures, centroids);
        Eigen::Vector4f temp_centroid;
        pcl::compute3DCentroid (*in, temp_centroid);

        Eigen::Vector3f temp_centroid_3f(temp_centroid[0], temp_centroid[1], temp_centroid[2]);
        centroids.push_back(temp_centroid_3f);

        if(!computedNormals()) {
          normals_.reset(new pcl::PointCloud<pcl::Normal>);
          normal_estimator_->estimate (in, processed, normals_);
        } else {
          this->getNormals(normals_);
        }

        crh_histograms_.resize(centroids.size());

        typedef typename pcl::CRHEstimation<PointInT, pcl::Normal, pcl::Histogram<90> > CRHEstimation;
        CRHEstimation crh;
        crh.setInputCloud(processed);
        crh.setInputNormals(normals_);

        cerr << "Input cloud has size " << processed->size() << endl;
        cerr << "Input cloud has size " << processed->points.size() << endl;
        cerr << "CRH histograms has size " << crh_histograms_.size() << endl;
        cerr << "Centroids has size " << centroids.size() << endl;

        for (size_t idx = 0; idx < centroids.size (); idx++)
        {
          cerr << "Centroid number " << idx << endl;
          Eigen::Vector4f centroid4f(centroids[idx][0],centroids[idx][1],centroids[idx][2],0);
          cerr << "Constructed new " << endl;
          crh.setCentroid(centroid4f);
          cerr << "Set centroid " << endl;
          crh_histograms_[idx].reset(new CRHPointCloud());
          cerr << "Done reset " << endl;
          crh.compute (*crh_histograms_[idx]);
          cerr << "Computed signature " << idx << endl;
          cerr << "...it has size " << crh_histograms_[idx]->size() << endl;
        }

      }

      void getCRHHistograms(std::vector< CRHPointCloud::Ptr > & crh_histograms) {
        crh_histograms = crh_histograms_;
      }

      bool
      computedNormals ()
      {
        return true;
      }
    };
  }
}

#endif /* REC_FRAMEWORK_CVFH_ESTIMATOR_H_ */
