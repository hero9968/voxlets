./apps/3d_rec_framework/tools/apps/src/local_recognition_mian_dataset.cpp

... is the one which really has the detail!

BUT - local_recognition it turns out is designed for local features, which is then used with some kind of correspondence grouping. 

I think I really want one of the GLOBAL methods, as I can then segment my scene via whatever method, and use the global method to find the matches for each region...

GlobalNNCVFHRecognizer
could be the way to go. However, there are no examples of its use - I might have to work this out myself!

  <mfirman2:Michael> 0 [07-14 11:45] ~/builds/pcl_and_dep/pcl/apps ($)
  ! grep -rIl GlobalNNCVFHRecognizer .
  ./3d_rec_framework/include/pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h
  ./3d_rec_framework/include/pcl/apps/3d_rec_framework/pipeline/impl/global_nn_recognizer_cvfh.hpp
  ./3d_rec_framework/src/pipeline/global_nn_recognizer_cvfh.cpp

So far I seem to be using 

  #include <pcl/apps/3d_rec_framework/feature_wrapper/global/cvfh_estimator.h>

which I'm not sure does pose estimation.

I'm still coding in recognise_global, using CVFHEstimation.

GlobalNNCVFHRecognizer

Although the GlobalNNCVFHRecognizer uses hypothesis verificatio, I'm fairly sure it is only supposed to be used for a single region of the scan at a time.

It seems to be suggesting that it wants to use the camera roll histogram (CRH) - will look into using this next!

This seems to be built into the cvfh code. Seems to require a roll pose to be found from the 


global_nn_recognizer_crh.hpp uses crh_estimator like so:

  crh_estimator_->estimate (view, processed, signatures, centroids);


Written my own train_crh.cpp now.
Seems to almost compile, but something is fucking up.
Think I need to try a minimum working example to get this to work...





 GlobalNNCVFHRecognizer ()
        {
          ICP_iterations_ = 0;
          noisify_ = false;
          compute_scale_ = false;
          use_single_categories_ = false;
        }

        ~GlobalNNCVFHRecognizer ()
        {
        }

        void
        getDescriptorDistances (std::vector<float> & ds)
        {
          ds = descriptor_distances_;
        }

        void
        setComputeScale (bool d)
        {
          compute_scale_ = d;
        }

        void
        setCategoriesToUseForRecognition (std::vector<std::string> & cats_to_use)
        {
          categories_to_be_searched_.clear ();
          categories_to_be_searched_ = cats_to_use;
        }

        void setUseSingleCategories(bool b) {
          use_single_categories_ = b;
        }

        void
        setNoise (float n)
        {
          noisify_ = true;
          noise_ = n;
        }

        void
        setNN (int nn)
        {
          NN_ = nn;
        }

        void
        setICPIterations (int it)
        {
          ICP_iterations_ = it;
        }

        /**
         * \brief Initializes the FLANN structure from the provided source
         */

        void
        initialize (bool force_retrain = false);

        /**
         * \brief Sets the model data source_
         */
        void
        setDataSource (typename boost::shared_ptr<Source<PointInT> > & source)
        {
          source_ = source;
        }

        /**
         * \brief Sets the model data source_
         */

        void
        setFeatureEstimator (typename boost::shared_ptr<OURCVFHEstimator<PointInT, FeatureT> > & feat)
        {
          micvfh_estimator_ = feat;
        }

        /**
         * \brief Sets the HV algorithm
         */
        void
        setHVAlgorithm (typename boost::shared_ptr<HypothesisVerification<PointInT, PointInT> > & alg)
        {
          hv_algorithm_ = alg;
        }

        void
        setIndices (std::vector<int> & indices)
        {
          indices_ = indices;
        }

        /**
         * \brief Sets the input cloud to be classified
         */
        void
        setInputCloud (const PointInTPtr & cloud)
        {
          input_ = cloud;
        }

        void
        setDescriptorName (std::string & name)
        {
          descr_name_ = name;
        }

        void
        setTrainingDir (std::string & dir)
        {
          training_dir_ = dir;
        }

        /**
         * \brief Performs recognition on the input cloud
         */

        void
        recognize ();

        boost::shared_ptr<std::vector<ModelT> >
        getModels ()
        {
          return models_;
        }

        boost::shared_ptr<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > >
        getTransforms ()
        {
          return transforms_;
        }

        void
        setUseCache (bool u)
        {
          use_cache_ = u;
        }