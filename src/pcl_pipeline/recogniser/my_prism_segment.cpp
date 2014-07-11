#include <string.h>
#include <time.h>

#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
//#include <pcl/search/organized.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/surface/convex_hull.h>


struct plane
{
  pcl::PointIndices::Ptr inliers;
  pcl::ModelCoefficients::Ptr coefficients;
  float score;
};


// my shitty function for doing prism segmentation
std::pair<std::vector <pcl::PointCloud<pcl::PointXYZ>::Ptr>, 
          std::vector <pcl::PointIndices> >
myPrismSegmenter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudl, 
              pcl::PointCloud<pcl::Normal>::Ptr cloud_normals)
{    

  // convert cloud to labelled cloud
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);
  // for (size_t i = 0; i < cloudl->size(); ++i)
  // {
  //   pcl::PointXYZL temp_point(cloudl->at(i).x, cloudl->at(i).y, cloudl->at(i).z, 0);
  //   cloud->push_back(temp_point);
  // }
  pcl::copyPointCloud(*cloudl, *cloud);

  for (size_t i = 0; i < cloud->size(); ++i)
    cloud->at(i).label = i;


  printf("There are %lu points in cloud\n", cloud->points.size());

  clock_t time1, time2;
  double t_diff;
  time1 = clock();

  // create required objects
  //pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);
  //pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  // extracting parameters
  int minsize = 500;
  int maxsize = 1000000;
  int num_neighbours = 50;
  double smoothness_threshold = 0.1222;
  double curvature_threshold = 1;

  // creating kdtree
  pcl::search::Search<pcl::PointXYZL>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZL> > (new pcl::search::KdTree<pcl::PointXYZL>);
  //pcl::search::Search<pcl::PointXYZL>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZL> > (new pcl::search::OrganizedNeighbor<pcl::PointXYZL>);

//  printf("Minx size %d and max size %d\n", *minsize, *maxsize);
//  printf("Neighbours %d\n", *num_neighbours);
//  printf("Smooth %f and curve %f\n", *smoothness_threshold, *curvature_threshold);

  t_diff = (double)(clock() - time1)/CLOCKS_PER_SEC;
  printf("...Mex input done: %f\n", t_diff);
  time1 = clock();
  t_diff = (double)(clock() - time1)/CLOCKS_PER_SEC;
  printf("...Joke test: %f\n", t_diff);
  time1 = clock();
  /*****************************************/
  /* Detecting and removing large planes   */
  /*****************************************/

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZL> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZL> ());
  pcl::PointCloud<pcl::PointXYZL>::Ptr remaining_points (new pcl::PointCloud<pcl::PointXYZL>);

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (200);
  seg.setDistanceThreshold (0.01);

  // create a vectpr of plane structures to store the inliers and the plane parameters
  std::vector<plane> plane_vect;

  int i=0, nr_points = (int) cloud->points.size ();
  *remaining_points = *cloud;
  int counter = 0;
  while (remaining_points->points.size () > 0.8 * nr_points && counter < 5)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (remaining_points);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      printf("Could not estimate a planar model for the given dataset.\n");
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZL> extract;
    extract.setInputCloud (remaining_points);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    printf("PointCloud representing the planar component: %lu data points.\n", cloud_plane->points.size ());

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*remaining_points);

    printf("Size of cloud: %lu \n", cloud->points.size ());
    printf("Size of remaining_points: %lu \n", remaining_points->points.size ());

    // for this plane store the indices of the *original points*
    pcl::PointIndices::Ptr plane_inliers (new pcl::PointIndices);
    for (size_t j = 0; j < inliers->indices.size(); ++j)
    {
      plane_inliers->indices.push_back(cloud_plane->points.at(j).label);
     // printf("Adding %d\n", (int)cloud_plane->points.at(j).label);
    }

    // adding
    plane this_plane;
    this_plane.coefficients = coefficients;
    this_plane.inliers = plane_inliers;
    plane_vect.push_back(this_plane);
    
    counter++;
  }

  t_diff = (double)(clock() - time1)/CLOCKS_PER_SEC;
  printf("...Plane detection done: %f\n", t_diff);
  printf("Found %lu planes", plane_vect.size());
  time1 = clock();

  /*****************************************/
  /* Choosing most upright plane  */
  /*****************************************/

  // here should choose the plane which really looks like an upright plane, and do the prism extraction on it...
  float best_score = 0;
  size_t best_plane = 0;
  for (size_t i = 0; i < plane_vect.size(); ++i)
  {
    // the score is simply a dot product between the 'best_vector' and the up direction of the plane
    float best_vector[3] = {0, -0.848, -0.530};
    plane_vect.at(i).score = 0;

    for (size_t j = 0; j < 3; ++j)
    {
      plane_vect.at(i).score += plane_vect.at(i).coefficients->values[j] * best_vector[j];
      printf("%f\t", plane_vect.at(i).coefficients->values[j]);
    } 

    printf("\nThis score is %f\n", plane_vect.at(i).score);

    // updating the best score
    if (plane_vect.at(i).score > best_score)
    {
      best_score = plane_vect.at(i).score;
      best_plane = i;
    }
  }
  printf("The best plane is %zu with a score of %f\n", best_plane, best_score);

  t_diff = (double)(clock() - time1)/CLOCKS_PER_SEC;
  printf("...Choosing best plane done: %f\n", t_diff);
  time1 = clock();

  /*******************************************/
  // extracting points from convex hull of best plane
  /*******************************************/

  // for (size_t i = 0; i < 1000; ++i)
  //   std::cout << cloud->at(i).x << " " << cloud->at(i).y << " " << cloud->at(i).z << std::endl;


  // get the xyz points associated with the best plane  
  pcl::PointCloud<pcl::PointXYZL>::Ptr plane_inliers (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::ExtractIndices<pcl::PointXYZL> extract;
  extract.setInputCloud (cloud);  
  extract.setIndices (plane_vect.at(best_plane).inliers);
  extract.setNegative (false);
  extract.filter (*plane_inliers);

  //for (size_t i = 0; i < 1000; ++i)
    //std::cout << plane_vect.at(best_plane).inliers->indices[i] << std::endl;

//     std::cout << plane_inliers->at(i).x << " " << plane_inliers->at(i).y << " " << plane_inliers->at(i).z << std::endl;



  // Project the model inliers
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::ProjectInliers<pcl::PointXYZL> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  //proj.setIndices (inliers);
  proj.setInputCloud (plane_inliers);
  proj.setModelCoefficients (plane_vect.at(best_plane).coefficients);
  proj.filter (*cloud_projected);
  std::cerr << "PointCloud after projection has: "
            << cloud_projected->points.size () << " data points." << std::endl;

    // copying projected cloud to new cloud
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_projected2 (new pcl::PointCloud<pcl::PointXYZL>);  
  pcl::copyPointCloud(*cloud_projected, *cloud_projected2);     
//<pcl::PointXYZL, pcl::PointXYZ>
   //for (size_t i = 0; i < cloud_projected2->size(); ++i)
     //std::cout << cloud_projected2->at(i).x << " " << cloud_projected2->at(i).y << " " << cloud_projected2->at(i).z << std::endl;
  std::cerr << "After copying" << std::endl;

  // Computing the convex hull of the projected points
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZL>);
  pcl::ConvexHull<pcl::PointXYZL> chull;
  chull.setDimension (2);
  chull.setInputCloud (cloud_projected2);
  chull.reconstruct (*cloud_hull);

  std::cerr << "After convex  hull" << std::endl;

  // now extracting the prism of points relating to the best plane
  pcl::ExtractPolygonalPrismData<pcl::PointXYZL> prism;
  prism.setInputCloud (remaining_points);
  prism.setInputPlanarHull (cloud_hull);
  prism.setHeightLimits( -0.01, +0.5 );
  pcl::PointIndices::Ptr output (new pcl::PointIndices);
  prism.segment (*output);
  std::cerr << "After prism" << std::endl;


  pcl::PointCloud<pcl::PointXYZL>::Ptr tabletop_points (new pcl::PointCloud<pcl::PointXYZL>);
  extract.setInputCloud (remaining_points);  
  extract.setIndices (output);
  extract.setNegative (false);
  extract.filter (*tabletop_points);

  std::cerr << "After extraction" << std::endl;


  // extracting the equivalent normals
  pcl::PointCloud<pcl::Normal>::Ptr remaining_normals (new pcl::PointCloud<pcl::Normal>);
  for (size_t i = 0; i < tabletop_points->points.size(); ++i)
  {
    size_t this_idx = tabletop_points->at(i).label;
    remaining_normals->push_back(cloud_normals->at(this_idx));
  }

  t_diff = (double)(clock() - time1)/CLOCKS_PER_SEC;
  printf("...Chull done: %f\n", t_diff);
  time1 = clock();

  /*******************************************/
  // segmenting the remaining points
  /*******************************************/

  pcl::RegionGrowing<pcl::PointXYZL, pcl::Normal> reg;
  reg.setMinClusterSize (minsize);
  reg.setMaxClusterSize (maxsize);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (num_neighbours);
  reg.setInputCloud (tabletop_points);
  reg.setInputNormals (remaining_normals);
  reg.setSmoothnessThreshold (smoothness_threshold);
  reg.setCurvatureThreshold (curvature_threshold);

  std::vector <pcl::PointIndices> cluster_idxs;
  reg.extract (cluster_idxs);

  // extracting the 3D clusters
  std::vector <pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  std::vector <pcl::PointIndices> global_cluster_idxs;

  for (size_t i = 0; i < cluster_idxs.size(); ++i)
  {
    pcl::PointCloud<pcl::PointXYZL>::Ptr temp (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp2 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointIndices temp_idxs;

    for (size_t j = 0; j < cluster_idxs[i].indices.size(); ++j)
    {

      size_t this_point_idx_in_tabletop_points = cluster_idxs[i].indices[j];

      // convert the index to the index in the full cloud
      size_t this_point_idx = tabletop_points->at(this_point_idx_in_tabletop_points).label;
      //out_ptr[this_point_idx] = clust_idx;
      temp->push_back(cloud->at(this_point_idx));
      temp_idxs.indices.push_back(this_point_idx);

    }

    pcl::copyPointCloud(*temp, *temp2);
    clusters.push_back(temp2);
    global_cluster_idxs.push_back(temp_idxs);
  }

  return std::make_pair(clusters, global_cluster_idxs);


}