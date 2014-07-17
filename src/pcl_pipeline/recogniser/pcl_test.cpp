//#include <iostream>
#include <pcl/io/pcd_io.h>

int main()
{
	pcl::Histogram<190> crh_histogram; //(new pcl::Histogram<190>());
	//pcl::PointCloud<pcl::PointXYZ >::Ptr crh_histogram (new pcl::PointCloud<pcl::PointXYZ >());
	pcl::io::savePCDFileASCII ("test_file", crh_histogram);
}