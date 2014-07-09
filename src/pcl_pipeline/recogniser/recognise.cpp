#include <iostream>

#include <pcl/apps/3d_rec_framework/pipeline/global_nn_recognizer_cvfh.h>

using std::cout;
using std::endl;

int main()
{

	cout << "In main" << endl;
	pcl::rec_3d_framework::GlobalNNCVFHRecognizer<flann::L1, pcl::PointXYZ> cvfh_recogniser;
	cvfh_recogniser.setICPIterations(10);

}