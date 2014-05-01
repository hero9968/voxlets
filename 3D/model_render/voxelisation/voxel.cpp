/*
Ok I'm going to try to voxelise a mesh

http://tech.unige.ch/cvmlcpp/source/doc/Voxelizer.html

	template <typename Tg, typename voxel_type>
	  bool voxelize(const Geometry<Tg> &geometry,
			matrix_type<voxel_type, 3> &voxels,
			const value_type voxelSize = 1.0,
			const std::size_t	 pad = 0u,
			const voxel_type inside = 1,
			const voxel_type outside = 0);


g++ voxel.cpp -lcvmlcpp

*/

#include <iostream>
#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>

//using namespace cvmlcpp;

int main(int argc, char **argv)
{
	bool voxelise_return = 0;
	bool read_stl_return = 0;

	cvmlcpp::Matrix<int, 3u> voxels;
	cvmlcpp::Geometry<float> geometry;

	if (!cvmlcpp::readSTL(geometry, "cube.stl"))
	{
		std::cout << "Could not read file" << std::endl;
		return(1);
	}

	if (!cvmlcpp::voxelize(geometry, voxels, 0.05))
	{
		std::cout << "Could not voxelise" << std::endl;
		return(1);
	}
	
	// printing voxels to disk
	for ( cvmlcpp::Matrix<int, 3u>::iterator it = voxels.begin(); it != voxels.end(); ++it )
	{
		std::cout << *it << ", ";
	}
	std::cout << std::endl;

	

	return 0;
}
