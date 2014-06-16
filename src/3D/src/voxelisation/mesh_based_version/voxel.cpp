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
#include <vector>

//using namespace cvmlcpp;
using std::endl;
using std::cout;


template <typename T>
bool readOBJ(cvmlcpp::Geometry<T> &geometry, const std::string filename)
{
	
	// am now doing some kind of loading in from an .obj file... with some kind of custom loader?
    std::ifstream file;
    file.open(filename);
    if (!file.good())
    {
		cout << "Could not read obj file" << endl;
		return(0);
	}
	
    std::string line;
	
	size_t f1, f2, f3;
	size_t temp_idx;
	std::vector<size_t> point_keys;

    while(!file.eof()) //while we are still in the file
    {
    	getline(file, line);
    	std::istringstream iss( line );
    	
    	char nextidx;
		iss >> nextidx;
		
    	switch (nextidx)
    	{
    		case 'v':
	    		float v1, v2, v3;
    			iss >> v1 >> v2 >> v3;
    			temp_idx = geometry.addPoint(v1, v2, v3);
    			point_keys.push_back(temp_idx);
    			break;
    		case 'f':
	    		iss >> f1 >> f2 >> f3;
	    		assert(f1!=f2);
	    		assert(f2!=f3);
	    		assert(f1!=f3);
	    		if (point_keys.at(f1-1)!=point_keys.at(f2-1) && 
	    			point_keys.at(f1-1)!=point_keys.at(f3-1) &&
	    			point_keys.at(f3-1)!=point_keys.at(f2-1))
	    		{
		    		geometry.addFacet(point_keys.at(f1-1), point_keys.at(f2-1), point_keys.at(f3-1));
		    	}
    			break;
    		default:
    			cout << "Skipping\n";
    	}
    	
    }
    
    cout << "Read " << geometry.nrPoints() << " points and " << geometry.nrFacets() << " faces" << endl;

	return(1);
}


int main(int argc, char **argv)
{
//	std::string filename = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/centred/f7281caf8ed6d597b50d0c6a0c254040.obj";
	std::string filename = "/Users/Michael/Desktop/temp_chair.obj";

	cvmlcpp::Matrix<int, 3u> voxels;
	cvmlcpp::Geometry<float> geometry;
	

	if (!readOBJ(geometry, filename))
	{
		cout << "Could not read file" << endl;
		return(1);
	}

	if (!cvmlcpp::voxelize(geometry, voxels, 0.01))
	{
		cout << "Could not voxelise" << endl;
		return(1);
	}

	// finding size of the matrix
	std::tr1::array<std::size_t, D>::const_iterator extents = voxels.extents();
	size_t X =  extents[2];
	size_t Y =  extents[1];
	size_t Z =  extents[0];

	// outputting the matrix to disk somehow...
	cout << extents[0] << " " << extents[1] << " " << extents[2] << endl;
		// printing voxels to disk
	for ( cvmlcpp::Matrix<int, 3u>::iterator it = voxels.begin(); it != voxels.end(); ++it )
	{
		cout << *it << ", ";
	}
	cout << endl;


	return 0;
}
