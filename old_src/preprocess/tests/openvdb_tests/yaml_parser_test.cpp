#include <iostream>
#include <yaml-cpp/yaml.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Composite.h>
//#include <string>

using std::cout;
using std::cerr;
using std::endl;
std::string fullpath = "/Users/Michael/projects/shape_sharing/data/3D/basis_models/voxelised_vdb/";

// helper function to convert nodes containing R and T 
// components into openvdb transformation matrix format
openvdb::Mat4R extract_matrix(const YAML::Node R, const YAML::Node T)
{
	double zero = 0;
	double one = 1;
	openvdb::Mat4R trans( R[0][0].as<double>(), R[0][1].as<double>(), R[0][2].as<double>(), T[0].as<double>(),
							R[1][0].as<double>(), R[1][1].as<double>(), R[1][2].as<double>(), T[1].as<double>(),
							R[2][0].as<double>(), R[2][1].as<double>(), R[2][2].as<double>(), T[2].as<double>(),
							zero, zero, zero, one/100);
	trans = trans.transpose();
	return trans;
}


int main()
{
	openvdb::initialize();
	YAML::Node transforms = YAML::LoadFile("../../3D/src/test.yaml");

	// the final output grid...
	openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create();
	openvdb::FloatGrid::Ptr grid;

	// loop over each object to be loaded in
	for (size_t i = 0; i < transforms.size(); ++i)
	{
		cerr << "Model number " << i << ": " << transforms[i]["name"] << endl;
		
		// load in the vdb voxel grid for this model
		std::string fullstring = fullpath + transforms[i]["name"].as<std::string>() + ".vdb";
		cerr << "Loading " << fullstring << endl;
		openvdb::io::File file(fullstring);
		file.open();
		openvdb::GridBase::Ptr baseGrid = file.readGrid("voxelgrid");
		file.close();

		// cast the baseGrid to a double grid
		grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

		for (size_t j = 0; j < transforms[i]["transform"].size(); ++j)
		{
			cerr << "Transforming " << endl;

			openvdb::Mat4R this_transform = extract_matrix(transforms[i]["transform"][j]["R"], transforms[i]["transform"][j]["T"]);
			cerr << this_transform << endl;
			openvdb::FloatGrid::Ptr gridCopy = grid->deepCopy();
			openvdb::FloatGrid::Ptr targetGrid = openvdb::FloatGrid::create();

			openvdb::tools::GridTransformer transformer(this_transform);

			// Resample using nearest-neighbor interpolation.
			transformer.transformGrid<openvdb::tools::PointSampler, openvdb::FloatGrid>(
			    *gridCopy, *targetGrid);

			// add into main grid (compositinbg modifies the frit grid and leaves the second empty)
			openvdb::tools::compSum(*outputGrid, *targetGrid);
			cerr << "Done transformation " << endl;
		}

	}

	// print the entire output grid
	for (openvdb::FloatGrid::ValueOnCIter iter = outputGrid->cbeginValueOn(); iter; ++iter)
	{
		std::stringstream ss;
		ss << iter.getCoord();
		std::string str = ss.str();
		str.erase(std::remove(str.begin(), str.end(), '['), str.end());
		str.erase(std::remove(str.begin(), str.end(), ']'), str.end());	
		cout << str << ", " << iter.getValue() << endl;
	}

	// saving the grid to file
	openvdb::io::File fileout("outputgrid.vdb");
	// Add the grid pointer to a container.
	openvdb::GridPtrVec grids;
	grids.push_back(outputGrid);
	// Write out the contents of the container.
	fileout.write(grids);
	fileout.close();	

}