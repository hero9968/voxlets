/* simple script to convert a text file representation of a voxel grid to a vdb version

 Takes as input the path to a text file of the format:

    size_x size_y size_z
    lin_idx1
    lin_idx2
    lin_idx3
    ...

where each lin_idx represents the 1-indexed linear index of an occupied voxel in the grid

Converts this to openvdb format and saves to the second argument

Used a lot of code from:
http://www.openvdb.org/documentation/doxygen/codeExamples.html
*/

#include <openvdb/openvdb.h>
#include <iostream>
#include <fstream>

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Require one arguments!" << endl;
        cerr << "Usage: " << endl;
        cerr << "./vdb2txt vdb_file_int.vdb > text_file_out.txt " << endl;
        return 0;
    }

    // Initialize the OpenVDB library.  This must be called at least
    // once per program and may safely be called multiple times.
    openvdb::initialize();

    // Create an empty floating-point grid with background value 0. 
    //openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    // Get an accessor for coordinate-based access to voxels.
    //openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    // Create a VDB file object.
    openvdb::io::File file(argv[1]);
    // Open the file.  This reads the file header, but not any grids.
    file.open();
    // Loop over all grids in the file and retrieve a shared pointer
    // to the one named "LevelSetSphere".  (This can also be done
    // more simply by calling file.readGrid("LevelSetSphere").)
    openvdb::GridBase::Ptr baseGrid = file.readGrid("voxelgrid");
    file.close();

    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);


    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter)
    {
        std::stringstream ss;
        ss << iter.getCoord();
        std::string str = ss.str();
        str.erase(std::remove(str.begin(), str.end(), '['), str.end());
        str.erase(std::remove(str.begin(), str.end(), ']'), str.end()); 
        cout << str << endl;
    }

}
