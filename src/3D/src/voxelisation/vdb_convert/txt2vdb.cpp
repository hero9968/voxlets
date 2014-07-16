// simple script to convert a text file representation of a voxel grid to a vdb version

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
        cerr << "Require one argument!" << endl;
        return 0;
    }

    // Initialize the OpenVDB library.  This must be called at least
    // once per program and may safely be called multiple times.
    openvdb::initialize();

    // Create an empty floating-point grid with background value 0. 
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    //std::cout << "Testing random access:" << std::endl;
    // Get an accessor for coordinate-based access to voxels.
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    // open the text file
    std::ifstream infile;
    //cout << "Opening " << argv[1] << endl;
    infile.open(argv[1]);
    if (!infile.is_open())
    {
        cerr << "Failed to open file!" << endl;
        return 0;
    }

    // reading the header - TODO interpret these values
    std::string header;
    infile >> header >> header >> header;
    //cout << "header is " << header;
    
    size_t dim_x = 100;
    size_t dim_y = 100;
    size_t dim_z = 100;
    size_t lin_idx;
    // 
    while(!infile.eof())
    {
        infile >> lin_idx;
        lin_idx--; // file is saved with matlab indexing - here convert to 0-indexing
        size_t z = lin_idx / (dim_x * dim_y);
        size_t y = (lin_idx - z * dim_x * dim_y) / (dim_x);
        size_t x = lin_idx - z * dim_x * dim_y - y * dim_x;
        //cout << "Lin idx = " << lin_idx << endl;
        std::cout << x << "," << y << "," << z << endl;        
        openvdb::Coord xyz(x, y, z);
        accessor.setValue(xyz, 1.0);

//        std::cout << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
    }

    infile.close();
        
    // Verify that the voxel value at (1000, -200000000, 30000000) is 1.
    
        
    // std::cout << "Testing sequential access:" << std::endl;
    // // Print all active ("on") voxels by means of an iterator.
    // for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
    //     std::cout << "Grid" << iter.getCoord() << " = " << *iter << std::endl;
    // }


}