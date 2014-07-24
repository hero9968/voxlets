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
    if (argc < 3)
    {
        cerr << "Require two arguments!" << endl;
        cerr << "Usage: " << endl;
        cerr << "./txt2vdb text_file_in.txt vdb_file_out.vdb" << endl;
        return 0;
    }

    // Initialize the OpenVDB library.  This must be called at least
    // once per program and may safely be called multiple times.
    openvdb::initialize();

    // Create an empty floating-point grid with background value 0. 
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
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

    // reading the header
    size_t dim_x, dim_y, dim_z;
    infile >> dim_x >> dim_y >> dim_z;
    cerr << dim_x << " " << dim_y << " " << dim_z << endl;
    
    
    // reading each idx and adding to the voxel grid
    while(!infile.eof())
    {
        size_t lin_idx;
        infile >> lin_idx;
        lin_idx--; // file is saved with matlab indexing - here convert to 0-indexing

        size_t z = lin_idx / (dim_x * dim_y);
        size_t y = (lin_idx - z * dim_x * dim_y) / (dim_x);
        size_t x = lin_idx - z * dim_x * dim_y - y * dim_x;

        //cerr << "Lin idx = " << lin_idx << endl;
        //cerr << x << "," << y << "," << z << endl;
        openvdb::Coord xyz(x, y, z);
        accessor.setValue(xyz, 1.0);

        //cerr << "Grid" << xyz << " = " << accessor.getValue(xyz) << std::endl;
    }

    infile.close();

    // reading the transform from the file
    const openvdb::math::Transform &sourceXform = grid->transform();
    cerr << sourceXform << endl;

    // adding transform to the grid to transform into world coordinates
    /*
    openvdb::math::Mat4d transform_mat = openvdb::math::Mat4d::identity();
    transform_mat.preScale(openvdb::math::Vec3d(0.01,0.01,0.01));
    transform_mat.postTranslate(openvdb::math::Vec3d(-0.5,-0.5,-0.5));

    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(1.0);
    transform->postMult(transform_mat);
    grid->setTransform(transform);

    // displaying the applied transform to the user
    //cerr << "Original: " << sourceXform << endl; // this doesn' work once the transform has been changed! (seg fault)
    cerr << "Latest: " << grid->transform() << endl;
    cerr << "Voxside is " << grid->voxelSize() << endl;
    */

    // spitting out the transformed voxel coordinates, i.e. transformed into world coordinates
    //openvdb::FloatGrid::Ptr floatgrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    for (openvdb::FloatGrid::ValueOnCIter iter = grid->cbeginValueOn(); iter; ++iter) {
        //openvdb::math::Vec3d temp;
        //std::cout << "baseGrid " << grid->indexToWorld(iter.getCoord()) << " = " << *iter << std::endl;
        
       // std::cout << grid->indexToWorld(iter.getCoord()).x() << " "
        //          << grid->indexToWorld(iter.getCoord()).y() << " "
         //         << grid->indexToWorld(iter.getCoord()).z() << std::endl;
    }

    // saving the grid to disk
    grid->setName("voxelgrid");
    // Create a VDB file object.
    openvdb::io::File file(argv[2]);
    // Add the grid pointer to a container.
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    // Write out the contents of the container.
    file.write(grids);
    file.close();


    /*
    // now read the grid back in
    // Create a VDB file object.
    openvdb::io::File file2(argv[2]);

    // Open the file.  This reads the file header, but not any grids.
    file2.open();
    openvdb::GridBase::Ptr baseGrid = file2.readGrid("voxelgrid");

    // must cast the pointer to floatgrid type before accessing the elements
    openvdb::FloatGrid::Ptr floatgrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    for (openvdb::FloatGrid::ValueOnCIter iter = floatgrid->cbeginValueOn(); iter; ++iter) {
        std::cout << "baseGrid " << iter.getCoord() << " = " << *iter << std::endl;
    }

    file2.close();
    */

}
