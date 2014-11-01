#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <openvdb/openvdb.h>
//#include <openvdb/tools/GridSampling.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/Composite.h>

using std::cout;
using std::endl;
std::string centres_path = "/Users/Michael/projects/shape_sharing/data/voxlets/shoebox_dictionary_training.vdb";


int main()
{
    openvdb::initialize();

    //---------------------------------------------------
    cout << "Reading in cluster centers" << endl;
    //---------------------------------------------------
    openvdb::io::File file(centres_path);
    file.open();

    openvdb::GridBase::Ptr baseGrid;
    std::vector<openvdb::FloatGrid::Ptr> voxlets;

    for (openvdb::io::File::NameIterator nameIter = file.beginName();
        nameIter != file.endName(); ++nameIter)
    {
        //std::cout << "Reading grid " << nameIter.gridName() << std::endl;
        baseGrid = file.readGrid(nameIter.gridName());

        // From the example above, "LevelSetSphere" is known to be a FloatGrid,
        // so cast the generic grid pointer to a FloatGrid pointer.
        openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        grid->tree().prune();
        voxlets.push_back(grid);
    }
    file.close();

    cout << "Now reading in the config file" << endl;
    YAML::Node config = YAML::LoadFile("./config.yaml");

    cout << "Now create the output grid space" << endl;
    openvdb::FloatGrid::Ptr output = openvdb::FloatGrid::create(/*background value=*/0.0);
    output->setName("output");

    //---------------------------------------------------
    // for each item in the config file
    //---------------------------------------------------
    for (size_t i = 0; i < config.size(); ++i)
    {

        // load the cluster idx and associated transform
        size_t voxlet_idx = config[i]["cluster_idx"].as<size_t>();

        double transform[16];
        for (size_t j = 0; j < 16; ++j) {
            transform[j] = config[i]["transform"][j].as<double>();
        }

        cout << "Dealing with item " << i << " which has centre idx " << voxlet_idx << endl;
        cout << transform[0] <<transform[1] << endl;
        
        // convert transform to openvdb
        openvdb::math::Mat4d mat(transform);
        cout << mat << endl;
        openvdb::math::Transform::Ptr linearTransform =
            openvdb::math::Transform::createLinearTransform(mat);

       // openvdb::Mat4R xform = linearTransform;
        openvdb::tools::GridTransformer transformer(mat);

        // make a copy of the associated grid and transform according to the transform
        openvdb::FloatGrid::Ptr voxlet_copy = voxlets[voxlet_idx];
        openvdb::FloatGrid::Ptr voxlet_trans = openvdb::FloatGrid::create(/*background value=*/0.0);

        transformer.transformGrid<openvdb::tools::PointSampler, openvdb::FloatGrid>(
             *voxlet_copy, *voxlet_trans);

        // now add to the output grid
        openvdb::tools::compSum(*output, *voxlet_trans);

        // Prune the target tree for optimal sparsity.
//        output->tree().prune();

    }

    //---------------------------------------------------
    cout << "Now save the output grid" << endl;
    //---------------------------------------------------

    openvdb::io::File fileout("combined.vdb");
    // Add the grid pointer to a container.
    openvdb::GridPtrVec grids;
    grids.push_back(output);
    // Write out the contents of the container.
    fileout.write(grids);
    fileout.close();
    
    return 0;
}
