#include <iostream>
#include <openvdb/openvdb.h>
//#include <openvdb/tools/GridSampling.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/MeshToVolume.h>
#include <string>



using std::cout;
using std::stoi;
using std::endl;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

int main()
{
    cout << "At start" << endl;

    // loading the points
    std::vector<openvdb::Vec3s> pointList;
    std::vector<openvdb::Vec3I> polygonList;
    std::vector<openvdb::Vec4I> quadList;


    std::ifstream myfile ("cube.obj");
    if (myfile.is_open())
    {
        std::string line;
        while ( getline (myfile,line) )
        {

            std::vector<std::string> x = split(line, ' ');
            //if (std::strcmp(x[0], 'f'))

            if (x.at(0)=="f")
            {
                cout << x.size() << endl;
                polygonList.push_back(openvdb::Vec3I(stoi(x[1]),stoi(x[2]),stoi(x[3])));
            }
            else if (x.at(0)=="v")
            {
                cout << x.size() << endl;
                pointList.push_back(openvdb::Vec3s(stod(x[3]),stod(x[2]),stod(x[1])));
               // cout << x[1] << " " << x[2] << " " << x[3] << endl;
            }
          //cout << line << '\n';
      }
      myfile.close();
    }
    else
    {
        cout << "Cannot read file..." << endl;
    }

    cout << pointList.size() << endl;

 float voxelSize = 0.1; //the bounding box of mesh has unit dimension

  openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform( voxelSize );// mVoxelSize 
  
      openvdb::tools::MeshToVolume< openvdb::FloatGrid > voxelizer( transform );   
  
      voxelizer.convertToLevelSet( pointList , polygonList  , 0.1f , 0.1f );

    openvdb::FloatGrid::Ptr grid = voxelizer.distGridPtr()->deepCopy();

    std::cout<< "aACtice" << grid->activeVoxelCount() <<std::endl;


    const openvdb::BBoxd bbox(/*min=*/openvdb::math::Vec3d(0,0,0),
                                    /*max=*/openvdb::math::Vec3d(100,50,120));
    // The far plane of the frustum will be twice as big as the near plane.
    const double taper = 2;
    // The depth of the frustum will be 10 times the x-width of the near plane.
    const double depth = 10;
    // The x-width of the frustum in world space units
    const double xWidth = 100;
    // Construct a frustum transform that results in a frustum whose
    // near plane is centered on the origin in world space.
    openvdb::math::Transform::Ptr frustumTransform =
        openvdb::math::Transform::createFrustumTransform(
            bbox, taper, depth, xWidth);
    // The frustum shape can be rotated, scaled, translated and even
    // sheared if desired.  For example, the following call translates
    // the frustum by 10,15,0 in world space:
    frustumTransform->postTranslate(openvdb::math::Vec3d(10,15,0));
    // Compute the world space image of a given point within
    // the index space bounding box that defines the frustum.
    openvdb::Coord ijk(20,10,18);
    openvdb::Coord worldLocation = frustumTransform->indexToWorld(ijk);

    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    std::cout << "Grid" << worldLocation << " = " << accessor.getValue(worldLocation) << std::endl;
    

    cout << " start look " << endl;

    // Compute the value of the grid at a location in world space.
    //openvdb::FloatGrid::ValueType worldValue = sampler.wsSample(openvdb::Vec3R(0.0276615470648, 0.0102880448103, 0.084502518177));
    
/*
cout << " s3 look " << endl;

    // loop over each image pixel
    for (float u = 0; u < 15; u+=0.1)
    {
        //openvdb::FloatGrid::ValueType worldValue = sampler.wsSample(openvdb::Vec3R(0.0, 0.0, u));
        openvdb::FloatGrid::ValueType worldValue  = openvdb::tools::PointSampler::sample(grid->tree(), openvdb::Vec3R(0.0, 0.0, u));
        cout << "vale " << u << " is " << worldValue << endl;
    }
*/


    cout << " s2 look " << endl;







    // now i want to do a line of voxels from the camera for perhaps 2m and see what they intersect...
    float f = 100;
    float c_x = 100;
    float c_y = 100;
    int im_w = 200;
    int im_y = 200;

    // loop over each image pixel
    for (int u = 0; u < im_w; ++u)
    {
        for (int v = 0; v < im_y; ++v)
        {
            // loop over each item along the voxel ray



        }
    }



    // now trace through each voxel individually...

    //openvdb::tools::MeshToVolume<openvdb::FloatGrid> voxelizer(xform);
    //voxelizer.convertToLevelSet(pointList, polygonList, 5.0f, 5.0f);
    //openvdb::Coord xyz(16,16,16);
    //std::cout << voxelizer.distGridPtr()->getAccessor().getValue(xyz) << std::endl;

/*
    meshToLevelSet(
   78     const openvdb::math::Transform& xform,
   79     const std::vector<Vec3s>& points,
   80     const std::vector<Vec3I>& triangles,
   81     float halfWidth = float(LEVEL_SET_HALF_WIDTH));
*/

}