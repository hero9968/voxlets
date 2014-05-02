#include "mex.h"

#include <iostream>
#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <vector>

//using namespace cvmlcpp;
using std::endl;
using std::cout;



//mex mymex.cpp /usr/lib/libcvmlcpp.dylib -I/usr/include  -I/Library/Developer/CommandLineTools/usr/lib/c++/v1/

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


void
mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
	// input checks
	if(nrhs != 2)
		mexErrMsgTxt("Need two input arguments, filename and scalar voxelsize");
		
	// getting the filename
    size_t buflen = mxGetN(prhs[0])*sizeof(mxChar)+1;
    char *buf;
    buf = (char*)mxMalloc(buflen);
    
    // Copy the string data into buf.  (returns 1 on failure)
    if (mxGetString(prhs[0], buf, (mwSize)buflen))
    {
    	mexErrMsgTxt("Cannot understand string argument. First argument should be a filename.");
    }
	std::string filename(buf);

	// get the size of the voxels
	float voxelsize;
	if(!mxIsDouble(prhs[1]))
		mexErrMsgTxt("Voxelsize must be a real double scalar.");
	else
		voxelsize = mxGetScalar(prhs[1]);


	cvmlcpp::Matrix<int, 3u> voxels;
	cvmlcpp::Geometry<float> geometry;
	
	if (!readOBJ(geometry, filename))
	{
		mexErrMsgTxt("Could not read file");
	}

	if (!cvmlcpp::voxelize(geometry, voxels, voxelsize))
	{
		mexErrMsgTxt("Could not voxelize");
	}

	// finding size of the matrix
	std::tr1::array<std::size_t, D>::const_iterator extents = voxels.extents();
	size_t X =  extents[2];
	size_t Y =  extents[1];
	size_t Z =  extents[0];
	
	int extents_array[3] = {(int)X, (int)Y, (int)Z};
	
	plhs[0] = mxCreateNumericArray(3, extents_array, mxDOUBLE_CLASS, mxREAL);
	double *V;
	V = mxGetPr(plhs[0]);
	int counter = 0;
	
	for ( cvmlcpp::Matrix<int, 3u>::iterator it = voxels.begin(); it != voxels.end(); ++it, ++counter)
	{
		V[counter] = *it;
	}

/*
	// outputting the matrix to disk somehow...
	cout << extents[0] << " " << extents[1] << " " << extents[2] << endl;
		// printing voxels to disk

	cout << endl;
*/

	//return 0;


}