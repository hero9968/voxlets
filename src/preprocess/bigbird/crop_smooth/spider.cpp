/*
spider features mex file
*/


#include "mex.h"
#include <iostream>
#include <string.h>
#include <math.h>

#define A(i,j,channel) A[(i) + (j)*H + (channel)*H*W]
#define OUT_IM(i,j, channel) out_stack[(i) + (j)*H + (channel)*H*W]
//#define GEODESIC_IM(i,j) out_stack[(i) + (j)*H]
//sqrt(pow(X(i1,j1)-X(i1, j2), 2) + pow(Y(i1,j1)-Y(i2,j2), 2) + pow(Z(i1,j1)-Z(i2,j2), 2))

#define EDGE (0)
#define X (1)
#define Y (2)
#define D (3)
#define NX (4)
#define NY (5)
#define NZ (6)

#define SQDIST3D(i1, j1, i2, j2) pow(A(i1,j1,X)-A(i2,j2,X), 2) + pow(A(i1,j1,Y)-A(i2,j2,Y), 2)  + pow(A(i1,j1,D)-A(i2,j2,D), 2)

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{
	// input checks
	if (nrhs != 1)
		mexErrMsgTxt("Expected 1 input arguments");

	// get cloud and normals from mex input arguments...
	const mwSize *dim_array;
	dim_array = mxGetDimensions(prhs[0]);
	int H = dim_array[0];
	int W = dim_array[1];
	int channels = dim_array[2];
	double *A = mxGetPr(prhs[0]);

	size_t skip = 5; // when measuring geodesic, how many pixels to skip

	// set out output arrays
	// plhs[0] = mxCreateDoubleMatrix( (mwSize)H, (mwSize)W, mxREAL);
	// double *pixel_image = mxGetPr(plhs[0]);

	// plhs[1] = mxCreateDoubleMatrix( (mwSize)H, (mwSize)W, mxREAL);
	// double *geodesic_im = mxGetPr(plhs[1]);

	const mwSize dims[3] = {H, W, 12};
	plhs[0] = mxCreateNumericArray((mwSize)3, dims, mxDOUBLE_CLASS, mxREAL);
	double *out_stack = mxGetPr(plhs[0]);


	mexPrintf("Starting loop\n");

	/*
	 do it in the east direction
	 */
	for (size_t row = 0; row < H; ++row)
	{
		int pixel_count = -1;
		double geo_dist = 0;
		double perp_dist = 0;
		size_t edge_col = -1;

		for (size_t col = 0; col < W; ++col)
		{
			if (A(row, col, EDGE)==1)
			{
				pixel_count = 0;
				geo_dist = 0;
				perp_dist = 0;
				edge_col = col; // keeps track of the column at which the last edge occurred
			}
			else if (pixel_count>=0)
			{
				pixel_count++;
				if (pixel_count%skip==0) geo_dist += sqrt(SQDIST3D(row, col, row, col-skip));
				perp_dist = A(row, col, NX) * (A(row, edge_col, X) - A(row, col, X))
						  + A(row, col, NY) * (A(row, edge_col, Y) - A(row, col, Y))
						  + A(row, col, NZ) * (A(row, edge_col, D) - A(row, col, D));

			}
			OUT_IM(row, col, 0) = pixel_count;
			OUT_IM(row, col, 1) = geo_dist;
			OUT_IM(row, col, 2) = perp_dist;
		}
	}

	/*
	 do it in the west direction
	 */
	for (size_t row = 0; row < H; ++row)
	{
		int pixel_count = -1;
		double geo_dist = 0;
		double perp_dist = 0;
		size_t edge_col = -1;

		for (int col = W-1; col >= 0; --col)
		{
			if (A(row, col, EDGE)==1)
			{
				pixel_count = 0;
				geo_dist = 0;
				perp_dist = 0;
				edge_col = col; // keeps track of the column at which the last edge occurred
			}
			else if (pixel_count>=0)
			{
				pixel_count++;
				if (pixel_count%skip==0) geo_dist += sqrt(SQDIST3D(row, col, row, col+skip));
				perp_dist = A(row, col, NX) * (A(row, edge_col, X) - A(row, col, X))
						  + A(row, col, NY) * (A(row, edge_col, Y) - A(row, col, Y))
						  + A(row, col, NZ) * (A(row, edge_col, D) - A(row, col, D));
			}
			OUT_IM(row, col, 3) = pixel_count;
			OUT_IM(row, col, 4) = geo_dist;
			OUT_IM(row, col, 5) = perp_dist;
		}
	}



	/*
	 do it in the north direction
	 */
	for (size_t col = 0; col < W; ++col)
	{
		int pixel_count = -1;
		double geo_dist = 0;
		double perp_dist = 0;
		size_t edge_row = -1;

		for (size_t row = 0; row < H; ++row)
		{
			if (A(row, col, EDGE)==1)
			{
				pixel_count = 0;
				geo_dist = 0;
				perp_dist = 0;
				edge_row = row; // keeps track of the column at which the last edge occurred
			}
			else if (pixel_count>=0)
			{
				pixel_count++;
				if (pixel_count%skip==0) geo_dist += sqrt(SQDIST3D(row, col, row-skip, col));
				perp_dist = A(row, col, NX) * (A(edge_row, col, X) - A(row, col, X))
						  + A(row, col, NY) * (A(edge_row, col, Y) - A(row, col, Y))
						  + A(row, col, NZ) * (A(edge_row, col, D) - A(row, col, D));
			}
			OUT_IM(row, col, 6) = pixel_count;
			OUT_IM(row, col, 7) = geo_dist;
			OUT_IM(row, col, 8) = perp_dist;
		}
	}

	/*
	 do it in the south direction
	 */
	for (size_t col = 0; col < W; ++col)
	{
		int pixel_count = -1;
		double geo_dist = 0;
		double perp_dist = 0;
		size_t edge_row = -1;

		for (int row = H-1; row >= 0; --row)
		{
			if (A(row, col, EDGE)==1)
			{
				pixel_count = 0;
				geo_dist = 0;
				perp_dist = 0;
				edge_row = row; // keeps track of the column at which the last edge occurred
			}
			else if (pixel_count>=0)
			{
				pixel_count++;
				if (pixel_count%skip==0) geo_dist += sqrt(SQDIST3D(row, col, row+skip, col));
				perp_dist = A(row, col, NX) * (A(edge_row, col, X) - A(row, col, X))
						  + A(row, col, NY) * (A(edge_row, col, Y) - A(row, col, Y))
						  + A(row, col, NZ) * (A(edge_row, col, D) - A(row, col, D));
			}
			OUT_IM(row, col, 9) = pixel_count;
			OUT_IM(row, col, 10) = geo_dist;
			OUT_IM(row, col, 11) = perp_dist;
		}
	}

	mexPrintf("Done\n");

	



}