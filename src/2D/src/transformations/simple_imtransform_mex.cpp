#include "mex.h"
#include <iostream>
#include <string.h>

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{

  // input checks
  if (nrhs != 4)
    mexErrMsgTxt("Expected 4 input arguments");

  // get image from mex input arguments...
  int rows = mxGetM(prhs[0]);
  int cols = mxGetN(prhs[0]);

  double *data = mxGetPr(prhs[0]);
  //for (size_t i = 0; i < rows; ++i)
   // for (size_t j = 0; j < cols; ++j)
      //mexPrintf("Position %d and %d has value %f\n", i, j, data[i + j*rows]);

  // getting the transformation matrix
  int T_rows = mxGetM(prhs[1]);
  int T_cols = mxGetN(prhs[1]);
  if (T_cols != 3 || T_rows != 3)
    mexErrMsgTxt("Transformation matrix should be 3x3");

  double *T = mxGetPr(prhs[1]);
  /*
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      mexPrintf("T %d and %d has value %f\n", i, j, T[i + j*T_rows]);
  */
  // getting the height and width of the output image
  double *width_out = mxGetPr(prhs[2]);
  double *height_out = mxGetPr(prhs[3]);
  int width_int = (int)(*width_out);
  int height_int = (int)(*height_out);

  // computing the image
  plhs[0] = mxCreateDoubleMatrix( (mwSize)height_int, (mwSize)width_int, mxREAL);
  double *image_out_ptr = mxGetPr(plhs[0]);
  for (size_t j = 0; j < width_int; ++j)
    for (size_t i = 0; i < height_int; ++i)
    {
      double input_col = T[0] * (double)j + T[3] * (double)i + T[6];
      double input_row = T[1] * (double)j + T[4] * (double)i + T[7];
      //mexPrintf("T 6 is %f\n", T[6]);
      //mexPrintf("input row is %f and input col is %f\n", input_row, input_col);
      if (input_row >= 0 && input_row < rows && input_col >= 0 && input_col < cols)
      {
        int position = (int)input_row + (int)input_col * rows;
        //mexPrintf("Position is  %d, trans is %f\n", position, T[0, 2]);
        image_out_ptr[i + j * height_int] = data[position];
      }
      else
        image_out_ptr[i + j * height_int] = 0;
    }

  // 



  mexPrintf("Done\n");

}