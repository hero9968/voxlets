
/*
Mex function for segmentation
mex segment_mex.cpp -I/usr/include/pcl-1.5/ -I/usr/local/include/eigen3/ -L/usr/lib/ -lpcl_search -lpcl_kdtree -lpcl_common -lpcl_features

Use like this:

*/

#include "mex.h"
#include <iostream>
#include <string.h>
#include <math.h>

void
mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])  
{

  // input checks
  if (nrhs != 11)
    mexErrMsgTxt("Expected 11 input arguments");
  
  // indsM, grayImg,absImgNsx,  len, nWin, i, j, rows, cols, gvals

  // get cloud and normals from mex input arguments...
  int H = mxGetM(prhs[0]);
  int W = mxGetN(prhs[0]);
  size_t numPix = H * W;

  double *indsM = mxGetPr(prhs[0]);
  double *grayImg = mxGetPr(prhs[1]);

  double absImgNdx = mxGetScalar(prhs[2]);
  double len = mxGetScalar(prhs[3]);
  double nWin = mxGetScalar(prhs[4]);
  double i = mxGetScalar(prhs[5]);
  double j = mxGetScalar(prhs[6]);

  double *rows = mxGetPr(prhs[7]);
  double *cols = mxGetPr(prhs[8]);
  double *gvals = mxGetPr(prhs[9]);
  double *vals = mxGetPr(prhs[10]);

  int winRad = 1;
  
  
  //size_t nWin = 0; // Counts the number of points in the current window.

  int start_ii = std::max(0, (int)i-winRad);
  int end_ii = std::min(H-1, (int)i+winRad);
  int start_jj = std::max(0, (int)j-winRad);
  int end_jj = std::min(W-1, (int)j+winRad);
  //mexPrintf("i=%d, j=%d\n", (int)i, (int)j);
  //mexPrintf("start_ii=%d, end_ii=%d, start_jj=%d, end_jj=%d\n", start_ii, end_ii, start_jj, end_jj);
  //mexPrintf("START: len=%d, nWin=%d\n", (int)len, (int)nWin);

  for (size_t ii = start_ii; ii <= end_ii; ii++)
  {
    for (size_t jj = start_jj; jj <= end_jj; jj++)
    {

      if (ii == i && jj == j) { continue; }

      rows[(int)len] = absImgNdx;
      cols[(int)len] = indsM[ii + jj * H];
      gvals[(int)nWin] = grayImg[ii + jj * H];

      len++;
      nWin++;

      //mexPrintf("i=%d, j=%d, ii=%d, jj=%d, nwin=%d, gvals[nWin]=%f\n", i, j, ii, jj, nWin, gvals[nWin]);
      //mexPrintf("cols[len]=%d\n", cols[len]);
    }
  }
  //mexPrintf("END: len=%d, nWin=%d\n", (int)len, (int)nWin);
  double curVal = grayImg[(int)i + (int)j * H];
  //mexPrintf("curVal[%d, %d] = %f", (int)i, (int)j, curVal);
  gvals[(int)nWin] = curVal;

  double sum = 0;
  for (size_t counter = 0; counter < nWin+1; counter++) 
    { sum += gvals[counter];}
  double mean = sum / (nWin+1);

  //mexPrintf("mean=%f\n", mean);

  // computing men of squared values
  double sum2 = 0;
  for (size_t counter = 0; counter < nWin+1; counter++) { sum2 += pow(gvals[counter], 2);}
  double squared_mean = sum2 / (nWin+1);

  // computing the variance
  double c_var = squared_mean - mean * mean;

  // computing csig
  double csig = c_var*0.6;
  double mgv = 1000000000;
  for (size_t counter = 0; counter < nWin; counter++) 
    { 
      double poss_value = pow(gvals[counter]-curVal, 2);
      if (poss_value < mgv)
        mgv = poss_value;
    }
  //mexPrintf("curval=%f\n", curVal);
  //mexPrintf("mgv=%f\n", mgv);
  
  if (csig < (-mgv/log(0.01)))
    csig = -mgv/log(0.01);
  
  if (csig < 0.000002)
    csig = 0.000002;

//gvals(1:nWin) = exp(-(gvals(1:nWin)-curVal).^2/csig);
  double sum_gvals = 0;
  for (size_t counter = 0; counter < nWin; counter++) 
  {
    double exponent = -pow(gvals[counter]-curVal, 2) / csig;
    gvals[counter] = exp(exponent);
    sum_gvals += gvals[counter];
  }

  //gvals(1:nWin) = gvals(1:nWin) / sum(gvals(1:nWin));
  for (size_t counter = 0; counter < nWin; counter++) 
    gvals[counter] /= sum_gvals;
  
  //
  int counter2=0;
   for (size_t counter = len-nWin; counter < len; counter++) 
   {
      vals[counter] = -gvals[counter2];
      counter2++;
    }
  //vals(len-nWin+1 : len) = -gvals(1:nWin);

  rows[(int)len] = absImgNdx;
  cols[(int)len] = absImgNdx;
  vals[(int)len] = 1;
  len++;


  plhs[0] = mxCreateDoubleScalar(len);
  plhs[1] = mxCreateDoubleScalar(nWin);
  plhs[2] = mxCreateDoubleScalar(c_var);
  plhs[3] = mxCreateDoubleScalar(csig);
}