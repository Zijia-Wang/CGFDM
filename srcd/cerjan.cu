#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "common.h"
#include "params.h"

//#define ND 20

/*

   // modified in "main.cu"

extern void init_cerjan_host(float *damp);
extern __global__ void apply_cerjan(Wave w, float *damp);

  float *damp_h = (float *) malloc(sizeof(float)*nx*ny*nz);
  memset(damp_h, 0, sizeof(float)*nx*ny*nz);
  float *damp;
  cudaMalloc((float **) &damp, sizeof(float)*nx*ny*nz);

  init_cerjan_host(damp_h);
  cudaMemcpy(damp, damp_h, sizeof(float)*nx*ny*nz, cudaMemcpyHostToDevice);

  // to use in each runge-kutta process:

      apply_cerjan <<< grid3, block3 >>> (w, damp);

  */

void init_cerjan_host(float *damp) {

  int ND = hostParams.DAMP_N;
  float coef = 0.92;
  //float alpha = 0.015f;
  float alpha = sqrt(-log(coef))/ND;
  if(masternode){
    printf("Cerjan ND = %d\n"
      "Cerjan alpha = %f\n", ND, alpha);
  }

  int i, j, k;
  int pos;
  int ni1 = 3;
  int nj1 = 3;
  int nk1 = 3;
  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;
  int ni2 = nx - 3;
  int nj2 = ny - 3;
  int nk2 = nz - 3;

  float *tmp = (float *) malloc(sizeof(float)*nx*ny*nz);

  for (i = 0; i < nx; i++){
    for (j = 0; j < ny; j++){
      for (k = 0; k < nz; k++){
        damp[i*ny*nz + k*ny + j] = 1.0;
      }
    }
  }

  // x1
  if(neigxid[0] == MPI_PROC_NULL){
    for (i = 0; i < ND; i++){
      for (j = nj1; j < nj2; j++){
        for (k = nk1; k < nk2; k++){
          pos = (ni1+i)*ny*nz + k*ny + j;
          damp[pos] = expf( -powf( alpha * (ND - i), 2) );
        }
      }
    }
  }
  // x2
  if(neigxid[1] == MPI_PROC_NULL){
    for (i = 0; i < ND; i++){
      for (j = nj1; j < nj2; j++){
        for (k = nk1; k < nk2; k++){
          pos = (ni2-i-1)*ny*nz + k*ny + j;
          damp[pos] = expf( -powf( alpha * (ND - i), 2) );
        }
      }
    }
  }

  memcpy(tmp, damp, nx*ny*nz*sizeof(float));

  // y1
  if(neigyid[0] == MPI_PROC_NULL){
    for (j = 0; j < ND; j++){
      for (i = ni1; i < ni2; i++){
        for (k = nk1; k < nk2; k++){
          pos = i*ny*nz + k*ny + (ni1+j);
          damp[pos] = expf( -powf( alpha * (ND - j), 2) );
          damp[pos] = MIN( damp[pos], tmp[pos] );
        }
      }
    }
  }

  // y2
  if(neigyid[1] == MPI_PROC_NULL){
    for (j = 0; j < ND; j++){
      for (i = ni1; i < ni2; i++){
        for (k = nk1; k < nk2; k++){
          pos = i*ny*nz + k*ny + (nj2-j-1);
          damp[pos] = expf( -powf( alpha * (ND - j), 2) );
          damp[pos] = MIN( damp[pos], tmp[pos] );
        }
      }
    }
  }

  memcpy(tmp, damp, nx*ny*nz*sizeof(float));

  // z1
  if(neigzid[0] == MPI_PROC_NULL){
    for (k = 0; k < ND; k++){
      for (i = ni1; i < ni2; i++){
        for (j = nj1; j < nj2; j++){
          pos = i*ny*nz + (k+nk1)*ny + j;
          damp[pos] = expf( -powf( alpha * (ND - k), 2) );
          damp[pos] = MIN( damp[pos], tmp[pos] );
        }
      }
    }
  }
  // z2
  if(neigzid[1] == MPI_PROC_NULL && (!freenode)){
    for (k = 0; k < ND; k++){
      for (i = ni1; i < ni2; i++){
        for (j = nj1; j < nj2; j++){
          pos = i*ny*nz + (nk2-k-1)*ny + j;
          damp[pos] = expf( -powf( alpha * (ND - k), 2) );
          damp[pos] = MIN( damp[pos], tmp[pos] );
        }
      }
    }
  }
  //}

//  TEST! I don't want to multiply it!
//  for (i = ni1+ND; i < ni2-ND; i++)
//    for (j = nj1+ND; j < nj2-ND; j++)
//      for (k = nk1+ND; k < nk2-ND; k++){
//        damp[i*ny*nz + j*nz + k] = 0.0;
//      }
//  TEST! I want to multiply it!
//  for (i = 0; i < nx; i++){
//    for (j = 0; j < ny; j++){
//      for (k = 0; k < nz; k++){
//        if( i < 3 || i >= nx-3 ||
//            j < 3 || j >= ny-3 ||
//            k < 3 || k >= nz-3 ){
//        damp[i*ny*nz + j*nz + k] = 0.0;
//        }
//      }
//    }
//  }

  free(tmp); tmp = NULL;
  return;
}

__global__ void apply_cerjan_cu(Rectangle R, Wave W, float *damp)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int i1 = i + 3 + R.istart;
  int j1 = j + 3 + R.jstart;
  int k1 = k + 3 + R.kstart;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  //int ni = par.ni;
  //int nj = par.nj;
  //int nk = par.nk;

  //int ND = par.DAMP_N;

  if (i < R.icount && j < R.jcount && k < R.kcount)
  {
    //if(!(i >= ND && i < ni-ND &&
    //     j >= ND && j < nj-ND &&
    //     k >= ND && k < nk-ND ) )
    //{
      for (int m = 0; m < 9; m++)
      {
        long pos = i1*ny*nz + k1*ny + j1;
        W.W[pos + m*nx*ny*nz] *= damp[pos];
      }
    //}
  }
  return ;
}

void apply_cerjan(Wave W, float *damp)
{
  //           _________________________
  //          /|                       /|
  //         / |         z2           / |
  //        /  |                     /  |
  //       /   |           y2       /   |
  //      /____|___________________/    |
  //      |    |                   |    |
  //      | x1 |                   | x2 |
  //      |    |                   |    |
  //      |    |___________________|____|
  //      |   /       y1           |   /
  //      |  /                     |  /
  //      | /           z1         | /
  //      |/_______________________|/
  //
  //
  //                   Y-axis
  //         Z-axis   /|
  //          /|\    /
  //           |    /
  //           |   /
  //           |  /
  //           | /
  //           |/______________\  X-axis
  //         O /               /
  //          /
  //
  // we divide the damp region into 26 blocks
  // including:
  // 3x2 = 6 surfaces
  // x1, x2, y1, y2, z1, z2
  // 3x4 = 12 edges
  // x1y1, x1y2, x2y1, x2y2
  // x1z1, x1z2, x2z1, x2z2
  // y1z1, y1z2, y2z1, y2z2
  // 2^3 = 8 vertexes
  // x1y1z1, x1y1z2, x1y2z1, x1y2z2
  // x2y1z1, x2y1z2, x2y2z1, x2y2z2

  Rectangle R[26];

  int ND = hostParams.DAMP_N;
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int istart, icount, jstart, jcount, kstart, kcount;

  for (int i = 0; i < 26; i++){
    R[i].isx = 0;
    R[i].isy = 0;
    R[i].isz = 0;
  }

#define SETR(R)  \
  R.istart = istart; \
  R.jstart = jstart; \
  R.kstart = kstart; \
  R.icount = icount; \
  R.jcount = jcount; \
  R.kcount = kcount;

  // x1
  istart = 0;     icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[0])
  // x2
  istart = ni-ND; icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[1])
  // y1
  istart = ND;    icount = ni-2*ND;
  jstart = 0;     jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[2])
  // y2
  istart = ND;    icount = ni-2*ND;
  jstart = nj-ND; jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[3])
  // z1
  istart = ND;    icount = ni-2*ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = 0;     kcount = ND;
  SETR(R[4])
  // z2
  istart = ND;    icount = ni-2*ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[5])
  // x1y1
  istart = 0;     icount = ND;
  jstart = 0;     jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[6])
  // x1y2
  istart = 0;     icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[7])
  // x2y1
  istart = ni-ND; icount = ND;
  jstart = 0;     jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[8])
  // x2y2
  istart = ni-ND; icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = ND;    kcount = nk-2*ND;
  SETR(R[9])
  // x1z1
  istart = 0;     icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = 0;     kcount = ND;
  SETR(R[10])
  // x1z2
  istart = 0;     icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[11])
  // x2z1
  istart = ni-ND; icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = 0;     kcount = ND;
  SETR(R[12])
  // x2z2
  istart = ni-ND; icount = ND;
  jstart = ND;    jcount = nj-2*ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[13])
  // y1z1
  istart = ND;    icount = ni-2*ND;
  jstart = 0;     jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[14])
  // y1z2
  istart = ND;    icount = ni-2*ND;
  jstart = 0;     jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[15])
  // y2z1
  istart = ND;    icount = ni-2*ND;
  jstart = nj-ND; jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[16])
  // y2z2
  istart = ND;    icount = ni-2*ND;
  jstart = nj-ND; jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[17])
  // x1y1z1
  istart = 0;     icount = ND;
  jstart = 0;     jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[18])
  // x1y1z2
  istart = 0;     icount = ND;
  jstart = 0;     jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[19])
  // x1y2z1
  istart = 0;     icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[20])
  // x1y2z2
  istart = 0;     icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[21])
  // x2y1z1
  istart = ni-ND; icount = ND;
  jstart = 0;     jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[22])
  // x2y1z2
  istart = ni-ND; icount = ND;
  jstart = 0;     jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[23])
  // x2y2z1
  istart = ni-ND; icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = 0;     kcount = ND;
  SETR(R[24])
  // x2y2z2
  istart = ni-ND; icount = ND;
  jstart = nj-ND; jcount = ND;
  kstart = nk-ND; kcount = ND;
  SETR(R[25])
#undef SETR

  Rectangle R0;
  R0.istart = 0;
  R0.jstart = 0;
  R0.kstart = 0;
  R0.icount = ni;
  R0.jcount = nj;
  R0.kcount = nk;
  dim3 block(16, 4, 4);
  dim3 grid;
  grid.x = (hostParams.nj + block.x - 1) / block.x;
  grid.y = (hostParams.nk + block.y - 1) / block.y;
  grid.z = (hostParams.ni + block.z - 1) / block.z;
  apply_cerjan_cu <<<grid, block>>> (R0, W, damp);

///  // 26 blocks way is not faster enough
///  for (int i = 0; i < 26; i++)
///  {
///    if(
///        neigxid[0] == MPI_PROC_NULL || neigxid[1] == MPI_PROC_NULL ||
///        neigyid[0] == MPI_PROC_NULL || neigyid[1] == MPI_PROC_NULL ||
///        neigzid[0] == MPI_PROC_NULL || neigzid[1] == MPI_PROC_NULL )
///    {
///      dim3 block(16, 4, 4);
///      dim3 grid;
///      grid.x = (R[i].jcount + block.x - 1) / block.x;
///      grid.y = (R[i].kcount + block.y - 1) / block.y;
///      grid.z = (R[i].icount + block.z - 1) / block.z;
///      apply_cerjan_cu <<<grid, block>>> (R[i], W, damp);
///    }
///  }

  return;
}
