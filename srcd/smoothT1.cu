#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

#define SMOOTH_HALF_WIDTH 3
#define SMOOTH_HALF_WIDTH1 (SMOOTH_HALF_WIDTH+1)
#define SMOOTH_WIDTH (SMOOTH_HALF_WIDTH*2+1)

extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);

__global__
void smooth_gauss_volume0(Fault f, int nfault)
{

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int k = nk - 1;
  int j1 = j + 3;
  int k1 = k + 3;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int Tsize = nfault * ny*nz*7;

  // extend at free surface
  if(par.freenode && j < nj){ 
    // f.T11, f.T12, f.T13 (ny, nz, 3)
    for (int n = 1; n < SMOOTH_HALF_WIDTH1; n++){
      int pos_free  = 3 * ny * nz + (k1    ) * ny + j1 + Tsize;
      int pos_outer = 3 * ny * nz + (k1 + n) * ny + j1 + Tsize;
      int pos_inner = 3 * ny * nz + (k1 - n) * ny + j1 + Tsize;
      f.T11[pos_outer] = 2.0 * f.T11[pos_free] - f.T11[pos_inner];
      f.T12[pos_outer] = 2.0 * f.T12[pos_free] - f.T12[pos_inner];
      f.T13[pos_outer] = 2.0 * f.T13[pos_free] - f.T13[pos_inner];
    }

  } // end freenode
}

__global__
void smooth_gauss_volume1(Fault f, int nfault)
{

  //real_t wp[7][7][7], wp1[4][7][7], wp2[4][7][7];
  real_t wp[SMOOTH_WIDTH][SMOOTH_WIDTH][SMOOTH_WIDTH];
  real_t wp1[SMOOTH_HALF_WIDTH1][SMOOTH_WIDTH][SMOOTH_WIDTH];
  real_t wp2[SMOOTH_HALF_WIDTH1][SMOOTH_WIDTH][SMOOTH_WIDTH];
  real_t matE[3][3], matS[3][3], matT[3][3];
  real_t gauss_width = 0.7;
  real_t pmax1 = 0.4;
  real_t gauss_width2 = gauss_width * gauss_width;

  for (int ii = -SMOOTH_HALF_WIDTH; ii <= SMOOTH_HALF_WIDTH; ii++){
    for (int jj = -SMOOTH_HALF_WIDTH; jj <= SMOOTH_HALF_WIDTH; jj++){
      for (int kk = -SMOOTH_HALF_WIDTH; kk <= SMOOTH_HALF_WIDTH; kk++){
        real_t r2 = (real_t)(ii*ii+jj*jj+kk*kk)/gauss_width2;
        wp[ii+SMOOTH_HALF_WIDTH][jj+SMOOTH_HALF_WIDTH][kk+SMOOTH_HALF_WIDTH] = exp(-r2);
      }
    }
  }

  // left half
  real_t sum1 = 0;
  for (int ii = 0; ii < SMOOTH_HALF_WIDTH1; ii++){
    for (int jj = 0; jj < SMOOTH_WIDTH; jj++){
      for (int kk = 0; kk < SMOOTH_WIDTH; kk++){
        wp1[ii][jj][kk] = wp[SMOOTH_HALF_WIDTH-ii][jj][kk] * ((ii==0) ? 1.0 : pmax1);
        sum1 += wp1[ii][jj][kk];
      }
    }
  }
  // right half
  real_t sum2 = 0;
  for (int ii = 0; ii < SMOOTH_HALF_WIDTH1; ii++){
    for (int jj = 0; jj < SMOOTH_WIDTH; jj++){
      for (int kk = 0; kk < SMOOTH_WIDTH; kk++){
        wp2[ii][jj][kk] = wp[SMOOTH_HALF_WIDTH+ii][jj][kk] * ((ii==0) ? 1.0 : pmax1);
        sum2 += wp2[ii][jj][kk];
      }
    }
  }
  // normalization
  for (int ii = 0; ii < SMOOTH_HALF_WIDTH1; ii++){
    for (int jj = 0; jj < SMOOTH_WIDTH; jj++){
      for (int kk = 0; kk < SMOOTH_WIDTH; kk++){
        wp1[ii][jj][kk] /= sum1;
        wp2[ii][jj][kk] /= sum2;
        //if(par.rankx == 0 && par.ranky == 0 && par.rankz == 0){
        //  printf("wp1(%d %d %d) %g\n", ii, jj, kk, wp1[ii][jj][kk]);
        //  printf("wp2(%d %d %d) %g\n", ii, jj, kk, wp2[ii][jj][kk]);
        //}
      }
    }
  }

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int Tsize = nfault * ny*nz*7;
  int mpiFsize = nfault * nj*nk;

  // smooth split nodes
  if(j < nj && k < nk){
    int pos;

    real_t T_s1 = 0.0;
    real_t T_s2 = 0.0;
    real_t T_s3 = 0.0;

    real_t *t11 = f.T11 + Tsize;
    real_t *t12 = f.T12 + Tsize;
    real_t *t13 = f.T13 + Tsize;

    real_t *t11s = f.T11S + mpiFsize;
    real_t *t12s = f.T12S + mpiFsize;
    real_t *t13s = f.T13S + mpiFsize;

    ////////////////////////// T1 not split ////////////////////////////
    // for (int m = 0; m < SMOOTH_WIDTH; m++){
    //   for (int n = 0; n < SMOOTH_WIDTH; n++){
    //     int jj = j1 + n - SMOOTH_HALF_WIDTH;
    //     int kk = k1 + m - SMOOTH_HALF_WIDTH;
    //     T_s1 += t11[kk * ny + jj] * wp1[0][m][n];
    //     T_s2 += t12[kk * ny + jj] * wp1[0][m][n];
    //     T_s3 += t13[kk * ny + jj] * wp1[0][m][n];
    //   }
    // }
    //////////////////////// other points //////////////////////////////
    for (int l = -3; l < SMOOTH_HALF_WIDTH1; l++){
      for (int m = 0; m < SMOOTH_WIDTH; m++){
        for (int n = 0; n < SMOOTH_WIDTH; n++){
          // int ii = i0 - l;
          int ii = l+3;  // 0 1 2 3 4 5 6
          int jj = j1 + n - SMOOTH_HALF_WIDTH;
          int kk = k1 + m - SMOOTH_HALF_WIDTH;
          pos = ii * ny * nz + kk * ny + jj;
          
          T_s1 += t11[pos] * wp1[l][m][n];
          T_s2 += t12[pos] * wp1[l][m][n];
          T_s3 += t13[pos] * wp1[l][m][n];
        }
      }
    }

    pos = k * nj + j;
    t11s[pos] = T_s1;
    t12s[pos] = T_s2;
    t13s[pos] = T_s3;
  }
}

__global__
void smooth_gauss_volume2(Fault f, int nfault, int Faultgrid[])
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  //int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int mpiFsize = nfault * nj*nk;

  // smooth split nodes
  if(j < nj && k < nk){
    int gj = par.ranky * nj + j + 1;
    int gk = par.rankz * nk + k + 1;
    real_t beta = 0;

    if (gj >= Faultgrid[0 + 4*nfault] && gj <= Faultgrid[1 + 4*nfault] &&
        gk >= Faultgrid[2 + 4*nfault] && gk <= Faultgrid[3 + 4*nfault] ){
    // if (gj >= par.Fault_grid[0] && gj <= par.Fault_grid[1] &&
        // gk >= par.Fault_grid[2] && gk <= par.Fault_grid[3] ){
      real_t c = 0.0f;
      beta = MIN(c, 1.0);
    }else{
      beta = 1.0;
    }

    int pos = 3 * ny * nz + k1 * ny + j1 + nfault * (ny*nz*7);
    int posf = j + k * nj + mpiFsize;
        // f.W[pos] = beta * f.Ws[pos] + (1.0-beta) * f.W[pos];
    f.T11[pos] = beta * f.T11S[posf] + (1.0-beta) * f.T11[pos];
    f.T12[pos] = beta * f.T12S[posf] + (1.0-beta) * f.T12[pos];
    f.T13[pos] = beta * f.T13S[posf] + (1.0-beta) * f.T13[pos];
  }
}

void smooth_T1(Fault f, int nfault, int Faultgrid[])
{
  dim3 block1(128, 1, 1);
  dim3 grid1(
      (hostParams.nj+block1.x-1)/block1.x, 1, 1);
  dim3 block(8, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);

  smooth_gauss_volume0 <<<grid, block>>> (f, nfault);
  CUDACHECK(cudaGetLastError());
  smooth_gauss_volume1 <<<grid, block>>> (f, nfault);
  CUDACHECK(cudaGetLastError());
  smooth_gauss_volume2 <<<grid, block>>> (f, nfault, Faultgrid);
  CUDACHECK(cudaGetLastError());
}
