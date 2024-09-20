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
void smooth_gauss_volume0(Wave w, Fault f, real_t *M)
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
  int i0 = nx/2;

  // extend at free surface
  if(par.freenode && j < nj){
    // f.W(ny, nz, 2, FSIZE)
    for (int l = 0; l < FSIZE; l++){
      for (int m = 0; m < 2; m++){
        for (int n = 1; n < SMOOTH_HALF_WIDTH1; n++){
          int pos_free  = l * ny * nz * 2 + m * ny * nz + (k1    ) * ny + j1;
          int pos_outer = l * ny * nz * 2 + m * ny * nz + (k1 + n) * ny + j1;
          int pos_inner = l * ny * nz * 2 + m * ny * nz + (k1 - n) * ny + j1;
          f.W[pos_outer] = 2.0 * f.W[pos_free] - f.W[pos_inner];
        }
      }
    }
    // f.T11, f.T12, f.T13 (ny, nz, 3)
    for (int n = 1; n < SMOOTH_HALF_WIDTH1; n++){
      int pos_free  = 3 * ny * nz + (k1    ) * ny + j1;
      int pos_outer = 3 * ny * nz + (k1 + n) * ny + j1;
      int pos_inner = 3 * ny * nz + (k1 - n) * ny + j1;
      f.T11[pos_outer] = 2.0 * f.T11[pos_free] - f.T11[pos_inner];
      f.T12[pos_outer] = 2.0 * f.T12[pos_free] - f.T12[pos_inner];
      f.T13[pos_outer] = 2.0 * f.T13[pos_free] - f.T13[pos_inner];
    }

    for (int l = 0; l < WSIZE; l++){
      for (int i1 = i0-SMOOTH_HALF_WIDTH; i1 <= i0+SMOOTH_HALF_WIDTH; i1++){
        for (int n = 1; n < SMOOTH_HALF_WIDTH1; n++){
          int pos_free  = l * ny * nz * nx + i1 * ny * nz + (k1    ) * ny + j1;
          int pos_outer = l * ny * nz * nx + i1 * ny * nz + (k1 + n) * ny + j1;
          int pos_inner = l * ny * nz * nx + i1 * ny * nz + (k1 - n) * ny + j1;
          w.W[pos_outer] = 2.0 * w.W[pos_free] - w.W[pos_inner];
        }
      }
    }
  } // end freenode
}

__global__
void smooth_gauss_volume1(Wave w, Fault f, real_t *M)
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
  int i0 = nx/2;

  real_t *xix = M + 0 * nx * ny * nz;
  real_t *xiy = M + 1 * nx * ny * nz;
  real_t *xiz = M + 2 * nx * ny * nz;
  real_t *etx = M + 3 * nx * ny * nz;
  real_t *ety = M + 4 * nx * ny * nz;
  real_t *etz = M + 5 * nx * ny * nz;
  real_t *ztx = M + 6 * nx * ny * nz;
  real_t *zty = M + 7 * nx * ny * nz;
  real_t *ztz = M + 8 * nx * ny * nz;
  real_t *jac = M + 9 * nx * ny * nz;

  // smooth split nodes
  if(j > 0 && j < nj-0 && k > 0 && k < nk-0){
    int pos;

    //ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    // minus side
    //ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    real_t T_s1 = 0.0;
    real_t T_s2 = 0.0;
    real_t T_s3 = 0.0;
    real_t T_s4 = 0.0;
    real_t T_s5 = 0.0;
    real_t T_s6 = 0.0;

    real_t *t21 = f.W + 3 * ny * nz * 2;
    real_t *t22 = f.W + 4 * ny * nz * 2;
    real_t *t23 = f.W + 5 * ny * nz * 2;
    real_t *t31 = f.W + 6 * ny * nz * 2;
    real_t *t32 = f.W + 7 * ny * nz * 2;
    real_t *t33 = f.W + 8 * ny * nz * 2;

    real_t *t21s = f.Ws + 3 * ny * nz * 2;
    real_t *t22s = f.Ws + 4 * ny * nz * 2;
    real_t *t23s = f.Ws + 5 * ny * nz * 2;
    real_t *t31s = f.Ws + 6 * ny * nz * 2;
    real_t *t32s = f.Ws + 7 * ny * nz * 2;
    real_t *t33s = f.Ws + 8 * ny * nz * 2;

    real_t *txx = w.W + 3 * ny * nz * nx;
    real_t *tyy = w.W + 4 * ny * nz * nx;
    real_t *tzz = w.W + 5 * ny * nz * nx;
    real_t *txy = w.W + 6 * ny * nz * nx;
    real_t *txz = w.W + 7 * ny * nz * nx;
    real_t *tyz = w.W + 8 * ny * nz * nx;

#ifndef Filter
    for (int m = 0; m < SMOOTH_WIDTH; m++){
      for (int n = 0; n < SMOOTH_WIDTH; n++){
        int jj = j1 + n - SMOOTH_HALF_WIDTH;
        int kk = k1 + m - SMOOTH_HALF_WIDTH;
        T_s1 += t21[kk * ny + jj] * wp1[0][m][n];
        T_s2 += t22[kk * ny + jj] * wp1[0][m][n];
        T_s3 += t23[kk * ny + jj] * wp1[0][m][n];
        T_s4 += t31[kk * ny + jj] * wp1[0][m][n];
        T_s5 += t32[kk * ny + jj] * wp1[0][m][n];
        T_s6 += t33[kk * ny + jj] * wp1[0][m][n];
      }
    }
    for (int l = 1; l < SMOOTH_HALF_WIDTH1; l++){
      for (int m = 0; m < SMOOTH_WIDTH; m++){
        for (int n = 0; n < SMOOTH_WIDTH; n++){
          int ii = i0 - l;
          int jj = j1 + n - SMOOTH_HALF_WIDTH;
          int kk = k1 + m - SMOOTH_HALF_WIDTH;
          int pos = ii * ny * nz + kk * ny + jj;

          matE[0][0] = xix[pos]; matE[0][1] = etx[pos]; matE[0][2] = ztx[pos];
          matE[1][0] = xiy[pos]; matE[1][1] = ety[pos]; matE[1][2] = zty[pos];
          matE[2][0] = xiz[pos]; matE[2][1] = etz[pos]; matE[2][2] = ztz[pos];

          matS[0][0] = txx[pos]; matS[0][1] = txy[pos]; matS[0][2] = txz[pos];
          matS[1][0] = txy[pos]; matS[1][1] = tyy[pos]; matS[1][2] = tyz[pos];
          matS[2][0] = txz[pos]; matS[2][1] = tyz[pos]; matS[2][2] = tzz[pos];

          matmul3x3(matS, matE, matT);

          T_s1 += matT[0][1] * jac[pos] * wp1[l][m][n];
          T_s2 += matT[1][1] * jac[pos] * wp1[l][m][n];
          T_s3 += matT[2][1] * jac[pos] * wp1[l][m][n];
          T_s4 += matT[0][2] * jac[pos] * wp1[l][m][n];
          T_s5 += matT[1][2] * jac[pos] * wp1[l][m][n];
          T_s6 += matT[2][2] * jac[pos] * wp1[l][m][n];
        }
      }
    }
#else

    T_s1 += f7pd0 * (t21s[k1*ny + j1])
         +  f7pd1 * (t21s[k1*ny + (j1+1)] + t21s[k1*ny + (j1-1)])
         +  f7pd2 * (t21s[k1*ny + (j1+2)] + t21s[k1*ny + (j1-2)])
         +  f7pd3 * (t21s[k1*ny + (j1+3)] + t21s[k1*ny + (j1-3)]);
    T_s1 += f7pd0 * (t21s[k1*ny + j1])
         +  f7pd1 * (t21s[(k1+1)*ny + j1] + t21s[(k1-1)*ny + j1])
         +  f7pd2 * (t21s[(k1+2)*ny + j1] + t21s[(k1-2)*ny + j1])
         +  f7pd3 * (t21s[(k1+3)*ny + j1] + t21s[(k1-3)*ny + j1]);
    T_s2 += f7pd0 * (t22s[k1*ny + j1])
         +  f7pd1 * (t22s[k1*ny + (j1+1)] + t22s[k1*ny + (j1-1)])
         +  f7pd2 * (t22s[k1*ny + (j1+2)] + t22s[k1*ny + (j1-2)])
         +  f7pd3 * (t22s[k1*ny + (j1+3)] + t22s[k1*ny + (j1-3)]);
    T_s2 += f7pd0 * (t22s[k1*ny + j1])
         +  f7pd1 * (t22s[(k1+1)*ny + j1] + t22s[(k1-1)*ny + j1])
         +  f7pd2 * (t22s[(k1+2)*ny + j1] + t22s[(k1-2)*ny + j1])
         +  f7pd3 * (t22s[(k1+3)*ny + j1] + t22s[(k1-3)*ny + j1]);
    T_s3 += f7pd0 * (t23s[k1*ny + j1])
         +  f7pd1 * (t23s[k1*ny + (j1+1)] + t23s[k1*ny + (j1-1)])
         +  f7pd2 * (t23s[k1*ny + (j1+2)] + t23s[k1*ny + (j1-2)])
         +  f7pd3 * (t23s[k1*ny + (j1+3)] + t23s[k1*ny + (j1-3)]);
    T_s3 += f7pd0 * (t23s[k1*ny + j1])
         +  f7pd1 * (t23s[(k1+1)*ny + j1] + t23s[(k1-1)*ny + j1])
         +  f7pd2 * (t23s[(k1+2)*ny + j1] + t23s[(k1-2)*ny + j1])
         +  f7pd3 * (t23s[(k1+3)*ny + j1] + t23s[(k1-3)*ny + j1]);
    T_s4 += f7pd0 * (t31s[k1*ny + j1])
         +  f7pd1 * (t31s[k1*ny + (j1+1)] + t31s[k1*ny + (j1-1)])
         +  f7pd2 * (t31s[k1*ny + (j1+2)] + t31s[k1*ny + (j1-2)])
         +  f7pd3 * (t31s[k1*ny + (j1+3)] + t31s[k1*ny + (j1-3)]);
    T_s4 += f7pd0 * (t31s[k1*ny + j1])
         +  f7pd1 * (t31s[(k1+1)*ny + j1] + t31s[(k1-1)*ny + j1])
         +  f7pd2 * (t31s[(k1+2)*ny + j1] + t31s[(k1-2)*ny + j1])
         +  f7pd3 * (t31s[(k1+3)*ny + j1] + t31s[(k1-3)*ny + j1]);
    T_s5 += f7pd0 * (t32s[k1*ny + j1])
         +  f7pd1 * (t32s[k1*ny + (j1+1)] + t32s[k1*ny + (j1-1)])
         +  f7pd2 * (t32s[k1*ny + (j1+2)] + t32s[k1*ny + (j1-2)])
         +  f7pd3 * (t32s[k1*ny + (j1+3)] + t32s[k1*ny + (j1-3)]);
    T_s5 += f7pd0 * (t32s[k1*ny + j1])
         +  f7pd1 * (t32s[(k1+1)*ny + j1] + t32s[(k1-1)*ny + j1])
         +  f7pd2 * (t32s[(k1+2)*ny + j1] + t32s[(k1-2)*ny + j1])
         +  f7pd3 * (t32s[(k1+3)*ny + j1] + t32s[(k1-3)*ny + j1]);
    T_s6 += f7pd0 * (t33s[k1*ny + j1])
         +  f7pd1 * (t33s[k1*ny + (j1+1)] + t33s[k1*ny + (j1-1)])
         +  f7pd2 * (t33s[k1*ny + (j1+2)] + t33s[k1*ny + (j1-2)])
         +  f7pd3 * (t33s[k1*ny + (j1+3)] + t33s[k1*ny + (j1-3)]);
    T_s6 += f7pd0 * (t33s[k1*ny + j1])
         +  f7pd1 * (t33s[(k1+1)*ny + j1] + t33s[(k1-1)*ny + j1])
         +  f7pd2 * (t33s[(k1+2)*ny + j1] + t33s[(k1-2)*ny + j1])
         +  f7pd3 * (t33s[(k1+3)*ny + j1] + t33s[(k1-3)*ny + j1]);
#endif

    pos = k1 * ny + j1;
    t21s[pos] = T_s1;
    t22s[pos] = T_s2;
    t23s[pos] = T_s3;
    t31s[pos] = T_s4;
    t32s[pos] = T_s5;
    t33s[pos] = T_s6;

    //ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
    // plus side
    //ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    T_s1 = 0.0;
    T_s2 = 0.0;
    T_s3 = 0.0;
    T_s4 = 0.0;
    T_s5 = 0.0;
    T_s6 = 0.0;

    t21 += ny * nz;
    t22 += ny * nz;
    t23 += ny * nz;
    t31 += ny * nz;
    t32 += ny * nz;
    t33 += ny * nz;

    t21s += ny * nz;
    t22s += ny * nz;
    t23s += ny * nz;
    t31s += ny * nz;
    t32s += ny * nz;
    t33s += ny * nz;

#ifndef Filter
    for (int m = 0; m < SMOOTH_WIDTH; m++){
      for (int n = 0; n < SMOOTH_WIDTH; n++){
        int jj = j1 + n - SMOOTH_HALF_WIDTH;
        int kk = k1 + m - SMOOTH_HALF_WIDTH;
        T_s1 += t21[kk * ny + jj] * wp2[0][m][n];
        T_s2 += t22[kk * ny + jj] * wp2[0][m][n];
        T_s3 += t23[kk * ny + jj] * wp2[0][m][n];
        T_s4 += t31[kk * ny + jj] * wp2[0][m][n];
        T_s5 += t32[kk * ny + jj] * wp2[0][m][n];
        T_s6 += t33[kk * ny + jj] * wp2[0][m][n];
      }
    }

    for (int l = 1; l < SMOOTH_HALF_WIDTH1; l++){
      for (int m = 0; m < SMOOTH_WIDTH; m++){
        for (int n = 0; n < SMOOTH_WIDTH; n++){
          int ii = i0 + l; // plus side
          int jj = j1 + n - SMOOTH_HALF_WIDTH;
          int kk = k1 + m - SMOOTH_HALF_WIDTH;
          int pos = ii * ny * nz + kk * ny + jj;

          matE[0][0] = xix[pos]; matE[0][1] = etx[pos]; matE[0][2] = ztx[pos];
          matE[1][0] = xiy[pos]; matE[1][1] = ety[pos]; matE[1][2] = zty[pos];
          matE[2][0] = xiz[pos]; matE[2][1] = etz[pos]; matE[2][2] = ztz[pos];

          matS[0][0] = txx[pos]; matS[0][1] = txy[pos]; matS[0][2] = txz[pos];
          matS[1][0] = txy[pos]; matS[1][1] = tyy[pos]; matS[1][2] = tyz[pos];
          matS[2][0] = txz[pos]; matS[2][1] = tyz[pos]; matS[2][2] = tzz[pos];

          matmul3x3(matS, matE, matT);

          T_s1 += matT[0][1] * jac[pos] * wp2[l][m][n];
          T_s2 += matT[1][1] * jac[pos] * wp2[l][m][n];
          T_s3 += matT[2][1] * jac[pos] * wp2[l][m][n];
          T_s4 += matT[0][2] * jac[pos] * wp2[l][m][n];
          T_s5 += matT[1][2] * jac[pos] * wp2[l][m][n];
          T_s6 += matT[2][2] * jac[pos] * wp2[l][m][n];
        }
      }
    }
#else
    T_s1 += f7pd0 * (t21s[k1*ny + j1])
         +  f7pd1 * (t21s[k1*ny + (j1+1)] + t21s[k1*ny + (j1-1)])
         +  f7pd2 * (t21s[k1*ny + (j1+2)] + t21s[k1*ny + (j1-2)])
         +  f7pd3 * (t21s[k1*ny + (j1+3)] + t21s[k1*ny + (j1-3)]);
    T_s1 += f7pd0 * (t21s[k1*ny + j1])
         +  f7pd1 * (t21s[(k1+1)*ny + j1] + t21s[(k1-1)*ny + j1])
         +  f7pd2 * (t21s[(k1+2)*ny + j1] + t21s[(k1-2)*ny + j1])
         +  f7pd3 * (t21s[(k1+3)*ny + j1] + t21s[(k1-3)*ny + j1]);
    T_s2 += f7pd0 * (t22s[k1*ny + j1])
         +  f7pd1 * (t22s[k1*ny + (j1+1)] + t22s[k1*ny + (j1-1)])
         +  f7pd2 * (t22s[k1*ny + (j1+2)] + t22s[k1*ny + (j1-2)])
         +  f7pd3 * (t22s[k1*ny + (j1+3)] + t22s[k1*ny + (j1-3)]);
    T_s2 += f7pd0 * (t22s[k1*ny + j1])
         +  f7pd1 * (t22s[(k1+1)*ny + j1] + t22s[(k1-1)*ny + j1])
         +  f7pd2 * (t22s[(k1+2)*ny + j1] + t22s[(k1-2)*ny + j1])
         +  f7pd3 * (t22s[(k1+3)*ny + j1] + t22s[(k1-3)*ny + j1]);
    T_s3 += f7pd0 * (t23s[k1*ny + j1])
         +  f7pd1 * (t23s[k1*ny + (j1+1)] + t23s[k1*ny + (j1-1)])
         +  f7pd2 * (t23s[k1*ny + (j1+2)] + t23s[k1*ny + (j1-2)])
         +  f7pd3 * (t23s[k1*ny + (j1+3)] + t23s[k1*ny + (j1-3)]);
    T_s3 += f7pd0 * (t23s[k1*ny + j1])
         +  f7pd1 * (t23s[(k1+1)*ny + j1] + t23s[(k1-1)*ny + j1])
         +  f7pd2 * (t23s[(k1+2)*ny + j1] + t23s[(k1-2)*ny + j1])
         +  f7pd3 * (t23s[(k1+3)*ny + j1] + t23s[(k1-3)*ny + j1]);
    T_s4 += f7pd0 * (t31s[k1*ny + j1])
         +  f7pd1 * (t31s[k1*ny + (j1+1)] + t31s[k1*ny + (j1-1)])
         +  f7pd2 * (t31s[k1*ny + (j1+2)] + t31s[k1*ny + (j1-2)])
         +  f7pd3 * (t31s[k1*ny + (j1+3)] + t31s[k1*ny + (j1-3)]);
    T_s4 += f7pd0 * (t31s[k1*ny + j1])
         +  f7pd1 * (t31s[(k1+1)*ny + j1] + t31s[(k1-1)*ny + j1])
         +  f7pd2 * (t31s[(k1+2)*ny + j1] + t31s[(k1-2)*ny + j1])
         +  f7pd3 * (t31s[(k1+3)*ny + j1] + t31s[(k1-3)*ny + j1]);
    T_s5 += f7pd0 * (t32s[k1*ny + j1])
         +  f7pd1 * (t32s[k1*ny + (j1+1)] + t32s[k1*ny + (j1-1)])
         +  f7pd2 * (t32s[k1*ny + (j1+2)] + t32s[k1*ny + (j1-2)])
         +  f7pd3 * (t32s[k1*ny + (j1+3)] + t32s[k1*ny + (j1-3)]);
    T_s5 += f7pd0 * (t32s[k1*ny + j1])
         +  f7pd1 * (t32s[(k1+1)*ny + j1] + t32s[(k1-1)*ny + j1])
         +  f7pd2 * (t32s[(k1+2)*ny + j1] + t32s[(k1-2)*ny + j1])
         +  f7pd3 * (t32s[(k1+3)*ny + j1] + t32s[(k1-3)*ny + j1]);
    T_s6 += f7pd0 * (t33s[k1*ny + j1])
         +  f7pd1 * (t33s[k1*ny + (j1+1)] + t33s[k1*ny + (j1-1)])
         +  f7pd2 * (t33s[k1*ny + (j1+2)] + t33s[k1*ny + (j1-2)])
         +  f7pd3 * (t33s[k1*ny + (j1+3)] + t33s[k1*ny + (j1-3)]);
    T_s6 += f7pd0 * (t33s[k1*ny + j1])
         +  f7pd1 * (t33s[(k1+1)*ny + j1] + t33s[(k1-1)*ny + j1])
         +  f7pd2 * (t33s[(k1+2)*ny + j1] + t33s[(k1-2)*ny + j1])
         +  f7pd3 * (t33s[(k1+3)*ny + j1] + t33s[(k1-3)*ny + j1]);
#endif

    pos = k1 * ny + j1;
    t21s[pos] = T_s1;
    t22s[pos] = T_s2;
    t23s[pos] = T_s3;
    t31s[pos] = T_s4;
    t32s[pos] = T_s5;
    t33s[pos] = T_s6;
  }
}

__global__
void smooth_gauss_volume2(Wave w, Fault f, real_t *M)
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

  // smooth split nodes
  if(j < nj && k < nk){
    int gj = par.ranky * nj + j + 1;
    int gk = par.rankz * nk + k + 1;
    real_t beta = 0;

#if defined TPV103 || defined TPV104
    if (!f.united[k * nj + j]){
      real_t RS_L = 0.4;
      real_t c = f.slip[k * nj + j] / RS_L;
#else
    if (gj >= par.Fault_grid[0] && gj <= par.Fault_grid[1] &&
        gk >= par.Fault_grid[2] && gk <= par.Fault_grid[3] ){
      real_t c = f.slip[k * nj + j] / par.Dc;
#endif
      beta = MIN(c, 1.0);
    }else{
      beta = 1.0;
    }

    for (int l = 3; l < FSIZE; l++){
      for (int m = 0; m < 2; m++){
        int pos = l * ny * nz * 2 + m * ny * nz + k1 * ny + j1 ;
#ifdef Filter
        f.W[pos] = f.W[pos] - 0.5 * f.Ws[pos];
#else
        f.W[pos] = beta * f.Ws[pos] + (1.0-beta) * f.W[pos];
#endif
      }
    }
  }
}

void fault_filter(Wave w, Fault f, real_t *M)
{
  dim3 block1(128, 1, 1);
  dim3 grid1(
      (hostParams.nj+block1.x-1)/block1.x, 1, 1);
  dim3 block(8, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);

  smooth_gauss_volume0 <<<grid, block>>> (w, f, M);
  smooth_gauss_volume1 <<<grid, block>>> (w, f, M);
  CUDACHECK(cudaGetLastError());
  smooth_gauss_volume2 <<<grid, block>>> (w, f, M);
}
