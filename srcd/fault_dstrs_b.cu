#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

//#define DEBUG

extern __device__ real_t norm3(real_t *A);
extern __device__ real_t dot_product(real_t *A, real_t *B);
extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void invert3x3(real_t m[][3]);
extern __device__ real_t Fr_func(const real_t r, const real_t R);
extern __device__ real_t Gt_func(const real_t t, const real_t T);

__global__
void fault_dstrs_b_cu(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ, int i0, int nfault)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int i;
  int nj = par.nj;
  int nk = par.nk;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  real_t rDH = par.rDH;
  // OUTPUT
  int stride = nx * ny * nz;
  int nyz = ny * nz;
  int nyz2 = nyz * 2;

  int faultsize = nfault * nyz2 * 9;
  int mpifaultsize = nfault * nj*nk;
  int Tsize = nfault * nyz * 7;
  int matsize = nfault * ny * 3*3;
  int Dsize = matsize * nz;
  int matsize2 = Dsize;

  real_t *w_Vx  = W.W + 0 * stride;
  real_t *w_Vy  = W.W + 1 * stride;
  real_t *w_Vz  = W.W + 2 * stride;
  //real_t *w_Txx = W.W + 3 * stride;
  //real_t *w_Tyy = W.W + 4 * stride;
  //real_t *w_Tzz = W.W + 5 * stride;
  //real_t *w_Txy = W.W + 6 * stride;
  //real_t *w_Txz = W.W + 7 * stride;
  //real_t *w_Tyz = W.W + 8 * stride;

  //real_t *w_hVx  = W.hW + 0 * stride;
  //real_t *w_hVy  = W.hW + 1 * stride;
  //real_t *w_hVz  = W.hW + 2 * stride;
  real_t *w_hTxx = W.hW + 3 * stride;
  real_t *w_hTyy = W.hW + 4 * stride;
  real_t *w_hTzz = W.hW + 5 * stride;
  real_t *w_hTxy = W.hW + 6 * stride;
  real_t *w_hTxz = W.hW + 7 * stride;
  real_t *w_hTyz = W.hW + 8 * stride;

  // INPUT
  real_t *XIX = M + 0 * stride;
  real_t *XIY = M + 1 * stride;
  real_t *XIZ = M + 2 * stride;
  real_t *ETX = M + 3 * stride;
  real_t *ETY = M + 4 * stride;
  real_t *ETZ = M + 5 * stride;
  real_t *ZTX = M + 6 * stride;
  real_t *ZTY = M + 7 * stride;
  real_t *ZTZ = M + 8 * stride;
  //real_t *JAC = M + 9 * stride;
  real_t *LAM = M + 10 * stride;
  real_t *MIU = M + 11 * stride;
  //real_t *RHO = M + 12 * stride;

  // Split nodes
  //stride = ny * nz * 2; // y vary first

  // INPUT
  real_t *f_Vx  = F.W + 0 * nyz2 + faultsize;
  real_t *f_Vy  = F.W + 1 * nyz2 + faultsize;
  real_t *f_Vz  = F.W + 2 * nyz2 + faultsize;
  //real_t *f_T21 = F.W + 3 * nyz2;
  //real_t *f_T22 = F.W + 4 * nyz2;
  //real_t *f_T23 = F.W + 5 * nyz2;
  //real_t *f_T31 = F.W + 6 * nyz2;
  //real_t *f_T32 = F.W + 7 * nyz2;
  //real_t *f_T33 = F.W + 8 * nyz2;

  //real_t *f_hVx  = F.hW + 0 * nyz2;
  //real_t *f_hVy  = F.hW + 1 * nyz2;
  //real_t *f_hVz  = F.hW + 2 * nyz2;
  real_t *f_hT21 = F.hW + 3 * nyz2 + faultsize;
  real_t *f_hT22 = F.hW + 4 * nyz2 + faultsize;
  real_t *f_hT23 = F.hW + 5 * nyz2 + faultsize;
  real_t *f_hT31 = F.hW + 6 * nyz2 + faultsize;
  real_t *f_hT32 = F.hW + 7 * nyz2 + faultsize;
  real_t *f_hT33 = F.hW + 8 * nyz2 + faultsize;

  real_t xix, xiy, xiz;
  real_t etx, ety, etz;
  real_t ztx, zty, ztz;

  real_t mu, lam, lam2mu;
  //real_t rrhojac;

  real_t DxVx[8],DxVy[8],DxVz[8];
  real_t DyVx[8],DyVy[8],DyVz[8];
  real_t DzVx[8],DzVy[8],DzVz[8];

  //real_t vecT11[7], vecT12[7], vecT13[7];
  //real_t vecT21[7], vecT22[7], vecT23[7];
  //real_t vecT31[7], vecT32[7], vecT33[7];
  //real_t DxTx[3],DyTy[3],DzTz[3];

  real_t vec_3[3], vec_5[5];
  real_t mat1[3][3], mat2[3][3], mat3[3][3];
  real_t vec1[3], vec2[3], vec3[3];
  real_t vecg1[3], vecg2[3], vecg3[3];

  real_t matT1toVxm[3][3];
  real_t matVytoVxm[3][3];
  real_t matVztoVxm[3][3];
  real_t matT1toVxp[3][3];
  real_t matVytoVxp[3][3];
  real_t matVztoVxp[3][3];
  real_t dtT1[3];
  real_t matT1toVxfm[3][3];
  real_t matVytoVxfm[3][3];
  real_t matT1toVxfp[3][3];
  real_t matVytoVxfp[3][3];

  real_t matMin2Plus1[3][3];
  real_t matMin2Plus2[3][3];
  real_t matMin2Plus3[3][3];
  real_t matMin2Plus4[3][3];
  real_t matMin2Plus5[3][3];
  //real_t matPlus2Min1[3][3];
  //real_t matPlus2Min2[3][3];
  //real_t matPlus2Min3[3][3];
  //real_t matPlus2Min4[3][3];
  //real_t matPlus2Min5[3][3];
  real_t dxV1[3], dyV1[3], dzV1[3];
  real_t dxV2[3], dyV2[3], dzV2[3];
  real_t out1[3], out2[3], out3[3], out4[3], out5[3];
  real_t matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  //real_t matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  real_t matVx2Vz1[3][3], matVy2Vz1[3][3];
  real_t matVx2Vz2[3][3], matVy2Vz2[3][3];

  //int ii, jj, mm, l, n;
  int ii, jj, l, n;

  //int pos, pos_m, slice, segment;
  int pos, slice, segment;
  //int pos0, pos1, pos2;
  int pos0, pos1;
  int idx;
  int pos_f;

//  int i0 = nx/2;       ***********wangzj
  if (j < nj && k < nk ) { 
    // non united
    if(F.united[j + k * nj + mpifaultsize]) return;
    //  km = NZ -(thisid[2]*nk+k-3);
    //int km = nz - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    //
    //     Update Stress (in Zhang's Thesis, 2014)
    // ---V-----V-----V-----0-----0-----V-----V-----V---  (grid point)
    //    G     F     E     D-    D+    C     B     A     (grid name in thesis)
    //    i0-3  i0-2  i0-1  i0-0  i0+0  i0+1  i0+2  i0+3  (3D grid index)
    //    0     1     2     3     4     5     6     7     (vec grid index)
    //    -3    -2    -1    -0    +0    1     2     3     (offset from fault)

    //n = 0; l = -3; get DxV[0]
    pos = j1 + k1 * ny + (i0-3) * ny * nz;
    slice = ny*nz;
    DxVx[0] = LB(w_Vx, pos, slice)*rDH;
    DxVy[0] = LB(w_Vy, pos, slice)*rDH;
    DxVz[0] = LB(w_Vz, pos, slice)*rDH;

    //n = 1; l = -2; get DxV[1]
    pos = j1 + k1 * ny + (i0-2) * ny * nz;
    slice = ny*nz;
#ifdef VHOC
    DxVx[1] = rDH*compact_a1*(w_Vx[pos]-w_Vx[pos-slice])-compact_a2*DxVx[0];
    DxVy[1] = rDH*compact_a1*(w_Vy[pos]-w_Vy[pos-slice])-compact_a2*DxVy[0];
    DxVz[1] = rDH*compact_a1*(w_Vz[pos]-w_Vz[pos-slice])-compact_a2*DxVz[0];
    DxVx[1] = rDH*(0.25*w_Vx[pos+nyz]+w_Vx[pos]-1.25*w_Vx[pos-nyz])-0.5*DxVx[0];
    DxVy[1] = rDH*(0.25*w_Vy[pos+nyz]+w_Vy[pos]-1.25*w_Vy[pos-nyz])-0.5*DxVy[0];
    DxVz[1] = rDH*(0.25*w_Vz[pos+nyz]+w_Vz[pos]-1.25*w_Vz[pos-nyz])-0.5*DxVz[0];
    if(par.freenode && km<=3){
    DxVx[1] = L24B(w_Vx, pos, slice)*rDH;
    DxVy[1] = L24B(w_Vy, pos, slice)*rDH;
    DxVz[1] = L24B(w_Vz, pos, slice)*rDH;
    }
#else
    DxVx[1] = L24B(w_Vx, pos, slice)*rDH;
    DxVy[1] = L24B(w_Vy, pos, slice)*rDH;
    DxVz[1] = L24B(w_Vz, pos, slice)*rDH;
#endif
#ifdef DxV_NCFD
    pos = j1 + k1 * ny + (i0-1) * ny * nz;
    pos_f = j1 + k1 * ny + 0 * ny * nz;
    DxVx[1] = FD24_1*f_Vx[pos_f]+
              FD24_2*w_Vx[pos]+
              FD24_3*w_Vx[pos-1*nyz]+
              FD24_4*w_Vx[pos-2*nyz]+
              FD24_5*w_Vx[pos-3*nyz]+
              FD24_6*w_Vx[pos-4*nyz]+
              FD24_7*w_Vx[pos-5*nyz];
    DxVy[1] = FD24_1*f_Vy[pos_f]+
              FD24_2*w_Vy[pos]+
              FD24_3*w_Vy[pos-1*nyz]+
              FD24_4*w_Vy[pos-2*nyz]+
              FD24_5*w_Vy[pos-3*nyz]+
              FD24_6*w_Vy[pos-4*nyz]+
              FD24_7*w_Vy[pos-5*nyz];
    DxVz[1] = FD24_1*f_Vz[pos_f]+
              FD24_2*w_Vz[pos]+
              FD24_3*w_Vz[pos-1*nyz]+
              FD24_4*w_Vz[pos-2*nyz]+
              FD24_5*w_Vz[pos-3*nyz]+
              FD24_6*w_Vz[pos-4*nyz]+
              FD24_7*w_Vz[pos-5*nyz];
    DxVx[1] *= rDH;
    DxVy[1] *= rDH;
    DxVz[1] *= rDH;
#endif

    //n = 2; l = -1; get DxV[2]
    pos = j1 + k1 * ny + (i0-1) * ny * nz;
    pos_f = j1 + k1 * ny + 0 * ny * nz;
    slice = ny*nz;
#ifdef VHOC
    DxVx[2] = rDH*compact_a1*(w_Vx[pos]-w_Vx[pos-slice])-compact_a2*DxVx[1];
    DxVy[2] = rDH*compact_a1*(w_Vy[pos]-w_Vy[pos-slice])-compact_a2*DxVy[1];
    DxVz[2] = rDH*compact_a1*(w_Vz[pos]-w_Vz[pos-slice])-compact_a2*DxVz[1];
    DxVx[2] = rDH*(0.25*f_Vx[pos_f]+w_Vx[pos]-1.25*w_Vx[pos-nyz])-0.5*DxVx[1];
    DxVy[2] = rDH*(0.25*f_Vy[pos_f]+w_Vy[pos]-1.25*w_Vy[pos-nyz])-0.5*DxVy[1];
    DxVz[2] = rDH*(0.25*f_Vz[pos_f]+w_Vz[pos]-1.25*w_Vz[pos-nyz])-0.5*DxVz[1];
    if(par.freenode && km<=3){
    DxVx[2] = L22B(w_Vx, pos, slice)*rDH;
    DxVy[2] = L22B(w_Vy, pos, slice)*rDH;
    DxVz[2] = L22B(w_Vz, pos, slice)*rDH;
    }
#else
    DxVx[2] = L22B(w_Vx, pos, slice)*rDH;
    DxVy[2] = L22B(w_Vy, pos, slice)*rDH;
    DxVz[2] = L22B(w_Vz, pos, slice)*rDH;
#endif
#ifdef DxV_NCFD
    pos = j1 + k1 * ny + (i0-1) * ny * nz;
    pos_f = j1 + k1 * ny + 0 * ny * nz;
    DxVx[2] = FD15_1*f_Vx[pos_f]+
              FD15_2*w_Vx[pos]+
              FD15_3*w_Vx[pos-1*nyz]+
              FD15_4*w_Vx[pos-2*nyz]+
              FD15_5*w_Vx[pos-3*nyz]+
              FD15_6*w_Vx[pos-4*nyz]+
              FD15_7*w_Vx[pos-5*nyz];
    DxVy[2] = FD15_1*f_Vy[pos_f]+
              FD15_2*w_Vy[pos]+
              FD15_3*w_Vy[pos-1*nyz]+
              FD15_4*w_Vy[pos-2*nyz]+
              FD15_5*w_Vy[pos-3*nyz]+
              FD15_6*w_Vy[pos-4*nyz]+
              FD15_7*w_Vy[pos-5*nyz];
    DxVz[2] = FD15_1*f_Vz[pos_f]+
              FD15_2*w_Vz[pos]+
              FD15_3*w_Vz[pos-1*nyz]+
              FD15_4*w_Vz[pos-2*nyz]+
              FD15_5*w_Vz[pos-3*nyz]+
              FD15_6*w_Vz[pos-4*nyz]+
              FD15_7*w_Vz[pos-5*nyz];
    DxVx[2] *= rDH;
    DxVy[2] *= rDH;
    DxVz[2] *= rDH;
#endif

    // Split nodes {{{
    slice = nz*FSIZE; //segment = FSIZE;
#ifdef RupSensor
    if(F.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){
#else
    if(F.rup_index_y[j + k * nj + mpifaultsize] % 7){
#endif
      pos = j1 + k1 * ny + 0 * ny * nz;
      DyVx[3] = L22(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[3] = L22(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[3] = L22(f_Vz, pos, 1, FlagY)*rDH;
      pos = j1 + k1 * ny + 1 * ny * nz;
      DyVx[4] = L22(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[4] = L22(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[4] = L22(f_Vz, pos, 1, FlagY)*rDH;
    }else{
      pos = j1 + k1 * ny + 0 * ny * nz;
      DyVx[3] = L(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[3] = L(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[3] = L(f_Vz, pos, 1, FlagY)*rDH;
      pos = j1 + k1 * ny + 1 * ny * nz;
      DyVx[4] = L(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[4] = L(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[4] = L(f_Vz, pos, 1, FlagY)*rDH;
    }
#ifdef DyzV_center
    real_t DyVxc[8];
    real_t DyVyc[8];
    real_t DyVzc[8];
    real_t DzVxc[8];
    real_t DzVyc[8];
    real_t DzVzc[8];

    pos0 = j1 + k1 * ny + 0 * ny * nz;
    pos1 = j1 + k1 * ny + 1 * ny * nz;
    DyVxc[3] = 0.5*(L22(f_Vx,pos0,1 ,-FlagY)+L22(f_Vx,pos0,1 ,-FlagY))*rDH;
    DyVyc[3] = 0.5*(L22(f_Vy,pos0,1 ,-FlagY)+L22(f_Vy,pos0,1 ,-FlagY))*rDH;
    DyVzc[3] = 0.5*(L22(f_Vz,pos0,1 ,-FlagY)+L22(f_Vz,pos0,1 ,-FlagY))*rDH;

    DyVxc[4] = 0.5*(L22(f_Vx,pos1,1 ,-FlagY)+L22(f_Vx,pos1,1 ,-FlagY))*rDH;
    DyVyc[4] = 0.5*(L22(f_Vy,pos1,1 ,-FlagY)+L22(f_Vy,pos1,1 ,-FlagY))*rDH;
    DyVzc[4] = 0.5*(L22(f_Vz,pos1,1 ,-FlagY)+L22(f_Vz,pos1,1 ,-FlagY))*rDH;

    DzVxc[3] = 0.5*(L22(f_Vx,pos0,ny,-FlagZ)+L22(f_Vx,pos0,ny,-FlagZ))*rDH;
    DzVyc[3] = 0.5*(L22(f_Vy,pos0,ny,-FlagZ)+L22(f_Vy,pos0,ny,-FlagZ))*rDH;
    DzVzc[3] = 0.5*(L22(f_Vz,pos0,ny,-FlagZ)+L22(f_Vz,pos0,ny,-FlagZ))*rDH;

    DzVxc[4] = 0.5*(L22(f_Vx,pos1,ny,-FlagZ)+L22(f_Vx,pos1,ny,-FlagZ))*rDH;
    DzVyc[4] = 0.5*(L22(f_Vy,pos1,ny,-FlagZ)+L22(f_Vy,pos1,ny,-FlagZ))*rDH;
    DzVzc[4] = 0.5*(L22(f_Vz,pos1,ny,-FlagZ)+L22(f_Vz,pos1,ny,-FlagZ))*rDH;
#endif

    //km = NZ -(thisid[2]*nk+k-3);
    //km = nz -(k-3);

    // calculate backward on the minus side directly
    pos = j1 + k1 * ny + (i0-1) * ny * nz;
    pos0 = j1 + k1 * ny + 0 * ny * nz;
#ifdef VHOC
    DxVx[3] = rDH*compact_a1*(f_Vx[pos0]-w_Vx[pos])-compact_a2*DxVx[2];
    DxVy[3] = rDH*compact_a1*(f_Vy[pos0]-w_Vy[pos])-compact_a2*DxVy[2];
    DxVz[3] = rDH*compact_a1*(f_Vz[pos0]-w_Vz[pos])-compact_a2*DxVz[2];
    DxVx[3] = rDH*(19./9.*f_Vx[pos0]-37./9.*w_Vx[pos]+19./6.*w_Vx[pos-nyz]-13./9.*w_Vx[pos-2*nyz]+5./18.*w_Vx[pos-3*nyz]);
    DxVy[3] = rDH*(19./9.*f_Vy[pos0]-37./9.*w_Vy[pos]+19./6.*w_Vy[pos-nyz]-13./9.*w_Vy[pos-2*nyz]+5./18.*w_Vy[pos-3*nyz]);
    DxVz[3] = rDH*(19./9.*f_Vz[pos0]-37./9.*w_Vz[pos]+19./6.*w_Vz[pos-nyz]-13./9.*w_Vz[pos-2*nyz]+5./18.*w_Vz[pos-3*nyz]);
    if(par.freenode && km<=3){
    DxVx[3] = ( f_Vx[pos0] - w_Vx[pos] )*rDH;
    DxVy[3] = ( f_Vy[pos0] - w_Vy[pos] )*rDH;
    DxVz[3] = ( f_Vz[pos0] - w_Vz[pos] )*rDH; // minus side
    }
#else
    DxVx[3] = ( f_Vx[pos0] - w_Vx[pos] )*rDH;
    DxVy[3] = ( f_Vy[pos0] - w_Vy[pos] )*rDH;
    DxVz[3] = ( f_Vz[pos0] - w_Vz[pos] )*rDH; // minus side
    //DxVx[3] = (7./6.*f_Vx[pos0]-4./3.*w_Vx[pos]+1./6.*w_Vx[pos-ny*nz])*rDH;
    //DxVy[3] = (7./6.*f_Vy[pos0]-4./3.*w_Vy[pos]+1./6.*w_Vy[pos-ny*nz])*rDH;
    //DxVz[3] = (7./6.*f_Vz[pos0]-4./3.*w_Vz[pos]+1./6.*w_Vz[pos-ny*nz])*rDH; // minus side
    //DxVx[3] = rDH*(19./9.*f_Vx[pos0]-37./9.*w_Vx[pos]+19./6.*w_Vx[pos-nyz]-13./9.*w_Vx[pos-2*nyz]+5./18.*w_Vx[pos-3*nyz]);
    //DxVy[3] = rDH*(19./9.*f_Vy[pos0]-37./9.*w_Vy[pos]+19./6.*w_Vy[pos-nyz]-13./9.*w_Vy[pos-2*nyz]+5./18.*w_Vy[pos-3*nyz]);
    //DxVz[3] = rDH*(19./9.*f_Vz[pos0]-37./9.*w_Vz[pos]+19./6.*w_Vz[pos-nyz]-13./9.*w_Vz[pos-2*nyz]+5./18.*w_Vz[pos-3*nyz]);
#endif
#ifdef DxV_24
    DxVx[3] = (7.0*f_Vx[pos0]-8.0*w_Vx[pos]+w_Vx[pos-nyz])/6.0*rDH;
    DxVy[3] = (7.0*f_Vy[pos0]-8.0*w_Vy[pos]+w_Vy[pos-nyz])/6.0*rDH;
    DxVz[3] = (7.0*f_Vz[pos0]-8.0*w_Vz[pos]+w_Vz[pos-nyz])/6.0*rDH; // minus side
#endif
#ifdef DxV_NCFD
    pos = j1 + k1 * ny + (i0-1) * ny * nz;
    pos_f = j1 + k1 * ny + 0 * ny * nz;
    DxVx[3] = FD06_1*f_Vx[pos_f]+
              FD06_2*w_Vx[pos]+
              FD06_3*w_Vx[pos-1*nyz]+
              FD06_4*w_Vx[pos-2*nyz]+
              FD06_5*w_Vx[pos-3*nyz]+
              FD06_6*w_Vx[pos-4*nyz]+
              FD06_7*w_Vx[pos-5*nyz];
    DxVy[3] = FD06_1*f_Vy[pos_f]+
              FD06_2*w_Vy[pos]+
              FD06_3*w_Vy[pos-1*nyz]+
              FD06_4*w_Vy[pos-2*nyz]+
              FD06_5*w_Vy[pos-3*nyz]+
              FD06_6*w_Vy[pos-4*nyz]+
              FD06_7*w_Vy[pos-5*nyz];
    DxVz[3] = FD06_1*f_Vz[pos_f]+
              FD06_2*w_Vz[pos]+
              FD06_3*w_Vz[pos-1*nyz]+
              FD06_4*w_Vz[pos-2*nyz]+
              FD06_5*w_Vz[pos-3*nyz]+
              FD06_6*w_Vz[pos-4*nyz]+
              FD06_7*w_Vz[pos-5*nyz];
    DxVx[3] *= rDH;
    DxVy[3] *= rDH;
    DxVz[3] *= rDH;
#endif

    if(par.freenode && km==1){
      // get DxV[4], DzV[3], DzV[4] {{{
      for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < 3; jj++){
          matVx2Vz1    [ii][jj] = F.matVx2Vz1    [j1*9 + ii*3 + jj + matsize];
          matVx2Vz2    [ii][jj] = F.matVx2Vz2    [j1*9 + ii*3 + jj + matsize];
          matVy2Vz1    [ii][jj] = F.matVy2Vz1    [j1*9 + ii*3 + jj + matsize];
          matVy2Vz2    [ii][jj] = F.matVy2Vz2    [j1*9 + ii*3 + jj + matsize];
          matMin2Plus1f[ii][jj] = F.matMin2Plus1f[j1*9 + ii*3 + jj + matsize];
          matMin2Plus2f[ii][jj] = F.matMin2Plus2f[j1*9 + ii*3 + jj + matsize];
          matMin2Plus3f[ii][jj] = F.matMin2Plus3f[j1*9 + ii*3 + jj + matsize];

          matT1toVxfm[ii][jj] = F.matT1toVxfm[j1*9 + ii*3 + jj + matsize];
          matVytoVxfm[ii][jj] = F.matVytoVxfm[j1*9 + ii*3 + jj + matsize];
          matT1toVxfp[ii][jj] = F.matT1toVxfp[j1*9 + ii*3 + jj + matsize];
          matVytoVxfp[ii][jj] = F.matVytoVxfp[j1*9 + ii*3 + jj + matsize];
        }
      }

      dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3];
      dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3];
      dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3];

                         dyV2[0] = DyVx[4];
                         dyV2[1] = DyVy[4];
                         dyV2[2] = DyVz[4];

      matmul3x1(matMin2Plus1f, dxV1, out1);
      matmul3x1(matMin2Plus2f, dyV1, out2);
      matmul3x1(matMin2Plus3f, dyV2, out3);

      DxVx[4] = out1[0] + out2[0] - out3[0];
      DxVy[4] = out1[1] + out2[1] - out3[1];
      DxVz[4] = out1[2] + out2[2] - out3[2];
#ifdef DxV_T1f
      // =======================================================
      if(F.faultgrid[j+k*nj + mpifaultsize]){
        dtT1[0] = F.hT11[j + k * nj + mpifaultsize];
        dtT1[1] = F.hT12[j + k * nj + mpifaultsize];
        dtT1[2] = F.hT13[j + k * nj + mpifaultsize];
      }else{
        dtT1[0] = 0.0;
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;
      }
        dtT1[0] = 0.0;
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;

      // Minus side --------------------------------------------
      dyV1[0] = DyVx[3];
      dyV1[1] = DyVy[3];
      dyV1[2] = DyVz[3];

      matmul3x1(matT1toVxfm, dtT1, out1);
      matmul3x1(matVytoVxfm, dyV1, out2);

      DxVx[3] = out1[0] - out2[0];
      DxVy[3] = out1[1] - out2[1];
      DxVz[3] = out1[2] - out2[2];

      // Plus side --------------------------------------------
      dyV1[0] = DyVx[4];
      dyV1[1] = DyVy[4];
      dyV1[2] = DyVz[4];

      matmul3x1(matT1toVxfp, dtT1, out1);
      matmul3x1(matVytoVxfp, dyV1, out2);

      DxVx[4] = out1[0] - out2[0];
      DxVy[4] = out1[1] - out2[1];
      DxVz[4] = out1[2] - out2[2];
#endif
#ifdef DxV_OSD
    // calculate forward on the plus side directly
      if(F.faultgrid[k*nj+j + mpifaultsize]){
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = ( w_Vx[pos] - f_Vx[pos1] )*rDH;
      DxVy[4] = ( w_Vy[pos] - f_Vy[pos1] )*rDH;
      DxVz[4] = ( w_Vz[pos] - f_Vz[pos1] )*rDH;
      }
      //DxVx[4] = DxVx[3];//( w_Vx[pos] - f_Vx[pos1] )*rDH;
      //DxVy[4] = DxVy[3];//( w_Vy[pos] - f_Vy[pos1] )*rDH;
      //DxVz[4] = DxVz[3];//( w_Vz[pos] - f_Vz[pos1] )*rDH;
      //DxVx[4] = rDH*(-37./18.*f_Vx[pos1]+35./9.*w_Vx[pos]-17./6.*w_Vx[pos+nyz]+11./9.*w_Vx[pos+2*nyz]-2./9.*w_Vx[pos+3*nyz]);
      //DxVy[4] = rDH*(-37./18.*f_Vy[pos1]+35./9.*w_Vy[pos]-17./6.*w_Vy[pos+nyz]+11./9.*w_Vy[pos+2*nyz]-2./9.*w_Vy[pos+3*nyz]);
      //DxVz[4] = rDH*(-37./18.*f_Vz[pos1]+35./9.*w_Vz[pos]-17./6.*w_Vz[pos+nyz]+11./9.*w_Vz[pos+2*nyz]-2./9.*w_Vz[pos+3*nyz]);
#endif
#ifdef DxV_OSD1
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = (-2.0*f_Vx[pos1]+3.0*w_Vx[pos]-w_Vx[pos+nyz])*rDH;
      DxVy[4] = (-2.0*f_Vy[pos1]+3.0*w_Vy[pos]-w_Vy[pos+nyz])*rDH;
      DxVz[4] = (-2.0*f_Vz[pos1]+3.0*w_Vz[pos]-w_Vz[pos+nyz])*rDH;
#endif
#ifdef DxV_NCFD
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos_f = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = FD06_1*f_Vx[pos_f]+
                FD06_2*w_Vx[pos]+
                FD06_3*w_Vx[pos+1*nyz]+
                FD06_4*w_Vx[pos+2*nyz]+
                FD06_5*w_Vx[pos+3*nyz]+
                FD06_6*w_Vx[pos+4*nyz]+
                FD06_7*w_Vx[pos+5*nyz];
      DxVy[4] = FD06_1*f_Vy[pos_f]+
                FD06_2*w_Vy[pos]+
                FD06_3*w_Vy[pos+1*nyz]+
                FD06_4*w_Vy[pos+2*nyz]+
                FD06_5*w_Vy[pos+3*nyz]+
                FD06_6*w_Vy[pos+4*nyz]+
                FD06_7*w_Vy[pos+5*nyz];
      DxVz[4] = FD06_1*f_Vz[pos_f]+
                FD06_2*w_Vz[pos]+
                FD06_3*w_Vz[pos+1*nyz]+
                FD06_4*w_Vz[pos+2*nyz]+
                FD06_5*w_Vz[pos+3*nyz]+
                FD06_6*w_Vz[pos+4*nyz]+
                FD06_7*w_Vz[pos+5*nyz];
      DxVx[4] *= rDH;
      DxVy[4] *= rDH;
      DxVz[4] *= rDH;
#endif

      dxV2[0] = DxVx[4];
      dxV2[1] = DxVy[4];
      dxV2[2] = DxVz[4];

      matmul3x1(matVx2Vz1, dxV1, out1);
      matmul3x1(matVy2Vz1, dyV1, out2);

      DzVx[3] = out1[0] + out2[0];
      DzVy[3] = out1[1] + out2[1];
      DzVz[3] = out1[2] + out2[2];

      matmul3x1(matVx2Vz2, dxV2, out1);
      matmul3x1(matVy2Vz2, dyV2, out2);

      DzVx[4] = out1[0] + out2[0];
      DzVy[4] = out1[1] + out2[1];
      DzVz[4] = out1[2] + out2[2];
      //}}}
    } // end of par.freenode and km==1


    if( par.freenode && (km==2 || km==3) ){
      if(km==2){
        // get DzV[3], DzV[4] {{{
        segment = ny;
        pos = j1 + k1 * ny + 0 * ny * nz;
        DzVx[3] = L22(f_Vx, pos, segment, FlagZ)*rDH;
        DzVy[3] = L22(f_Vy, pos, segment, FlagZ)*rDH;
        DzVz[3] = L22(f_Vz, pos, segment, FlagZ)*rDH;
        pos = j1 + k1 * ny + 1 * ny * nz;
        DzVx[4] = L22(f_Vx, pos, segment, FlagZ)*rDH;
        DzVy[4] = L22(f_Vy, pos, segment, FlagZ)*rDH;
        DzVz[4] = L22(f_Vz, pos, segment, FlagZ)*rDH;
        //}}}
      }else{ // km==3
        // get DzV[3], DzV[4] {{{
        segment = ny;
        pos = j1 + k1 * ny + 0 * ny * nz;
        DzVx[3] = L24(f_Vx, pos, segment, FlagZ)*rDH;
        DzVy[3] = L24(f_Vy, pos, segment, FlagZ)*rDH;
        DzVz[3] = L24(f_Vz, pos, segment, FlagZ)*rDH;
        pos = j1 + k1 * ny + 1 * ny * nz;
        DzVx[4] = L24(f_Vx, pos, segment, FlagZ)*rDH;
        DzVy[4] = L24(f_Vy, pos, segment, FlagZ)*rDH;
        DzVz[4] = L24(f_Vz, pos, segment, FlagZ)*rDH;
        //}}}
      }

      // get DxV[4] {{{
      idx = (j1 + k1 * ny)*3*3;
      for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < 3; jj++){
          matMin2Plus1[ii][jj] = F.matMin2Plus1[idx + ii*3 + jj + matsize2];
          matMin2Plus2[ii][jj] = F.matMin2Plus2[idx + ii*3 + jj + matsize2];
          matMin2Plus3[ii][jj] = F.matMin2Plus3[idx + ii*3 + jj + matsize2];
          matMin2Plus4[ii][jj] = F.matMin2Plus4[idx + ii*3 + jj + matsize2];
          matMin2Plus5[ii][jj] = F.matMin2Plus5[idx + ii*3 + jj + matsize2];

          matT1toVxm[ii][jj] = F.matT1toVxm[idx + ii*3 + jj + matsize2];
          matVytoVxm[ii][jj] = F.matVytoVxm[idx + ii*3 + jj + matsize2];
          matVztoVxm[ii][jj] = F.matVztoVxm[idx + ii*3 + jj + matsize2];
          matT1toVxp[ii][jj] = F.matT1toVxp[idx + ii*3 + jj + matsize2];
          matVytoVxp[ii][jj] = F.matVytoVxp[idx + ii*3 + jj + matsize2];
          matVztoVxp[ii][jj] = F.matVztoVxp[idx + ii*3 + jj + matsize2];
        }
      }

      dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
      dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
      dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                         dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                         dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                         dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

      matmul3x1(matMin2Plus1, dxV1, out1);
      matmul3x1(matMin2Plus2, dyV1, out2);
      matmul3x1(matMin2Plus3, dzV1, out3);
      matmul3x1(matMin2Plus4, dyV2, out4);
      matmul3x1(matMin2Plus5, dzV2, out5);

      DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
      DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
      DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
#ifdef DxV_T1
      // =======================================================
      if(F.faultgrid[j+k*nj + mpifaultsize]){
        dtT1[0] = F.hT11[j + k * nj + mpifaultsize];
        dtT1[1] = F.hT12[j + k * nj + mpifaultsize];
        dtT1[2] = F.hT13[j + k * nj + mpifaultsize];
      }else{
        dtT1[0] = 0.0;
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;
      }

      // Minus side --------------------------------------------
      dyV1[0] = DyVx[3];
      dyV1[1] = DyVy[3];
      dyV1[2] = DyVz[3];
      dzV1[0] = DzVx[3];
      dzV1[1] = DzVy[3];
      dzV1[2] = DzVz[3];

      matmul3x1(matT1toVxm, dtT1, out1);
      matmul3x1(matVytoVxm, dyV1, out2);
      matmul3x1(matVztoVxm, dzV1, out3);

      DxVx[3] = out1[0] - out2[0] - out3[0];
      DxVy[3] = out1[1] - out2[1] - out3[1];
      DxVz[3] = out1[2] - out2[2] - out3[2];

      // Plus side --------------------------------------------
      dyV1[0] = DyVx[4];
      dyV1[1] = DyVy[4];
      dyV1[2] = DyVz[4];
      dzV1[0] = DzVx[4];
      dzV1[1] = DzVy[4];
      dzV1[2] = DzVz[4];

      matmul3x1(matT1toVxp, dtT1, out1);
      matmul3x1(matVytoVxp, dyV1, out2);
      matmul3x1(matVztoVxp, dzV1, out3);

      DxVx[4] = out1[0] - out2[0] - out3[0];
      DxVy[4] = out1[1] - out2[1] - out3[1];
      DxVz[4] = out1[2] - out2[2] - out3[2];
#endif
#ifdef DxV_OSD
    // calculate forward on the plus side directly

      if(F.faultgrid[k*nj+j + mpifaultsize]){
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = ( w_Vx[pos] - f_Vx[pos1] )*rDH;
      DxVy[4] = ( w_Vy[pos] - f_Vy[pos1] )*rDH;
      DxVz[4] = ( w_Vz[pos] - f_Vz[pos1] )*rDH;
      }
      //DxVx[4] = DxVx[3];// ( w_Vx[pos] - f_Vx[pos1] )*rDH;
      //DxVy[4] = DxVy[3];// ( w_Vy[pos] - f_Vy[pos1] )*rDH;
      //DxVz[4] = DxVz[3];// ( w_Vz[pos] - f_Vz[pos1] )*rDH;
      //DxVx[4] = rDH*(-37./18.*f_Vx[pos1]+35./9.*w_Vx[pos]-17./6.*w_Vx[pos+nyz]+11./9.*w_Vx[pos+2*nyz]-2./9.*w_Vx[pos+3*nyz]);
      //DxVy[4] = rDH*(-37./18.*f_Vy[pos1]+35./9.*w_Vy[pos]-17./6.*w_Vy[pos+nyz]+11./9.*w_Vy[pos+2*nyz]-2./9.*w_Vy[pos+3*nyz]);
      //DxVz[4] = rDH*(-37./18.*f_Vz[pos1]+35./9.*w_Vz[pos]-17./6.*w_Vz[pos+nyz]+11./9.*w_Vz[pos+2*nyz]-2./9.*w_Vz[pos+3*nyz]);
#endif
#ifdef DxV_OSD1
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = (-2.0*f_Vx[pos1]+3.0*w_Vx[pos]-w_Vx[pos+nyz])*rDH;
      DxVy[4] = (-2.0*f_Vy[pos1]+3.0*w_Vy[pos]-w_Vy[pos+nyz])*rDH;
      DxVz[4] = (-2.0*f_Vz[pos1]+3.0*w_Vz[pos]-w_Vz[pos+nyz])*rDH;
#endif
#ifdef DxV_NCFD
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos_f = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = FD06_1*f_Vx[pos_f]+
                FD06_2*w_Vx[pos]+
                FD06_3*w_Vx[pos+1*nyz]+
                FD06_4*w_Vx[pos+2*nyz]+
                FD06_5*w_Vx[pos+3*nyz]+
                FD06_6*w_Vx[pos+4*nyz]+
                FD06_7*w_Vx[pos+5*nyz];
      DxVy[4] = FD06_1*f_Vy[pos_f]+
                FD06_2*w_Vy[pos]+
                FD06_3*w_Vy[pos+1*nyz]+
                FD06_4*w_Vy[pos+2*nyz]+
                FD06_5*w_Vy[pos+3*nyz]+
                FD06_6*w_Vy[pos+4*nyz]+
                FD06_7*w_Vy[pos+5*nyz];
      DxVz[4] = FD06_1*f_Vz[pos_f]+
                FD06_2*w_Vz[pos]+
                FD06_3*w_Vz[pos+1*nyz]+
                FD06_4*w_Vz[pos+2*nyz]+
                FD06_5*w_Vz[pos+3*nyz]+
                FD06_6*w_Vz[pos+4*nyz]+
                FD06_7*w_Vz[pos+5*nyz];
      DxVx[4] *= rDH;
      DxVy[4] *= rDH;
      DxVz[4] *= rDH;
#endif
      //}}}
    } // par.freenode and (km== 2 or 3)

    if(!(par.freenode && km<=3)){
      // get DzV[3], DzV[4], DxV[4] {{{
      pos0 = j1 + k1 * ny + 0 * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      segment = ny;
#ifdef RupSensor
      if(F.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){
#else
      if(F.rup_index_z[j + k * nj + mpifaultsize] % 7){
#endif
        DzVx[3] = L22(f_Vx, pos0, segment, FlagZ)*rDH;
        DzVy[3] = L22(f_Vy, pos0, segment, FlagZ)*rDH;
        DzVz[3] = L22(f_Vz, pos0, segment, FlagZ)*rDH;
        DzVx[4] = L22(f_Vx, pos1, segment, FlagZ)*rDH;
        DzVy[4] = L22(f_Vy, pos1, segment, FlagZ)*rDH;
        DzVz[4] = L22(f_Vz, pos1, segment, FlagZ)*rDH;
      }else{
        DzVx[3] = L(f_Vx, pos0, segment, FlagZ)*rDH;
        DzVy[3] = L(f_Vy, pos0, segment, FlagZ)*rDH;
        DzVz[3] = L(f_Vz, pos0, segment, FlagZ)*rDH;
        DzVx[4] = L(f_Vx, pos1, segment, FlagZ)*rDH;
        DzVy[4] = L(f_Vy, pos1, segment, FlagZ)*rDH;
        DzVz[4] = L(f_Vz, pos1, segment, FlagZ)*rDH;
      }

      idx = (j1 + k1 * ny)*3*3;
      for (ii = 0; ii < 3; ii++){
        for (jj = 0; jj < 3; jj++){
          matMin2Plus1[ii][jj] = F.matMin2Plus1[idx + ii*3 + jj + matsize2];
          matMin2Plus2[ii][jj] = F.matMin2Plus2[idx + ii*3 + jj + matsize2];
          matMin2Plus3[ii][jj] = F.matMin2Plus3[idx + ii*3 + jj + matsize2];
          matMin2Plus4[ii][jj] = F.matMin2Plus4[idx + ii*3 + jj + matsize2];
          matMin2Plus5[ii][jj] = F.matMin2Plus5[idx + ii*3 + jj + matsize2];

          matT1toVxm[ii][jj] = F.matT1toVxm[idx + ii*3 + jj + matsize2];
          matVytoVxm[ii][jj] = F.matVytoVxm[idx + ii*3 + jj + matsize2];
          matVztoVxm[ii][jj] = F.matVztoVxm[idx + ii*3 + jj + matsize2];
          matT1toVxp[ii][jj] = F.matT1toVxp[idx + ii*3 + jj + matsize2];
          matVytoVxp[ii][jj] = F.matVytoVxp[idx + ii*3 + jj + matsize2];
          matVztoVxp[ii][jj] = F.matVztoVxp[idx + ii*3 + jj + matsize2];
        }
      }

      dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
      dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
      dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                         dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                         dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                         dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];
#ifdef DyzV_center
      dyV1[0] = DyVxc[3]; dzV1[0] = DzVxc[3];
      dyV1[1] = DyVyc[3]; dzV1[1] = DzVyc[3];
      dyV1[2] = DyVzc[3]; dzV1[2] = DzVzc[3];

      dyV2[0] = DyVxc[4]; dzV2[0] = DzVxc[4];
      dyV2[1] = DyVyc[4]; dzV2[1] = DzVyc[4];
      dyV2[2] = DyVzc[4]; dzV2[2] = DzVzc[4];
#endif


      matmul3x1(matMin2Plus1, dxV1, out1);
      matmul3x1(matMin2Plus2, dyV1, out2);
      matmul3x1(matMin2Plus3, dzV1, out3);
      matmul3x1(matMin2Plus4, dyV2, out4);
      matmul3x1(matMin2Plus5, dzV2, out5);

      DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
      DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
      DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
#ifdef DxV_T1
      // =======================================================
      if(F.faultgrid[j+k*nj + mpifaultsize]){
        dtT1[0] = F.hT11[j + k * nj + mpifaultsize];
        dtT1[1] = F.hT12[j + k * nj + mpifaultsize];
        dtT1[2] = F.hT13[j + k * nj + mpifaultsize];
      }else{
        dtT1[0] = 0.0;
        dtT1[1] = 0.0;
        dtT1[2] = 0.0;
      }

      // Minus side --------------------------------------------
      dyV1[0] = DyVx[3];
      dyV1[1] = DyVy[3];
      dyV1[2] = DyVz[3];
      dzV1[0] = DzVx[3];
      dzV1[1] = DzVy[3];
      dzV1[2] = DzVz[3];

      matmul3x1(matT1toVxm, dtT1, out1);
      matmul3x1(matVytoVxm, dyV1, out2);
      matmul3x1(matVztoVxm, dzV1, out3);

      DxVx[3] = out1[0] - out2[0] - out3[0];
      DxVy[3] = out1[1] - out2[1] - out3[1];
      DxVz[3] = out1[2] - out2[2] - out3[2];

      // Plus side --------------------------------------------
      dyV1[0] = DyVx[4];
      dyV1[1] = DyVy[4];
      dyV1[2] = DyVz[4];
      dzV1[0] = DzVx[4];
      dzV1[1] = DzVy[4];
      dzV1[2] = DzVz[4];

      matmul3x1(matT1toVxp, dtT1, out1);
      matmul3x1(matVytoVxp, dyV1, out2);
      matmul3x1(matVztoVxp, dzV1, out3);

      DxVx[4] = out1[0] - out2[0] - out3[0];
      DxVy[4] = out1[1] - out2[1] - out3[1];
      DxVz[4] = out1[2] - out2[2] - out3[2];
#endif
      //}}}
#ifdef DxV_OSD
    // calculate forward on the plus side directly
      if(F.faultgrid[k*nj+j + mpifaultsize]){
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = ( w_Vx[pos] - f_Vx[pos1] )*rDH;
      DxVy[4] = ( w_Vy[pos] - f_Vy[pos1] )*rDH;
      DxVz[4] = ( w_Vz[pos] - f_Vz[pos1] )*rDH;
      }
      //DxVx[4] = DxVx[3];//( w_Vx[pos] - f_Vx[pos1] )*rDH;
      //DxVy[4] = DxVy[3];//( w_Vy[pos] - f_Vy[pos1] )*rDH;
      //DxVz[4] = DxVz[3];//( w_Vz[pos] - f_Vz[pos1] )*rDH;
      //DxVx[4] = rDH*(-37./18.*f_Vx[pos1]+35./9.*w_Vx[pos]-17./6.*w_Vx[pos+nyz]+11./9.*w_Vx[pos+2*nyz]-2./9.*w_Vx[pos+3*nyz]);
      //DxVy[4] = rDH*(-37./18.*f_Vy[pos1]+35./9.*w_Vy[pos]-17./6.*w_Vy[pos+nyz]+11./9.*w_Vy[pos+2*nyz]-2./9.*w_Vy[pos+3*nyz]);
      //DxVz[4] = rDH*(-37./18.*f_Vz[pos1]+35./9.*w_Vz[pos]-17./6.*w_Vz[pos+nyz]+11./9.*w_Vz[pos+2*nyz]-2./9.*w_Vz[pos+3*nyz]);
#endif
#ifdef DxV_OSD1
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = (-2.0*f_Vx[pos1]+3.0*w_Vx[pos]-w_Vx[pos+nyz])*rDH;
      DxVy[4] = (-2.0*f_Vy[pos1]+3.0*w_Vy[pos]-w_Vy[pos+nyz])*rDH;
      DxVz[4] = (-2.0*f_Vz[pos1]+3.0*w_Vz[pos]-w_Vz[pos+nyz])*rDH;
#endif
#ifdef DxV_NCFD
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      pos_f = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = FD06_1*f_Vx[pos_f]+
                FD06_2*w_Vx[pos]+
                FD06_3*w_Vx[pos+1*nyz]+
                FD06_4*w_Vx[pos+2*nyz]+
                FD06_5*w_Vx[pos+3*nyz]+
                FD06_6*w_Vx[pos+4*nyz]+
                FD06_7*w_Vx[pos+5*nyz];
      DxVy[4] = FD06_1*f_Vy[pos_f]+
                FD06_2*w_Vy[pos]+
                FD06_3*w_Vy[pos+1*nyz]+
                FD06_4*w_Vy[pos+2*nyz]+
                FD06_5*w_Vy[pos+3*nyz]+
                FD06_6*w_Vy[pos+4*nyz]+
                FD06_7*w_Vy[pos+5*nyz];
      DxVz[4] = FD06_1*f_Vz[pos_f]+
                FD06_2*w_Vz[pos]+
                FD06_3*w_Vz[pos+1*nyz]+
                FD06_4*w_Vz[pos+2*nyz]+
                FD06_5*w_Vz[pos+3*nyz]+
                FD06_6*w_Vz[pos+4*nyz]+
                FD06_7*w_Vz[pos+5*nyz];
      DxVx[4] *= rDH;
      DxVy[4] *= rDH;
      DxVz[4] *= rDH;
#endif
    } // end of !(par.freenode and km<=3)

    //############################### Minus ####################################
    vec1[0] = DxVx[3]; vec1[1] = DxVy[3]; vec1[2] = DxVz[3];
    vec2[0] = DyVx[3]; vec2[1] = DyVy[3]; vec2[2] = DyVz[3];
    vec3[0] = DzVx[3]; vec3[1] = DzVy[3]; vec3[2] = DzVz[3];
#ifdef DxV_hT1
    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D11_1[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D12_1[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D13_1[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    real_t hT11m = vecg1[0] + vecg2[0] + vecg3[0];
    real_t hT12m = vecg1[1] + vecg2[1] + vecg3[1];
    real_t hT13m = vecg1[2] + vecg2[2] + vecg3[2];
#endif
    // calculate F->hT2, F->hT3 on the Minus side {{{
    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D21_1[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D22_1[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D23_1[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    pos0 = j1 + k1 * ny + 0 * ny * nz;
    f_hT21[pos0] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[pos0] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[pos0] = vecg1[2] + vecg2[2] + vecg3[2];

    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D31_1[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D32_1[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D33_1[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    pos0 = j1 + k1 * ny + 0 * ny * nz;
    f_hT31[pos0] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[pos0] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[pos0] = vecg1[2] + vecg2[2] + vecg3[2];
    // }}}

    //############################### Plus  ####################################
    vec1[0] = DxVx[4]; vec1[1] = DxVy[4]; vec1[2] = DxVz[4];
    vec2[0] = DyVx[4]; vec2[1] = DyVy[4]; vec2[2] = DyVz[4];
    vec3[0] = DzVx[4]; vec3[1] = DzVy[4]; vec3[2] = DzVz[4];
#ifdef DxV_hT1
    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D11_2[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D12_2[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D13_2[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    real_t hT11p = vecg1[0] + vecg2[0] + vecg3[0];
    real_t hT12p = vecg1[1] + vecg2[1] + vecg3[1];
    real_t hT13p = vecg1[2] + vecg2[2] + vecg3[2];

    F.hT11[j + k * nj + mpifaultsize] = 0.5*(hT11m + hT11p);
    F.hT12[j + k * nj + mpifaultsize] = 0.5*(hT12m + hT12p);
    F.hT13[j + k * nj + mpifaultsize] = 0.5*(hT13m + hT13p);
#endif
    // calculate F->hT2, F->hT3 on the Plus side {{{
    ////////////////////////////////////////////////////////////////////////////
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D21_2[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D22_2[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D23_2[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    pos1 = j1 + k1 * ny + 1 * ny * nz;
    f_hT21[pos1] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[pos1] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[pos1] = vecg1[2] + vecg2[2] + vecg3[2];

    ////////////////////////////////////////////////////////////////////////////
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D31_2[idx + ii*3 + jj + Dsize];
        mat2[ii][jj] = F.D32_2[idx + ii*3 + jj + Dsize];
        mat3[ii][jj] = F.D33_2[idx + ii*3 + jj + Dsize];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    pos1 = j1 + k1 * ny + 1 * ny * nz;
    f_hT31[pos1] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[pos1] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[pos1] = vecg1[2] + vecg2[2] + vecg3[2];
    // Split nodes  }}}

    // n = 5; l = +1; get DxV[5]
    pos = j1 + k1 * ny + 1 * ny * nz;
    pos1 = j1 + k1 * ny + (i0+1) * ny * nz;
#ifdef VHOC
    pos = j1 + k1 * ny + (i0+1) * ny * nz;
    pos_f = j1 + k1 * ny + 1 * ny * nz;
    slice = ny*nz;
    DxVx[5] = rDH*compact_a1*(w_Vx[pos]-f_Vx[pos_f])-compact_a2*DxVx[4];
    DxVy[5] = rDH*compact_a1*(w_Vy[pos]-f_Vy[pos_f])-compact_a2*DxVy[4];
    DxVz[5] = rDH*compact_a1*(w_Vz[pos]-f_Vz[pos_f])-compact_a2*DxVz[4];
    DxVx[5] = rDH*(0.25*w_Vx[pos+nyz]+w_Vx[pos]-1.25*f_Vx[pos_f])-0.5*DxVx[4];
    DxVy[5] = rDH*(0.25*w_Vy[pos+nyz]+w_Vy[pos]-1.25*f_Vy[pos_f])-0.5*DxVy[4];
    DxVz[5] = rDH*(0.25*w_Vz[pos+nyz]+w_Vz[pos]-1.25*f_Vz[pos_f])-0.5*DxVz[4];
    if(par.freenode && km<=3){
    DxVx[5] = (w_Vx[pos1] - f_Vx[pos])*rDH; // backward
    DxVy[5] = (w_Vy[pos1] - f_Vy[pos])*rDH; // backward
    DxVz[5] = (w_Vz[pos1] - f_Vz[pos])*rDH; // backward
    }
#else
    DxVx[5] = (w_Vx[pos1] - f_Vx[pos])*rDH; // backward
    DxVy[5] = (w_Vy[pos1] - f_Vy[pos])*rDH; // backward
    DxVz[5] = (w_Vz[pos1] - f_Vz[pos])*rDH; // backward
#endif
#ifdef DxV_NCFD
    pos = j1 + k1 * ny + (i0+1) * ny * nz;
    pos_f = j1 + k1 * ny + 1 * ny * nz;
    DxVx[5] = FD15_1*f_Vx[pos_f]+
              FD15_2*w_Vx[pos]+
              FD15_3*w_Vx[pos+1*nyz]+
              FD15_4*w_Vx[pos+2*nyz]+
              FD15_5*w_Vx[pos+3*nyz]+
              FD15_6*w_Vx[pos+4*nyz]+
              FD15_7*w_Vx[pos+5*nyz];
    DxVy[5] = FD15_1*f_Vy[pos_f]+
              FD15_2*w_Vy[pos]+
              FD15_3*w_Vy[pos+1*nyz]+
              FD15_4*w_Vy[pos+2*nyz]+
              FD15_5*w_Vy[pos+3*nyz]+
              FD15_6*w_Vy[pos+4*nyz]+
              FD15_7*w_Vy[pos+5*nyz];
    DxVz[5] = FD15_1*f_Vz[pos_f]+
              FD15_2*w_Vz[pos]+
              FD15_3*w_Vz[pos+1*nyz]+
              FD15_4*w_Vz[pos+2*nyz]+
              FD15_5*w_Vz[pos+3*nyz]+
              FD15_6*w_Vz[pos+4*nyz]+
              FD15_7*w_Vz[pos+5*nyz];
    DxVx[5] *= rDH;
    DxVy[5] *= rDH;
    DxVz[5] *= rDH;
#endif

    //n = 6; l = +2; get DxV[6]
#ifdef VHOC
    pos = j1 + k1 * ny + (i0+2) * ny * nz;
    slice = ny * nz;
    DxVx[6] = rDH*compact_a1*(w_Vx[pos]-w_Vx[pos-slice])-compact_a2*DxVx[5];
    DxVy[6] = rDH*compact_a1*(w_Vy[pos]-w_Vy[pos-slice])-compact_a2*DxVy[5];
    DxVz[6] = rDH*compact_a1*(w_Vz[pos]-w_Vz[pos-slice])-compact_a2*DxVz[5];
    DxVx[6] = rDH*(0.25*w_Vx[pos+nyz]+w_Vx[pos]-1.25*w_Vx[pos-nyz])-0.5*DxVx[5];
    DxVy[6] = rDH*(0.25*w_Vy[pos+nyz]+w_Vy[pos]-1.25*w_Vy[pos-nyz])-0.5*DxVy[5];
    DxVz[6] = rDH*(0.25*w_Vz[pos+nyz]+w_Vz[pos]-1.25*w_Vz[pos-nyz])-0.5*DxVz[5];
    if(par.freenode && km<=3){
    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vx[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vx[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vx[pos];
    DxVx[6] = vec_L24B(vec_3,2)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vy[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vy[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vy[pos];
    DxVy[6] = vec_L24B(vec_3,2)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vz[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vz[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vz[pos];
    DxVz[6] = vec_L24B(vec_3,2)*rDH;
    }
#else
    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vx[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vx[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vx[pos];
    DxVx[6] = vec_L24B(vec_3,2)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vy[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vy[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vy[pos];
    DxVy[6] = vec_L24B(vec_3,2)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vz[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vz[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vz[pos];
    DxVz[6] = vec_L24B(vec_3,2)*rDH;
#endif
#ifdef DxV_NCFD
    pos = j1 + k1 * ny + (i0+1) * ny * nz;
    pos_f = j1 + k1 * ny + 1 * ny * nz;
    DxVx[6] = FD24_1*f_Vx[pos_f]+
              FD24_2*w_Vx[pos]+
              FD24_3*w_Vx[pos+1*nyz]+
              FD24_4*w_Vx[pos+2*nyz]+
              FD24_5*w_Vx[pos+3*nyz]+
              FD24_6*w_Vx[pos+4*nyz]+
              FD24_7*w_Vx[pos+5*nyz];
    DxVy[6] = FD24_1*f_Vy[pos_f]+
              FD24_2*w_Vy[pos]+
              FD24_3*w_Vy[pos+1*nyz]+
              FD24_4*w_Vy[pos+2*nyz]+
              FD24_5*w_Vy[pos+3*nyz]+
              FD24_6*w_Vy[pos+4*nyz]+
              FD24_7*w_Vy[pos+5*nyz];
    DxVz[6] = FD24_1*f_Vz[pos_f]+
              FD24_2*w_Vz[pos]+
              FD24_3*w_Vz[pos+1*nyz]+
              FD24_4*w_Vz[pos+2*nyz]+
              FD24_5*w_Vz[pos+3*nyz]+
              FD24_6*w_Vz[pos+4*nyz]+
              FD24_7*w_Vz[pos+5*nyz];
    DxVx[6] *= rDH;
    DxVy[6] *= rDH;
    DxVz[6] *= rDH;
#endif

    //n = 7; l = +3; get DxV[7]
    pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vx[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vx[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vx[pos];
    pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vx[pos];
    pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vx[pos];
    DxVx[7] = vec_LB(vec_5,3)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vy[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vy[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vy[pos];
    pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vy[pos];
    pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vy[pos];
    DxVy[7] = vec_LB(vec_5,3)*rDH;

    pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vz[pos];
    pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vz[pos];
    pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vz[pos];
    pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vz[pos];
    pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vz[pos];
    DxVz[7] = vec_LB(vec_5,3)*rDH;

    // update surrounding points {{{
    for (n = 0; n < 8 ; n++){
      // n =  0  1  2  3  4  5  6  7
      // l = -3 -2 -1 -0 +0 +1 +2 +3
      if(n < 4){
        l = n-3;
      }else{
        l = n-4;
      }
      if(l==0) continue; // do not update i0
      i = i0+l;

      pos = j1 + k1 * ny + i * ny * nz;
      DyVx[n] = L(w_Vx, pos, 1, FlagY)*rDH;
      DyVy[n] = L(w_Vy, pos, 1, FlagY)*rDH;
      DyVz[n] = L(w_Vz, pos, 1, FlagY)*rDH;

      // get Dz by Dx, Dy or directly
      idx = (j1 + i * ny)*9;

      if(par.freenode && km==1){
        DzVx[n] = W.matVx2Vz[idx + 0*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 0*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 0*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 0*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 0*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 0*3 + 2] * DyVz[n];

        DzVy[n] = W.matVx2Vz[idx + 1*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 1*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 1*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 1*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 1*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 1*3 + 2] * DyVz[n];

        DzVz[n] = W.matVx2Vz[idx + 2*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 2*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 2*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 2*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 2*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 2*3 + 2] * DyVz[n] ;
      }else if(par.freenode && km==2){
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L22(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L22(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L22(w_Vz, pos, ny, FlagZ)*rDH;
      }else if(par.freenode && km==3){
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L24(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L24(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L24(w_Vz, pos, ny, FlagZ)*rDH;
      }else{
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L(w_Vz, pos, ny, FlagZ)*rDH;
      }

      pos = j1 + k1 * ny + i * ny * nz;
      lam = LAM[pos]; mu = MIU[pos];
      lam2mu  = lam + 2.0f*mu;
      xix = XIX[pos]; xiy = XIY[pos]; xiz = XIZ[pos];
      etx = ETX[pos]; ety = ETY[pos]; etz = ETZ[pos];
      ztx = ZTX[pos]; zty = ZTY[pos]; ztz = ZTZ[pos];

      w_hTxx[pos] = (
          lam2mu*xix*DxVx[n] + lam*xiy*DxVy[n] + lam*xiz*DxVz[n] +
          lam2mu*etx*DyVx[n] + lam*ety*DyVy[n] + lam*etz*DyVz[n] +
          lam2mu*ztx*DzVx[n] + lam*zty*DzVy[n] + lam*ztz*DzVz[n] );

      w_hTyy[pos] = (
          lam*xix*DxVx[n] + lam2mu*xiy*DxVy[n] + lam*xiz*DxVz[n] +
          lam*etx*DyVx[n] + lam2mu*ety*DyVy[n] + lam*etz*DyVz[n] +
          lam*ztx*DzVx[n] + lam2mu*zty*DzVy[n] + lam*ztz*DzVz[n] );

      w_hTzz[pos] = (
          lam*xix*DxVx[n] + lam*xiy*DxVy[n] + lam2mu*xiz*DxVz[n] +
          lam*etx*DyVx[n] + lam*ety*DyVy[n] + lam2mu*etz*DyVz[n] +
          lam*ztx*DzVx[n] + lam*zty*DzVy[n] + lam2mu*ztz*DzVz[n] );

      w_hTxy[pos] = (
          xiy*DxVx[n] + xix*DxVy[n] +
          ety*DyVx[n] + etx*DyVy[n] +
          zty*DzVx[n] + ztx*DzVy[n] ) * mu;

      w_hTxz[pos] = (
          xiz*DxVx[n] + xix*DxVz[n] +
          etz*DyVx[n] + etx*DyVz[n] +
          ztz*DzVx[n] + ztx*DzVz[n] ) * mu;

      w_hTyz[pos] = (
          xiz*DxVy[n] + xiy*DxVz[n] +
          etz*DyVy[n] + ety*DyVz[n] +
          ztz*DzVy[n] + zty*DzVz[n] ) * mu;

    } // end loop of n   }}}

  } // end j k
  return;
}

void fault_dstrs_b(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ, int i0, int nfault)
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj + block.x - 1) / block.x,
      (hostParams.nk + block.y - 1) / block.y,
      1);
  fault_dstrs_b_cu <<<grid, block>>> (W, F, M, FlagX, FlagY, FlagZ, i0, nfault);
  return;
}
