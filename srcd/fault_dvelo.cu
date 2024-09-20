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
void fault_dvelo_cu(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ, int i0, int nfault)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int i;
  int nj = par.nj;
  int nk = par.nk;
///////////////////////////////////
  int gj = par.ranky * nj + j;
  int gk = par.rankz * nk + k;
  int gj1 = gj + 1;
  int gk1 = gk + 1;
///////////////////////////////////
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

  //real_t *w_Vx  = W.W + 0 * stride;
  //real_t *w_Vy  = W.W + 1 * stride;
  //real_t *w_Vz  = W.W + 2 * stride;
  real_t *w_Txx = W.W + 3 * stride;
  real_t *w_Tyy = W.W + 4 * stride;
  real_t *w_Tzz = W.W + 5 * stride;
  real_t *w_Txy = W.W + 6 * stride;
  real_t *w_Txz = W.W + 7 * stride;
  real_t *w_Tyz = W.W + 8 * stride;

  real_t *w_hVx  = W.hW + 0 * stride;
  real_t *w_hVy  = W.hW + 1 * stride;
  real_t *w_hVz  = W.hW + 2 * stride;
  //real_t *w_hTxx = W.hW + 3 * stride;
  //real_t *w_hTyy = W.hW + 4 * stride;
  //real_t *w_hTzz = W.hW + 5 * stride;
  //real_t *w_hTxy = W.hW + 6 * stride;
  //real_t *w_hTxz = W.hW + 7 * stride;
  //real_t *w_hTyz = W.hW + 8 * stride;

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
  real_t *JAC = M + 9 * stride;
  //real_t *LAM = M + 10 * stride;
  //real_t *MIU = M + 11 * stride;
  real_t *RHO = M + 12 * stride;

  // Split nodes
  //stride = ny * nz * 2; // y vary first

  // INPUT
  //real_t *f_Vx  = F.W + 0 * nyz2;
  //real_t *f_Vy  = F.W + 1 * nyz2;
  //real_t *f_Vz  = F.W + 2 * nyz2;
  real_t *f_T21 = F.W + 3 * nyz2 + faultsize;
  real_t *f_T22 = F.W + 4 * nyz2 + faultsize;
  real_t *f_T23 = F.W + 5 * nyz2 + faultsize;
  real_t *f_T31 = F.W + 6 * nyz2 + faultsize;
  real_t *f_T32 = F.W + 7 * nyz2 + faultsize;
  real_t *f_T33 = F.W + 8 * nyz2 + faultsize;

  real_t *f_hVx  = F.hW + 0 * nyz2 + faultsize;
  real_t *f_hVy  = F.hW + 1 * nyz2 + faultsize;
  real_t *f_hVz  = F.hW + 2 * nyz2 + faultsize;
  //real_t *f_hT21 = F.hW + 3 * nyz2;
  //real_t *f_hT22 = F.hW + 4 * nyz2;
  //real_t *f_hT23 = F.hW + 5 * nyz2;
  //real_t *f_hT31 = F.hW + 6 * nyz2;
  //real_t *f_hT32 = F.hW + 7 * nyz2;
  //real_t *f_hT33 = F.hW + 8 * nyz2;

  //real_t xix, xiy, xiz;
  //real_t etx, ety, etz;
  //real_t ztx, zty, ztz;

  //real_t mu, lam, lam2mu;
  real_t rrhojac;

  //real_t DxVx[8],DxVy[8],DxVz[8];
  //real_t DyVx[8],DyVy[8],DyVz[8];
  //real_t DzVx[8],DzVy[8],DzVz[8];

  real_t vecT11[7], vecT12[7], vecT13[7];
  real_t vecT21[7], vecT22[7], vecT23[7];
  real_t vecT31[7], vecT32[7], vecT33[7];
  real_t DxTx[3],DyTy[3],DzTz[3];

  //real_t vec_3[3], vec_5[5];
  //real_t mat1[3][3], mat2[3][3], mat3[3][3];
  //real_t vec1[3], vec2[3], vec3[3];
  //real_t vecg1[3], vecg2[3], vecg3[3];

  //real_t matMin2Plus1[3][3];
  //real_t matMin2Plus2[3][3];
  //real_t matMin2Plus3[3][3];
  //real_t matMin2Plus4[3][3];
  //real_t matMin2Plus5[3][3];
  //real_t matPlus2Min1[3][3];
  //real_t matPlus2Min2[3][3];
  //real_t matPlus2Min3[3][3];
  //real_t matPlus2Min4[3][3];
  //real_t matPlus2Min5[3][3];
  //real_t dxV1[3], dyV1[3], dzV1[3];
  //real_t dxV2[3], dyV2[3], dzV2[3];
  //real_t out1[3], out2[3], out3[3], out4[3], out5[3];
  //real_t matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  //real_t matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  //real_t matVx2Vz1[3][3], matVy2Vz1[3][3];
  //real_t matVx2Vz2[3][3], matVy2Vz2[3][3];

  //int ii, jj, mm, l, n;
  int mm, l, n;

  //int pos, pos_m, slice, segment;
  int pos, pos_m;
  int pos0, pos1, pos2;
  int pos_f = j1 + k1*ny + nfault * (nyz*7);   //*********wangzj
  //int idx;

  //int i0 = nx/2;                  ***************wangzj

  if (j < nj && k < nk ) { 
    if(F.united[j + k * nj + mpifaultsize]) return;
    //  km = NZ -(thisid[2]*nk+k-3);
    //int km = nz - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    // update velocity at surrounding points
    for (i = i0-3; i <= i0+3; i++) {
      //n = i - i0; // -3, -2, -1, +1, +2, +3, not at fault plane
      n = i0 - i; // +3, +2, +1, -1, -2, -3, not at fault plane
      if(n==0) continue; // skip Split nodes

      for (l = -3; l <= 3; l++){

        pos = j1 + k1 * ny + (i+l) * ny * nz;
        vecT11[l+3] = JAC[pos]*(XIX[pos]*w_Txx[pos] + XIY[pos]*w_Txy[pos] + XIZ[pos]*w_Txz[pos]);
        vecT12[l+3] = JAC[pos]*(XIX[pos]*w_Txy[pos] + XIY[pos]*w_Tyy[pos] + XIZ[pos]*w_Tyz[pos]);
        vecT13[l+3] = JAC[pos]*(XIX[pos]*w_Txz[pos] + XIY[pos]*w_Tyz[pos] + XIZ[pos]*w_Tzz[pos]);

        pos = (j1+l) + k1 * ny + i * ny * nz;
        vecT21[l+3] = JAC[pos]*(ETX[pos]*w_Txx[pos] + ETY[pos]*w_Txy[pos] + ETZ[pos]*w_Txz[pos]);
        vecT22[l+3] = JAC[pos]*(ETX[pos]*w_Txy[pos] + ETY[pos]*w_Tyy[pos] + ETZ[pos]*w_Tyz[pos]);
        vecT23[l+3] = JAC[pos]*(ETX[pos]*w_Txz[pos] + ETY[pos]*w_Tyz[pos] + ETZ[pos]*w_Tzz[pos]);

        pos = j1 + (k1+l) * ny + i * ny * nz;
        vecT31[l+3] = JAC[pos]*(ZTX[pos]*w_Txx[pos] + ZTY[pos]*w_Txy[pos] + ZTZ[pos]*w_Txz[pos]);
        vecT32[l+3] = JAC[pos]*(ZTX[pos]*w_Txy[pos] + ZTY[pos]*w_Tyy[pos] + ZTZ[pos]*w_Tyz[pos]);
        vecT33[l+3] = JAC[pos]*(ZTX[pos]*w_Txz[pos] + ZTY[pos]*w_Tyz[pos] + ZTZ[pos]*w_Tzz[pos]);
      }

      //pos = j1 + k1 * ny + 1 * ny * nz;
      pos = j1 + k1 * ny + 3 * ny * nz + nfault * (nyz*7);
      vecT11[n+3] = F.T11[pos];// F->T11[1][j][k];
      vecT12[n+3] = F.T12[pos];// F->T12[1][j][k];
      vecT13[n+3] = F.T13[pos];// F->T13[1][j][k];

//#ifdef TractionLow
      // reduce order
      if(abs(n)==1){
        DxTx[0] = vec_L22(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L22(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L22(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==2){
        DxTx[0] = vec_L24(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L24(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L24(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==3){
        DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
      }
//#endif
#ifdef TractionImg
      if (n==2) { // i0-2
        vecT11[6] = 2.0*vecT11[5] - vecT11[4];
        vecT12[6] = 2.0*vecT12[5] - vecT12[4];
        vecT13[6] = 2.0*vecT13[5] - vecT13[4];
      }
      if (n==1) { // i0-1
        vecT11[5] = 2.0*vecT11[4] - vecT11[3];
        vecT12[5] = 2.0*vecT12[4] - vecT12[3];
        vecT13[5] = 2.0*vecT13[4] - vecT13[3];
        vecT11[6] = 2.0*vecT11[4] - vecT11[2];
        vecT12[6] = 2.0*vecT12[4] - vecT12[2];
        vecT13[6] = 2.0*vecT13[4] - vecT13[2];
      }
      if (n==-1) { // i0+1
        vecT11[0] = 2.0*vecT11[2] - vecT11[4];
        vecT12[0] = 2.0*vecT12[2] - vecT12[4];
        vecT13[0] = 2.0*vecT13[2] - vecT13[4];
        vecT11[1] = 2.0*vecT11[2] - vecT11[3];
        vecT12[1] = 2.0*vecT12[2] - vecT12[3];
        vecT13[1] = 2.0*vecT13[2] - vecT13[3];
      }
      if (n==-2) { // i0+2
        vecT11[0] = 2.0*vecT11[1] - vecT11[2];
        vecT12[0] = 2.0*vecT12[1] - vecT12[2];
        vecT13[0] = 2.0*vecT13[1] - vecT13[2];
      }

      DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
      DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
      DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
#endif
      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }
      }

      DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
      DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
      DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
      DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
      DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      pos = j1 + k1 * ny + i * ny * nz;

      rrhojac = 1.0 / (RHO[pos] * JAC[pos]);
      w_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      w_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      w_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;

    } // end of loop i

    // update velocity at the fault plane
    // 0 for minus side on the fault
    // 1 for plus  side on the fault
    for (mm = 0; mm < 2; mm++){
      //km = NZ -(thisid[2]*nk+k-3);
      //rrhojac = F->rrhojac_f[mm][j][k];

//#ifdef TractionLow
      pos0 = j1 + k1 * ny + (3-1) * ny * nz + nfault * (nyz*7);
      pos1 = j1 + k1 * ny + (3  ) * ny * nz + nfault * (nyz*7);
      pos2 = j1 + k1 * ny + (3+1) * ny * nz + nfault * (nyz*7);
      if(mm==0){
        DxTx[0] = (F.T11[pos1] - F.T11[pos0])*rDH;
        DxTx[1] = (F.T12[pos1] - F.T12[pos0])*rDH;
        DxTx[2] = (F.T13[pos1] - F.T13[pos0])*rDH;
      }else{
        DxTx[0] = (F.T11[pos2] - F.T11[pos1])*rDH;
        DxTx[1] = (F.T12[pos2] - F.T12[pos1])*rDH;
        DxTx[2] = (F.T13[pos2] - F.T13[pos1])*rDH;
      }
//#endif
#ifdef TractionImg
      
      real_t a0p,a0m;
      if(FlagX==FWD){
        a0p = a_0pF;
        a0m = a_0mF;
      }else{
        a0p = a_0pB;
        a0m = a_0mB;
      }
      if(mm==0){ // "-" side
        DxTx[0] = rDH*(
            a0m*F.T11[3*nyz + pos_f] -
            a_1*F.T11[2*nyz + pos_f] -
            a_2*F.T11[1*nyz + pos_f] -
            a_3*F.T11[0*nyz + pos_f] );
        DxTx[1] = rDH*(
            a0m*F.T12[3*nyz + pos_f] -
            a_1*F.T12[2*nyz + pos_f] -
            a_2*F.T12[1*nyz + pos_f] -
            a_3*F.T12[0*nyz + pos_f] );
        DxTx[2] = rDH*(
            a0m*F.T13[3*nyz + pos_f] -
            a_1*F.T13[2*nyz + pos_f] -
            a_2*F.T13[1*nyz + pos_f] -
            a_3*F.T13[0*nyz + pos_f] );
      }else{ // "+" side
        DxTx[0] = rDH*(
            a0p*F.T11[3*nyz + pos_f] +
            a_1*F.T11[4*nyz + pos_f] +
            a_2*F.T11[5*nyz + pos_f] +
            a_3*F.T11[6*nyz + pos_f] );
        DxTx[1] = rDH*(
            a0p*F.T12[3*nyz + pos_f] +
            a_1*F.T12[4*nyz + pos_f] +
            a_2*F.T12[5*nyz + pos_f] +
            a_3*F.T12[6*nyz + pos_f] );
        DxTx[2] = rDH*(
            a0p*F.T13[3*nyz + pos_f] +
            a_1*F.T13[4*nyz + pos_f] +
            a_2*F.T13[5*nyz + pos_f] +
            a_3*F.T13[6*nyz + pos_f] );
      }
#endif

      for (l = -3; l <= 3; l++){
        pos = (j1+l) + k1 * ny + mm * ny * nz;
        vecT21[l+3] = f_T21[pos];
        vecT22[l+3] = f_T22[pos];
        vecT23[l+3] = f_T23[pos];
        pos = j1 + (k1+l) * ny + mm * ny * nz;
        vecT31[l+3] = f_T31[pos];
        vecT32[l+3] = f_T32[pos];
        vecT33[l+3] = f_T33[pos];
      }

      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }

        DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
        DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
        DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      }else{
#ifdef RupSensor
        if(F.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){
#else
        if(F.rup_index_z[j + k * nj + mpifaultsize] % 7){
#endif
          DzTz[0] = vec_L22(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L22(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L22(vecT33,3,FlagZ)*rDH;
        }else{
          DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;
        }
      }

#ifdef RupSensor
      if(F.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){
#else
      if(F.rup_index_y[j + k * nj + mpifaultsize] % 7){
#endif
        DyTy[0] = vec_L22(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L22(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L22(vecT23,3,FlagY)*rDH;
      }else{
        DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      }

      pos_m = j1 + k1 * ny + i0 * ny * nz;
      pos = j1 + k1 * ny + mm * ny * nz + nfault * nyz2;      //wangzj
      rrhojac = 1.0 / (F.rho_f[pos] * JAC[pos_m]);
    // if( gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] &&
    //     gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault]){
     
      pos = j1 + k1 * ny + mm * ny * nz; // mm = 0, 1
      f_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      f_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      f_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;
      // }else{
      //   int pos1 = j1 + k1 * ny + 0 * ny * nz; // mm = 0, 1
      //   f_hVx[pos1] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      //   f_hVy[pos1] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      //   f_hVz[pos1] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;
      //   int pos2 = j1 + k1 * ny + 1 * ny * nz; // mm = 0, 1
      //   f_hVx[pos2] = f_hVx[pos1];
      //   f_hVy[pos2] = f_hVy[pos1];
      //   f_hVz[pos2] = f_hVz[pos1];
      // }
    } // end of loop mm  update fault plane

  } // end j k
  return;
}

void fault_dvelo(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ, int i0, int nfault)
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj + block.x - 1) / block.x,
      (hostParams.nk + block.y - 1) / block.y,
      1);
  fault_dvelo_cu <<<grid, block>>> (W, F, M, FlagX, FlagY, FlagZ, i0, nfault);
  return;
}
