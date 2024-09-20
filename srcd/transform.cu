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

__global__ void wave2fault_cu(Wave w, Fault f, realptr_t M, int i0, int nfault)
{
  return;
  // transform
  //          wave (Txx, Tyy, ..., Tyz, Vx, ... (at i0))
  // to
  //          fault (T11, T12, T13 (at i0), T21, ..., T31, ..., Vx, ...)
  // and
  // transform
  //          wave (hTxx, hTyy, ..., hTyz)
  // to
  //          fault (hT21, ..., hT31, ...)
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
  //real_t DH = par.DH;
  //real_t DT = par.DT;
  int stride = nx * ny * nz;
  int nyz = ny * nz;
  int nyz2 = nyz * 2;
  int faultsize = nfault * 9 * nyz2;  //******************wangzj

  // INPUT
  real_t *w_Vx  = w.W + 0 * stride;
  real_t *w_Vy  = w.W + 1 * stride;
  real_t *w_Vz  = w.W + 2 * stride;
  real_t *w_Txx = w.W + 3 * stride;
  real_t *w_Tyy = w.W + 4 * stride;
  real_t *w_Tzz = w.W + 5 * stride;
  real_t *w_Txy = w.W + 6 * stride;
  real_t *w_Txz = w.W + 7 * stride;
  real_t *w_Tyz = w.W + 8 * stride;

  real_t *w_hTxx = w.hW + 3 * stride;
  real_t *w_hTyy = w.hW + 4 * stride;
  real_t *w_hTzz = w.hW + 5 * stride;
  real_t *w_hTxy = w.hW + 6 * stride;
  real_t *w_hTxz = w.hW + 7 * stride;
  real_t *w_hTyz = w.hW + 8 * stride;

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
  //real_t *LAM = JAC + stride;
  //real_t *MIU = LAM + stride;
  //real_t *RHO = MIU + stride;

  // Split nodes
  //stride = ny * nz * 2; // y vary first

  // OUTPUT
  real_t *f_Vx  = f.W + 0 * nyz2 + faultsize;  //********wangzj
  real_t *f_Vy  = f.W + 1 * nyz2 + faultsize;
  real_t *f_Vz  = f.W + 2 * nyz2 + faultsize;
  real_t *f_T21 = f.W + 3 * nyz2 + faultsize;
  real_t *f_T22 = f.W + 4 * nyz2 + faultsize;
  real_t *f_T23 = f.W + 5 * nyz2 + faultsize;
  real_t *f_T31 = f.W + 6 * nyz2 + faultsize;
  real_t *f_T32 = f.W + 7 * nyz2 + faultsize;
  real_t *f_T33 = f.W + 8 * nyz2 + faultsize;

  real_t *f_hT21 = f.hW + 3 * nyz2 + faultsize;
  real_t *f_hT22 = f.hW + 4 * nyz2 + faultsize;
  real_t *f_hT23 = f.hW + 5 * nyz2 + faultsize;
  real_t *f_hT31 = f.hW + 6 * nyz2 + faultsize;
  real_t *f_hT32 = f.hW + 7 * nyz2 + faultsize;
  real_t *f_hT33 = f.hW + 8 * nyz2 + faultsize;

  real_t jac;
  real_t metric[3][3], stress[3][3], traction[3][3];
  // int i0  = nx/2; // local Nucleate source i index
  int pos, pos_f;

  int m;
//  if( j < ny - 3 && k < nz - 3){
//#ifdef FreeSurface
//    if(j < 30+3 || j > ny-31-3 || k < 30+3 ) { // united
//#else
//    if(j < 30+3 || j > ny-31-3 || k < 30+3 || k > nz-31-3) { // united
//#endif
  if( j < nj && k < nk){
//#ifdef FreeSurface
//    if(j < 30 || j > nj-31 || k < 30 ) { // united
//#else
//    if(j < 30 || j > nj-31 || k < 30 || k > nk-31) { // united
//#endif
    if(f.united[j + k * nj + nfault*nj*nk]) {
      //metric[0][0]=xi_x;metric[0][1]=et_x;metric[0][2]=zt_x;
      //metric[1][0]=xi_y;metric[1][1]=et_y;metric[1][2]=zt_y;
      //metric[2][0]=xi_z;metric[2][1]=et_z;metric[2][2]=zt_z;
      //pos = (i0*ny*nz + j1*nz + k1)*MSIZE;
      //metric[0][0]=M[pos+0];metric[0][1]=M[pos+3];metric[0][2]=M[pos+6];
      //metric[1][0]=M[pos+1];metric[1][1]=M[pos+4];metric[1][2]=M[pos+7];
      //metric[2][0]=M[pos+2];metric[2][1]=M[pos+5];metric[2][2]=M[pos+8];
      //real_t jac = M[pos+9];
      pos = j1 + k1 * ny + i0 * ny * nz;
      metric[0][0]=XIX[pos];metric[0][1]=ETX[pos];metric[0][2]=ZTX[pos];
      metric[1][0]=XIY[pos];metric[1][1]=ETY[pos];metric[1][2]=ZTY[pos];
      metric[2][0]=XIZ[pos];metric[2][1]=ETZ[pos];metric[2][2]=ZTZ[pos];
      jac = JAC[pos];

      // /Txx 3/ /Tyy 4/ /Tzz 5/ /Txy 6/ /Txz 7/ /Tyz 8/
      //stress[0][0]=Txx;stress[0][1]=Txy;stress[0][2]=Txz;
      //stress[1][0]=Txy;stress[1][1]=Tyy;stress[1][2]=Tyz;
      //stress[2][0]=Txz;stress[2][1]=Tyz;stress[2][2]=Tzz;
      //pos = (i0*ny*nz + j1*nz + k1)*WSIZE;
      //stress[0][0]=w.W[pos+3];stress[0][1]=w.W[pos+6];stress[0][2]=w.W[pos+7];
      //stress[1][0]=w.W[pos+6];stress[1][1]=w.W[pos+4];stress[1][2]=w.W[pos+8];
      //stress[2][0]=w.W[pos+7];stress[2][1]=w.W[pos+8];stress[2][2]=w.W[pos+5];
      stress[0][0]=w_Txx[pos];stress[0][1]=w_Txy[pos];stress[0][2]=w_Txz[pos];
      stress[1][0]=w_Txy[pos];stress[1][1]=w_Tyy[pos];stress[1][2]=w_Tyz[pos];
      stress[2][0]=w_Txz[pos];stress[2][1]=w_Tyz[pos];stress[2][2]=w_Tzz[pos];

      matmul3x3(stress, metric, traction);

      int ii, jj;
#pragma unroll
      for (ii = 0; ii < 3; ii++)
#pragma unroll
        for (jj = 0; jj < 3; jj++)
          traction[ii][jj] *= jac;

      // Non Split, 0: left, 1: middle, 2: right
      //pos = 1*ny*nz + j1*nz + k1;
      //pos = j1 + k1 * ny + 1 * ny * nz;
      pos = j1 + k1 * ny + 3 * ny * nz + nfault * (ny*nz*7);
      f.T11[pos] = traction[0][0];
      f.T12[pos] = traction[1][0];
      f.T13[pos] = traction[2][0];

      pos = j1 + k1 * ny + i0 * ny * nz;
      for (m = 0; m < 2; m++){
        // Split nodes => 0: minus side, 1: plus side
        //pos_f = (m*ny*nz + j1*nz + k1)*FSIZE;
        //pos = (i0*ny*nz + j1*nz + k1)*WSIZE;
        pos_f = j1 + k1 * ny + m * ny * nz; //+ nfault * ny * nz;
        f_Vx [pos_f] = w_Vx[pos];
        f_Vy [pos_f] = w_Vy[pos];
        f_Vz [pos_f] = w_Vz[pos];
        f_T21[pos_f] = traction[0][1];
        f_T22[pos_f] = traction[1][1];
        f_T23[pos_f] = traction[2][1];
        f_T31[pos_f] = traction[0][2];
        f_T32[pos_f] = traction[1][2];
        f_T33[pos_f] = traction[2][2];
      }

      //pos = (i0*ny*nz + j1*nz + k1)*WSIZE;
      //stress[0][0]=hTxx;stress[0][1]=hTxy;stress[0][2]=hTxz;
      //stress[1][0]=hTxy;stress[1][1]=hTyy;stress[1][2]=hTyz;
      //stress[2][0]=hTxz;stress[2][1]=hTyz;stress[2][2]=hTzz;
      //stress[0][0]=w.hW[pos+3];stress[0][1]=w.hW[pos+6];stress[0][2]=w.hW[pos+7];
      //stress[1][0]=w.hW[pos+6];stress[1][1]=w.hW[pos+4];stress[1][2]=w.hW[pos+8];
      //stress[2][0]=w.hW[pos+7];stress[2][1]=w.hW[pos+8];stress[2][2]=w.hW[pos+5];
      stress[0][0]=w_hTxx[pos];stress[0][1]=w_hTxy[pos];stress[0][2]=w_hTxz[pos];
      stress[1][0]=w_hTxy[pos];stress[1][1]=w_hTyy[pos];stress[1][2]=w_hTyz[pos];
      stress[2][0]=w_hTxz[pos];stress[2][1]=w_hTyz[pos];stress[2][2]=w_hTzz[pos];

      matmul3x3(stress, metric, traction);

#pragma unroll
      for (ii = 0; ii < 3; ii++)
#pragma unroll
        for (jj = 0; jj < 3; jj++){
          traction[ii][jj] *= jac;
        }

      for (m = 0; m < 2; m++){
        pos_f = j1 + k1 * ny + m * ny * nz; //+ nfault * ny * nz;
        f_hT21[pos_f] = traction[0][1];
        f_hT22[pos_f] = traction[1][1];
        f_hT23[pos_f] = traction[2][1];
        f_hT31[pos_f] = traction[0][2];
        f_hT32[pos_f] = traction[1][2];
        f_hT33[pos_f] = traction[2][2];
      }

      //for (m = 0; m < 2; m++){
      //  // 0: minus side, 1: plus side
      //  pos_f = (m*ny*nz + j1*nz + k1)*FSIZE;
      //  pos = (i0*ny*nz + j1*nz + k1)*WSIZE;
      //  f.hW[pos_f + 3] = traction[0][1];
      //  f.hW[pos_f + 4] = traction[1][1];
      //  f.hW[pos_f + 5] = traction[2][1];
      //  f.hW[pos_f + 6] = traction[0][2];
      //  f.hW[pos_f + 7] = traction[1][2];
      //  f.hW[pos_f + 8] = traction[2][2];
      //}
    } // end if of united
  }// end of loop j, k
  return;
}

__global__ void fault2wave_cu(Wave w, Fault f, realptr_t M, int i0, int nfault)
{
  // transform
  //          fault (T11 (at i0), ...
  //                 T21, ..., F->T31, ...)
  // to
  //          wave (Txx, Tyy, ..., Tyz (at i0)
  // and
  // transform
  //          fault (Vx_f, ..., F->Vz_f)
  // to
  //          wave (Vx, ..., Vz (at i0))
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
  //real_t DH = par.DH;
  //real_t DT = par.DT;
  int stride = nx * ny * nz;
  int nyz = ny * nz;
  int nyz2 = nyz * 2;
  int faultsize = nfault * (9 * nyz2);   //******************wangzj

  // OUTPUT
  real_t *w_Vx  = w.W + 0 * stride;
  real_t *w_Vy  = w.W + 1 * stride;
  real_t *w_Vz  = w.W + 2 * stride;
  real_t *w_Txx = w.W + 3 * stride;
  real_t *w_Tyy = w.W + 4 * stride;
  real_t *w_Tzz = w.W + 5 * stride;
  real_t *w_Txy = w.W + 6 * stride;
  real_t *w_Txz = w.W + 7 * stride;
  real_t *w_Tyz = w.W + 8 * stride;

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
  //real_t *LAM = JAC + stride;
  //real_t *MIU = LAM + stride;
  //real_t *RHO = MIU + stride;

  // Split nodes
  //stride = ny * nz * 2; // y vary first

  // INPUT
  real_t *f_Vx  = f.W + 0 * nyz2 + faultsize;
  real_t *f_Vy  = f.W + 1 * nyz2 + faultsize;
  real_t *f_Vz  = f.W + 2 * nyz2 + faultsize;
  real_t *f_T21 = f.W + 3 * nyz2 + faultsize;
  real_t *f_T22 = f.W + 4 * nyz2 + faultsize;
  real_t *f_T23 = f.W + 5 * nyz2 + faultsize;
  real_t *f_T31 = f.W + 6 * nyz2 + faultsize;
  real_t *f_T32 = f.W + 7 * nyz2 + faultsize;
  real_t *f_T33 = f.W + 8 * nyz2 + faultsize;


  real_t jac;
  real_t metric[3][3], stress[3][3], traction[3][3];
  // int i0  = nx / 2; // local Nucleate source i index
  int pos, pos1;//, pos0;

  int ii, jj;
//#ifdef FreeSurface
//  if( j >= 30+3 && j < ny-31-3 && k >= 30+3 && k < nz-3){ // not united
//#else
//  if( j >= 30+3 && j < ny-31-3 && k >= 30+3 && k < nz-31-3){ // not united
//#endif
//#ifdef FreeSurface
//  if( j >= 30 && j < nj-31 && k >= 30 && k < nk){ // not united
//#else
//  if( j >= 30 && j < nj-31 && k >= 30 && k < nk-31){ // not united
//#endif
  if( j < nj && k < nk){ // not united
    if(f.united[j + k * nj + nfault * nj * nk]) return;
    //pos = (i0*ny*nz + j1*nz + k1)*MSIZE;
    //metric[0][0]=M[pos+0];metric[0][1]=M[pos+3];metric[0][2]=M[pos+6];
    //metric[1][0]=M[pos+1];metric[1][1]=M[pos+4];metric[1][2]=M[pos+7];
    //metric[2][0]=M[pos+2];metric[2][1]=M[pos+5];metric[2][2]=M[pos+8];
    //jac = M[pos+9];
    pos = j1 + k1 * ny + i0 * ny * nz;
    metric[0][0]=XIX[pos];metric[0][1]=ETX[pos];metric[0][2]=ZTX[pos];
    metric[1][0]=XIY[pos];metric[1][1]=ETY[pos];metric[1][2]=ZTY[pos];
    metric[2][0]=XIZ[pos];metric[2][1]=ETZ[pos];metric[2][2]=ZTZ[pos];
    jac = 1.0/JAC[pos];

    // Non Split, 0: left, 1: middle, 2: right
    //pos = 1*ny*nz + j1*nz + k1;
    //pos = j1 + k1 * ny + 1 * ny * nz;
    pos = j1 + k1 * ny + 3 * ny * nz + nfault * (7 * ny * nz);
    traction[0][0] = f.T11[pos];
    traction[1][0] = f.T12[pos];
    traction[2][0] = f.T13[pos];
#ifdef DEBUG
    if(j == nj/2 && k == nk/2)
      printf("@fault2wave, f.T1 = %e %e %e\n", f.T11[pos], f.T12[pos], f.T13[pos]);
#endif
    // Split nodes => 0: minus side, 1: plus side
    //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
    //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
    //traction[0][1] = (f.W[pos0 + 3] + f.W[pos1 + 3]) * 0.5f; // T21
    //traction[1][1] = (f.W[pos0 + 4] + f.W[pos1 + 4]) * 0.5f; // T22
    //traction[2][1] = (f.W[pos0 + 5] + f.W[pos1 + 5]) * 0.5f; // T23
    //traction[0][2] = (f.W[pos0 + 6] + f.W[pos1 + 6]) * 0.5f; // T31
    //traction[1][2] = (f.W[pos0 + 7] + f.W[pos1 + 7]) * 0.5f; // T32
    //traction[2][2] = (f.W[pos0 + 8] + f.W[pos1 + 8]) * 0.5f; // T33
    //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
    //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
    pos = j1 + k1 * ny; //+ nfault * ny * nz;
    traction[0][1] = (f_T21[pos] + f_T21[pos + nyz]) * 0.5f; // T21
    traction[1][1] = (f_T22[pos] + f_T22[pos + nyz]) * 0.5f; // T22
    traction[2][1] = (f_T23[pos] + f_T23[pos + nyz]) * 0.5f; // T23
    traction[0][2] = (f_T31[pos] + f_T31[pos + nyz]) * 0.5f; // T31
    traction[1][2] = (f_T32[pos] + f_T32[pos + nyz]) * 0.5f; // T32
    traction[2][2] = (f_T33[pos] + f_T33[pos + nyz]) * 0.5f; // T33

    invert3x3(metric);
    matmul3x3(traction, metric, stress);

#pragma unroll
    for (ii = 0; ii < 3; ii++)
#pragma unroll
      for (jj = 0; jj < 3; jj++){
        stress[ii][jj] *= jac;
      }

    //pos = (i0*ny*nz + j1*nz + k1)*WSIZE;
    //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
    //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
    //w.W[pos + 0] = (f.W[pos0 + 0] + f.W[pos1 + 0]) * 0.5f;
    //w.W[pos + 1] = (f.W[pos0 + 1] + f.W[pos1 + 1]) * 0.5f;
    //w.W[pos + 2] = (f.W[pos0 + 2] + f.W[pos1 + 2]) * 0.5f;
    //w.W[pos + 3] = stress[0][0];
    //w.W[pos + 4] = stress[1][1];
    //w.W[pos + 5] = stress[2][2];
    //w.W[pos + 6] = (stress[0][1] + stress[1][0]) * 0.5f;
    //w.W[pos + 7] = (stress[0][2] + stress[2][0]) * 0.5f;
    //w.W[pos + 8] = (stress[1][2] + stress[2][1]) * 0.5f;
    pos1 = j1 + k1 * ny;
    pos = j1 + k1 * ny + i0 * ny * nz;
    w_Vx [pos] = (f_Vx[pos1] + f_Vx[pos1 + nyz]) * 0.5f;
    w_Vy [pos] = (f_Vy[pos1] + f_Vy[pos1 + nyz]) * 0.5f;
    w_Vz [pos] = (f_Vz[pos1] + f_Vz[pos1 + nyz]) * 0.5f;
    w_Txx[pos] = stress[0][0];
    w_Tyy[pos] = stress[1][1];
    w_Tzz[pos] = stress[2][2];
    w_Txy[pos] = (stress[0][1] + stress[1][0]) * 0.5f;
    w_Txz[pos] = (stress[0][2] + stress[2][0]) * 0.5f;
    w_Tyz[pos] = (stress[1][2] + stress[2][1]) * 0.5f;

//#ifdef DEBUG
//      if(j == nj/2 && k == nk/2)
//        printf("@fault2wave, W = %e %e %e %e %e %e %e %e %e\n",
//            w.W[pos + 0],
//            w.W[pos + 1],
//            w.W[pos + 2],
//            w.W[pos + 3],
//            w.W[pos + 4],
//            w.W[pos + 5],
//            w.W[pos + 6],
//            w.W[pos + 7],
//            w.W[pos + 8]);
//#endif
  } // if not united

  return;
}

void wave2fault(Wave w, Fault f, realptr_t M, int i0, int nfault)
{
  dim3 block(256, 1, 1);
  dim3 grid(
      (hostParams.nj + block.x-1)/block.x,
      (hostParams.nk + block.y-1)/block.y, 1);
  wave2fault_cu <<<grid, block>>> (w, f, M, i0, nfault);
}
void fault2wave(Wave w, Fault f, realptr_t M, int i0, int nfault)
{
  dim3 block(256, 1, 1);
  dim3 grid(
      (hostParams.nj + block.x-1)/block.x,
      (hostParams.nk + block.y-1)/block.y, 1);
  fault2wave_cu <<<grid, block>>> (w, f, M, i0, nfault);
}
