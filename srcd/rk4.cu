#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include <math.h>

#define RK_Syn_wrong \
  if(idx < len){ \
    if(irk == 0){ \
      tW[idx] = rkb * hW[idx] + W[idx]; \
    }else if(irk == 1 || irk == 2) { \
      tW[idx] += rkb * hW[idx]; \
    } \
    if(irk == 3){ \
      W[idx] = rkb * hW[idx] + tW[idx]; \
    }else { \
      hW[idx] = rka * hW[idx] + W[idx]; \
    } \
  }

#define RK_Syn \
  if(idx < len){ \
    if(irk == 0){ \
      mW[idx] = W[idx]; \
      tW[idx] = rkb * hW[idx] + mW[idx]; \
    }else if(irk == 1 || irk == 2) { \
      tW[idx] += rkb * hW[idx]; \
    } \
    if(irk == 3){ \
      W[idx] = rkb * hW[idx] + tW[idx]; \
    }else { \
      W[idx] = rka * hW[idx] + mW[idx]; \
    } \
  }

#define RK_Syn_without_mW \
  if(idx < len){ \
    if(irk == 0){ \
      tW[idx] = rkb * hW[idx] + mW[idx]; \
    }else if(irk == 1 || irk == 2) { \
      tW[idx] += rkb * hW[idx]; \
    } \
    if(irk == 3){ \
      W[idx] = rkb * hW[idx] + tW[idx]; \
    }else { \
      W[idx] = rka * hW[idx] + mW[idx]; \
    } \
  }


#define RK_Syn_1 \
  if(idx < len){ \
    real_t hw = hW[idx]; \
    if(irk == 0){ \
      mW[idx] = W[idx]; \
      tW[idx] = rkb * hw + mW[idx]; \
    }else if(irk == 1 || irk == 2) { \
      real_t hw = hW[idx]; \
      tW[idx] += rkb * hw; \
    } \
    if(irk == 3){ \
      W[idx] = rkb * hw + tW[idx]; \
    }else { \
      W[idx] = rka * hw + mW[idx]; \
    } \
  }

#define RK_Syn_2 \
  if(idx < len){ \
    if(irk == 0){ \
      W[idx] += hW[idx] * 0.5; \
      tW[idx] = hW[idx]; \
    }else if(irk == 1){ \
      W[idx] += hW[idx] * 0.5 - tW[idx] * 0.5; \
      tW[idx] *= (1.0/6.0); \
    }else if(irk == 2){ \
      W[idx] += hW[idx]; \
      tW[idx] -= hW[idx]; \
    }else{ \
      W[idx] += tW[idx] + hW[idx] * (1.0/6.0); \
    } \
  }

#define RK_Syn_3 \
  if(idx < len){ \
    if(irk == 0){ \
      W[idx] += tW[idx] * 0.5; \
    }else if(irk == 1){ \
      W[idx] += hW[idx] * 0.5 - tW[idx] * 0.5; \
    }else if(irk == 2){ \
      W[idx] += hW[idx]; \
      tW[idx] = tW[idx] * (1.0/6.0) - hW[idx]; \
    }else{ \
      W[idx] += tW[idx] + hW[idx] * (1.0/6.0); \
    } \
  }

__global__ void wave_rk_1(real_t *W, real_t *mW, real_t *hW, real_t *tW, int len, int irk){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  real_t rka, rkb;

  if(irk == 0){
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){
    rka = 1.0; rkb = 1.0/3.0;
  }else{
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= par.DT; rkb *= par.DT;

  RK_Syn;

  return;
}

__global__ void wave_rk_new(real_t *W, real_t *mW, real_t *hW, real_t *tW, int irk, int n){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  real_t rka, rkb;
  if(irk == 0){
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){
    rka = 1.0; rkb = 1.0/3.0;
  }else{
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= par.DT; rkb *= par.DT;

  if(idx < n){
    if(irk == 0){
      mW[idx] = W[idx];
      tW[idx] = rkb * hW[idx] + mW[idx];
    }else if(irk == 1 || irk == 2) {
      tW[idx] += rkb * hW[idx];
    }

    if(irk == 3){
      W[idx] = rkb * hW[idx] + tW[idx];
    }else {
      W[idx] = rka * hW[idx] + mW[idx];
    }
  }

  return;
}

__global__ void wave_rk_cu(Wave w, PML P, int irk){

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int len;

  real_t *W ;
  real_t *mW;
  real_t *hW;
  real_t *tW;

  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  real_t rka, rkb;

  //real_t RKa[4] = {0.5, 0.5, 1.0, 1e10};
  //real_t RKb[4] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};

  //__shared__ real_t RKa[4];
  //__shared__ real_t RKb[4];

  //RKa[0] = 0.5;
  //RKa[1] = 0.5;
  //RKa[2] = 1.0;
  //RKa[3] = 1e10;
  //RKb[0] = 1.0/6.0;
  //RKb[1] = 1.0/3.0;
  //RKb[2] = 1.0/3.0;
  //RKb[3] = 1.0/6.0;

  //rka = RKa[irk]; // Can't optimize in this way 
  //rkb = RKb[irk]; // because the compiler don't know the value of irk

  if(irk == 0){
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){
    rka = 1.0; rkb = 1.0/3.0;
  }else{
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= par.DT; rkb *= par.DT;
  //real_t coef = par.DT / par.DH;
  //rka *= coef; rkb *= coef;

  len = (ni+6)*(nj+6)*(nk+6)*WSIZE;
  mW = w.mW;
  hW = w.hW;
  tW = w.tW;
  W  = w.W;
  RK_Syn;
#ifdef usePML
  int N = par.PML_N;
  len = N*nj*nk*WSIZE;
  if(P.isx1){
    mW = P.mWx1;
    hW = P.hWx1;
    tW = P.tWx1;
    W  = P. Wx1;
    RK_Syn;
  }
  if(P.isx2){
    mW = P.mWx2;
    hW = P.hWx2;
    tW = P.tWx2;
    W  = P. Wx2;
    RK_Syn;
  }
  len = N*ni*nk*WSIZE;
  if(P.isy1){
    mW = P.mWy1;
    hW = P.hWy1;
    tW = P.tWy1;
    W  = P. Wy1;
    RK_Syn;
  }
  if(P.isy2){
    mW = P.mWy2;
    hW = P.hWy2;
    tW = P.tWy2;
    W  = P. Wy2;
    RK_Syn;
  }
  len = N*ni*nj*WSIZE;
  if(P.isz1){
    mW = P.mWz1;
    hW = P.hWz1;
    tW = P.tWz1;
    W  = P. Wz1;
    RK_Syn;
  }
  if(P.isz2){
    mW = P.mWz2;
    hW = P.hWz2;
    tW = P.tWz2;
    W  = P. Wz2;
    RK_Syn;
  }
#endif
  return;
}

void wave_rk(Wave w, PML P, int irk){
  dim3 block(256, 1, 1);
  int nx = hostParams.ni + 6;
  int ny = hostParams.nj + 6;
  int nz = hostParams.nk + 6;
  dim3 grid((nx*ny*nz*WSIZE+block.x-1)/block.x, 1, 1);
  wave_rk_cu <<<grid, block>>> (w, P, irk);
}

__global__ void fault_rk_cu0(Fault f, int irk){

  //int j = blockIdx.x * blockDim.x + threadIdx.x;
  //int k = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //int j1 = j + 3;
  //int k1 = k + 3;
  //int i;
  //int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  //real_t DH = par.DH;
  real_t DT = par.DT;

  int len;

  real_t *W ;//     = f.W;
  real_t *mW;//     = f.mW;
  real_t *hW;//     = f.hW;
  real_t *tW;//     = f.tW;
  //real_t *Vs1    = f.Vs1;
  //real_t *Vs2    = f.Vs2;
  //real_t *Ts1    = f.Ts1;
  //real_t *Ts2    = f.Ts2;
  //real_t *Tn     = f.Tn;
  //real_t *tTs1   = f.tTs1;
  //real_t *tTs2   = f.tTs2;
  //real_t *tTn    = f.tTn;
  //real_t *slip   = f.slip;
  //real_t *hslip  = f.hslip;

  // W[FSIZE][2][nz][ny]
  //int stride = nz*ny;
  //real_t *Vx_f0  = f.W;
  //real_t *Vx_f1 = Vx_f0 + stride;
  //real_t *Vy_f0 = Vx_f1 + stride;
  //real_t *Vy_f1 = Vy_f0 + stride;
  //real_t *Vz_f0 = Vy_f1 + stride;
  //real_t *Vz_f1 = Vz_f0 + stride;

  //real_t *State  = f.State;
  //real_t *hState = f.hState;
  //real_t *mState = f.mState;
  //real_t *tState = f.tState;

  real_t rka, rkb;
  if(irk == 0){ // RK_begin
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){ // RK_inner
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){ // RK_inner
    rka = 1.0; rkb = 1.0/3.0;
  }else{ // RK_finish
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= DT;
  rkb *= DT;

  //int pos, pos0, pos1;
  //int l;

  //real_t vec_n [3] = {1, 0, 0};
  //real_t vec_s1[3] = {0, 1, 0};
  //real_t vec_s2[3] = {0, 0, 1};
  //real_t vec_n [3];
  //real_t vec_s1[3];
  //real_t vec_s2[3];

  W = f.W; mW = f.mW; tW = f.tW; hW = f.hW;
  len = 2*ny*nz*FSIZE; RK_Syn;

  W = f.State; mW = f.mState; tW = f.tState; hW = f.hState;
  len = nj*nk; RK_Syn;
/*
  int j = idx / nk; 
  //int k = idx % nk;
  int k = idx - j * nk;

  j -= 2*nj;// -nj, -nj+1, -nj+2, ..., -1, 0, 1, 2, ..., nj

  int j1 = j + 3;
  int k1 = k + 3;
  
  if (j >= 0 && j < nj && k < nk) {

    //pos = (j1*nz + k1)*3;
    pos = (k1 * ny + j1) * 3;

    vec_s1[0] = fc.vec_s1[pos + 0];
    vec_s1[1] = fc.vec_s1[pos + 1];
    vec_s1[2] = fc.vec_s1[pos + 2];
    vec_s2[0] = fc.vec_s2[pos + 0];
    vec_s2[1] = fc.vec_s2[pos + 1];
    vec_s2[2] = fc.vec_s2[pos + 2];

    //for (i = 0; i < 2; i++) {
    //  pos = (i*ny*nz + j1*nz + k1)*FSIZE;
    //  // update tW
    //  if(irk == 0){ //RK_begin
    //    for (l = 0; l < FSIZE; l++){
    //      mW[pos + l] = W[pos + l];
    //      tW[pos + l] = mW[pos + l] + rkb * hW[pos + l];
    //    }
    //  }else if(irk == 1 || irk == 2) { // RK_inner
    //    for (l = 0; l < FSIZE; l++)
    //      tW[pos + l] += rkb * hW[pos + l];
    //  }
    //  // update W
    //  if(irk == 3) { // RK_finish
    //    for (l = 0; l < FSIZE; l++)
    //      W[pos + l] = tW[pos + l] + rkb * hW[pos + l];
    //  }else{ // RK_begin or RK_inner
    //    for (l = 0; l < FSIZE; l++)
    //      W[pos + l] = mW[pos + l] + rka * hW[pos + l];
    //  }

    //} // minus and plus

    // update State using rk4
    //pos = j*nk + k;
    //// update tState
    //if(irk == 0){ //RK_begin
    //  mState[pos] = State[pos];
    //  tState[pos] = mState[pos] + rkb * hState[pos];
    //}else if(irk == 1 || irk == 2) { // RK_inner
    //  tState[pos] += rkb * hState[pos];
    //}
    //// update State
    //if(irk == 3) { // RK_finish
    //  State[pos] = tState[pos] + rkb * hState[pos];
    //}else{ // RK_begin or RK_inner
    //  State[pos] = mState[pos] + rka * hState[pos];
    //}

    // update State using Euler Forward
    //pos = j*nk + k;
    //// update State
    //if(irk == 3) { // RK_finish
    //  State[pos] += DT * hState[pos];
    //}

    //pos0  = (0*ny*nz + j1*nz + k1)*FSIZE;
    //pos1  = (1*ny*nz + j1*nz + k1)*FSIZE;
    //pos = j*nk + k;
    pos = k * nj + j;

    real_t rate1, rate0;
    //rate0 = W[pos0 + 0] * vec_s1[0]
    //  + W[pos0 + 1] * vec_s1[1]
    //  + W[pos0 + 2] * vec_s1[2];
    //rate1 = W[pos1 + 0] * vec_s1[0]
    //  + W[pos1 + 1] * vec_s1[1]
    //  + W[pos1 + 2] * vec_s1[2];
    //Vs1[pos] = rate1 - rate0;
    //rate0 = W[pos0 + 0] * vec_s2[0]
    //  + W[pos0 + 1] * vec_s2[1]
    //  + W[pos0 + 2] * vec_s2[2];
    //rate1 = W[pos1 + 0] * vec_s2[0]
    //  + W[pos1 + 1] * vec_s2[1]
    //  + W[pos1 + 2] * vec_s2[2];
    //Vs2[pos] = rate1 - rate0;
    Vs1[pos] = (Vx_f1[pos] - Vx_f0[pos]) * vec_s1[0]
             + (Vy_f1[pos] - Vy_f0[pos]) * vec_s1[1]
             + (Vz_f1[pos] - Vz_f0[pos]) * vec_s1[2];
    Vs2[pos] = (Vx_f1[pos] - Vx_f0[pos]) * vec_s2[0]
             + (Vy_f1[pos] - Vy_f0[pos]) * vec_s2[1]
             + (Vz_f1[pos] - Vz_f0[pos]) * vec_s2[2];

    hslip[pos] = sqrt(Vs2[pos]*Vs2[pos] + Vs1[pos]*Vs1[pos]);

    if (irk == 0){ // RK_begin
      Ts1[pos] = rkb * tTs1[pos];
      Ts2[pos] = rkb * tTs2[pos];
      Tn [pos] = rkb * tTn [pos];
    }else if(irk == 1 || irk == 2){ // RK_inner
      Ts1[pos] += rkb * tTs1[pos];
      Ts2[pos] += rkb * tTs2[pos];
      Tn [pos] += rkb * tTn [pos];
    }else{ // RK_end
      Ts1[pos] += rkb * tTs1[pos];
      Ts2[pos] += rkb * tTs2[pos];
      Tn [pos] += rkb * tTn [pos];
      Ts1[pos] /= DT;
      Ts2[pos] /= DT;
      Tn [pos] /= DT;
      slip[pos] += hslip[pos] * DT; // Euler forward
    }
  } // end j k
*/
  return;
}


__global__ void fault_rk_stage2_cu(Fault f, int irk, int nfault){

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  int ny = nj + 6;
  int nz = nk + 6;
  int mpifaultsize = nfault * nj * nk;   //****wangzj
  int faultsize = nfault * (ny*nz*9*2);          //****wangzj
  real_t DT = par.DT;

  real_t *Vs1    = f.Vs1 + mpifaultsize;
  real_t *Vs2    = f.Vs2 + mpifaultsize;
  real_t *Ts1    = f.Ts1 + mpifaultsize;
  real_t *Ts2    = f.Ts2 + mpifaultsize;
  real_t *Tn     = f.Tn + mpifaultsize;
  real_t *tTs1   = f.tTs1 + mpifaultsize;
  real_t *tTs2   = f.tTs2 + mpifaultsize;
  real_t *tTn    = f.tTn + mpifaultsize;
  real_t *slip   = f.slip + mpifaultsize;
  real_t *hslip  = f.hslip + mpifaultsize;

  // W[FSIZE][2][nz][ny]
  int stride = nz*ny;
  
  real_t *Vx_f0  = f.W + faultsize;
  real_t *Vx_f1 = Vx_f0 + stride;
  real_t *Vy_f0 = Vx_f1 + stride;
  real_t *Vy_f1 = Vy_f0 + stride;
  real_t *Vz_f0 = Vy_f1 + stride;
  real_t *Vz_f1 = Vz_f0 + stride;

  real_t rka, rkb;
  if(irk == 0){ // RK_begin
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){ // RK_inner
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){ // RK_inner
    rka = 1.0; rkb = 1.0/3.0;
  }else{ // RK_finish
    rka = 99999.99; rkb = 1.0/6.0;
  }

  rka *= DT;
  rkb *= DT;

  int pos;

  real_t vec_s1_x, vec_s1_y, vec_s1_z;
  real_t vec_s2_x, vec_s2_y, vec_s2_z;

  real_t dVx, dVy, dVz;
  real_t vs1, vs2;

  if (j < nj && k < nk) {

    pos = (k1 * ny + j1) * 3 + nfault * (ny*nz*3);  //**********wangzj
    vec_s1_x = f.vec_s1[pos + 0];
    vec_s1_y = f.vec_s1[pos + 1];
    vec_s1_z = f.vec_s1[pos + 2];
    vec_s2_x = f.vec_s2[pos + 0];
    vec_s2_y = f.vec_s2[pos + 1];
    vec_s2_z = f.vec_s2[pos + 2];

    pos = k1 * ny + j1;
    dVx = Vx_f1[pos] - Vx_f0[pos];
    dVy = Vy_f1[pos] - Vy_f0[pos];
    dVz = Vz_f1[pos] - Vz_f0[pos];

    vs1 = dVx * vec_s1_x
        + dVy * vec_s1_y
        + dVz * vec_s1_z;
    vs2 = dVx * vec_s2_x
        + dVy * vec_s2_y
        + dVz * vec_s2_z;

    pos = k * nj + j;      
    hslip[pos] = sqrt(vs2*vs2+vs1*vs1);
    Vs1[pos] = vs1;
    Vs2[pos] = vs2;

    // pos = k * nj + j + mpifaultsize;         //**********wangzj
    real_t rkb1 = rkb/DT;
    if (irk == 0){ // RK_begin
      Ts1[pos] = rkb1 * tTs1[pos];
      Ts2[pos] = rkb1 * tTs2[pos];
      Tn [pos] = rkb1 * tTn [pos];
    }else if(irk == 1 || irk == 2){ // RK_inner
      Ts1[pos] += rkb1 * tTs1[pos];
      Ts2[pos] += rkb1 * tTs2[pos];
      Tn [pos] += rkb1 * tTn [pos];
    }else{ // RK_end
      Ts1[pos] += rkb1 * tTs1[pos];
      Ts2[pos] += rkb1 * tTs2[pos];
      Tn [pos] += rkb1 * tTn [pos];
      //Ts1[pos] /= DT;
      //Ts2[pos] /= DT;
      //Tn [pos] /= DT;
      //slip[pos] += hslip[pos] * DT; // Euler forward
      //Ts1[pos] = tTs1[pos];
      //Ts2[pos] = tTs2[pos];
      //Tn [pos] = tTn [pos];
    }
  } // end j k
  return;
}

__global__ void fault_rk_cu(Fault F, int irk, int nfault){

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int j1 = j + 3;
  int k1 = k + 3;
  //int i;
  //int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int nyz2 = 2*ny*nz;
  int faultsize = nfault * (nyz2*9);
  int mpifaultsize = nfault * (nj*nk);
  //real_t DH = par.DH;
  real_t DT = par.DT;

  real_t *f_Vx  = F.W + 0 * nyz2 + faultsize;
  real_t *f_Vy  = F.W + 1 * nyz2 + faultsize;
  real_t *f_Vz  = F.W + 2 * nyz2 + faultsize;
  real_t *f_T21 = F.W + 3 * nyz2 + faultsize;
  real_t *f_T22 = F.W + 4 * nyz2 + faultsize;
  real_t *f_T23 = F.W + 5 * nyz2 + faultsize;
  real_t *f_T31 = F.W + 6 * nyz2 + faultsize;
  real_t *f_T32 = F.W + 7 * nyz2 + faultsize;
  real_t *f_T33 = F.W + 8 * nyz2 + faultsize;

  real_t *f_hVx  = F.hW + 0 * nyz2 + faultsize;
  real_t *f_hVy  = F.hW + 1 * nyz2 + faultsize;
  real_t *f_hVz  = F.hW + 2 * nyz2 + faultsize;
  real_t *f_hT21 = F.hW + 3 * nyz2 + faultsize;
  real_t *f_hT22 = F.hW + 4 * nyz2 + faultsize;
  real_t *f_hT23 = F.hW + 5 * nyz2 + faultsize;
  real_t *f_hT31 = F.hW + 6 * nyz2 + faultsize;
  real_t *f_hT32 = F.hW + 7 * nyz2 + faultsize;
  real_t *f_hT33 = F.hW + 8 * nyz2 + faultsize;

  real_t *f_mVx  = F.mW + 0 * nyz2 + faultsize;
  real_t *f_mVy  = F.mW + 1 * nyz2 + faultsize;
  real_t *f_mVz  = F.mW + 2 * nyz2 + faultsize;
  real_t *f_mT21 = F.mW + 3 * nyz2 + faultsize;
  real_t *f_mT22 = F.mW + 4 * nyz2 + faultsize;
  real_t *f_mT23 = F.mW + 5 * nyz2 + faultsize;
  real_t *f_mT31 = F.mW + 6 * nyz2 + faultsize;
  real_t *f_mT32 = F.mW + 7 * nyz2 + faultsize;
  real_t *f_mT33 = F.mW + 8 * nyz2 + faultsize;

  real_t *f_tVx  = F.tW + 0 * nyz2 + faultsize;
  real_t *f_tVy  = F.tW + 1 * nyz2 + faultsize;
  real_t *f_tVz  = F.tW + 2 * nyz2 + faultsize;
  real_t *f_tT21 = F.tW + 3 * nyz2 + faultsize;
  real_t *f_tT22 = F.tW + 4 * nyz2 + faultsize;
  real_t *f_tT23 = F.tW + 5 * nyz2 + faultsize;
  real_t *f_tT31 = F.tW + 6 * nyz2 + faultsize;
  real_t *f_tT32 = F.tW + 7 * nyz2 + faultsize;
  real_t *f_tT33 = F.tW + 8 * nyz2 + faultsize;

  int stride = nz*ny;
  real_t *Vx_f0  = F.W + faultsize;
  real_t *Vx_f1 = Vx_f0 + stride;
  real_t *Vy_f0 = Vx_f1 + stride;
  real_t *Vy_f1 = Vy_f0 + stride;
  real_t *Vz_f0 = Vy_f1 + stride;
  real_t *Vz_f1 = Vz_f0 + stride;

  real_t rka, rkb;
  if(irk == 0){ // RK_begin
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){ // RK_inner
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){ // RK_inner
    rka = 1.0; rkb = 1.0/3.0;
  }else{ // RK_finish
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= DT;
  rkb *= DT;

  real_t viscosity = par.viscosity * DT;

  if (j < nj && k < nk && !F.united[j + k * nj + mpifaultsize]){
    for (int mm=0;mm<2;mm++){
      int idx = j1 + k1 * ny + mm * ny * nz;

      if(irk == 0){
        //mW[idx] = W[idx];
        //tW[idx] = rkb * hW[idx] + mW[idx];

        f_mVx [idx] = f_Vx [idx];
        f_mVy [idx] = f_Vy [idx];
        f_mVz [idx] = f_Vz [idx];
        f_mT21[idx] = f_T21[idx];
        f_mT22[idx] = f_T22[idx];
        f_mT23[idx] = f_T23[idx];
        f_mT31[idx] = f_T31[idx];
        f_mT32[idx] = f_T32[idx];
        f_mT33[idx] = f_T33[idx];

        f_tVx [idx] = rkb * f_hVx [idx] + f_mVx [idx];
        f_tVy [idx] = rkb * f_hVy [idx] + f_mVy [idx];
        f_tVz [idx] = rkb * f_hVz [idx] + f_mVz [idx];
        f_tT21[idx] = rkb * f_hT21[idx] + f_mT21[idx];
        f_tT22[idx] = rkb * f_hT22[idx] + f_mT22[idx];
        f_tT23[idx] = rkb * f_hT23[idx] + f_mT23[idx];
        f_tT31[idx] = rkb * f_hT31[idx] + f_mT31[idx];
        f_tT32[idx] = rkb * f_hT32[idx] + f_mT32[idx];
        f_tT33[idx] = rkb * f_hT33[idx] + f_mT33[idx];

      }else if(irk == 1 || irk == 2) {
        //tW[idx] += rkb * hW[idx];
        f_tVx [idx] += rkb * f_hVx [idx];
        f_tVy [idx] += rkb * f_hVy [idx];
        f_tVz [idx] += rkb * f_hVz [idx];
        f_tT21[idx] += rkb * f_hT21[idx];
        f_tT22[idx] += rkb * f_hT22[idx];
        f_tT23[idx] += rkb * f_hT23[idx];
        f_tT31[idx] += rkb * f_hT31[idx];
        f_tT32[idx] += rkb * f_hT32[idx];
        f_tT33[idx] += rkb * f_hT33[idx];
      }
      if(irk == 3){
        //W[idx] = rkb * hW[idx] + tW[idx];
        f_Vx [idx] = rkb * f_hVx [idx] + f_tVx [idx];
        f_Vy [idx] = rkb * f_hVy [idx] + f_tVy [idx];
        f_Vz [idx] = rkb * f_hVz [idx] + f_tVz [idx];
        f_T21[idx] = rkb * f_hT21[idx] + f_tT21[idx];
        f_T22[idx] = rkb * f_hT22[idx] + f_tT22[idx];
        f_T23[idx] = rkb * f_hT23[idx] + f_tT23[idx];
        f_T31[idx] = rkb * f_hT31[idx] + f_tT31[idx];
        f_T32[idx] = rkb * f_hT32[idx] + f_tT32[idx];
        f_T33[idx] = rkb * f_hT33[idx] + f_tT33[idx];
      }else{
        //W[idx] = rka * hW[idx] + mW[idx];
        f_Vx [idx] = rka * f_hVx [idx] + f_mVx [idx];
        f_Vy [idx] = rka * f_hVy [idx] + f_mVy [idx];
        f_Vz [idx] = rka * f_hVz [idx] + f_mVz [idx];
        f_T21[idx] = rka * f_hT21[idx] + f_mT21[idx];
        f_T22[idx] = rka * f_hT22[idx] + f_mT22[idx];
        f_T23[idx] = rka * f_hT23[idx] + f_mT23[idx];
        f_T31[idx] = rka * f_hT31[idx] + f_mT31[idx];
        f_T32[idx] = rka * f_hT32[idx] + f_mT32[idx];
        f_T33[idx] = rka * f_hT33[idx] + f_mT33[idx];
      }

      // add viscosity
      f_T21[idx] += viscosity * f_hT21[idx];
      f_T22[idx] += viscosity * f_hT22[idx];
      f_T23[idx] += viscosity * f_hT23[idx];
      f_T31[idx] += viscosity * f_hT31[idx];
      f_T32[idx] += viscosity * f_hT32[idx];
      f_T33[idx] += viscosity * f_hT33[idx];
    }

    int idx = j1 + k1 * ny + 3 * ny * nz;


    //// update State using rk4
    real_t vec_s1_x, vec_s1_y, vec_s1_z;
    real_t vec_s2_x, vec_s2_y, vec_s2_z;

    real_t dVx, dVy, dVz;
    real_t vs1, vs2;

    int pos = (k1 * ny + j1) * 3 + nfault * (ny*nz*3);  //****wangzj
    vec_s1_x = F.vec_s1[pos + 0];
    vec_s1_y = F.vec_s1[pos + 1];
    vec_s1_z = F.vec_s1[pos + 2];
    vec_s2_x = F.vec_s2[pos + 0];
    vec_s2_y = F.vec_s2[pos + 1];
    vec_s2_z = F.vec_s2[pos + 2];

    pos = k1 * ny + j1;
    dVx = Vx_f1[pos] - Vx_f0[pos];
    dVy = Vy_f1[pos] - Vy_f0[pos];
    dVz = Vz_f1[pos] - Vz_f0[pos];


    vs1 = dVx * vec_s1_x
        + dVy * vec_s1_y
        + dVz * vec_s1_z;
    vs2 = dVx * vec_s2_x
        + dVy * vec_s2_y
        + dVz * vec_s2_z;

    pos = k * nj + j + mpifaultsize;     //*******wangzj
    F.hslip[pos] = sqrt(vs2*vs2+vs1*vs1);
/*
    //int pos = j + k * nj;
    // update tState
    if(irk == 0){ //RK_begin
      F.mState[pos] = F.State[pos];
      F.tState[pos] = F.mState[pos] + rkb * F.hState[pos];

      F.mslip[pos] = F.slip[pos];
      F.tslip[pos] = F.mslip[pos] + rkb * F.hslip[pos];
    }else if(irk == 1 || irk == 2) { // RK_inner
      F.tState[pos] += rkb * F.hState[pos];

      F.tslip[pos] += rkb * F.hslip[pos];
    }
    // update State
    if(irk == 3) { // RK_finish
      F.State[pos] = F.tState[pos] + rkb * F.hState[pos];

      F.slip[pos] = F.tslip[pos] + rkb * F.hslip[pos];
    }else{ // RK_begin or RK_inner
      F.State[pos] = F.mState[pos] + rka * F.hState[pos];

      F.slip[pos] = F.mslip[pos] + rka * F.hslip[pos];
    }
    */
  } // end j k
  return;
}

__global__ void state_rk_cu(Fault F, int irk, int nfault, int Faultgrid[]){

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  //int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int j1 = j + 3;
  int k1 = k + 3;
  //int i;
  //int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  //////////////////////////////////
  int gj = par.ranky * nj + j;
  int gj1 = gj + 1;
  int gk = par.rankz * nk + k;
  int gk1 = gk + 1;
  //////////////////////////////////
  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  //real_t DH = par.DH;
  real_t DT = par.DT;

  int stride = nz*ny;
  real_t *Vx_f0  = F.W + nfault * (ny*nz*9*2);        //**************
  real_t *Vx_f1 = Vx_f0 + stride;
  real_t *Vy_f0 = Vx_f1 + stride;
  real_t *Vy_f1 = Vy_f0 + stride;
  real_t *Vz_f0 = Vy_f1 + stride;
  real_t *Vz_f1 = Vz_f0 + stride;

  real_t rka, rkb;
  if(irk == 0){ // RK_begin
    rka = 0.5; rkb = 1.0/6.0;
  }else if(irk == 1){ // RK_inner
    rka = 0.5; rkb = 1.0/3.0;
  }else if(irk == 2){ // RK_inner
    rka = 1.0; rkb = 1.0/3.0;
  }else{ // RK_finish
    rka = 1e10; rkb = 1.0/6.0;
  }

  rka *= DT;
  rkb *= DT;

  //real_t viscosity = par.viscosity * DT;

  if (j < nj && k < nk && !F.united[j + k * nj + nfault*(nj*nk)]){      //**********wangzj

    //// update State using rk4
    real_t vec_s1_x, vec_s1_y, vec_s1_z;
    real_t vec_s2_x, vec_s2_y, vec_s2_z;

    real_t dVx, dVy, dVz;
    real_t vs1, vs2;

    int pos = (k1 * ny + j1) * 3 + nfault * (ny*nz*3);   //**********wangzj
    vec_s1_x = F.vec_s1[pos + 0];
    vec_s1_y = F.vec_s1[pos + 1];
    vec_s1_z = F.vec_s1[pos + 2];
    vec_s2_x = F.vec_s2[pos + 0];
    vec_s2_y = F.vec_s2[pos + 1];
    vec_s2_z = F.vec_s2[pos + 2];

    pos = k1 * ny + j1;
    dVx = Vx_f1[pos] - Vx_f0[pos];
    dVy = Vy_f1[pos] - Vy_f0[pos];
    dVz = Vz_f1[pos] - Vz_f0[pos];


    vs1 = dVx * vec_s1_x
        + dVy * vec_s1_y
        + dVz * vec_s1_z;
    vs2 = dVx * vec_s2_x
        + dVy * vec_s2_y
        + dVz * vec_s2_z;

    pos = k * nj + j + nfault * (nj*nk);    //*******wangzj
    
    double V = F.hslip[pos];
    V = MAX(V, 1e-20);
        
    double state = F.State[pos];

    double RS_a = F.a[pos];
    double RS_b = F.b[pos];
    double RS_f0 = 0.6;
    double RS_V0 = 1e-6;
    double RS_L = 0.02;
    double RS_Vw = F.Vw[pos];

    // ageing law
#if defined TPV101 || defined TPV102
    RS_f0 = 0.6;
    RS_V0 = 1e-6;
    RS_L = 0.02;
    F.hState[pos] = RS_b * RS_V0 / RS_L * (exp((RS_f0 - state)/RS_b) - V/RS_V0);
#endif
#if defined TPV103 || defined TPV104
    RS_f0 = 0.6;
    RS_V0 = 1e-6;
    RS_L = 0.4;
    double RS_fw = 0.2;

    double RS_flv = RS_f0 - (RS_b-RS_a)*log(V/RS_V0);
    double RS_fss = RS_fw + (RS_flv - RS_fw)/pow((1.+pow(V/RS_Vw, 8)),0.125);
    double psiss = RS_a*(log(sinh(RS_fss/RS_a)) + log(2.*(RS_V0/V)));

    F.hState[pos] = -V/RS_L*(state-psiss);
    //f.State[pos] = (f.State[pos]-psiss)*exp(-V*DT/RS_L) + psiss;
#endif

    RS_f0 = par.f0;
    RS_V0 = par.V0;
    RS_L = par.L;
    RS_L = F.L[j+k*nj + nfault * (nj*nk)];                //******wangzj
    if(par.Friction_type == 1){
      F.hState[pos] = RS_b * RS_V0 / RS_L * (exp((RS_f0 - state)/RS_b) - V/RS_V0);
    }else if (
        par.Friction_type == 2 ||
        par.Friction_type == 3
        ){
      // if(gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] &&
      //    gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault]){
        double RS_fw = par.fw;
        double RS_flv = RS_f0 - (RS_b-RS_a)*log(V/RS_V0);
        double RS_fss = RS_fw + (RS_flv - RS_fw)/pow((1.+pow(V/RS_Vw, 8)),0.125);
        double psiss = RS_a*(log(sinh(RS_fss/RS_a)) + log(2.*(RS_V0/V)));
        F.hState[pos] = -V/RS_L*(state-psiss);
        // if( gj1 == 51 && gk1 == 51){
        //   printf("RS_flv=%lf\n", RS_flv);
        //   printf("RS_fss=%lf\n", RS_fss);
        //   printf("psiss=%lf\n", psiss);
        //   printf("state=%lf\n", F.hState[pos]);
        // }
        // }else{
        //   F.hState[pos] = state;
          // F.hslip[pos] = 0;
        // }
    }
    
    //int pos = j + k * nj;
    // update tState
    if(irk == 0){ //RK_begin
      F.mState[pos] = F.State[pos];
      F.tState[pos] = F.mState[pos] + rkb * F.hState[pos];

      F.mslip[pos] = F.slip[pos];
      F.tslip[pos] = F.mslip[pos] + rkb * F.hslip[pos];
    }else if(irk == 1 || irk == 2) { // RK_inner
      F.tState[pos] += rkb * F.hState[pos];

      F.tslip[pos] += rkb * F.hslip[pos];
    }
    // update State
    if(irk == 3) { // RK_finish
      F.State[pos] = F.tState[pos] + rkb * F.hState[pos];

      //F.slip[pos] = F.tslip[pos] + rkb * F.hslip[pos];
      F.slip[pos] += sqrt(vs1*vs1 + vs2*vs2) * DT;
      F.slip1[pos] += vs1 * DT;
      F.slip2[pos] += vs2 * DT;
     
      F.rake[pos] = atan2(vs2, vs1);
    }else{ // RK_begin or RK_inner
      F.State[pos] = F.mState[pos] + rka * F.hState[pos];

      // F.slip[pos] = F.mslip[pos] + rka * F.hslip[pos];
    }
    // if((gj1 < Faultgrid[0 + 4*nfault] || gj1 > Faultgrid[1 + 4*nfault]) &&
    //    (gk1 < Faultgrid[2 + 4*nfault] || gk1 > Faultgrid[3 + 4*nfault])){
    //     // F.slip[pos] = 0;
    //     F.State[pos] = state;
    //   }
  } // end j k
  return;
}

void fault_rk(Fault F, int irk, int nfault)
{
  // launch as one dimension
  int ny = hostParams.ny;
  int nz = hostParams.nz;
  dim3 block(256, 1, 1);
  dim3 grid((2*ny*nz*FSIZE+block.x-1)/block.x, 1, 1);
  // launch as two dimension
  dim3 block2(16, 8, 1);
  dim3 grid2(
      (hostParams.nj+block2.x-1)/block2.x,
      (hostParams.nk+block2.y-1)/block2.y, 1);

  fault_rk_cu        <<<grid2, block2>>> (F, irk, nfault);
  fault_rk_stage2_cu <<<grid2, block2>>> (F, irk, nfault);
  return;
}

void state_rk(Fault F, int irk, int nfault, int Faultgrid[])
{
  // launch as two dimension
  dim3 block2(16, 8, 1);
  dim3 grid2(
      (hostParams.nj+block2.x-1)/block2.x,
      (hostParams.nk+block2.y-1)/block2.y, 1);

  state_rk_cu <<<grid2, block2>>> (F, irk, nfault, Faultgrid);
  return;
}
