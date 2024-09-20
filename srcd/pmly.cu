#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "params.h"
#include "common.h"
#include "macdrp.h"

__global__ void abs_deriv_y_cu(Wave w, real_t *M, PML P, int Flag, int idx0)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int i1 = i + 3; // with ghost points
  int j1 = j + 3;
  int k1 = k + 3;

  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  int stride = nx * ny * nz;

  //real_t *XIX = M + 0 * stride;
  //real_t *XIY = M + 1 * stride;
  //real_t *XIZ = M + 2 * stride;
  real_t *ETX = M + 3 * stride;
  real_t *ETY = M + 4 * stride;
  real_t *ETZ = M + 5 * stride;
#ifdef FreeSurface
  real_t *ZTX = M + 6 * stride;
  real_t *ZTY = M + 7 * stride;
  real_t *ZTZ = M + 8 * stride;
#endif
  //real_t *JAC = M + 9 * stride;
  real_t *LAM = M + 10 * stride;
  real_t *MIU = M + 11 * stride;
  real_t *RHO = M + 12 * stride;

  real_t *Vx  = w.W;
  real_t *Vy  = Vx  + stride;
  real_t *Vz  = Vy  + stride;
  real_t *Txx = Vz  + stride;
  real_t *Tyy = Txx + stride;
  real_t *Tzz = Tyy + stride;
  real_t *Txy = Tzz + stride;
  real_t *Txz = Txy + stride;
  real_t *Tyz = Txz + stride;

  real_t *hVx  = w.hW;
  real_t *hVy  = hVx  + stride;
  real_t *hVz  = hVy  + stride;
  real_t *hTxx = hVz  + stride;
  real_t *hTyy = hTxx + stride;
  real_t *hTzz = hTyy + stride;
  real_t *hTxy = hTzz + stride;
  real_t *hTxz = hTxy + stride;
  real_t *hTyz = hTxz + stride;

  real_t *aux_Vx;
  real_t *aux_hVx;
  int pni = ni;
  int pnj = par.PML_N;
  int pnk = nk;
  if (0 == idx0){
    aux_Vx  = P.Wy1;
    aux_hVx = P.hWy1;
  }else{
    aux_Vx  = P.Wy2;
    aux_hVx = P.hWy2;
  }
  stride = pni * pnj * pnk;

  real_t *aux_Vy  = aux_Vx  + stride;
  real_t *aux_Vz  = aux_Vy  + stride;
  real_t *aux_Txx = aux_Vz  + stride;
  real_t *aux_Tyy = aux_Txx + stride;
  real_t *aux_Tzz = aux_Tyy + stride;
  real_t *aux_Txy = aux_Tzz + stride;
  real_t *aux_Txz = aux_Txy + stride;
  real_t *aux_Tyz = aux_Txz + stride;

  real_t *aux_hVy  = aux_hVx  + stride;
  real_t *aux_hVz  = aux_hVy  + stride;
  real_t *aux_hTxx = aux_hVz  + stride;
  real_t *aux_hTyy = aux_hTxx + stride;
  real_t *aux_hTzz = aux_hTyy + stride;
  real_t *aux_hTxy = aux_hTzz + stride;
  real_t *aux_hTxz = aux_hTxy + stride;
  real_t *aux_hTyz = aux_hTxz + stride;

  real_t rDH = 1.0/par.DH;
  //real_t DT = par.DT;

  int pos;
  int pos_a; // position of auxiliary variables

  real_t DxVx, DxVy, DxVz, DxTxx, DxTyy, DxTzz, DxTxy, DxTxz, DxTyz;
  //real_t DyVx, DyVy, DyVz, DyTxx, DyTyy, DyTzz, DyTxy, DyTxz, DyTyz;
  //real_t DzVx, DzVy, DzVz, DzTxx, DzTyy, DzTzz, DzTxy, DzTxz, DzTyz;

  real_t xix, xiy, xiz;
  //real_t etx, ety, etz;
  real_t lam, mu, lam2mu;
  real_t rrho;

  real_t rb1;//, rb2, rb3;
  real_t d1;//, d2, d3;
  real_t ad1;//, ad2, ad3;

  real_t b1;//, b2, b3;

#ifdef FreeSurface
  real_t ztx, zty, ztz;
  real_t DzVx1;
  real_t DzVy1;
  real_t DzVz1;
#endif

  real_t Deriv[WSIZE];

  if (i < pni && j < pnj && k < pnk) {

    // idx0 = 0 or ni-ND or nj-ND or nk-ND
    pos = (j1 + idx0) + k1 * ny + i1 * ny * nz;
    stride = Flag;

    DxVx  = LF(Vx , pos, stride) * Flag * rDH;
    DxVy  = LF(Vy , pos, stride) * Flag * rDH;
    DxVz  = LF(Vz , pos, stride) * Flag * rDH;
    DxTxx = LF(Txx, pos, stride) * Flag * rDH;
    DxTyy = LF(Tyy, pos, stride) * Flag * rDH;
    DxTzz = LF(Tzz, pos, stride) * Flag * rDH;
    DxTxy = LF(Txy, pos, stride) * Flag * rDH;
    DxTxz = LF(Txz, pos, stride) * Flag * rDH;
    DxTyz = LF(Tyz, pos, stride) * Flag * rDH;

    xix = ETX[pos]; xiy = ETY[pos]; xiz = ETZ[pos];
    b1 = P.By[j + idx0];
    d1 = P.Dy[j + idx0];
    ad1 = -(P.Ay[j + idx0] + d1);

    lam = LAM[pos]; mu = MIU[pos];
    rrho = 1.0f/RHO[pos];
    lam2mu = lam + 2.0f*mu;
    rb1 = 1.0f/b1;

    Deriv[0] = (DxTxx*xix + DxTxy*xiy + DxTxz*xiz)*rrho;
    Deriv[1] = (DxTxy*xix + DxTyy*xiy + DxTyz*xiz)*rrho;
    Deriv[2] = (DxTxz*xix + DxTyz*xiy + DxTzz*xiz)*rrho;
    Deriv[3] = (DxVx*xix*lam2mu + DxVy*xiy*lam    + DxVz*xiz*lam   );
    Deriv[4] = (DxVx*xix*lam    + DxVy*xiy*lam2mu + DxVz*xiz*lam   );
    Deriv[5] = (DxVx*xix*lam    + DxVy*xiy*lam    + DxVz*xiz*lam2mu);
    Deriv[6] = (DxVx*xiy + DxVy*xix)*mu;
    Deriv[7] = (DxVx*xiz + DxVz*xix)*mu;
    Deriv[8] = (DxVy*xiz + DxVz*xiy)*mu;

    // Auxiliary Equations
    pos_a = j + k * pnj + i * pnj * pnk;

    hVx [pos] += -aux_Vx [pos_a] * rb1;
    hVy [pos] += -aux_Vy [pos_a] * rb1;
    hVz [pos] += -aux_Vz [pos_a] * rb1;
    hTxx[pos] += -aux_Txx[pos_a] * rb1;
    hTyy[pos] += -aux_Tyy[pos_a] * rb1;
    hTzz[pos] += -aux_Tzz[pos_a] * rb1;
    hTxy[pos] += -aux_Txy[pos_a] * rb1;
    hTxz[pos] += -aux_Txz[pos_a] * rb1;
    hTyz[pos] += -aux_Tyz[pos_a] * rb1;

    aux_hVx [pos_a] = ad1 * aux_Vx [pos_a] + d1 * Deriv[0];
    aux_hVy [pos_a] = ad1 * aux_Vy [pos_a] + d1 * Deriv[1];
    aux_hVz [pos_a] = ad1 * aux_Vz [pos_a] + d1 * Deriv[2];
    aux_hTxx[pos_a] = ad1 * aux_Txx[pos_a] + d1 * Deriv[3];
    aux_hTyy[pos_a] = ad1 * aux_Tyy[pos_a] + d1 * Deriv[4];
    aux_hTzz[pos_a] = ad1 * aux_Tzz[pos_a] + d1 * Deriv[5];
    aux_hTxy[pos_a] = ad1 * aux_Txy[pos_a] + d1 * Deriv[6];
    aux_hTxz[pos_a] = ad1 * aux_Txz[pos_a] + d1 * Deriv[7];
    aux_hTyz[pos_a] = ad1 * aux_Tyz[pos_a] + d1 * Deriv[8];

#ifdef FreeSurface
    if(par.freenode && k == nk-1) {
      // Get Lz by Lx and Ly
      int ij = ((j1 + idx0) + i1 * ny) * 9;
      DzVx1 = w.matVy2Vz[ij+3*0+0] * DxVx
            + w.matVy2Vz[ij+3*0+1] * DxVy
            + w.matVy2Vz[ij+3*0+2] * DxVz;

      DzVy1 = w.matVy2Vz[ij+3*1+0] * DxVx
            + w.matVy2Vz[ij+3*1+1] * DxVy
            + w.matVy2Vz[ij+3*1+2] * DxVz;

      DzVz1 = w.matVy2Vz[ij+3*2+0] * DxVx
            + w.matVy2Vz[ij+3*2+1] * DxVy
            + w.matVy2Vz[ij+3*2+2] * DxVz;

      ztx = ZTX[pos]; zty = ZTY[pos]; ztz = ZTZ[pos];

      aux_hTxx[pos_a] += d1*b1*(lam2mu*ztx*DzVx1 + lam   *zty*DzVy1 + lam   *ztz*DzVz1);
      aux_hTyy[pos_a] += d1*b1*(lam   *ztx*DzVx1 + lam2mu*zty*DzVy1 + lam   *ztz*DzVz1);
      aux_hTzz[pos_a] += d1*b1*(lam   *ztx*DzVx1 + lam   *zty*DzVy1 + lam2mu*ztz*DzVz1);
      aux_hTxy[pos_a] += d1*b1*mu*(zty*DzVx1 + ztx*DzVy1);
      aux_hTxz[pos_a] += d1*b1*mu*(ztz*DzVx1 + ztx*DzVz1);
      aux_hTyz[pos_a] += d1*b1*mu*(ztz*DzVy1 + zty*DzVz1);
    }
#endif
  } // end of i, j, k

  return;
}

void abs_deriv_y(Wave w, real_t *M, PML P, int Flag, int idx0)
{
  dim3 block(16, 16, 1);
  dim3 grid(
        (hostParams.PML_N + block.x-1)/block.x,
        (hostParams.nk    + block.y-1)/block.y,
        (hostParams.ni    + block.z-1)/block.z);

  abs_deriv_y_cu <<<grid, block>>> (w, M, P, Flag, idx0);
  return;
}