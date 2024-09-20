#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "params.h"
#include "common.h"
#include "macdrp.h"

__global__
void wave_deriv_cu(Wave W, realptr_t M, PML P, int FlagX, int FlagY, int FlagZ)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;

  int i1 = i + 3; // with ghost points
  int j1 = j + 3;
  int k1 = k + 3;

  int ni = par.ni;
  int nj = par.nj;
  int nk = par.nk;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  long stride = nx * ny * nz;

  real_t rDH = 1.0/par.DH;
  real_t coef;

  real_t *Vx  = W.W;
  real_t *Vy  = W.W + 1 * stride;
  real_t *Vz  = W.W + 2 * stride;
  real_t *Txx = W.W + 3 * stride;
  real_t *Tyy = W.W + 4 * stride;
  real_t *Tzz = W.W + 5 * stride;
  real_t *Txy = W.W + 6 * stride;
  real_t *Txz = W.W + 7 * stride;
  real_t *Tyz = W.W + 8 * stride;

  real_t *hVx  = W.hW;
  real_t *hVy  = W.hW + 1 * stride;
  real_t *hVz  = W.hW + 2 * stride;
  real_t *hTxx = W.hW + 3 * stride;
  real_t *hTyy = W.hW + 4 * stride;
  real_t *hTzz = W.hW + 5 * stride;
  real_t *hTxy = W.hW + 6 * stride;
  real_t *hTxz = W.hW + 7 * stride;
  real_t *hTyz = W.hW + 8 * stride;

  real_t *XIX = M;
  real_t *XIY = M + 1 * stride;
  real_t *XIZ = M + 2 * stride;
  real_t *ETX = M + 3 * stride;
  real_t *ETY = M + 4 * stride;
  real_t *ETZ = M + 5 * stride;
  real_t *ZTX = M + 6 * stride;
  real_t *ZTY = M + 7 * stride;
  real_t *ZTZ = M + 8 * stride;
#ifdef FreeSurface
  real_t *JAC = M + 9 * stride;
#endif
  real_t *LAM = M + 10 * stride;
  real_t *MIU = M + 11 * stride;
  real_t *RHO = M + 12 * stride;

  //int pos;//, slice;

  real_t DxTxx, DxTyy, DxTzz, DxTxy, DxTxz, DxTyz;
  real_t DyTxx, DyTyy, DyTzz, DyTxy, DyTxz, DyTyz;
  real_t DzTxx, DzTyy, DzTzz, DzTxy, DzTxz, DzTyz;
  real_t DxVx, DxVy, DxVz;
  real_t DyVx, DyVy, DyVz;
  real_t DzVx, DzVy, DzVz;

  real_t xix, xiy, xiz;
  real_t etx, ety, etz;
  real_t ztx, zty, ztz;
  real_t lam, mu, lam2mu;
  real_t rrho;

  real_t rb1, rb2, rb3;

#ifdef FreeSurface
  real_t vecT11[7],vecT12[7],vecT13[7];
  real_t vecT21[7],vecT22[7],vecT23[7];
  real_t vecT31[7],vecT32[7],vecT33[7];
  real_t DxT11, DxT12, DxT13;
  real_t DyT21, DyT22, DyT23;
  real_t DzT31, DzT32, DzT33;
  real_t DzVx1, DzVx2;
  real_t DzVy1, DzVy2;
  real_t DzVz1, DzVz2;
#endif

  real_t Dx[WSIZE], Dy[WSIZE], Dz[WSIZE];

  if (i < ni && j < nj && k < nk) {

    long pos = j1 + k1 * ny + i1 * ny * nz;

    lam = LAM[pos]; mu = MIU[pos];
    rrho = 1.0f/RHO[pos];
    lam2mu = lam + 2.0f*mu;

    //=========================================================================
    // X-derivatives
    stride = FlagX * ny * nz; coef = FlagX * rDH;
    DxTxx = MacCormack(Txx, pos, stride, coef);
    DxTyy = MacCormack(Tyy, pos, stride, coef);
    DxTzz = MacCormack(Tzz, pos, stride, coef);
    DxTxy = MacCormack(Txy, pos, stride, coef);
    DxTxz = MacCormack(Txz, pos, stride, coef);
    DxTyz = MacCormack(Tyz, pos, stride, coef);
    DxVx  = MacCormack(Vx , pos, stride, coef);
    DxVy  = MacCormack(Vy , pos, stride, coef);
    DxVz  = MacCormack(Vz , pos, stride, coef);

    //stride = ny*nz;
    //DxTxx = L(Txx, pos, stride, FlagX) * rDH;
    //DxTyy = L(Tyy, pos, stride, FlagX) * rDH;
    //DxTzz = L(Tzz, pos, stride, FlagX) * rDH;
    //DxTxy = L(Txy, pos, stride, FlagX) * rDH;
    //DxTxz = L(Txz, pos, stride, FlagX) * rDH;
    //DxTyz = L(Tyz, pos, stride, FlagX) * rDH;
    //DxVx  = L(Vx , pos, stride, FlagX) * rDH;
    //DxVy  = L(Vy , pos, stride, FlagX) * rDH;
    //DxVz  = L(Vz , pos, stride, FlagX) * rDH;

    xix = XIX[pos]; xiy = XIY[pos]; xiz = XIZ[pos];

    Dx[0] = (DxTxx*xix + DxTxy*xiy + DxTxz*xiz)*rrho;
    Dx[1] = (DxTxy*xix + DxTyy*xiy + DxTyz*xiz)*rrho;
    Dx[2] = (DxTxz*xix + DxTyz*xiy + DxTzz*xiz)*rrho;
    Dx[3] = (DxVx*xix*lam2mu + DxVy*xiy*lam    + DxVz*xiz*lam   );
    Dx[4] = (DxVx*xix*lam    + DxVy*xiy*lam2mu + DxVz*xiz*lam   );
    Dx[5] = (DxVx*xix*lam    + DxVy*xiy*lam    + DxVz*xiz*lam2mu);
    Dx[6] = (DxVx*xiy + DxVy*xix)*mu;
    Dx[7] = (DxVx*xiz + DxVz*xix)*mu;
    Dx[8] = (DxVy*xiz + DxVz*xiy)*mu;

    //=========================================================================
    // Y-derivatives
    stride = FlagY; coef = FlagY * rDH;
    DyTxx = MacCormack(Txx, pos, stride, coef);
    DyTyy = MacCormack(Tyy, pos, stride, coef);
    DyTzz = MacCormack(Tzz, pos, stride, coef);
    DyTxy = MacCormack(Txy, pos, stride, coef);
    DyTxz = MacCormack(Txz, pos, stride, coef);
    DyTyz = MacCormack(Tyz, pos, stride, coef);
    DyVx  = MacCormack(Vx , pos, stride, coef);
    DyVy  = MacCormack(Vy , pos, stride, coef);
    DyVz  = MacCormack(Vz , pos, stride, coef);

    //stride = 1;
    //DyTxx = L(Txx, pos, stride, FlagY);
    //DyTyy = L(Tyy, pos, stride, FlagY);
    //DyTzz = L(Tzz, pos, stride, FlagY);
    //DyTxy = L(Txy, pos, stride, FlagY);
    //DyTxz = L(Txz, pos, stride, FlagY);
    //DyTyz = L(Tyz, pos, stride, FlagY);
    //DyVx  = L(Vx , pos, stride, FlagY);
    //DyVy  = L(Vy , pos, stride, FlagY);
    //DyVz  = L(Vz , pos, stride, FlagY);

    etx = ETX[pos]; ety = ETY[pos]; etz = ETZ[pos];

    Dy[0] = (DyTxx*etx + DyTxy*ety + DyTxz*etz)*rrho;
    Dy[1] = (DyTxy*etx + DyTyy*ety + DyTyz*etz)*rrho;
    Dy[2] = (DyTxz*etx + DyTyz*ety + DyTzz*etz)*rrho;
    Dy[3] = (DyVx*etx*lam2mu + DyVy*ety*lam    + DyVz*etz*lam);
    Dy[4] = (DyVx*etx*lam    + DyVy*ety*lam2mu + DyVz*etz*lam);
    Dy[5] = (DyVx*etx*lam    + DyVy*ety*lam    + DyVz*etz*lam2mu);
    Dy[6] = (DyVx*ety + DyVy*etx)*mu;
    Dy[7] = (DyVx*etz + DyVz*etx)*mu;
    Dy[8] = (DyVy*etz + DyVz*ety)*mu;

    //=========================================================================
    // Z-derivatives
    stride = FlagZ * ny; coef = FlagZ * rDH;
    DzTxx = MacCormack(Txx, pos, stride, coef);
    DzTyy = MacCormack(Tyy, pos, stride, coef);
    DzTzz = MacCormack(Tzz, pos, stride, coef);
    DzTxy = MacCormack(Txy, pos, stride, coef);
    DzTxz = MacCormack(Txz, pos, stride, coef);
    DzTyz = MacCormack(Tyz, pos, stride, coef);
    DzVx  = MacCormack(Vx , pos, stride, coef);
    DzVy  = MacCormack(Vy , pos, stride, coef);
    DzVz  = MacCormack(Vz , pos, stride, coef);

    //stride = ny;
    //DzTxx = L(Txx, pos, stride, FlagZ);
    //DzTyy = L(Tyy, pos, stride, FlagZ);
    //DzTzz = L(Tzz, pos, stride, FlagZ);
    //DzTxy = L(Txy, pos, stride, FlagZ);
    //DzTxz = L(Txz, pos, stride, FlagZ);
    //DzTyz = L(Tyz, pos, stride, FlagZ);
    //DzVx  = L(Vx , pos, stride, FlagZ);
    //DzVy  = L(Vy , pos, stride, FlagZ);
    //DzVz  = L(Vz , pos, stride, FlagZ);

    ztx = ZTX[pos]; zty = ZTY[pos]; ztz = ZTZ[pos];

#ifdef FreeSurface
    // Get Lz by Lx and Ly
    if(par.freenode && k == nk-1) {
      int ij = (j1 + i1 * ny) * 9;
      DzVx1 = W.matVx2Vz[ij+3*0+0] * DxVx
            + W.matVx2Vz[ij+3*0+1] * DxVy
            + W.matVx2Vz[ij+3*0+2] * DxVz;
      DzVx2 = W.matVy2Vz[ij+3*0+0] * DyVx
            + W.matVy2Vz[ij+3*0+1] * DyVy
            + W.matVy2Vz[ij+3*0+2] * DyVz;

      DzVy1 = W.matVx2Vz[ij+3*1+0] * DxVx
            + W.matVx2Vz[ij+3*1+1] * DxVy
            + W.matVx2Vz[ij+3*1+2] * DxVz;
      DzVy2 = W.matVy2Vz[ij+3*1+0] * DyVx
            + W.matVy2Vz[ij+3*1+1] * DyVy
            + W.matVy2Vz[ij+3*1+2] * DyVz;

      DzVz1 = W.matVx2Vz[ij+3*2+0] * DxVx
            + W.matVx2Vz[ij+3*2+1] * DxVy
            + W.matVx2Vz[ij+3*2+2] * DxVz;
      DzVz2 = W.matVy2Vz[ij+3*2+0] * DyVx
            + W.matVy2Vz[ij+3*2+1] * DyVy
            + W.matVy2Vz[ij+3*2+2] * DyVz;

      DzVx = DzVx1 + DzVx2;
      DzVy = DzVy1 + DzVy2;
      DzVz = DzVz1 + DzVz2;
    } else if(par.freenode && k == nk-2) {
      // Get Lz directly
      stride = ny * FlagZ; coef = FlagZ * rDH;
      DzVx = MacCormack22(Vx, pos, stride, coef);
      DzVy = MacCormack22(Vy, pos, stride, coef);
      DzVz = MacCormack22(Vz, pos, stride, coef);
    } else if(par.freenode && k == nk-3) {
      // Get Lz directly
      stride = ny * FlagZ; coef = FlagZ * rDH;
      DzVx = MacCormack24(Vx, pos, stride, coef);
      DzVy = MacCormack24(Vy, pos, stride, coef);
      DzVz = MacCormack24(Vz, pos, stride, coef);
    }
#endif

    Dz[0] = (DzTxx*ztx + DzTxy*zty + DzTxz*ztz)*rrho;
    Dz[1] = (DzTxy*ztx + DzTyy*zty + DzTyz*ztz)*rrho;
    Dz[2] = (DzTxz*ztx + DzTyz*zty + DzTzz*ztz)*rrho;
    Dz[3] = (DzVx*ztx*lam2mu + DzVy*zty*lam    + DzVz*ztz*lam   );
    Dz[4] = (DzVx*ztx*lam    + DzVy*zty*lam2mu + DzVz*ztz*lam   );
    Dz[5] = (DzVx*ztx*lam    + DzVy*zty*lam    + DzVz*ztz*lam2mu);
    Dz[6] = (DzVx*zty + DzVy*ztx)*mu;
    Dz[7] = (DzVx*ztz + DzVz*ztx)*mu;
    Dz[8] = (DzVy*ztz + DzVz*zty)*mu;

#ifdef usePML
    rb1 = 1.0f/P.Bx[i];
    rb2 = 1.0f/P.By[j];
    rb3 = 1.0f/P.Bz[k];
#else
    rb1 = 1.0f;
    rb2 = 1.0f;
    rb3 = 1.0f;
#endif

    hVx [pos] = Dx[0] * rb1 + Dy[0] * rb2 + Dz[0] * rb3;
    hVy [pos] = Dx[1] * rb1 + Dy[1] * rb2 + Dz[1] * rb3;
    hVz [pos] = Dx[2] * rb1 + Dy[2] * rb2 + Dz[2] * rb3;
    hTxx[pos] = Dx[3] * rb1 + Dy[3] * rb2 + Dz[3] * rb3;
    hTyy[pos] = Dx[4] * rb1 + Dy[4] * rb2 + Dz[4] * rb3;
    hTzz[pos] = Dx[5] * rb1 + Dy[5] * rb2 + Dz[5] * rb3;
    hTxy[pos] = Dx[6] * rb1 + Dy[6] * rb2 + Dz[6] * rb3;
    hTxz[pos] = Dx[7] * rb1 + Dy[7] * rb2 + Dz[7] * rb3;
    hTyz[pos] = Dx[8] * rb1 + Dy[8] * rb2 + Dz[8] * rb3;

#ifdef Conservative
    real_t rrhojac = rrho/JAC[pos];

    for (int l = 0; l < 7; l++) {

      int l3 = l - 3; // -3 -2 -1 0 +1 +2 +3
      // changed "pos" here, which is a dangerous operation, remember
      // to reset "pos" when updating wavefield later

      pos = j1 + k1 * ny + (i1 + l3) * ny * nz;
      vecT11[l] = ( XIX[pos] * Txx[pos]
                  + XIY[pos] * Txy[pos]
                  + XIZ[pos] * Txz[pos] ) * JAC[pos];
      vecT12[l] = ( XIX[pos] * Txy[pos]
                  + XIY[pos] * Tyy[pos]
                  + XIZ[pos] * Tyz[pos] ) * JAC[pos];
      vecT13[l] = ( XIX[pos] * Txz[pos]
                  + XIY[pos] * Tyz[pos]
                  + XIZ[pos] * Tzz[pos] ) * JAC[pos];

      pos = (j1 + l3) + k1 * ny + i1 * ny * nz;
      vecT21[l] = ( ETX[pos] * Txx[pos]
                  + ETY[pos] * Txy[pos]
                  + ETZ[pos] * Txz[pos] ) * JAC[pos];
      vecT22[l] = ( ETX[pos] * Txy[pos]
                  + ETY[pos] * Tyy[pos]
                  + ETZ[pos] * Tyz[pos] ) * JAC[pos];
      vecT23[l] = ( ETX[pos] * Txz[pos]
                  + ETY[pos] * Tyz[pos]
                  + ETZ[pos] * Tzz[pos] ) * JAC[pos];

      pos = j1 + (k1 + l3) * ny + i1 * ny * nz;
      vecT31[l] = ( ZTX[pos] * Txx[pos]
                  + ZTY[pos] * Txy[pos]
                  + ZTZ[pos] * Txz[pos] ) * JAC[pos];
      vecT32[l] = ( ZTX[pos] * Txy[pos]
                  + ZTY[pos] * Tyy[pos]
                  + ZTZ[pos] * Tyz[pos] ) * JAC[pos];
      vecT33[l] = ( ZTX[pos] * Txz[pos]
                  + ZTY[pos] * Tyz[pos]
                  + ZTZ[pos] * Tzz[pos] ) * JAC[pos];
    }

    stride = FlagX; coef = FlagX * rDH;
    DxT11 = MacCormack(vecT11,3,stride,coef);
    DxT12 = MacCormack(vecT12,3,stride,coef);
    DxT13 = MacCormack(vecT13,3,stride,coef);

    stride = FlagY; coef = FlagY * rDH;
    DyT21 = MacCormack(vecT21,3,stride,coef);
    DyT22 = MacCormack(vecT22,3,stride,coef);
    DyT23 = MacCormack(vecT23,3,stride,coef);

    stride = FlagZ; coef = FlagZ * rDH;
    DzT31 = MacCormack(vecT31,3,stride,coef);
    DzT32 = MacCormack(vecT32,3,stride,coef);
    DzT33 = MacCormack(vecT33,3,stride,coef);

    pos = j1 + k1 * ny + i1 * ny * nz; // reset the position of index !!!

    hVx[pos] = ( DxT11*rb1 + DyT21*rb2 + DzT31*rb3 ) * rrhojac;
    hVy[pos] = ( DxT12*rb1 + DyT22*rb2 + DzT32*rb3 ) * rrhojac;
    hVz[pos] = ( DxT13*rb1 + DyT23*rb2 + DzT33*rb3 ) * rrhojac;
#endif

    //hVx [pos] = Vx [pos];
    //hVy [pos] = Vy [pos];
    //hVz [pos] = Vz [pos];
    //hTxx[pos] = Txx[pos];
    //hTyy[pos] = Tyy[pos];
    //hTzz[pos] = Tzz[pos];
    //hTxy[pos] = Txy[pos];
    //hTxz[pos] = Txz[pos];
    //hTyz[pos] = Tyz[pos];
    //xix = 1.0; xiy = 0.0; xiz = 0.0;
    //etx = 0.0; ety = 1.0; etz = 0.0;
    //ztx = 0.0; zty = 0.0; ztz = 1.0;
/*
    // Moment equation
    hVx[pos] = ( (DxTxx*xix + DxTxy*xiy + DxTxz*xiz)*rb1
               + (DyTxx*etx + DyTxy*ety + DyTxz*etz)*rb2
               + (DzTxx*ztx + DzTxy*zty + DzTxz*ztz)*rb3 ) * rrho;

    hVy[pos] = ( (DxTxy*xix + DxTyy*xiy + DxTyz*xiz)*rb1
               + (DyTxy*etx + DyTyy*ety + DyTyz*etz)*rb2
               + (DzTxy*ztx + DzTyy*zty + DzTyz*ztz)*rb3 ) * rrho;

    hVz[pos] = ( (DxTxz*xix + DxTyz*xiy + DxTzz*xiz)*rb1
               + (DyTxz*etx + DyTyz*ety + DyTzz*etz)*rb2
               + (DzTxz*ztx + DzTyz*zty + DzTzz*ztz)*rb3 ) * rrho;

    // Hooke's law
    hTxx[pos] = (lam2mu*DxVx*xix + lam*DxVy*xiy + lam*DxVz*xiz)*rb1
              + (lam2mu*DyVx*etx + lam*DyVy*ety + lam*DyVz*etz)*rb2
              + (lam2mu*DzVx*ztx + lam*DzVy*zty + lam*DzVz*ztz)*rb3;

    hTyy[pos] = (lam*DxVx*xix + lam2mu*DxVy*xiy + lam*DxVz*xiz)*rb1
              + (lam*DyVx*etx + lam2mu*DyVy*ety + lam*DyVz*etz)*rb2
              + (lam*DzVx*ztx + lam2mu*DzVy*zty + lam*DzVz*ztz)*rb3;

    hTzz[pos] = (lam*DxVx*xix + lam*DxVy*xiy + lam2mu*DxVz*xiz)*rb1
              + (lam*DyVx*etx + lam*DyVy*ety + lam2mu*DyVz*etz)*rb2
              + (lam*DzVx*ztx + lam*DzVy*zty + lam2mu*DzVz*ztz)*rb3;

    hTxy[pos] = mu * ( (DxVx*xiy + DxVy*xix)*rb1
                     + (DyVx*ety + DyVy*etx)*rb2
                     + (DzVx*zty + DzVy*ztx)*rb3 );

    hTxz[pos] = mu * ( (DxVx*xiz + DxVz*xix)*rb1
                     + (DyVx*etz + DyVz*etx)*rb2
                     + (DzVx*ztz + DzVz*ztx)*rb3 );

    hTyz[pos] = mu * ( (DxVy*xiz + DxVz*xiy)*rb1
                     + (DyVy*etz + DyVz*ety)*rb2
                     + (DzVy*ztz + DzVz*zty)*rb3 );
*/
#ifdef FreeSurface
    // ==============================================================
    //              Traction Image Method
    // ==============================================================
    if(par.freenode && (k >= (nk-3)) && (k <= (nk-1)) ) {
      //  if not free surface then do nothing!
      //
      //  use Traction Image method to calculate fd of stress compoents,
      //  then assemble the right hand side to update velocities
      //
      int ik = nk - 1 - k; // ik = 0, 1, 2
      real_t rrhojac = rrho/JAC[pos];

      for (int l = 0; l < 7; l++) {

        int l3 = l - 3; // -3 -2 -1 0 +1 +2 +3
        // changed "pos" here, which is a dangerous operation, remember
        // to reset "pos" when updating wavefield later

        pos = j1 + k1 * ny + (i1 + l3) * ny * nz;
        vecT11[l] = ( XIX[pos] * Txx[pos]
                    + XIY[pos] * Txy[pos]
                    + XIZ[pos] * Txz[pos] ) * JAC[pos];
        vecT12[l] = ( XIX[pos] * Txy[pos]
                    + XIY[pos] * Tyy[pos]
                    + XIZ[pos] * Tyz[pos] ) * JAC[pos];
        vecT13[l] = ( XIX[pos] * Txz[pos]
                    + XIY[pos] * Tyz[pos]
                    + XIZ[pos] * Tzz[pos] ) * JAC[pos];

        pos = (j1 + l3) + k1 * ny + i1 * ny * nz;
        vecT21[l] = ( ETX[pos] * Txx[pos]
                    + ETY[pos] * Txy[pos]
                    + ETZ[pos] * Txz[pos] ) * JAC[pos];
        vecT22[l] = ( ETX[pos] * Txy[pos]
                    + ETY[pos] * Tyy[pos]
                    + ETZ[pos] * Tyz[pos] ) * JAC[pos];
        vecT23[l] = ( ETX[pos] * Txz[pos]
                    + ETY[pos] * Tyz[pos]
                    + ETZ[pos] * Tzz[pos] ) * JAC[pos];

        pos = j1 + (k1 + l3) * ny + i1 * ny * nz;
        vecT31[l] = ( ZTX[pos] * Txx[pos]
                    + ZTY[pos] * Txy[pos]
                    + ZTZ[pos] * Txz[pos] ) * JAC[pos];
        vecT32[l] = ( ZTX[pos] * Txy[pos]
                    + ZTY[pos] * Tyy[pos]
                    + ZTZ[pos] * Tyz[pos] ) * JAC[pos];
        vecT33[l] = ( ZTX[pos] * Txz[pos]
                    + ZTY[pos] * Tyz[pos]
                    + ZTZ[pos] * Tzz[pos] ) * JAC[pos];
      }

      vecT31[ik+3] = 0.0f;               // FreeSurface set to zero
      vecT32[ik+3] = 0.0f;               // FreeSurface set to zero
      vecT33[ik+3] = 0.0f;               // FreeSurface set to zero
      for (int l = 0; l < 3 - ik; l++) {
        vecT31[6-l] = -vecT31[l+2*ik];   // Traction Image
        vecT32[6-l] = -vecT32[l+2*ik];   // Traction Image
        vecT33[6-l] = -vecT33[l+2*ik];   // Traction Image
      }

      //DxT11 = vec_L(vecT11,3,FlagX) * rDH;
      //DxT12 = vec_L(vecT12,3,FlagX) * rDH;
      //DxT13 = vec_L(vecT13,3,FlagX) * rDH;
      //DyT21 = vec_L(vecT21,3,FlagY) * rDH;
      //DyT22 = vec_L(vecT22,3,FlagY) * rDH;
      //DyT23 = vec_L(vecT23,3,FlagY) * rDH;
      //DzT31 = vec_L(vecT31,3,FlagZ) * rDH;
      //DzT32 = vec_L(vecT32,3,FlagZ) * rDH;
      //DzT33 = vec_L(vecT33,3,FlagZ) * rDH;

      stride = FlagX; coef = FlagX * rDH;
      DxT11 = MacCormack(vecT11,3,stride,coef);
      DxT12 = MacCormack(vecT12,3,stride,coef);
      DxT13 = MacCormack(vecT13,3,stride,coef);

      stride = FlagY; coef = FlagY * rDH;
      DyT21 = MacCormack(vecT21,3,stride,coef);
      DyT22 = MacCormack(vecT22,3,stride,coef);
      DyT23 = MacCormack(vecT23,3,stride,coef);

      stride = FlagZ; coef = FlagZ * rDH;
      DzT31 = MacCormack(vecT31,3,stride,coef);
      DzT32 = MacCormack(vecT32,3,stride,coef);
      DzT33 = MacCormack(vecT33,3,stride,coef);

      pos = j1 + k1 * ny + i1 * ny * nz; // reset the position of index !!!

      hVx[pos] = ( DxT11*rb1 + DyT21*rb2 + DzT31*rb3 ) * rrhojac;
      hVy[pos] = ( DxT12*rb1 + DyT22*rb2 + DzT32*rb3 ) * rrhojac;
      hVz[pos] = ( DxT13*rb1 + DyT23*rb2 + DzT33*rb3 ) * rrhojac;

    } // end of freenode, Traction Image Method
#endif
    // ===================================================================

  } // end if of i, j, k

  return;
}

//extern "C"
void wave_deriv(Wave W, realptr_t M, PML P, int FlagX, int FlagY, int FlagZ)
{
  dim3 block(16, 4, 4);
  dim3 grid;
  grid.x = (hostParams.nj + block.x -1) / block.x;
  grid.y = (hostParams.nk + block.y -1) / block.y;
  grid.z = (hostParams.ni + block.z -1) / block.z;

  wave_deriv_cu <<<grid, block>>> (W, M, P, FlagX, FlagY, FlagZ);
  return;
}
