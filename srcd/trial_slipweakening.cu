#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

#ifdef TPV29
#define SW_rcrit 4000.0
#define SW_t0 0.5
#define Vs 3464.0
#endif

#if defined TPV22 || defined TPV23
#define SW_rcrit 3000.0
#define SW_t0 0.5
#define Vs 3464.0
#endif

//#define DEBUG
//extern __device__ __constant__ int Flags[8][3];
extern __device__ real_t norm3(real_t *A);
extern __device__ real_t dot_product(real_t *A, real_t *B);
extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void invert3x3(real_t m[][3]);
extern __device__ real_t Fr_func(const real_t r, const real_t R);
extern __device__ real_t Gt_func(const real_t t, const real_t T);

__global__ void trial_sw_cu(Wave w, Fault f, real_t *M,
    int it, int irk, int FlagX, int FlagY, int FlagZ, int i0, int nfault, int Faultgrid[])
{
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
//////////////////////////
  int gj = par.ranky * nj + j;
  int gj1 = gj + 1;
  int gk = par.rankz * nk + k;
  int gk1 = gk + 1;
//////////////////////////
  int nxyz = nx * ny * nz;
  int nyz = ny * nz;
  int nyz2 = 2 * ny * nz;
  int faultsize = nfault * nyz2 * 9;      //******wangzj
  int Tsize = nfault * nyz * 7;
  int mpifaultsize = nfault * nj*nk;

  real_t *XIX = M;
  real_t *XIY = XIX + nxyz;
  real_t *XIZ = XIY + nxyz;
  real_t *JAC = M + 9 * nxyz;
  //real_t *RHO = M + 12 * nxyz;

  //stride = nx * ny * nz; // x vary first
  //real_t *w_Vx  = w.W + 0 * nxyz;
  //real_t *w_Vy  = w.W + 1 * nxyz;
  //real_t *w_Vz  = w.W + 2 * nxyz;
  real_t *w_Txx = w.W + 3 * nxyz;
  real_t *w_Tyy = w.W + 4 * nxyz;
  real_t *w_Tzz = w.W + 5 * nxyz;
  real_t *w_Txy = w.W + 6 * nxyz;
  real_t *w_Txz = w.W + 7 * nxyz;
  real_t *w_Tyz = w.W + 8 * nxyz;

  // Split nodes
  //stride = ny * nz * 2; // y vary first
  real_t *f_Vx  = f.W + 0 * nyz2 + faultsize;
  real_t *f_Vy  = f.W + 1 * nyz2 + faultsize;
  real_t *f_Vz  = f.W + 2 * nyz2 + faultsize;
  real_t *f_T21 = f.W + 3 * nyz2 + faultsize;
  real_t *f_T22 = f.W + 4 * nyz2 + faultsize;
  real_t *f_T23 = f.W + 5 * nyz2 + faultsize;
  real_t *f_T31 = f.W + 6 * nyz2 + faultsize;
  real_t *f_T32 = f.W + 7 * nyz2 + faultsize;
  real_t *f_T33 = f.W + 8 * nyz2 + faultsize;

  real_t *f_mVx  = f.mW + 0 * nyz2 + faultsize;
  real_t *f_mVy  = f.mW + 1 * nyz2 + faultsize;
  real_t *f_mVz  = f.mW + 2 * nyz2 + faultsize;

  real_t *f_tVx  = f.tW + 0 * nyz2 + faultsize;
  real_t *f_tVy  = f.tW + 1 * nyz2 + faultsize;
  real_t *f_tVz  = f.tW + 2 * nyz2 + faultsize;

  real_t DH = par.DH;
  real_t rDH = 1.0/DH;
  real_t DT = par.DT;

  //int istep = it % 8;
  //int sign1 = irk % 2;
  //int FlagX = Flags[istep][0];
  //int FlagY = Flags[istep][1];
  //int FlagZ = Flags[istep][2];
  //int i0 = nx/2;                         *****wangzj

  real_t xix, xiy, xiz;
  //real_t etx, ety, etz;
  //real_t ztx, zty, ztz;
  real_t jac;
  real_t vec_n0;
  real_t jacvec;
  //real_t lam, mu;
  //real_t lam2mu;
  //real_t rrho;
  real_t rho;

  real_t Mrho[2], Rx[2], Ry[2], Rz[2];
  real_t DH2 = DH*DH;
  real_t T11, T12, T13;
  int i;
  int pos, pos_f, pos1;
  //int slice;
  real_t DyT21, DyT22, DyT23;
  real_t DzT31, DzT32, DzT33;
  real_t vecT31[7];
  real_t vecT32[7];
  real_t vecT33[7];

  //if(sign1) { FlagX *= -1; FlagY *= -1; FlagZ *= -1; }

  /* check RK flag */
  //if(100==j && 100 == k)printf("(%d, %d):%+d%+d%+d\n", it, irk, FlagX, FlagY, FlagZ);

//#ifdef FreeSurface
//  if ( j >= 30+3 && j < ny-31-3 && k >=30+3 && k < nz-3){  // not united
//#else
//  if ( j >= 30+3 && j < ny-31-3 && k >=30+3 && k < nz-31-3){  // not united
//#endif
//#ifdef FreeSurface
//  if ( j >= 30 && j < nj-31 && k >=30 && k < nk){  // not united
//#else
//  if ( j >= 30 && j < nj-31 && k >=30 && k < nk-31){  // not united
//#endif
  if ( j < nj && k < nk && f.united[j + k * nj + mpifaultsize] == 0){  // not united

    pos1 = j1 + k1 * ny + 3*nyz + Tsize;
    pos = j + k * nj + mpifaultsize;           //*****wangzj
    if(irk == 0){
      f.mT11[pos] = f.T11[pos1];
      f.mT12[pos] = f.T12[pos1];
      f.mT13[pos] = f.T13[pos1];
    }

    //km = NZ - (thisid[2]*nk+k-3);
    //int km = (nz - 6) - (k-3); // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)

    for (int m = 0; m < 2; m++){

      i = i0 + 2*m - 1; // i0-1, i0+1
      //pos_m = (i*ny*nz+j1*nz+k1)*MSIZE;
      //pos   = (i*ny*nz+j1*nz+k1)*WSIZE;
      //pos = k1 * ny * nx + j1 * nx + i;
      //pos = i * ny * nz + j1 * nz + k1;
      pos = j1 + k1 * ny + i * ny * nz;

      xix = XIX[pos];
      xiy = XIY[pos];
      xiz = XIZ[pos];
      // jac = JAC[pos];
      jac = JAC[j1 + k1 * ny + i0 * ny * nz];
      //rho = RHO[pos];
      // bimaterial
      rho = f.rho_f[j1 + k1 * ny + m * ny * nz + nfault * (nyz2)];     //***wangzj
      
      Mrho[m] = 0.5f*jac*rho*DH*DH2;
      //xix = M[pos_m + 0]; xiy = M[pos_m + 1]; xiz = M[pos_m + 2];
      //jac = M[pos_m + 9];
      //rho = M[pos_m + 12];

      // {Txx 3} {Tyy 4} {Tzz 5} {Txy 6} {Txz 7} {Tyz 8}
      // T1 is continuous!
      //T11 = jac*(xix * w.W[pos + 3] + xiy * w.W[pos + 6] + xiz * w.W[pos + 7]);
      //T12 = jac*(xix * w.W[pos + 6] + xiy * w.W[pos + 4] + xiz * w.W[pos + 8]);
      //T13 = jac*(xix * w.W[pos + 7] + xiy * w.W[pos + 8] + xiz * w.W[pos + 5]);

      //T11 = jac*(xix * w_Txx[pos] + xiy * w_Txy[pos] + xiz * w_Txz[pos]);
      //T12 = jac*(xix * w_Txy[pos] + xiy * w_Tyy[pos] + xiz * w_Tyz[pos]);
      //T13 = jac*(xix * w_Txz[pos] + xiy * w_Tyz[pos] + xiz * w_Tzz[pos]);

      // 0 or 2 ==> i0-1 or i0+1
      //f.T11[(2*m)*ny*nz+j1*nz+k1] = T11;
      //f.T12[(2*m)*ny*nz+j1*nz+k1] = T12;
      //f.T13[(2*m)*ny*nz+j1*nz+k1] = T13;
      //pos_f = j1 + k1 * ny;
      //f.T11[(2*m)*nyz + pos_f] = T11;
      //f.T12[(2*m)*nyz + pos_f] = T12;
      //f.T13[(2*m)*nyz + pos_f] = T13;
      pos_f = j1 + k1 * ny + Tsize;
      for (int l = 1; l <= 3; l++){
        pos = j1 + k1*ny + (i0+(2*m-1)*l)*ny*nz;
        xix = XIX[pos];
        xiy = XIY[pos];
        xiz = XIZ[pos];
        jac = JAC[pos];
        T11 = jac*(xix * w_Txx[pos] + xiy * w_Txy[pos] + xiz * w_Txz[pos]);
        T12 = jac*(xix * w_Txy[pos] + xiy * w_Tyy[pos] + xiz * w_Tyz[pos]);
        T13 = jac*(xix * w_Txz[pos] + xiy * w_Tyz[pos] + xiz * w_Tzz[pos]);
        f.T11[(3+(2*m-1)*l)*nyz + pos_f] = T11;
        f.T12[(3+(2*m-1)*l)*nyz + pos_f] = T12;
        f.T13[(3+(2*m-1)*l)*nyz + pos_f] = T13;
      }

      // {T21 3} {T22 4} {T23 5} {T31 6} {T32 7} {T33 8}
      //slice = nz*FSIZE;
      // bug fixed
      //pos_f = (m*ny*nz+j1*nz+k1)*FSIZE;
      //pos_f = k1 * ny + j1;
      real_t *t21 = f_T21 + m*nyz;
      real_t *t22 = f_T22 + m*nyz;
      real_t *t23 = f_T23 + m*nyz;
#ifdef RupSensor
      if(f.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){     //wangzj
#else
      if(f.rup_index_y[j + k * nj + mpifaultsize] % 7){         //wangzj
#endif
        //DyT21 = L22(f.W, (pos_f + 3), slice, FlagY) / DH;
        //DyT22 = L22(f.W, (pos_f + 4), slice, FlagY) / DH;
        //DyT23 = L22(f.W, (pos_f + 5), slice, FlagY) / DH;
        pos_f = j1 + k1 * ny;                                    //wangzj
        DyT21 = L22(t21, pos_f, 1, FlagY) * rDH;
        DyT22 = L22(t22, pos_f, 1, FlagY) * rDH;
        DyT23 = L22(t23, pos_f, 1, FlagY) * rDH;
      }else{
        pos_f = j1 + k1 * ny;                                    //wangzj
        DyT21 = L(t21, pos_f, 1, FlagY) * rDH;
        DyT22 = L(t22, pos_f, 1, FlagY) * rDH;
        DyT23 = L(t23, pos_f, 1, FlagY) * rDH;
        //DyT21 = L(f.W, (pos_f + 3), slice, FlagY) / DH;
        //DyT22 = L(f.W, (pos_f + 4), slice, FlagY) / DH;
        //DyT23 = L(f.W, (pos_f + 5), slice, FlagY) / DH;
      }

      for (int l = -3; l <=3 ; l++){
        //vecT31[l+3] = F->T31[m][j][k+l];
        //vecT32[l+3] = F->T32[m][j][k+l];
        //vecT33[l+3] = F->T33[m][j][k+l];
        //pos = (m*ny*nz + j1*nz + k1+l)*FSIZE;
        pos = j1 + (k1 + l) * ny;
        vecT31[l+3] = f_T31[pos + m*nyz];//f.W[pos + 6];
        vecT32[l+3] = f_T32[pos + m*nyz];//f.W[pos + 7];
        vecT33[l+3] = f_T33[pos + m*nyz];//f.W[pos + 8];
      }

      if(par.freenode && km<=3){
        ///extendvect(vecT31,km+2,0.0);
        ///extendvect(vecT32,km+2,0.0);
        ///extendvect(vecT33,km+2,0.0);
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (int l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }
      } // end par.freenode
      //else
      //{
      //  DzT31 = L(f.W, (pos_f + 6), segment, FlagZ) / DH;
      //  DzT32 = L(f.W, (pos_f + 7), segment, FlagZ) / DH;
      //  DzT33 = L(f.W, (pos_f + 8), segment, FlagZ) / DH;
      //}
#ifdef RupSensor
      if(f.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){           //wangzj
#else
      if(f.rup_index_z[j + k * nj + mpifaultsize] % 7){                     //wangzj
#endif
        DzT31 = vec_L22(vecT31, 3, FlagZ) * rDH;
        DzT32 = vec_L22(vecT32, 3, FlagZ) * rDH;
        DzT33 = vec_L22(vecT33, 3, FlagZ) * rDH;
      }else{
        DzT31 = vec_L(vecT31, 3, FlagZ) * rDH;
        DzT32 = vec_L(vecT32, 3, FlagZ) * rDH;
        DzT33 = vec_L(vecT33, 3, FlagZ) * rDH;
      }
      pos_f = j1 + k1 * ny + Tsize;
      T11 = f.T11[(3+2*m-1)*nyz+pos_f];
      T12 = f.T12[(3+2*m-1)*nyz+pos_f];
      T13 = f.T13[(3+2*m-1)*nyz+pos_f];

      Rx[m] = 0.5f*( (2*m-1)*T11 + (DyT21 + DzT31)*DH )*DH2;
      Ry[m] = 0.5f*( (2*m-1)*T12 + (DyT22 + DzT32)*DH )*DH2;
      Rz[m] = 0.5f*( (2*m-1)*T13 + (DyT23 + DzT33)*DH )*DH2;
#ifdef TractionImg
      if (m == 0){ // "-" side
        Rx[m] =
          a_1 * f.T11[2*nyz+pos_f] +
          a_2 * f.T11[1*nyz+pos_f] +
          a_3 * f.T11[0*nyz+pos_f] ;
        Ry[m] =
          a_1 * f.T12[2*nyz+pos_f] +
          a_2 * f.T12[1*nyz+pos_f] +
          a_3 * f.T12[0*nyz+pos_f] ;
        Rz[m] =
          a_1 * f.T13[2*nyz+pos_f] +
          a_2 * f.T13[1*nyz+pos_f] +
          a_3 * f.T13[0*nyz+pos_f] ;
      }else{ // "+" side
        Rx[m] =
          a_1 * f.T11[4*nyz+pos_f] +
          a_2 * f.T11[5*nyz+pos_f] +
          a_3 * f.T11[6*nyz+pos_f] ;
        Ry[m] =
          a_1 * f.T12[4*nyz+pos_f] +
          a_2 * f.T12[5*nyz+pos_f] +
          a_3 * f.T12[6*nyz+pos_f] ;
        Rz[m] =
          a_1 * f.T13[4*nyz+pos_f] +
          a_2 * f.T13[5*nyz+pos_f] +
          a_3 * f.T13[6*nyz+pos_f] ;
      }

      Rx[m] = 0.5f*( (2*m-1)*Rx[m] + (DyT21 + DzT31)*DH )*DH2;
      Ry[m] = 0.5f*( (2*m-1)*Ry[m] + (DyT22 + DzT32)*DH )*DH2;
      Rz[m] = 0.5f*( (2*m-1)*Rz[m] + (DyT23 + DzT33)*DH )*DH2;
#endif

      

    } // end m

    real_t t;
    if(irk == 0){
      t = it*DT;
    }else if (irk == 1 || irk == 2){
      t = (it+0.5)*DT;
    }else{
      t = (it+1)*DT;
    }
    if (1 == par.INPORT_STRESS_TYPE){
      pos = j + k * nj + mpifaultsize;          //********wangzj
      real_t Gt = Gt_func(t, 1.0);
      Gt = Gt_func(t, par.smooth_load_T);
      f.str_init_x[pos] = f.T0x[pos] + Gt * f.dT0x[pos];
      f.str_init_y[pos] = f.T0y[pos] + Gt * f.dT0y[pos];
      f.str_init_z[pos] = f.T0z[pos] + Gt * f.dT0z[pos];
    }
    //pos  = (0*ny*nz+j1*nz+k1)*FSIZE;
    //pos1 = (1*ny*nz+j1*nz+k1)*FSIZE;
    //real_t Vx1 = f.mW[pos1 + 0] - f.mW[pos + 0];
    //real_t Vy1 = f.mW[pos1 + 1] - f.mW[pos + 1];
    //real_t Vz1 = f.mW[pos1 + 2] - f.mW[pos + 2];
    //real_t dVx = f.W[pos1 + 0] - f.W[pos + 0];
    //real_t dVy = f.W[pos1 + 1] - f.W[pos + 1] + 1e-12;
    //real_t dVz = f.W[pos1 + 2] - f.W[pos + 2];
    pos = j1 + k1 * ny;
    real_t dVx = f_mVx[pos + nyz] - f_mVx[pos];
    real_t dVy = f_mVy[pos + nyz] - f_mVy[pos];
    real_t dVz = f_mVz[pos + nyz] - f_mVz[pos];
#ifdef RKtrial
    if (irk == 3){
      dVx = f_tVx[pos + nyz] - f_tVx[pos];
      dVy = f_tVy[pos + nyz] - f_tVy[pos];
      dVz = f_tVz[pos + nyz] - f_tVz[pos];
    }
#endif

    real_t Trial[3]; // stress variation
    real_t Trial_local[3]; // + init background stress
    real_t Trial_s[3]; // shear stress

    real_t dt1 = DT;
#ifdef RKtrial
    if(irk == 0 || irk == 1){
      dt1 *= 0.5;
    }else if (irk == 3){
      dt1 /= 6.0;
    }
#endif

    Trial[0] = (Mrho[0]*Mrho[1]*dVx/dt1 + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(DH2*(Mrho[0]+Mrho[1]))*2.0f;
    Trial[1] = (Mrho[0]*Mrho[1]*dVy/dt1 + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(DH2*(Mrho[0]+Mrho[1]))*2.0f;
    Trial[2] = (Mrho[0]*Mrho[1]*dVz/dt1 + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(DH2*(Mrho[0]+Mrho[1]))*2.0f;
#ifdef TractionImg
    real_t a0p,a0m;
    if(FlagX==FWD){
      a0p = a_0pF;
      a0m = a_0mF;
    }else{
      a0p = a_0pB;
      a0m = a_0mB;
    }
    Trial[0] = -(Mrho[0]*Mrho[1]*dVx/dt1 + Mrho[0]*Rx[1] - Mrho[1]*Rx[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;
    Trial[1] = -(Mrho[0]*Mrho[1]*dVy/dt1 + Mrho[0]*Ry[1] - Mrho[1]*Ry[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;
    Trial[2] = -(Mrho[0]*Mrho[1]*dVz/dt1 + Mrho[0]*Rz[1] - Mrho[1]*Rz[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1]))*2.0f;
#endif

    //if(j==nj/2 && k==nk/2) printf("Trial = %e %e %e\n", Trial[0], Trial[1], Trial[2]);
    //real_t jacvec = 1; // J*|xi|

    //pos = j*nz + k;
    //Trial_local[0] = Trial[0]/jacvec + f.str_init_x[pos];
    //Trial_local[1] = Trial[1]/jacvec + f.str_init_y[pos];
    //Trial_local[2] = Trial[2]/jacvec + f.str_init_z[pos];

    //real_t vec_n [3] = {1, 0, 0};
    //real_t vec_s1[3] = {0, 1, 0};
    //real_t vec_s2[3] = {0, 0, 1};
    real_t vec_n [3];
    real_t vec_s1[3];
    real_t vec_s2[3];


    //vec_n [0] = 1.f;//f.vec_n[pos + 0];
    //vec_n [1] = 0.f;//f.vec_n[pos + 1];
    //vec_n [2] = 0.f;//f.vec_n[pos + 2];
    //vec_s1[0] = 0.f;//f.vec_s1[pos + 0];
    //vec_s1[1] = 1.f;//f.vec_s1[pos + 1];
    //vec_s1[2] = 0.f;//f.vec_s1[pos + 2];
    //vec_s2[0] = 0.f;//f.vec_s2[pos + 0];
    //vec_s2[1] = 0.f;//f.vec_s2[pos + 1];
    //vec_s2[2] = 1.f;//f.vec_s2[pos + 2];

    //pos = (j1*nz + k1)*3;
    pos = (j1 + k1 * ny) * 3 + nfault * (nyz*3);        //********wangzj

    //vec_n [0] = f.vec_n[pos + 0];
    //vec_n [1] = f.vec_n[pos + 1];
    //vec_n [2] = f.vec_n[pos + 2];
    vec_s1[0] = f.vec_s1[pos + 0];
    vec_s1[1] = f.vec_s1[pos + 1];
    vec_s1[2] = f.vec_s1[pos + 2];
    vec_s2[0] = f.vec_s2[pos + 0];
    vec_s2[1] = f.vec_s2[pos + 1];
    vec_s2[2] = f.vec_s2[pos + 2];

    //pos = (i0 * ny * nz + j1 * nz + k1) * MSIZE;
    //pos = k1 * ny * nx + j1 * nx + i0;
    pos = j1 + k1 * ny + i0 * ny * nz;
    vec_n[0] = XIX[pos];//M[pos + 0];
    vec_n[1] = XIY[pos];//M[pos + 1];
    vec_n[2] = XIZ[pos];//M[pos + 2];
    vec_n0 = norm3(vec_n);

    jacvec = JAC[pos] * vec_n0;

    for (int ii = 0; ii < 3; ++ii)
    {
        vec_n[ii] /= vec_n0;
    }

    pos = j + k * nj + mpifaultsize;        //*********wangzj
    Trial_local[0] = Trial[0]/jacvec + f.str_init_x[pos];
    Trial_local[1] = Trial[1]/jacvec + f.str_init_y[pos];
    Trial_local[2] = Trial[2]/jacvec + f.str_init_z[pos];

    real_t Trial_n = dot_product(Trial_local, vec_n);
    // Ts = T - n * Tn
    Trial_s[0] = Trial_local[0] - vec_n[0]*Trial_n;
    Trial_s[1] = Trial_local[1] - vec_n[1]*Trial_n;
    Trial_s[2] = Trial_local[2] - vec_n[2]*Trial_n;

#ifdef TPV29
    real_t depth = (par.NZ - (k + nk * par.rankz + 1)) * DH;
    real_t Pf = 9.8e3 * depth;
    Trial_n += Pf;
#endif

    real_t Trial_s0;
    //Trial_s0 = dot_product(Trial_s, Trial_s);
    //Trial_s0 = sqrtf(Trial_s0);
    Trial_s0 = norm3(Trial_s);

    real_t Tau_n = Trial_n;
    real_t Tau_s = Trial_s0;

    real_t ifchange = 0; // false
    if(Trial_n >= 0.0){
      // fault can not open
      Tau_n = 0.0;
      ifchange = 1;
    }else{
      Tau_n = Trial_n;
    }

//#define Regularize
#ifdef Regularize
    ////// Modified Prakash-Clifton regularization
    ////// #1 d(sigma_eff)/dt = -1/T_sigma * (sigma_eff - sigma)
    ////// #2 d(sigma_eff)/dt = -((abs(V)+V_ref)/L_sigma * (sigma_eff - sigma)
    //////pos = j + k * nj;
    //////real_t T_sigma = 0.02 * par.Dc;
    //////real_t coef = exp(-par.DT/T_sigma);
    //////real_t Tau_n_eff = Tau_n + coef * (f.tTn[pos] - Tau_n);

    //////// using effective normal stress
    ////////if(irk == 0) Tau_n = Tau_n_eff;
    //////real_t Vs1 = f.Vs1[pos];
    //////real_t Vs2 = f.Vs2[pos];
    //////real_t rate = sqrt(Vs1*Vs1+Vs2*Vs2);
    //////if(rate >1e-2) Tau_n = Tau_n_eff;
    ////// Modified Prakash-Clifton regularization
    ////// #1 d(sigma_eff)/dt = -1/T_sigma * (sigma_eff - sigma)
    ////// #2 d(sigma_eff)/dt = -((abs(V)+V_ref)/L_sigma * (sigma_eff - sigma)

    pos = j + k * nj + mpifaultsize;           //*****wangzj

    // Prakash-Clifton #1
    real_t V_ref = 0.2;
    real_t delta_L = 0.2 * f.Dc[pos];

    real_t T_sigma = delta_L / V_ref;
    real_t coef = exp(-par.DT/T_sigma);
    real_t Tau_n_eff = Tau_n + coef * (f.tTn[pos] - Tau_n);

    real_t Vs1 = f.Vs1[pos];
    real_t Vs2 = f.Vs2[pos];
    real_t rate = sqrt(Vs1*Vs1+Vs2*Vs2);
    //real_t rate = fabs(Vs1);

    // Prakash-Clifton #2
    coef = par.DT / delta_L;
    Tau_n_eff = Tau_n + exp(-(rate + V_ref)*coef) * (f.tTn[pos] - Tau_n);

    // using effective normal stress
    //if(irk == 0) Tau_n = Tau_n_eff;
    if(rate >1e-2) Tau_n = Tau_n_eff;
#endif


    pos = j + k * nj + mpifaultsize;       //******wangzj
    real_t mu_s1 = f.str_peak[pos];
    real_t slip = f.slip[pos];
    real_t friction;
    real_t Dc = f.Dc[pos];
    real_t mu_d = f.mu_d[pos];

    // slip weakening
    if(slip <= Dc){
      friction = mu_s1 - (mu_s1 - mu_d) * slip / Dc;
    }else{
      friction = mu_d;
    }
#ifdef TPV29
    real_t dist;
    real_t SW_T;
    real_t SW_f1, SW_f2;
    real_t cur_time = it * par.DT;
    if(irk == 0){
      cur_time = it * par.DT;
    }else if(irk == 1 || irk == 2){
      cur_time = (it + 0.5) * par.DT;
    }else{
      cur_time = (it + 1.0) * par.DT;
    }

    int gj = j + nj * par.ranky;
    int gk = k + nk * par.rankz;

    int gj1 = gj + 1;
    int gk1 = gk + 1;

    int srcj = par.NY / 2 - int(5e3/DH);
    int srck = par.NZ - int(10e3/DH);

    real_t jj = (gj1 - srcj) * DH;
    real_t kk = (gk1 - srck) * DH;
    dist = sqrt(jj*jj + kk*kk);

    if(dist < SW_rcrit){
      SW_T = dist/(0.7*Vs)+0.081*SW_rcrit/(0.7*Vs)*(1.0/(1.0-pow(dist/SW_rcrit,2))-1.0);
    }else{
      SW_T = 1e9;
    }

    if(slip < Dc){
      SW_f1 = slip/Dc;
    }else{
      SW_f1 = 1.0;
    }
    if(cur_time < SW_T){
      SW_f2 = 0.0;
    }else if(cur_time < SW_T + SW_t0){
      SW_f2 = (cur_time - SW_T)/SW_t0;
    }else{
      SW_f2 = 1.0;
    }

    real_t SW_f;
    SW_f = MAX(SW_f1, SW_f2);
    friction = mu_s1 - (mu_s1 - mu_d) * SW_f;

    //friction = mu_d;

#endif

#if defined TPV22 || defined TPV23
    real_t dist;
    real_t SW_T;
    real_t SW_f1, SW_f2;
    real_t cur_time = it * par.DT;
    if(irk == 0){
      cur_time = it * par.DT;
    }else if(irk == 1 || irk == 2){
      cur_time = (it + 0.5) * par.DT;
    }else{
      cur_time = (it + 1.0) * par.DT;
    }

    int gj = j + nj * par.ranky;
    int gk = k + nk * par.rankz;

    int gj1 = gj + 1;
    int gk1 = gk + 1;

    int srcj = par.NY / 2 - int(10e3/DH);
    int srck = par.NZ - int(10e3/DH);

    real_t jj = (gj1 - srcj) * DH;
    real_t kk = (gk1 - srck) * DH;
    dist = sqrt(jj*jj + kk*kk);
  #ifdef TPV22
    if(dist < SW_rcrit && nfault == 0)
  #endif
  #ifdef TPV23
    if(dist < SW_rcrit && nfault == 1)
  #endif
    {
      SW_T = dist/(0.7*Vs)+0.081*SW_rcrit/(0.7*Vs)*(1.0/(1.0-pow(dist/SW_rcrit,2))-1.0);
    }else{
      SW_T = 1e9;
    }

    if(slip < Dc){
      SW_f1 = slip/Dc;
    }else{
      SW_f1 = 1.0;
    }
    if(cur_time < SW_T){
      SW_f2 = 0.0;
    }else if(cur_time < SW_T + SW_t0){
      SW_f2 = (cur_time - SW_T)/SW_t0;
    }else{
      SW_f2 = 1.0;
    }

    real_t SW_f;
    SW_f = MAX(SW_f1, SW_f2);
    friction = mu_s1 - (mu_s1 - mu_d) * SW_f;

    //friction = mu_d;

#endif

    //real_t C0 = 0;
    real_t Tau_c = -friction * Tau_n + f.C0[pos];

    if(Trial_s0 >= Tau_c){
      Tau_s = Tau_c; // can not exceed shear strengh!
      //f.flag_rup[j*nk + k] = 1;
      f.flag_rup[j + k * nj + mpifaultsize] = 1;       //wangzj
      ifchange = 1;
    }else{
      Tau_s = Trial_s0;
      //f.flag_rup[j*nk + k] = 0;
      f.flag_rup[j + k * nj + mpifaultsize] = 0;       //wangzj
    }
#ifdef TPV29
    Tau_n -= Pf;
#endif

    real_t Tau[3];
    if(ifchange){
      // to avoid divide by 0, 1e-1 is a small value compared to stress
      if(fabsf(Trial_s0) < 1e-1){
        Tau[0] = Tau_n * vec_n[0];
        Tau[1] = Tau_n * vec_n[1];
        Tau[2] = Tau_n * vec_n[2];
      }else{
        //Tau[0] = Tau_n * vec_n[0] + Tau_s * Trial_s[0]/Trial_s0;
        //Tau[1] = Tau_n * vec_n[1] + Tau_s * Trial_s[1]/Trial_s0;
        //Tau[2] = Tau_n * vec_n[2] + Tau_s * Trial_s[2]/Trial_s0;
        Tau[0] = Tau_n * vec_n[0] + (Tau_s/Trial_s0) * Trial_s[0];
        Tau[1] = Tau_n * vec_n[1] + (Tau_s/Trial_s0) * Trial_s[1];
        Tau[2] = Tau_n * vec_n[2] + (Tau_s/Trial_s0) * Trial_s[2];
      }
    }else{
      Tau[0] = Trial_local[0];
      Tau[1] = Trial_local[1];
      Tau[2] = Trial_local[2];
    }
#ifdef TPV29
    Tau_n += Pf;
#endif

#ifdef DEBUG
    if(j == nj/2 && k == nk/2)
      printf("@trial Tau = %e %e %e\n", Tau[0], Tau[1], Tau[2]);
#endif

    //pos1 = 1*ny*nz + j1*nz + k1;
    //pos  = j1*nz + k1;
    //pos  = j*nk + k;
    //pos1 = j1 + k1 * ny + nyz;
    pos1 = j1 + k1 * ny + 3*nyz + Tsize;          //wangzj
    pos  = j + k * nj + mpifaultsize;
    // if(gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] &&
    //    gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault]){
      if(ifchange){
        f.T11[pos1] = (Tau[0] - f.str_init_x[pos])*jacvec;
        f.T12[pos1] = (Tau[1] - f.str_init_y[pos])*jacvec;
        f.T13[pos1] = (Tau[2] - f.str_init_z[pos])*jacvec;
      }else{
        f.T11[pos1] = Trial[0];
        f.T12[pos1] = Trial[1];
        f.T13[pos1] = Trial[2];
      }
    // }
    // else{
    //   f.T11[pos1] = Trial[0];
    //   f.T12[pos1] = Trial[1];
    //   f.T13[pos1] = Trial[2];
    //   Tau[0] = Trial_local[0];
    //   Tau[1] = Trial_local[1];
    //   Tau[2] = Trial_local[2];
    // }
// //////////////////////// Apply smooth near the strong boundary ///////////////////////
// //     real_t T11_smooth, T12_smooth, T13_smooth;
    // int pos0 = j1 + k1 * ny + 3*nyz + Tsize;
    // pos = j + k * nj;
    // f.T11S[pos] = f.T11[pos1];
    // f.T12S[pos] = f.T12[pos1];      
    // f.T13S[pos] = f.T13[pos1];

//     if(((gj1 < Faultgrid[0 + 4*nfault] + 3 && gj1 >= Faultgrid[0 + 4*nfault] - 10) ||
//         (gj1 > Faultgrid[1 + 4*nfault] - 3 && gj1 <= Faultgrid[1 + 4*nfault] + 10)) &&
//        ((gk1 < Faultgrid[2 + 4*nfault] + 3 && gk1 >= Faultgrid[2 + 4*nfault] - 10)
// #ifndef FreeSurface
//         || (gk1 > Faultgrid[3 + 4*nfault] - 3 && gk2 <= Faultgrid[3 + 4*nfault] + 10)
// #endif
//       )){
//       ////////////////////// Do smooth after rupture started /////////////////////////
//       if (f.flag_rup[pos]){
//         int pos1 = j1 + k1 * ny + 3*nyz + Tsize + 1;
//         int pos_1 = j1 + k1 * ny + 3*nyz + Tsize - 1;

//         f.T11[pos0] = (f.T11S[pos_1] + f.T11S[pos0] + f.T11S[pos1]) / 3;      
//         f.T12[pos0] = (f.T12S[pos_1] + f.T12S[pos0] + f.T12S[pos1]) / 3;      
//         f.T13[pos0] = (f.T13S[pos_1] + f.T13S[pos0] + f.T13S[pos1]) / 3;
//       }
//     }
/////////////////////////////////////////////////////////////////////////////////////////    
    real_t hT11, hT12, hT13;
    real_t viscosity = par.viscosity * DT;

    real_t DT2 = DT;
    if(irk == 0){
      DT2 = 0*DT;
    }else if(irk == 1){
      DT2 = 0.5*DT;
    }else if(irk == 2){
      DT2 = 0.5*DT;
    }else if(irk == 3){
      DT2 = 1.0*DT;
    }
    pos1 = j1 + k1 * ny + 3*nyz + Tsize;          //wangzj
    if(irk==0){
      hT11 = f.hT11[pos];         //wangzj
      hT12 = f.hT12[pos];
      hT13 = f.hT13[pos];
    }else{
      // update
      hT11 = (f.T11[pos1] - f.mT11[pos])/DT2;
      hT12 = (f.T12[pos1] - f.mT12[pos])/DT2;
      hT13 = (f.T13[pos1] - f.mT13[pos])/DT2;
    }
    hT11 = (f.T11[pos1] - f.mT11[pos])/DT;
    hT12 = (f.T12[pos1] - f.mT12[pos])/DT;
    hT13 = (f.T13[pos1] - f.mT13[pos])/DT;

#ifndef DxV_hT1
    if(irk==3){
    f.hT11[pos] = hT11;        //wangzj
    f.hT12[pos] = hT12;
    f.hT13[pos] = hT13;
    }
#endif
    //if(f.slip[j1+k1*ny] > par.Dc){
    //f.hT11[j1+k1*ny] = 0;//hT11;
    //f.hT12[j1+k1*ny] = 0;//hT12;
    //f.hT13[j1+k1*ny] = 0;//hT13;
    //}

    f.T11[pos1] += viscosity * hT11;
    f.T12[pos1] += viscosity * hT12;
    f.T13[pos1] += viscosity * hT13;

    //Tau[0] = hT11;
    //Tau[1] = hT12;
    //Tau[2] = hT13;

    //pos = j*nk + k;
    // pos = j + k * nj + mpifaultsize;           //wangzj
    f.tTs1[pos] = dot_product(Tau, vec_s1);
    f.tTs2[pos] = dot_product(Tau, vec_s2);
    f.tTn [pos] = Tau_n;
#ifdef DEBUG
    if(j == nj/2 && k == nk/2)
      printf("@trial tTs1, tTs2, tTn = %e %e %e\n", f.tTs1[pos], f.tTs2[pos], f.tTn[pos]);
#endif

    //pos = j*nk + k;
    pos = j + k * nj + mpifaultsize;          //wangzj
    if(!f.init_t0_flag[pos]) {
      if (f.hslip[pos] > 1e-3) {
        f.init_t0[pos] = it * DT;
        f.init_t0_flag[pos] = 1;
        f.flag_rup[pos] = 1;
      }
    }
// #ifdef FaultSmooth
//  pos = j + k*nj + mpifaultsize;            //wangzj
//     if(f.first_rup[pos] && f.smooth_flag[pos]){
//         f.smooth_flag[pos] += 1;
//       }
//     }
// #endif

#ifdef SelectStencil
    pos = j + k*nj + mpifaultsize;            //wangzj
    if(f.flag_rup[pos] && f.first_rup[pos]){
      for (int l = -3; l <=3; l++){
        int jj1 = j+l;
        jj1 = MAX(0, jj1);
        jj1 = MIN(nj-1, jj1);
        int pos2 = (jj1+k*nj + mpifaultsize);  //wangzj
        f.rup_index_y[pos2] += 1;
        int kk1 = k+l;
        kk1 = MAX(0, kk1);
        kk1 = MIN(nk-1, kk1);
        pos2 = (j+kk1*nj + mpifaultsize);   //wangzj
        f.rup_index_z[pos2] += 1;
      }
      f.first_rup[pos] = 0;
    }
#endif

  } // end j k
  return;
}

void trial_sw(Wave W, Fault F, real_t *M, int it, int irk, int FlagX, int FlagY, int FlagZ, int i0, int nfault, int Faultgrid[])
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);
  trial_sw_cu <<<grid, block>>> (W, F, M, it, irk, FlagX, FlagY, FlagZ, i0, nfault, Faultgrid);
}


// __global__ void smoothT1_cu(Fault f, int nfault, int Faultgrid[])
// {
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   int k = blockIdx.y * blockDim.y + threadIdx.y;

//   int j1 = j + 3;
//   int k1 = k + 3;

//   int nj = par.NY / par.PY;
//   int nk = par.NZ / par.PZ;

//   int ny = nj + 6;
//   int nz = nk + 6;
//   int nyz = ny * nz;
//   int Tsize = nfault * nyz * 7;
//   int mpifaultsize = nfault * nj*nk;

//   int gj = par.ranky * nj + j;
//   int gj1 = gj + 1;
//   int gk = par.rankz * nk + k;
//   int gk1 = gk + 1;

//   int p1 = Faultgrid[0 + 4*nfault];
//   int p2 = Faultgrid[1 + 4*nfault];
//   // int p3 = Faultgrid[2 + 4*nfault];
//   // int p4 = Faultgrid[3 + 4*nfault];

// ////////////////////// Do smooth after rupture started /////////////////////////
//   int smoothY = 0;
//   int start = 0;
//   if(gj1 <= p1 - 2 && gj1 >= p1 - 12){
//     smoothY = 1;
//     int posj = p1 - 1 - par.ranky * nj;
//     int pos = posj + k * nj +  + mpifaultsize;
//     if(f.init_t0_flag[pos]){
//     // if(f.hslip[pos] > 0.1){
//       start = 1;
//     }
//     // if(f.slip[pos] > 0.1){
//     //   start = 1;
//     // }
//   }
  
//   if(gj1 >= p2 - 2 && gj1 <= p2 + 12){
//     smoothY = 1;
//     int posj = p2 - 1 - par.ranky * nj;
//     int pos = posj + k * nj + mpifaultsize;
//     // printf("========posj=%d==========\n", posj);
//     if(f.init_t0_flag[pos]){
//     // if(f.hslip[pos] > 0.1){
//       start = 1;
//     }
//     // if(f.slip[pos] > 0.1){
//     //   start = 1;
//     // }
//   }
  
//   if(smoothY && start){
//     // printf("=======ranky=%d,==j=%d=======\n", par.ranky, j);  
//     int pos0 = j1 + k1 * ny + 3*nyz + Tsize; 
//     int pos = j + k * nj + mpifaultsize;
//     int pos1 = pos + 1;
//     int pos_1 = pos - 1;
//     int pos_2 = pos - 2;
//     int pos2 = pos + 2;
//     // f.T11[pos0] = (f.T11S[pos_2] + f.T11S[pos_1] + f.T11S[pos] + f.T11S[pos1] + f.T11S[pos2]) * 0.2;      
//     // f.T12[pos0] = (f.T12S[pos_2] + f.T12S[pos_1] + f.T12S[pos] + f.T12S[pos1] + f.T12S[pos2]) * 0.2;      
//     // f.T13[pos0] = (f.T13S[pos_2] + f.T13S[pos_1] + f.T13S[pos] + f.T13S[pos1] + f.T13S[pos2]) * 0.2;
//     f.T11[pos0] = (f.T11S[pos_1] + f.T11S[pos] + f.T11S[pos1]) / 3.0f;      
//     f.T12[pos0] = (f.T12S[pos_1] + f.T12S[pos] + f.T12S[pos1]) / 3.0f;      
//     f.T13[pos0] = (f.T13S[pos_1] + f.T13S[pos] + f.T13S[pos1]) / 3.0f;  
//   }
// }

// void smoothT1(Fault F, int nfault, int Faultgrid[])
// {
//   dim3 block(16, 8, 1);
//   dim3 grid(
//       (hostParams.nj+block.x-1)/block.x,
//       (hostParams.nk+block.y-1)/block.y, 1);
//   smoothT1_cu <<<grid, block>>> (F, nfault, Faultgrid);
// }