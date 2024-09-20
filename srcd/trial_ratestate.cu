#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"


//#define DEBUG
//extern __device__ __constant__ int Flags[8][3];
extern __device__ real_t norm3(real_t *A);
extern __device__ real_t dot_product(real_t *A, real_t *B);
extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void invert3x3(real_t m[][3]);
extern __device__ real_t Fr_func(const real_t r, const real_t R);
extern __device__ real_t Gt_func(const real_t t, const real_t T);

#define DOT3(A,B) (A[0]*B[0]+A[1]*B[1]+A[2]*B[2])

__device__ void NRsearch(double *xnew, int *iter, double *err,
    double x, double psi, double RS_a, double Vhat, double dt, double coef, int verbose){
  int iter_max = 100;
  double eps = 1e-12;

  double RS_V0 = 1e-6;
  //double Tn = 120e6;
  double f, df, d;  //f_{k-1}
  *iter = 0;
  *xnew = x; // initial guess
  while (*iter < iter_max) {

    //if(verbose){
    //  printf("x = %e, V = %e, psi = %e, RS_a = %e, Vhat = %e, dt = %e, coef = %e\n",
    //      x, 2.0*RS_V0*exp(-psi/RS_a)*sinh(x), psi, RS_a, Vhat, dt, coef);
    //}
    //f = 2.0 * RS_V0 * exp(-psi/RS_a) * sinh(x) + dt * coef * Tn * RS_a * x - Vhat;
    //df = 2.0 * RS_V0 * exp(-psi/RS_a) * cosh(x) + dt * coef * Tn * RS_a;


    // if (*iter == 0){
      f  = 2.0 * exp(-psi/RS_a) * RS_V0 * sinh(x) + coef * x - Vhat;
      df = 2.0 * exp(-psi/RS_a) * RS_V0 * cosh(x) + coef;

      d = -f/(df+1e-100);
      if (d > 100 || d < -150){
        if (d >  700 || d < - 750){
          d = -f/(df + 1);
        }else if(d >  500 || d < - 550){
          d = -f/(df + 1e-1);
        }else if(d >  300 || d < - 350){
          d = -f/(df + 1e-2);
        }else{
          d = -f/(df + 1e-3);
        }
      }
      *xnew = x + d;
      if (*xnew > 100){
        *xnew -= 100;
      }
      if (*xnew < -150){
        *xnew += 100;
      }
      // xk_1 = x;
      *err = fabs(d)/(fabs(x) + 1e-100);
    // } else{
    //   f  = 2.0 * exp(-psi/RS_a) * RS_V0 * sinh(x) + coef * x - Vhat;
    //   fk_1 = 2.0 * exp(-psi/RS_a) * RS_V0 * sinh(xk_1) + coef * xk_1 - Vhat;
    //   // if (f * fk_1 > 0 && *iter == 1){
    //   //   xk_1 = 500;
    //   //   fk_1 = 2.0 * exp(-psi/RS_a) * RS_V0 * sinh(xk_1) + coef * xk_1 - Vhat;
    //   // }
    //   d = -f * ((x - xk_1) / (f - fk_1 + 1e-12));
    //   *xnew = x + d;
    // }
///////////////////////////////////////////////////////////////////////////////////////////////////////
    if(verbose){
      printf("iter = %d, f = %e, df = %e, f/df = %e, xnew = %e, x = %e\n",
          *iter, f, df, d, *xnew, x);
    }
    // xk_1 = x;  // secant method
    x = *xnew; 
    (*iter)++;
    if (*err < eps) break;
  }

}

__global__ void trial_rs_cu(Wave w, Fault f, real_t *M,
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
  int faultsize = nfault * nyz2 * 9;      //***wangzj
  int mpifaultsize = nfault * nj * nk;
  int Tsize = nfault * nyz * 7;

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
  //int i0 = nx/2;                      **********wangzj

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
  // real_t R1[2], R2[2], R3[2];
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
  if ( j < nj && k < nk && f.united[j + k * nj + mpifaultsize] == 0){  // not united   **wangzj

    pos1 = j1 + k1 * ny + 3*nyz + Tsize;
    pos = j + k * nj + mpifaultsize;
    if(irk == 0){
      f.mT11[pos] = f.T11[pos1];
      f.mT12[pos] = f.T12[pos1];
      f.mT13[pos] = f.T13[pos1];
    }
    //km = NZ - (thisid[2]*nk+k-3);
    //int km = (nz - 6) - (k-3); // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)

    real_t vec_n[3];
    real_t vec_s1[3];
    real_t vec_s2[3];

    pos = (j1 + k1 * ny) * 3 + nfault * (nyz*3);
    vec_s1[0] = f.vec_s1[pos + 0];
    vec_s1[1] = f.vec_s1[pos + 1];
    vec_s1[2] = f.vec_s1[pos + 2];
    vec_s2[0] = f.vec_s2[pos + 0];
    vec_s2[1] = f.vec_s2[pos + 1];
    vec_s2[2] = f.vec_s2[pos + 2];

    pos = j1 + k1 * ny + i0 * ny * nz;
    vec_n[0] = XIX[pos];
    vec_n[1] = XIY[pos];
    vec_n[2] = XIZ[pos];
    vec_n0 = norm3(vec_n);
    jacvec = JAC[pos] * vec_n0;
    for (int ii = 0; ii < 3; ii++){
      vec_n[ii] /= vec_n0;
    }

    //vec_n [0] = 1;vec_n [1] = 0;vec_n [2] = 0;
    //vec_s1[0] = 0;vec_s1[1] = 1;vec_s1[2] = 0;
    //vec_s2[0] = 0;vec_s2[1] = 0;vec_s2[2] = 1;

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
      rho = f.rho_f[j1 + k1 * ny + m * ny * nz + nfault * nyz2];    //*****wangzj
      
      Mrho[m] = 0.5f*jac*rho*DH*DH2;
      //xix = M[pos_m + 0]; xiy = M[pos_m + 1]; xiz = M[pos_m + 2];
      //jac = M[pos_m + 9];
      //rho = M[pos_m + 12];

      // {Txx 3} {Tyy 4} {Tzz 5} {Txy 6} {Txz 7} {Tyz 8}
      // T1 is continuous!
      //T11 = jac*(xix * w.W[pos + 3] + xiy * w.W[pos + 6] + xiz * w.W[pos + 7]);
      //T12 = jac*(xix * w.W[pos + 6] + xiy * w.W[pos + 4] + xiz * w.W[pos + 8]);
      //T13 = jac*(xix * w.W[pos + 7] + xiy * w.W[pos + 8] + xiz * w.W[pos + 5]);
      //!!T11 = jac*(xix * w_Txx[pos] + xiy * w_Txy[pos] + xiz * w_Txz[pos]);
      //!!T12 = jac*(xix * w_Txy[pos] + xiy * w_Tyy[pos] + xiz * w_Tyz[pos]);
      //!!T13 = jac*(xix * w_Txz[pos] + xiy * w_Tyz[pos] + xiz * w_Tzz[pos]);
      pos_f = j1 + k1 * ny  + Tsize;         //****wangzj
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

      // 0 or 2 ==> i0-1 or i0+1
      //f.T11[(2*m)*ny*nz+j1*nz+k1] = T11;
      //f.T12[(2*m)*ny*nz+j1*nz+k1] = T12;
      //f.T13[(2*m)*ny*nz+j1*nz+k1] = T13;
      //pos_f = j1 + k1 * ny;
      //f.T11[(2*m)*nyz + pos_f] = T11;
      //f.T12[(2*m)*nyz + pos_f] = T12;
      //f.T13[(2*m)*nyz + pos_f] = T13;

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
      if(f.rup_index_y[j + k * nj + mpifaultsize] % 7){               //wangzj
#endif
        //DyT21 = L22(f.W, (pos_f + 3), slice, FlagY) / DH;
        //DyT22 = L22(f.W, (pos_f + 4), slice, FlagY) / DH;
        //DyT23 = L22(f.W, (pos_f + 5), slice, FlagY) / DH;
        pos_f = j1 + k1 * ny;                                       //wangzj
        DyT21 = L22(t21, pos_f, 1, FlagY); // * rDH;
        DyT22 = L22(t22, pos_f, 1, FlagY); // * rDH;
        DyT23 = L22(t23, pos_f, 1, FlagY); // * rDH;
      }else{
        pos_f = j1 + k1 * ny;                                     //wangzj
        DyT21 = L(t21, pos_f, 1, FlagY); // * rDH;
        DyT22 = L(t22, pos_f, 1, FlagY); // * rDH;
        DyT23 = L(t23, pos_f, 1, FlagY); // * rDH;
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
      if(f.rup_sensor[j + k * nj + mpifaultsize] > par.RupThres){     //wangzj
#else
      if(f.rup_index_z[j + k * nj + mpifaultsize] % 7){
#endif
        DzT31 = vec_L22(vecT31, 3, FlagZ); // * rDH;
        DzT32 = vec_L22(vecT32, 3, FlagZ); // * rDH;
        DzT33 = vec_L22(vecT33, 3, FlagZ); // * rDH;
      }else{
        DzT31 = vec_L(vecT31, 3, FlagZ); // * rDH;
        DzT32 = vec_L(vecT32, 3, FlagZ); // * rDH;
        DzT33 = vec_L(vecT33, 3, FlagZ); // * rDH;
      }
      pos_f = j1 + k1 * ny  + Tsize;                          //wangzj
      T11 = f.T11[(3+2*m-1)*nyz+pos_f];
      T12 = f.T12[(3+2*m-1)*nyz+pos_f];
      T13 = f.T13[(3+2*m-1)*nyz+pos_f];

      // Rx[m] = 0.5f*( (2*m-1)*T11 + (DyT21 + DzT31)*DH )*DH2;
      // Ry[m] = 0.5f*( (2*m-1)*T12 + (DyT22 + DzT32)*DH )*DH2;
      // Rz[m] = 0.5f*( (2*m-1)*T13 + (DyT23 + DzT33)*DH )*DH2;
      Rx[m] = 0.5f*( (2*m-1)*T11 + (DyT21 + DzT31) )*DH2;
      Ry[m] = 0.5f*( (2*m-1)*T12 + (DyT22 + DzT32) )*DH2;
      Rz[m] = 0.5f*( (2*m-1)*T13 + (DyT23 + DzT33) )*DH2;

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

      // Rx[m] = 0.5f*( (2*m-1)*Rx[m] + (DyT21 + DzT31)*DH )*DH2;
      // Ry[m] = 0.5f*( (2*m-1)*Ry[m] + (DyT22 + DzT32)*DH )*DH2;
      // Rz[m] = 0.5f*( (2*m-1)*Rz[m] + (DyT23 + DzT33)*DH )*DH2;
      Rx[m] = 0.5f*( (2*m-1)*Rx[m] + (DyT21 + DzT31) )*DH2;
      Ry[m] = 0.5f*( (2*m-1)*Ry[m] + (DyT22 + DzT32) )*DH2;
      Rz[m] = 0.5f*( (2*m-1)*Rz[m] + (DyT23 + DzT33) )*DH2;
#endif

      // R1[m] = vec_n [0]*Rx[m]+vec_n [1]*Ry[m]+vec_n [2]*Rz[m];
      // R2[m] = vec_s1[0]*Rx[m]+vec_s1[1]*Ry[m]+vec_s1[2]*Rz[m];
      // R3[m] = vec_s2[0]*Rx[m]+vec_s2[1]*Ry[m]+vec_s2[2]*Rz[m];

    } // end m

    //========================================================================
    // add perturbation
    //========================================================================
    // int gj = par.ranky * nj + j;
    // int gk = par.rankz * nk + k;
    real_t y1 = (gj-par.NY/2);
#ifdef FreeSurface
    real_t z1 = (par.NZ-1-gk) - (int)(7.5e3/DH);
#else
    real_t z1 = (gk-par.NZ/2);
#endif

    real_t r = sqrt(y1*y1+z1*z1) * DH;
    real_t dTau0 = -25e6;
    real_t Tau_ini = -75e6;
#if defined TPV103 || defined TPV104
    dTau0 = -45.0e6;
    Tau_ini = -40.0e6;
#endif
    real_t t;
    if(irk == 0){
      t = it*DT;
    }else if (irk == 1 || irk == 2){
      t = (it+0.5)*DT;
    }else{
      t = (it+1)*DT;
    }

    real_t Gt = Gt_func(t, 1.0);
    Gt = Gt_func(t, par.smooth_load_T);
    real_t Fr = Fr_func(r, 3.0e3);

    pos = j + k * nj + mpifaultsize;
    f.str_init_x[pos] = -120e6;
    f.str_init_y[pos] = dTau0*Fr*Gt + Tau_ini;
    f.str_init_z[pos] = 0.0;

    if (1 == par.INPORT_STRESS_TYPE){
      // overwrite by input stress data
      f.str_init_x[pos] = f.T0x[pos] + Gt * f.dT0x[pos];
      f.str_init_y[pos] = f.T0y[pos] + Gt * f.dT0y[pos];
      f.str_init_z[pos] = f.T0z[pos] + Gt * f.dT0z[pos];
    }

    real_t T0[3];
    real_t T0_local[3];

    T0[0] = f.str_init_x[pos];
    T0[1] = f.str_init_y[pos];
    T0[2] = f.str_init_z[pos];

    T0_local[0] = vec_n [0]*T0[0]+vec_n [1]*T0[1]+vec_n [2]*T0[2];
    T0_local[1] = vec_s1[0]*T0[0]+vec_s1[1]*T0[1]+vec_s1[2]*T0[2];
    T0_local[2] = vec_s2[0]*T0[0]+vec_s2[1]*T0[1]+vec_s2[2]*T0[2];


    //T0_local[0] = -120e6;
    //T0_local[1] = dTau0*Fr_func(r, 3.0e3)*Gt_func(t, 1.0f) + Tau_ini;
    //T0_local[2] = 0.0;
    //f.str_init_y[pos] = dTau0*Fr_func(r, 3.0e3) + Tau_ini;
    //if(j==nj/2 && k==nk/2)
    //  printf("it = %03d, irk = %d, str_init_y = %e\n",
    //      it, irk, f.str_init_y[pos]);

    //========================================================================
    //========================================================================
    //real_t c1[3];
    //c1[0] = (Rx[1] - Rx[0]) / Mrho[0];
    //c1[1] = (Ry[1] - Ry[0]) / Mrho[0];
    //c1[2] = (Rz[1] - Rz[0]) / Mrho[0];
    //c1[0] = (Rx[1]/Mrho[1] - Rx[0]/Mrho[0]);
    //c1[1] = (Ry[1]/Mrho[1] - Ry[0]/Mrho[0]);
    //c1[2] = (Rz[1]/Mrho[1] - Rz[0]/Mrho[0]);

    //real_t c2 = DH * DH / Mrho[0];

    pos = j1 + k1 * ny;

    double dVx = f_mVx[pos + nyz] - f_mVx[pos];
    double dVy = f_mVy[pos + nyz] - f_mVy[pos];
    double dVz = f_mVz[pos + nyz] - f_mVz[pos];
#ifdef RKtrial
    if (irk == 3){
      dVx = f_tVx[pos + nyz] - f_tVx[pos];
      dVy = f_tVy[pos + nyz] - f_tVy[pos];
      dVz = f_tVz[pos + nyz] - f_tVz[pos];
    }
#endif
    double dt1 = DT;
#ifdef RKtrial
    if (irk == 0 || irk == 1){
      dt1 *= 0.5;
    }else if (irk == 3){
      dt1 /= 6.0;
    }
#endif

    //real_t V = f.hslip[pos] + 1e-12;
    // rate
    //real_t V = sqrt(dVx * dVx + dVy * dVy + dVz * dVz);
    real_t V;// = fabs(dVy);
    double Ratex = dVx;
    double Ratey = dVy;
    double Ratez = dVz;
    double Vini = par.Vini;
#if defined TPV103 || defined TPV104
    Ratex = dVx;
    Ratey = dVy-1e-16;
    Ratez = dVz;
#endif
#if defined TPV101 || defined TPV102
    Ratex = dVx;
    Ratey = dVy-1e-12;
    Ratez = dVz;
#endif

    double Rate_local[3];
///////////////////////////////////////////////////////////////////////////////wangzj
    // if(gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] &&
    //    gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault]){
      Rate_local[0] = vec_n [0]*Ratex+vec_n [1]*Ratey+vec_n [2]*Ratez;
      Rate_local[1] = vec_s1[0]*Ratex+vec_s1[1]*Ratey+vec_s1[2]*Ratez;
      Rate_local[2] = vec_s2[0]*Ratex+vec_s2[1]*Ratey+vec_s2[2]*Ratez;

      Rate_local[1] -= Vini;

      // if( gj1 == 250 && gk1 == 200 && irk == 3){
      //   printf("Rate_normal=%.17lf\n", Rate_local[0]);
      // }
    // }
    // else{
    //   Rate_local[0] = 0;
    //   Rate_local[1] = 0;
    //   Rate_local[2] = 0;
    // }
    /////////////////////////////////////////////////////////////////////////
    Ratex = Rate_local[0] * vec_n [0] + Rate_local[1] * vec_s1[0] + Rate_local[2] * vec_s2[0];
    Ratey = Rate_local[0] * vec_n [1] + Rate_local[1] * vec_s1[1] + Rate_local[2] * vec_s2[1];
    Ratez = Rate_local[0] * vec_n [2] + Rate_local[1] * vec_s1[2] + Rate_local[2] * vec_s2[2];

    //real_t mRate = sqrt(Rate_local[1]*Rate_local[1]+Rate_local[2]*Rate_local[2]);

   // Rate_local[0] = 0;

#ifdef TractionImg
    double a0p,a0m;
    if(FlagX==FWD){
      a0p = a_0pF;
      a0m = a_0mF;
    }else{
      a0p = a_0pB;
      a0m = a_0mB;
    }
#endif
    //real_t dVhat_x = dVx + DT * ( c1[0] + c2 * f.str_init_x[j + k * nj]);
    //real_t dVhat_y = dVy + DT * ( c1[1] + c2 * f.str_init_y[j + k * nj]);
    //real_t dVhat_z = dVz + DT * ( c1[2] + c2 * f.str_init_z[j + k * nj]);
    // pos = j1 + k1 * ny + i * ny * nz;
    jac = JAC[j1 + k1 * ny + i0 * ny * nz];
    
    double Vhat_x = dVx + dt1 * ((Rx[1]/Mrho[1] - Rx[0]/Mrho[0]) + (2.0*f.str_init_x[j + k * nj + mpifaultsize])/(jac*rho*DH));
    double Vhat_y = dVy + dt1 * ((Ry[1]/Mrho[1] - Ry[0]/Mrho[0]) + (2.0*f.str_init_y[j + k * nj + mpifaultsize])/(jac*rho*DH));
    double Vhat_z = dVz + dt1 * ((Rz[1]/Mrho[1] - Rz[0]/Mrho[0]) + (2.0*f.str_init_z[j + k * nj + mpifaultsize])/(jac*rho*DH));

    //real_t Vhat_x = fabs(dVhat_x);
    //real_t Vhat_y = fabs(dVhat_y);
    //real_t Vhat_z = fabs(dVhat_z);
    //Vhat_x = (dVhat_x);
    //Vhat_y = (dVhat_y);
    //Vhat_z = (dVhat_z);

    //real_t Vhat  = sqrt(dVhat_y * dVhat_y + dVhat_z * dVhat_z);
    //real_t vec_n[3];
    //real_t vec_s1[3];
    //real_t vec_s2[3];

    //pos = (j1 + k1 * ny) * 3;
    //vec_s1[0] = f.vec_s1[pos + 0];
    //vec_s1[1] = f.vec_s1[pos + 1];
    //vec_s1[2] = f.vec_s1[pos + 2];
    //vec_s2[0] = f.vec_s2[pos + 0];
    //vec_s2[1] = f.vec_s2[pos + 1];
    //vec_s2[2] = f.vec_s2[pos + 2];

    //pos = j1 + k1 * ny + i0 * ny * nz;
    //vec_n[0] = XIX[pos];
    //vec_n[1] = XIY[pos];
    //vec_n[2] = XIZ[pos];
    //vec_n0 = norm3(vec_n);
    //jacvec = JAC[pos] * vec_n0;
    //for (int ii = 0; ii < 3; ii++)
    //  vec_n[ii] /= vec_n0;

    double Ttilde[3];
    Ttilde[0] = f.str_init_x[j+k*nj+mpifaultsize]+2.0*(Mrho[0]*Mrho[1]*dVx/dt1+Mrho[0]*Rx[1]-Mrho[1]*Rx[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);
    Ttilde[1] = f.str_init_y[j+k*nj+mpifaultsize]+2.0*(Mrho[0]*Mrho[1]*dVy/dt1+Mrho[0]*Ry[1]-Mrho[1]*Ry[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);
    Ttilde[2] = f.str_init_z[j+k*nj+mpifaultsize]+2.0*(Mrho[0]*Mrho[1]*dVz/dt1+Mrho[0]*Rz[1]-Mrho[1]*Rz[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);

    double Ttilde_local[3];
    Ttilde_local[0] = vec_n [0]*Ttilde[0]+vec_n [1]*Ttilde[1]+vec_n [2]*Ttilde[2];
    Ttilde_local[1] = vec_s1[0]*Ttilde[0]+vec_s1[1]*Ttilde[1]+vec_s1[2]*Ttilde[2];
    Ttilde_local[2] = vec_s2[0]*Ttilde[0]+vec_s2[1]*Ttilde[1]+vec_s2[2]*Ttilde[2];

    // Ttilde_local[0] = T0_local[0]+2.0*(Mrho[0]*Mrho[1]*Rate_local[0]/dt1+Mrho[0]*R1[1]-Mrho[1]*R1[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);
    // Ttilde_local[1] = T0_local[1]+2.0*(Mrho[0]*Mrho[1]*Rate_local[1]/dt1+Mrho[0]*R2[1]-Mrho[1]*R2[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);
    // Ttilde_local[2] = T0_local[2]+2.0*(Mrho[0]*Mrho[1]*Rate_local[2]/dt1+Mrho[0]*R3[1]-Mrho[1]*R3[0])/(DH2*(Mrho[0]+Mrho[1])*jacvec);
#ifdef TractionImg
    Ttilde[0] = T0[0]-2.0*(Mrho[0]*Mrho[1]*Ratex/dt1+Mrho[0]*Rx[1]-Mrho[1]*Rx[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);
    Ttilde[1] = T0[1]-2.0*(Mrho[0]*Mrho[1]*Ratey/dt1+Mrho[0]*Ry[1]-Mrho[1]*Ry[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);
    Ttilde[2] = T0[2]-2.0*(Mrho[0]*Mrho[1]*Ratez/dt1+Mrho[0]*Rz[1]-Mrho[1]*Rz[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);

    Ttilde_local[0] = vec_n [0]*Ttilde[0]+vec_n [1]*Ttilde[1]+vec_n [2]*Ttilde[2];
    Ttilde_local[1] = vec_s1[0]*Ttilde[0]+vec_s1[1]*Ttilde[1]+vec_s1[2]*Ttilde[2];
    Ttilde_local[2] = vec_s2[0]*Ttilde[0]+vec_s2[1]*Ttilde[1]+vec_s2[2]*Ttilde[2];
    // Ttilde_local[0] = T0_local[0]-2.0*(Mrho[0]*Mrho[1]*Rate_local[0]/dt1+Mrho[0]*R1[1]-Mrho[1]*R1[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);
    // Ttilde_local[1] = T0_local[1]-2.0*(Mrho[0]*Mrho[1]*Rate_local[1]/dt1+Mrho[0]*R2[1]-Mrho[1]*R2[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);
    // Ttilde_local[2] = T0_local[2]-2.0*(Mrho[0]*Mrho[1]*Rate_local[2]/dt1+Mrho[0]*R3[1]-Mrho[1]*R3[0])/(DH2*(Mrho[0]*a0p-a0m*Mrho[1])*jacvec);
#endif

    double Vtilde[3];
    Vtilde[0] = Vhat_x;
    Vtilde[1] = Vhat_y;
    Vtilde[2] = Vhat_z;

    double Vtilde_local[3];
    Vtilde_local[0] = vec_n [0]*Vtilde[0]+vec_n [1]*Vtilde[1]+vec_n [2]*Vtilde[2];
    Vtilde_local[1] = vec_s1[0]*Vtilde[0]+vec_s1[1]*Vtilde[1]+vec_s1[2]*Vtilde[2];
    Vtilde_local[2] = vec_s2[0]*Vtilde[0]+vec_s2[1]*Vtilde[1]+vec_s2[2]*Vtilde[2];

    double c = 0.5*dt1*DH2*jacvec/Mrho[0] + 0.5*dt1*DH2*jacvec/Mrho[1];
    //c = 2.0*DT*jacvec/(jac*rho*DH);
#ifdef TractionImg
    // c = -0.5*DT*DH2*jacvec*(a0p/Mrho[1]-a0m/Mrho[0]);
    // c = -(a0p-a0m)*dt1*jacvec/(jac*rho*DH);
    c = -(a0p-a0m)*vec_n0/(rho*DH)*dt1;
#endif
//////////////////////////////////////////////////////////////////////////////////////
    // Vtilde_local[0] = c*Ttilde_local[0];
    // Vtilde_local[1] = c*Ttilde_local[1];
    // Vtilde_local[2] = c*Ttilde_local[2];

    // Vtilde_local[0] = c*T0_local[0]+Rate_local[0]+dt1*(R1[1]/Mrho[1]-R1[0]/Mrho[0]);
    // Vtilde_local[1] = c*T0_local[1]+Rate_local[1]+dt1*(R2[1]/Mrho[1]-R2[0]/Mrho[0]);
    // Vtilde_local[2] = c*T0_local[2]+Rate_local[2]+dt1*(R3[1]/Mrho[1]-R3[0]/Mrho[0]);
///////////////////////////////////////////////////////////////////////////////////////
    //Rate_local[0] = Ratex;
    //Rate_local[1] = Ratey;
    //Rate_local[2] = Ratez;

    pos = j + k * nj + mpifaultsize;       //******wangzj
    //double V = f.hslip[pos];
    double state = f.State[pos];

    double RS_a = f.a[pos];
    double RS_b = f.b[pos];
    double RS_f0 = 0.6;
    double RS_V0 = 1e-6;
    double RS_L = 0.02;
    double RS_Vw = f.Vw[pos];
    int TP_n = par.TP_n;                       //*******wangzj

    //f.hState[pos] = RS_b * RS_V0 / RS_L * (exp((RS_f0 - state)/RS_b) - V/RS_V0);

    double Tau[3] = {0.,0.,0.};
    double Tau_n, Tau_s1, Tau_s2;// = f.Tn[j + k * nj];
    //Tau_n = -120e6;
    Tau_n = Ttilde_local[0];

    //real_t ifchange = 0; // false
    if(Ttilde_local[0] >= -1e-1){
      // fault can not open
      Tau_n = -1e-1;
      //ifchange = 1;
    }else{
      Tau_n = Ttilde_local[0];
    }

    Vtilde_local[0] = c*Ttilde_local[0];
    Vtilde_local[1] = c*Ttilde_local[1];
    Vtilde_local[2] = c*Ttilde_local[2];

//#ifdef TP
    real_t Pf;
    if(par.Friction_type == 3){
      Pf = f.TP_P[j+k*nj+0*nj*nk + mpifaultsize*TP_n];    //*****wangzj
      //Pf = 0;
      // it is effective normal stress
      //Ttilde_local[0] += Pf;
      //T0_local[0] += Pf;
      Tau_n += Pf;
    }
//#endif

    double xnew, err, x, coef;
    int iter;

    coef = fabs(2.0*(dt1*Tau_n)/(rho*DH*jac)*RS_a);
#ifdef TractionImg
    coef = c*RS_a*fabs(Tau_n);
#endif

    // search absolute V 
    // or search V1 and V2
    int SearchAbsolute = 0;                             // wangzj
    if(gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] 
       && gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault]
       ){
      SearchAbsolute = 1;
    }
    else{
      Tau_s1 = Ttilde_local[1];
      Tau_s2 = Ttilde_local[2];
      V = 0;
    }
#ifdef Barrier
    if(gj1 >= par.Barrier_grid[0] && gj1 <= par.Barrier_grid[1]
       && gk1 >= par.Barrier_grid[2] && gk1 <= par.Barrier_grid[3]
       ){
      SearchAbsolute = 0;
      Tau_s1 = Ttilde_local[1];
      Tau_s2 = Ttilde_local[2];
      V = 0;
    }
#endif
//#define SearchEach

//   #ifdef SearchEach
//       // search in the eta direction
//       x = asinh( exp(state/RS_a) * Rate_local[1]/(2.0*RS_V0) );
//       NRsearch(&xnew, &iter, &err, x, state, RS_a, Vtilde_local[1], dt1, coef, 0);
//       //NRsearch(&xnew, &iter, &err, x, state, RS_a, Vhat_y, DT, coef, 0);
//       //if(j==nj/2 && k==nk/2) {
//       //  printf("Vhat=%.10f,%.10f,%.10f\n",Vhat_x,Vhat_y,Vhat_z);
//       //  printf("Vtil=%.10f,%.10f,%.10f\n", Vtilde_local[0],Vtilde_local[1],Vtilde_local[2]);
//       //  printf("vec_n=%g,%g,%g\n",vec_n[0],vec_n[1],vec_n[2]);
//       //  printf("vec_s1=%g,%g,%g\n",vec_s1[0],vec_s1[1],vec_s1[2]);
//       //  printf("vec_s2=%g,%g,%g\n",vec_s2[0],vec_s2[1],vec_s2[2]);
//       //}
//       
//       Ratey = 2.0*RS_V0*exp(-state/RS_a)*sinh(xnew);
//       Tau_s1 = RS_a * xnew * fabs(Tau_n);
//       // search in the zeta direction
//       // ===========================================================================
//       x = asinh( exp(state/RS_a) * Rate_local[2]/(2.0*RS_V0) );
//       NRsearch(&xnew, &iter, &err, x, state, RS_a, Vtilde_local[2], dt1, coef, 0);
//       Ratez = 2.0*RS_V0*exp(-state/RS_a)*sinh(xnew);
//       Tau_s2 = RS_a * xnew * fabs(Tau_n);
//   
//       V = sqrt(Ratey*Ratey+Ratez*Ratez);
//   #endif

// #ifdef SearchAbsolute
  
  if(SearchAbsolute){
    double Ttilde_local_norm = sqrt(
        Ttilde_local[1]*Ttilde_local[1]+Ttilde_local[2]*Ttilde_local[2]);
    double Vtilde_local_norm = sqrt(
        Vtilde_local[1]*Vtilde_local[1]+Vtilde_local[2]*Vtilde_local[2]);
    double Rate_norm = sqrt(
        Rate_local[1]*Rate_local[1]+Rate_local[2]*Rate_local[2]);
    x = asinh( exp((double)state/RS_a) * (Rate_norm/(2.0*RS_V0)) );   
//#define Trapz
//  #ifdef Trapz
//      coef = 0.5*c*RS_a*fabs(Tau_n);
//      Vtilde_local_norm -= 0.5 * c * RS_a * x * fabs(Tau_n);
//  #endif
    int ver = 0;
    // if( gj1 == 850 && gk1 == 152 && nfault == 0){
    //   ver = 1;
    // }
    NRsearch(&xnew, &iter, &err, x, state, RS_a, Vtilde_local_norm, dt1, coef, ver);
    //SecantSearch(&xnew, &iter, &err, x, state, RS_a, Vtilde_local_norm, dt1, coef, 0);
    V = 2.0*RS_V0*exp(-(double)state/RS_a)*sinh((double)xnew);

    double T = RS_a * xnew * fabs(Tau_n + 0);

    f.friction[j+k*nj + mpifaultsize] = RS_a * xnew;
   
    //if(j==nj/2 && k==(nk-1-int(7.5e3/DH))) {
    //  //printf("irk=%d,j=%d,k=%d,Vtil=%e,a=%e,v0=%e,x=%e,xnew=%e,T=%e,V=%e\n", irk,j,k, Vtilde_local_norm, RS_a, RS_V0, x, xnew, T, V);
    //  printf("irk=%d,j=%d,k=%d,Vtil=%e,psi=%e,a=%e,v0=%e,Rate_norm=%e,x=%e\n", irk,j,k, Vtilde_local_norm,state, RS_a, RS_V0, Rate_norm, x);
    //}

    //Vtilde_local_norm = max(Vtilde_local_norm, 1e-30);
    //Tau_s1 = Vtilde_local[1]/Vtilde_local_norm*T;
    //Tau_s2 = Vtilde_local[2]/Vtilde_local_norm*T;
    Ttilde_local_norm = MAX(Ttilde_local_norm, 1e-8);
    Tau_s1 = Ttilde_local[1]/Ttilde_local_norm*T;
    Tau_s2 = Ttilde_local[2]/Ttilde_local_norm*T;
  }
// #endif

    V = MAX(V, 1e-30);

//    // ageing law
//#if defined TPV101 || defined TPV102
//    RS_f0 = 0.6;
//    RS_V0 = 1e-6;
//    RS_L = 0.02;
//    f.hState[pos] = RS_b * RS_V0 / RS_L * (exp((RS_f0 - state)/RS_b) - V/RS_V0);
//#endif
//#if defined TPV103 || defined TPV104
//    RS_f0 = 0.6;
//    RS_V0 = 1e-6;
//    RS_L = 0.4;
//    double RS_fw = 0.2;
//
//    double RS_flv = RS_f0 - (RS_b-RS_a)*log(V/RS_V0);
//    double RS_fss = RS_fw + (RS_flv - RS_fw)/pow((1.+pow(V/RS_Vw, 8)),0.125);
//    double psiss = RS_a*(log(sinh(RS_fss/RS_a)) + log(2.*(RS_V0/V)));
//
//    f.hState[pos] = -V/RS_L*(state-psiss);
//    //f.State[pos] = (f.State[pos]-psiss)*exp(-V*DT/RS_L) + psiss;
//#endif
//
//    RS_f0 = par.f0;
//    RS_V0 = par.V0;
//    RS_L = par.L;
//    RS_L = f.L[j+k*nj];
//    if(par.Friction_type == 1){
//      f.hState[pos] = RS_b * RS_V0 / RS_L * (exp((RS_f0 - state)/RS_b) - V/RS_V0);
//    }else if (par.Friction_type == 2){
//      double RS_fw = par.fw;
//      double RS_flv = RS_f0 - (RS_b-RS_a)*log(V/RS_V0);
//      double RS_fss = RS_fw + (RS_flv - RS_fw)/pow((1.+pow(V/RS_Vw, 8)),0.125);
//      double psiss = RS_a*(log(sinh(RS_fss/RS_a)) + log(2.*(RS_V0/V)));
//      f.hState[pos] = -V/RS_L*(state-psiss);
//    }

    //if(j == nj/2 && k == nk/2){
    //  printf("Tau = %e %e %e, friction = %e, Tn = %e\n",
    //      Tau[0], Tau[1], Tau[2], RS_a * xnew, Tau_n);
    //}

    //pos = (j1 + k1 * ny) * 3;

    //double vec_s1[3], vec_s2[3], vec_n[3];
    //vec_s1[0] = f.vec_s1[pos + 0];
    //vec_s1[1] = f.vec_s1[pos + 1];
    //vec_s1[2] = f.vec_s1[pos + 2];
    //vec_s2[0] = f.vec_s2[pos + 0];
    //vec_s2[1] = f.vec_s2[pos + 1];
    //vec_s2[2] = f.vec_s2[pos + 2];

    //pos = j1 + k1 * ny + i0 * ny * nz;
    //vec_n[0] = XIX[pos];//M[pos + 0];
    //vec_n[1] = XIY[pos];//M[pos + 1];
    //vec_n[2] = XIZ[pos];//M[pos + 2];
    //vec_n0 = norm3(vec_n);

    //jacvec = JAC[pos] * vec_n0;

    //for (int ii = 0; ii < 3; ++ii){
    //  vec_n[ii] /= vec_n0;
    //}

    //pos1 = j1 + k1 * ny + nyz;
    pos1 = j1 + k1 * ny + 3*nyz + Tsize;       //*****wangzj
    pos  = j + k * nj + mpifaultsize;
//#ifdef TP
    if(par.Friction_type == 3){
      Tau_n -= Pf; // not effective normal stress
    }
//#endif

    // transform back to x, y, z
    Tau[0] = Tau_n * vec_n[0] + Tau_s1 * vec_s1[0] + Tau_s2 * vec_s2[0];
    Tau[1] = Tau_n * vec_n[1] + Tau_s1 * vec_s1[1] + Tau_s2 * vec_s2[1];
    Tau[2] = Tau_n * vec_n[2] + Tau_s1 * vec_s1[2] + Tau_s2 * vec_s2[2];
    //Tau[0] = Tau_n
    // Tau[0] = f.str_init_x[pos]; // force the Tn perturb to 0

    f.T11[pos1] = (Tau[0] - f.str_init_x[pos])*jacvec;
    // f.T11[pos1] = 0;
    f.T12[pos1] = (Tau[1] - f.str_init_y[pos])*jacvec;
    f.T13[pos1] = (Tau[2] - f.str_init_z[pos])*jacvec;

    
    real_t viscosity = par.viscosity * DT;
    real_t hT11, hT12, hT13;
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

    if(irk==0){
      hT11 = f.hT11[pos];       //wangzj
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

    //pos = j + k * nj + mpifaultsize;               wangzj

    //Tau[0] = f.T11[pos1];
    //Tau[1] = f.T12[pos1];
    //Tau[2] = f.T13[pos1];

    //f.tTs1[pos] = dot_product(Tau, vec_s1);
    //f.tTs2[pos] = dot_product(Tau, vec_s2);
    //f.tTn [pos] = dot_product(Tau, vec_n);
    f.tTs1[pos] = DOT3(Tau, vec_s1);
    f.tTs2[pos] = DOT3(Tau, vec_s2);
    f.tTn [pos] = DOT3(Tau, vec_n);
    //f.tTn [pos] = Tau_n;
    //f.tTn [pos] = -120e6;

   // pos = j + k * nj;                            wangzj
    if(!f.init_t0_flag[pos]) {
      if (V > 1e-3) {
        f.init_t0[pos] = it * DT;
        f.init_t0_flag[pos] = 1;
        f.flag_rup[pos] = 1;
      }
    }

#ifdef SelectStencil
    pos = j + k*nj + mpifaultsize;                    //wangzj
    if(f.flag_rup[pos] && f.first_rup[pos]){
      for (int l = -3; l <=3; l++){
        int jj1 = j+l;
        jj1 = MAX(0, jj1);
        jj1 = MIN(nj-1, jj1);
        int pos2 = (jj1+k*nj + mpifaultsize);           //wangzj
        f.rup_index_y[pos2] += 1;
        int kk1 = k+l;
        kk1 = MAX(0, kk1);
        kk1 = MIN(nk-1, kk1);
        pos2 = (j+kk1*nj + mpifaultsize);               //wangzj
        f.rup_index_z[pos2] += 1;
      }
      f.first_rup[pos] = 0;
    }
#endif
  } // end j k
  return;
}

void trial_rs(Wave W,Fault F,real_t *M,int it,int irk,int FlagX,int FlagY,int FlagZ, int i0, int nfault, int Faultgrid[])
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);
  trial_rs_cu <<<grid, block>>> (W, F, M, it, irk, FlagX, FlagY, FlagZ, i0, nfault, Faultgrid);
}
