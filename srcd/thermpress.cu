#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

#define SQRT2PI 2.506628274631000

//#define DEBUG
//extern __device__ __constant__ int Flags[8][3];
//extern __device__ real_t norm3(real_t *A);
//extern __device__ real_t dot_product(real_t *A, real_t *B);
//extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
//extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
//extern __device__ void invert3x3(real_t m[][3]);
//extern __device__ real_t Fr_func(const real_t r, const real_t R);
//extern __device__ real_t Gt_func(const real_t t, const real_t T);
//
//#define DOT3(A,B) (A[0]*B[0]+A[1]*B[1]+A[2]*B[2])


__global__ void thermpress_cu(Wave W, Fault F, real_t *M,
    int it, int irk, int FlagX, int FlagY, int FlagZ, int nfault)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  //int j1 = j + 3;
  //int k1 = k + 3;

  //int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int mpifaultsize = nfault * nj*nk;
  //int nx = ni + 6;
  //int ny = nj + 6;
  //int nz = nk + 6;

  //int nxyz = nx * ny * nz;
  //int nyz = ny * nz;
  //int nyz2 = 2 * ny * nz;

  //real_t *XIX = M;
  //real_t *XIY = XIX + nxyz;
  //real_t *XIZ = XIY + nxyz;
  //real_t *JAC = M + 9 * nxyz;
  //real_t *RHO = M + 12 * nxyz;

  //stride = nx * ny * nz; // x vary first
  //real_t *w_Vx  = w.W + 0 * nxyz;
  //real_t *w_Vy  = w.W + 1 * nxyz;
  //real_t *w_Vz  = w.W + 2 * nxyz;
  //real_t *w_Txx = w.W + 3 * nxyz;
  //real_t *w_Tyy = w.W + 4 * nxyz;
  //real_t *w_Tzz = w.W + 5 * nxyz;
  //real_t *w_Txy = w.W + 6 * nxyz;
  //real_t *w_Txz = w.W + 7 * nxyz;
  //real_t *w_Tyz = w.W + 8 * nxyz;

  // Split nodes
  //stride = ny * nz * 2; // y vary first
  //real_t *f_Vx  = f.W + 0 * nyz2;
  //real_t *f_Vy  = f.W + 1 * nyz2;
  //real_t *f_Vz  = f.W + 2 * nyz2;
  //real_t *f_T21 = f.W + 3 * nyz2;
  //real_t *f_T22 = f.W + 4 * nyz2;
  //real_t *f_T23 = f.W + 5 * nyz2;
  //real_t *f_T31 = f.W + 6 * nyz2;
  //real_t *f_T32 = f.W + 7 * nyz2;
  //real_t *f_T33 = f.W + 8 * nyz2;

  //real_t *f_mVx  = f.mW + 0 * nyz2;
  //real_t *f_mVy  = f.mW + 1 * nyz2;
  //real_t *f_mVz  = f.mW + 2 * nyz2;

  //real_t *f_tVx  = f.tW + 0 * nyz2;
  //real_t *f_tVy  = f.tW + 1 * nyz2;
  //real_t *f_tVz  = f.tW + 2 * nyz2;

  //real_t DH = par.DH;
  //real_t rDH = 1.0/DH;
  //real_t DT = par.DT;

  //int i0 = nx/2;

  //real_t xix, xiy, xiz;
  ////real_t etx, ety, etz;
  ////real_t ztx, zty, ztz;
  //real_t jac;
  //real_t vec_n0;
  //real_t jacvec;
  ////real_t lam, mu;
  ////real_t lam2mu;
  ////real_t rrho;
  //real_t rho;

  //real_t Mrho[2], Rx[2], Ry[2], Rz[2];
  //real_t R1[2], R2[2], R3[2];
  //real_t DH2 = DH*DH;
  //real_t T11, T12, T13;
  //int i;
  //int pos, pos_f, pos1;
  ////int slice;
  //real_t DyT21, DyT22, DyT23;
  //real_t DzT31, DzT32, DzT33;
  //real_t vecT31[7];
  //real_t vecT32[7];
  //real_t vecT33[7];

  //real_t alphy_hy, tauV;
  real_t Lamb = 0.1e6, rho_c = 2.7e6, h = 20e-3, alpha_th = 1.0e-6;//, alpha_hy0 = 4e-4;
  real_t h2 = h*h;
  //real_t TP_x, omega, dT, dP;
  //real_t TP_dt, TP_dh, TP_w, alpha;
  //int nt_sub;

  if ( j < nj && k < nk && F.united[j + k * nj + mpifaultsize] == 0){  // not united  wangzj

    //int pos1 = j1 + k1 * ny + 3*nyz;

    //f.mT11[pos1] = f.T11[pos1];
    //f.mT12[pos1] = f.T12[pos1];
    //f.mT13[pos1] = f.T13[pos1];
    //if(irk == 0){
    //  f.mT11[pos1] = f.T11[pos1];
    //  f.mT12[pos1] = f.T12[pos1];
    //  f.mT13[pos1] = f.T13[pos1];
    //}
    //km = NZ - (thisid[2]*nk+k-3);
    //int km = (nz - 6) - (k-3); // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    //int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)

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
    //for (int ii = 0; ii < 3; ii++){
    //  vec_n[ii] /= vec_n0;
    //}

    int pos = j + k * nj + mpifaultsize;       //********wangzj
    //real_t tau  = sqrt(F.Ts1[pos]*F.Ts1[pos]+F.Ts2[pos]*F.Ts2[pos]);
    real_t rate = sqrt(F.Vs1[pos]*F.Vs1[pos]+F.Vs2[pos]*F.Vs2[pos]);
    real_t Tau_n = F.Tn[pos];
    real_t friction = F.friction[pos];
    

    //real_t tauV = rate * tau;
    real_t alpha_hy = F.TP_hy[pos];

    //real_t alpha_max = max(alpha_th, (alpha_hy0+1.0));
    real_t alpha_max = MAX(alpha_th, alpha_hy);
    real_t TP_w = sqrt(4.0*alpha_max*25.0);
    real_t TP_dh = TP_w / (double) (par.TP_n-1);
    //TP_dh = 1e-3;
    real_t TP_dh2 = TP_dh*TP_dh;
    real_t TP_dt = 0.1 * TP_dh2 / alpha_max;
    int nt_sub = (int)(par.DT / TP_dt) + 1;
    TP_dt = par.DT/(double)nt_sub;
    int TP_n = par.TP_n;

    //if(j == par.ny/2 && k == par.nz/2){
    //  printf("nt_sub=%d,TP_dt=%g,TP_dh=%g\n", nt_sub,TP_dt,TP_dh);
    //}

    // test
    //TP_n = 100;
    //nt_sub = 10;

    //real_t *dT, *dP;
    //cudaMalloc((real_t **)dT, sizeof(real_t)*TP_n);
    //cudaMalloc((real_t **)dP, sizeof(real_t)*TP_n);
    //cudaMemset(dT, 0, sizeof(real_t)*TP_n);
    //cudaMemset(dP, 0, sizeof(real_t)*TP_n);

    for (int it = 0; it < nt_sub; it++){
      real_t tau = -friction * (Tau_n + F.TP_P[j+k*nj+0*nj*nk + mpifaultsize*TP_n]);      //wangzj
      real_t tauV = tau * rate;
      for (int i = 0; i < TP_n; i++){
        real_t TP_x = i * TP_dh;
        real_t TP_x2 = TP_x * TP_x;
        real_t omega = exp(-0.5*TP_x2/h2) * (tauV / (rho_c*h*SQRT2PI) );

        long pos_m = j + k*nj + (i-1)*nj*nk + mpifaultsize * TP_n;        //wangzj
        long pos   = j + k*nj + (i  )*nj*nk + mpifaultsize * TP_n;
        long pos_p = j + k*nj + (i+1)*nj*nk + mpifaultsize * TP_n;
        if (i == 0){
          F.TP_dT[pos] = alpha_th * (F.TP_T[pos_p]-2.0*F.TP_T[pos]+F.TP_T[pos_p])/TP_dh2 + omega;
          F.TP_dP[pos] = alpha_hy * (F.TP_P[pos_p]-2.0*F.TP_P[pos]+F.TP_P[pos_p])/TP_dh2 + Lamb * F.TP_dT[pos];
        }else if (i == TP_n-1){
          F.TP_dT[pos] = alpha_th * (F.TP_T[pos_m]-2.0*F.TP_T[pos]+F.TP_T[pos_m])/TP_dh2 + omega;
          F.TP_dP[pos] = alpha_hy * (F.TP_P[pos_m]-2.0*F.TP_P[pos]+F.TP_P[pos_m])/TP_dh2 + Lamb * F.TP_dT[pos];
        }else{
          F.TP_dT[pos] = alpha_th * (F.TP_T[pos_p]-2.0*F.TP_T[pos]+F.TP_T[pos_m])/TP_dh2 + omega;
          F.TP_dP[pos] = alpha_hy * (F.TP_P[pos_p]-2.0*F.TP_P[pos]+F.TP_P[pos_m])/TP_dh2 + Lamb * F.TP_dT[pos];
        }

        //if(j == par.ny/2 - 40 && k == par.nz/2 && i > 0 && i < TP_n-1){
        //  printf("i=%d, exp=%g, tau=%g, V=%g, omega=%g, dT=%g\n", 
        //      i, exp(-0.5*TP_x2/h2), tau, rate, omega,
        //      F.TP_T[pos_p]-2.0*F.TP_T[pos]+F.TP_T[pos_m]
        //      );
        //}
        
      }

      for (int i = 0; i < TP_n; i++){
        long pos = j + k*nj + i*nj*nk + mpifaultsize * TP_n;
        F.TP_T[pos] += TP_dt * F.TP_dT[pos];
        F.TP_P[pos] += TP_dt * F.TP_dP[pos];
      }

      //F.str_init_x[j+k*nj] += F.TP_P[j1+k1*ny+0*ny*nz];
    } // end time substeps

    //CUDACHECK(cudaFree( dT ));
    //CUDACHECK(cudaFree( dP ));

  } // end j k

  return;
}

void thermpress(Wave W,Fault F,real_t *M,int it,int irk,int FlagX,int FlagY,int FlagZ, int nfault)
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);
  thermpress_cu <<<grid, block>>> (W, F, M, it, irk, FlagX, FlagY, FlagZ, nfault);
}
