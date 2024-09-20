#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"

//#include "netcdf.h"
//#define handle_err(e)                         \
//{                                             \
//  if (e != NC_NOERR) {                        \
//    printf("nc error: %s\n", nc_strerror(e)); \
//    exit(2);                                  \
//  }                                           \
//}

//#define InitialCondition
#ifdef DoublePrecision
#define nc_get_vara_real_t nc_get_vara_double
#define nc_put_vara_real_t nc_put_vara_double
#else
#define nc_get_vara_real_t nc_get_vara_float
#define nc_put_vara_real_t nc_put_vara_float
#endif

extern __device__ real_t Bfunc(const real_t x, const real_t W, const real_t w);
extern __device__ real_t Fr_func(const real_t r, const real_t R);
extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
extern __device__ real_t dot_product(real_t *A, real_t *B);
extern __device__ real_t norm3(real_t *A);

//__global__ void init_metric_2d(real_t *m){
//  int j = blockIdx.x * blockDim.x + threadIdx.x;
//  int k = blockIdx.y * blockDim.y + threadIdx.y;
//  int i, pos;
//  if( j < ny && k < nz ){
//    for (i = 0; i < nx; i++){
//      pos = (i*ny*nz + j*nz + k)*MSIZE;
//      m[pos + 0] = 1.0;
//      m[pos + 1] = 0.0;
//      m[pos + 2] = 0.0;
//      m[pos + 3] = 0.0;
//      m[pos + 4] = 1.0;
//      m[pos + 5] = 0.0;
//      m[pos + 6] = 0.0;
//      m[pos + 7] = 0.0;
//      m[pos + 8] = 1.0;
//      m[pos + 9] = 1.0; // jac
//      m[pos + 10] = 32e9;  // lam
//      m[pos + 11] = 32e9;  // mu
//      m[pos + 12] = 2700.0; // rho
//    }
//  }
//}
//
//__global__ void init_metric_3d(real_t *m){
//  int i = blockIdx.x * blockDim.x + threadIdx.x;
//  int j = blockIdx.y * blockDim.y + threadIdx.y;
//  int k = blockIdx.z * blockDim.z + threadIdx.z;
//  int  pos;
//  if( j < ny && k < nz && i < nx){
//      pos = (i*ny*nz + j*nz + k)*MSIZE;
//      m[pos + 0] = 1.0;
//      m[pos + 1] = 0.0;
//      m[pos + 2] = 0.0;
//      m[pos + 3] = 0.0;
//      m[pos + 4] = 1.0;
//      m[pos + 5] = 0.0;
//      m[pos + 6] = 0.0;
//      m[pos + 7] = 0.0;
//      m[pos + 8] = 1.0;
//      m[pos + 9] = 1.0; // jac
//      m[pos + 10] = rho1*(vp1*vp1-2.f*vs1*vs1);  // lam
//      m[pos + 11] = rho1*vs1*vs1;  // mu
//      m[pos + 12] = rho1; // rho
//  }
//}
//
__global__ void init_wave_3d_cu(real_t *W){
  //return;
//#ifdef InitialCondition
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;
  int i1 = i + 3;
  int j1 = j + 3;
  int k1 = k + 3;

  int ni = par.ni;
  int nj = par.nj;
  int nk = par.nk;
  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  int stride = nx*ny*nz;
  real_t *Vx  = W;
  real_t *Vy  = Vx  + stride;
  real_t *Vz  = Vy  + stride;
  real_t *Txx = Vz  + stride;
  real_t *Tyy = Txx + stride;
  real_t *Tzz = Tyy + stride;
  real_t *Txy = Tzz + stride;
  real_t *Txz = Txy + stride;
  real_t *Tyz = Txz + stride;

  //int pos = (i1*ny*nz + j1*nz + k1)*WSIZE;
  //int pos = i1 + j1 * nx + k1 * nx * ny;
  int pos = j1 + k1 * ny + i1 * ny * nz;

  int gi = par.rankx * ni + i;
  int gj = par.ranky * nj + j;
  int gk = par.rankz * nk + k;

//  if(0==i && 0==j && 0==k)
//    printf("rankxyz = %d %d %d\n", par.rankx, par.ranky, par.rankz);

  int NX = par.NX;
  int NY = par.NY;
  int NZ = par.NZ;

  //real_t r2 = powf(gi-NX/2, 2) + powf(gj-NY*0.5, 2) + powf(gk-NZ*0.9, 2);
  //real_t r2 = powf(i-ni/2, 2) + powf(j-nj/2, 2) + powf(k-nk*0.5, 2);
  //real_t a2 = powf(7.0, 2);
  //real_t M0 = 1e10 * expf(-r2/a2);

  //if(r2 > powf(15.0, 2)){
  //  M0 = 0;
  //}

  if(i < ni && j < nj && k < nk){

    Vx[pos] = 0;//M0 * 1.0e-10; // test
    Vy[pos] = 0;//M0 * 1.0e-10; // test
    Vz[pos] = 0;//M0 * 1.0e-10; // test

    //Txx[pos] = M0 * 1.0;
    //Tyy[pos] = M0 * 1.0;
    //Tzz[pos] = M0 * 1.0;
    //Txy[pos] = M0 * 0.0;
    //Txz[pos] = M0 * 0.0;
    //Tyz[pos] = M0 * 0.0;
    //if(i < ni/2){
    //  Vy[pos] = -5e-13;
    //}else if(i> ni/2){
    //  Vy[pos] = 5e-13;
    //}

  } // end j k
//#endif
}

void init_wave_3d(real_t *W){
  dim3 block(8, 4, 4);
  dim3 grid;
  grid.x = (hostParams.nj + block.x -1) / block.x;
  grid.y = (hostParams.nk + block.y -1) / block.y;
  grid.z = (hostParams.ni + block.z -1) / block.z;
  init_wave_3d_cu <<<grid, block>>> (W);
}

__global__ void init_fault_cu(Fault F, int nfault, int Faultgrid[]){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int pos, pos1;
  int pos_v;
  int NY = par.NY;
  int NZ = par.NZ;

  int nj = par.NY/par.PY;
  int nk = par.NZ/par.PZ;

  int ny = nj + 6;
  int nz = nk + 6;

  //int gi = par.rankx * ni + i;
  int gj = par.ranky * nj + j;
  int gk = par.rankz * nk + k;
  int gj1 = gj + 1;
  int gk1 = gk + 1;

  real_t DH = par.DH;

  if( j < nj && k < nk ){

    pos = j + k * nj + nfault * (nj*nk);    //*********wangzj
    pos1 = j1 + k1 * ny;

    F.rup_index_y[pos] = 0;
    F.rup_index_z[pos] = 0;
    F.flag_rup   [pos] = 0;
    F.first_rup  [pos] = 1;
    F.smooth_flag[pos] = 0;

    pos_v = (k1 * ny + j1) * 3 + nfault * (ny*nz*3);
    real_t vec_n[3], vec_s1[3], vec_s2[3];
    vec_n [0] = F.vec_n [pos_v + 0];
    vec_n [1] = F.vec_n [pos_v + 1];
    vec_n [2] = F.vec_n [pos_v + 2];
    vec_s1[0] = F.vec_s1[pos_v + 0];
    vec_s1[1] = F.vec_s1[pos_v + 1];
    vec_s1[2] = F.vec_s1[pos_v + 2];
    vec_s2[0] = F.vec_s2[pos_v + 0];
    vec_s2[1] = F.vec_s2[pos_v + 1];
    vec_s2[2] = F.vec_s2[pos_v + 2];

    //!!real_t T_global[3][3];
    //!!T_global[0][0] = -60e6;
    //!!T_global[1][1] = -60e6;
    //!!T_global[2][2] = -60e6;
    //!!T_global[0][1] = -29.38e6;
    //!!T_global[0][2] = 0;
    //!!T_global[1][2] = 0;

    //!!T_global[1][0] = T_global[0][1];
    //!!T_global[2][0] = T_global[0][2];
    //!!T_global[2][1] = T_global[1][2];

    //!!real_t temp_init_stress_xyz[3];
    //!!matmul3x1(T_global, vec_n, temp_init_stress_xyz);

    //!!real_t tn = dot_product(temp_init_stress_xyz, vec_n);
    //!!real_t ts_vec[3], ts_vec_new[3];
    //!!ts_vec[0] = temp_init_stress_xyz[0] - tn * vec_n[0];
    //!!ts_vec[1] = temp_init_stress_xyz[1] - tn * vec_n[1];
    //!!ts_vec[2] = temp_init_stress_xyz[2] - tn * vec_n[2];

    //!!real_t ts = norm3(ts_vec);

    //!!real_t y = (gj-NY*0.5) * DH;
    //!!real_t z = (NZ-1-gk) * DH - 7.5e3;
    //!!real_t r = sqrt(y*y+z*z);
    //!!real_t tau_nuke = 0;
    //!!if (r <= 1.4e3){
    //!!  tau_nuke = 11.6e6;
    //!!}else if (r <= 2.0e3){
    //!!  tau_nuke = 5.8e6 * (1.0 + cos(pi * (r - 1.4e3)/600.));
    //!!}else{
    //!!  tau_nuke = 0.0;
    //!!}

    //!!real_t ts_new = ts + tau_nuke;
    //!!//ts_new = ts;

    //!!ts = max(ts, 1.0);

    //!!ts_vec_new[0] = ts_new / ts * ts_vec[0] ;
    //!!ts_vec_new[1] = ts_new / ts * ts_vec[1] ;
    //!!ts_vec_new[2] = ts_new / ts * ts_vec[2] ;

    //!!//ts_vec_new[0] = ts_vec[0];
    //!!//ts_vec_new[1] = ts_vec[1];
    //!!//ts_vec_new[2] = ts_vec[2];

    //!!temp_init_stress_xyz[0] = tn * vec_n[0] + ts_vec_new[0];
    //!!temp_init_stress_xyz[1] = tn * vec_n[1] + ts_vec_new[1];
    //!!temp_init_stress_xyz[2] = tn * vec_n[2] + ts_vec_new[2];

    //!!//F.str_init_x[pos] = -60e6;
    //!!//F.str_init_y[pos] = -29.38e6;
    //!!//F.str_init_z[pos] = 0.0;
    //!!F.str_init_x[pos] = temp_init_stress_xyz[0];
    //!!F.str_init_y[pos] = temp_init_stress_xyz[1];
    //!!F.str_init_z[pos] = temp_init_stress_xyz[2];

    //int j0 = NY/2;
    //int k0 = NZ-(int)(7.5e3/DH);
    //int k0 = NZ/2;
    //int wid = (int)(1.5e3/DH);
    F.Tn[pos] = -120.0e6;
    //F.Ts1[pos] = -70.0e6;
    F.Ts1[pos] = F.Tn[pos] * par.mu_0;
    F.Ts2[pos] = 0.0;
    //if( gj >= j0 - wid && gj <= j0 + wid && gk >= k0 - wid && gk <= k0 + wid){
    //  F.Ts1[pos] = -81.6e6;
    //}
    if( gj1 >= par.Asp_grid[0] && gj1 <= par.Asp_grid[1] &&
        gk1 >= par.Asp_grid[2] && gk1 <= par.Asp_grid[3] )
    {
      //F.Ts1[pos] = -81.6e6;
      //F.Ts1[pos] = 1.005 * par.mu_s * F.Tn[pos];
      F.Ts1[pos] = (par.mu_s + (par.mu_s-par.mu_d) * 0.001 ) * F.Tn[pos] ;
      // add const
      //F.Ts1[pos] = (par.mu_s + 0.001 ) * F.Tn[pos] ;
    }

#if defined TPV6 || defined TPV7
    F.Ts1[pos] = -70.0e6;
    F.Ts2[pos] = 0.0;
    F.Tn[pos] = -120.0e6;
    if( gj1 >= par.Asp_grid[0] && gj1 <= par.Asp_grid[1] &&
        gk1 >= par.Asp_grid[2] && gk1 <= par.Asp_grid[3] ){
      F.Ts1[pos] = -81.6e6;
    }
#endif
//#define TPV3_hill
#if defined TPV3_hill
    F.str_init_y[pos] = -70.0e6;
    F.str_init_z[pos] = 0.0;
    F.str_init_x[pos] = -120.0e6;
    if( gj1 >= par.Asp_grid[0] && gj1 <= par.Asp_grid[1] &&
        gk1 >= par.Asp_grid[2] && gk1 <= par.Asp_grid[3] ){
      F.str_init_y[pos] = -81.6e6;
    }
    F.Tn [pos] = F.str_init_x[pos] * vec_n [0]
               + F.str_init_y[pos] * vec_n [1]
               + F.str_init_z[pos] * vec_n [2];
    F.Ts1[pos] = F.str_init_x[pos] * vec_s1[0]
               + F.str_init_y[pos] * vec_s1[1]
               + F.str_init_z[pos] * vec_s1[2];
    F.Ts2[pos] = F.str_init_x[pos] * vec_s2[0]
               + F.str_init_y[pos] * vec_s2[1]
               + F.str_init_z[pos] * vec_s2[2];
#endif
    /*
    // higher stress
    j0 = nj/2 - (int)(7.5e3/DH);
    if( j >= j0 - wid && j <= j0 + wid && k >= k0 - wid && k <= k0 + wid){
      F.Ts1[pos] = -78.0e6;
    }
    // lower stress
    j0 = nj/2 + (int)(7.5e3/DH);
    if( j >= j0 - wid && j <= j0 + wid && k >= k0 - wid && k <= k0 + wid){
      F.Ts1[pos] = -62.0e6;
    }
    */
    //j0 = NY/2; // reset j0 k0
    //int widy = (int)(15.0e3/DH);
    //int widy = (int)(105.0e3/DH);
    //int widz = (int)( 7.5e3/DH);
    //if( 
    //    gj >= j0 - widy &&
    //    gj <= j0 + widy &&
    //    gk >= k0 - widz &&
    //    gk <= k0 + widz 
    if( gj1 >= Faultgrid[0 + 4*nfault] && gj1 <= Faultgrid[1 + 4*nfault] && 
        gk1 >= Faultgrid[2 + 4*nfault] && gk1 <= Faultgrid[3 + 4*nfault] ){
    // if( gj1 >= par.Fault_grid[0] && gj1 <= par.Fault_grid[1] && 
    //     gk1 >= par.Fault_grid[2] && gk1 <= par.Fault_grid[3] ){
      F.str_peak[pos] = par.mu_s;
    }else{
      F.str_peak[pos] = 10000.;
    }

    //pos_v = (j1*nz + k1)*3;
    F.str_init_x[pos] = F.Tn [pos] * vec_n [0]
                      + F.Ts1[pos] * vec_s1[0] 
                      + F.Ts2[pos] * vec_s2[0];
    F.str_init_y[pos] = F.Tn [pos] * vec_n [1] 
                      + F.Ts1[pos] * vec_s1[1] 
                      + F.Ts2[pos] * vec_s2[1];
    F.str_init_z[pos] = F.Tn [pos] * vec_n [2] 
                      + F.Ts1[pos] * vec_s1[2] 
                      + F.Ts2[pos] * vec_s2[2];
//    F.Tn [pos] = F.str_init_x[pos] * vec_n [0]
//               + F.str_init_y[pos] * vec_n [1]
//               + F.str_init_z[pos] * vec_n [2];
//    F.Ts1[pos] = F.str_init_x[pos] * vec_s1[0]
//               + F.str_init_y[pos] * vec_s1[1]
//               + F.str_init_z[pos] * vec_s1[2];
//    F.Ts2[pos] = F.str_init_x[pos] * vec_s2[0]
//               + F.str_init_y[pos] * vec_s2[1]
//               + F.str_init_z[pos] * vec_s2[2];
//
    //if(j==nj/2 && k==nk/2) printf("str_init = %e %e %e\n", 
    //    F.str_init_x[pos],
    //    F.str_init_y[pos],
    //    F.str_init_z[pos]);
#ifdef TPV10
    int j0 = NY/2;
    int k0 = NZ-(int)(12e3/DH);
    int wid = (int)(1.5e3/DH);
    F.Tn[pos] = -7378.0*(NZ-gk1)*DH;
    if(gk1==NZ) F.Tn[pos] = -7378.0*DH/3.0;
    F.Ts1[pos] = 0.0;
    F.Ts2[pos] = 0.55*F.Tn[pos];
    F.C0[pos] = 0.2e6;
    //F.C0[pos] = 0.0;
    if( gj1 >= j0 - wid && gj1 <= j0 + wid &&
        gk1 >= k0 - wid && gk1 <= k0 + wid ){
      F.Ts2[pos] = (par.mu_s + 0.0057)* F.Tn[pos] - 0.2e6;
    }
    F.str_init_x[pos] = F.Tn [pos] * vec_n [0]
                      + F.Ts1[pos] * vec_s1[0]
                      + F.Ts2[pos] * vec_s2[0];
    F.str_init_y[pos] = F.Tn [pos] * vec_n [1]
                      + F.Ts1[pos] * vec_s1[1]
                      + F.Ts2[pos] * vec_s2[1];
    F.str_init_z[pos] = F.Tn [pos] * vec_n [2]
                      + F.Ts1[pos] * vec_s1[2]
                      + F.Ts2[pos] * vec_s2[2];
#endif

#ifdef TPV28
    real_t DH = par.DH;
    real_t depth = (NZ-gk1) * DH; // freesurface: z = 0
    real_t T_global[3][3];

    T_global[2][2] = 0.0;
    //if(par.freenode && gk1 == NZ) T_global[2][2] = -2670*9.8*DH/10.0;
    T_global[1][1] = -60.00e6;
    T_global[0][0] = -60.00e6;
    T_global[0][1] = -29.38e6;

    T_global[0][2] = 0;
    T_global[1][2] = 0;

    real_t C0 = 0.0;
    F.C0[pos] = C0 * 1e6;

    T_global[1][0] = T_global[0][1];
    T_global[2][0] = T_global[0][2];
    T_global[2][1] = T_global[1][2];

    real_t temp_init_stress_xyz[3];
    matmul3x1(T_global, vec_n, temp_init_stress_xyz);

    real_t tn = dot_product(temp_init_stress_xyz, vec_n);
    real_t ts_vec[3], ts_vec_new[3];
    ts_vec[0] = temp_init_stress_xyz[0] - tn * vec_n[0];
    ts_vec[1] = temp_init_stress_xyz[1] - tn * vec_n[1];
    ts_vec[2] = temp_init_stress_xyz[2] - tn * vec_n[2];

    real_t ts = norm3(ts_vec);

    real_t y = (gj1-NY*0.5) * DH - 0.0e3;
    real_t z = (NZ-gk1) * DH - 7.5e3;
#ifndef FreeSurface
    z = (gk1-NZ*0.5)*DH;
#endif
    real_t r = sqrt(y*y+z*z);
    real_t tau_nuke = 0;
    if (r <= 1.4e3){
      tau_nuke = 11.6e6;
    }else if (r <= 2.0e3){
      tau_nuke = 5.8e6 * (1.0 + cos(PI * (r - 1.4e3)/600.));
    }else{
      tau_nuke = 0.0;
    }

    real_t ts_new = ts + tau_nuke;

    ts = MAX(ts, 1.0);

    ts_vec_new[0] = ts_new / ts * ts_vec[0] ;
    ts_vec_new[1] = ts_new / ts * ts_vec[1] ;
    ts_vec_new[2] = ts_new / ts * ts_vec[2] ;

    temp_init_stress_xyz[0] = tn * vec_n[0] + ts_vec_new[0];
    temp_init_stress_xyz[1] = tn * vec_n[1] + ts_vec_new[1];
    temp_init_stress_xyz[2] = tn * vec_n[2] + ts_vec_new[2];

    F.str_init_x[pos] = temp_init_stress_xyz[0];
    F.str_init_y[pos] = temp_init_stress_xyz[1];
    F.str_init_z[pos] = temp_init_stress_xyz[2];

    F.Tn [pos] = F.str_init_x[pos] * vec_n [0]
               + F.str_init_y[pos] * vec_n [1]
               + F.str_init_z[pos] * vec_n [2];
    F.Ts1[pos] = F.str_init_x[pos] * vec_s1[0]
               + F.str_init_y[pos] * vec_s1[1]
               + F.str_init_z[pos] * vec_s1[2];
    F.Ts2[pos] = F.str_init_x[pos] * vec_s2[0]
               + F.str_init_y[pos] * vec_s2[1]
               + F.str_init_z[pos] * vec_s2[2];

    int j0 = NY/2; // reset j0 k0
    int k0 = NZ-int(7.5e3/DH);
#ifndef FreeSurface
    k0 = NZ/2;
#endif
    int widy = (int)(18.0e3/DH);
    int widz = (int)(7.5e3/DH);
    if(
        gj1 >= j0 - widy &&
        gj1 <= j0 + widy &&
        gk1 >= k0 - widz
#ifndef FreeSurface
        && gk1 <= k0 + widz
#endif
        ){
      F.str_peak[pos] = par.mu_s;
    }else{
      F.str_peak[pos] = 10000.;
    }
#endif

#ifdef TPV29
    real_t DH = par.DH;
    real_t omeg, Pf;
    real_t depth = (NZ-gk1) * DH; // freesurface: z = 0
    real_t T_global[3][3];

    if(depth <= 17e3){
      omeg = 1.0;
    }else if(depth <= 22e3){
      omeg = (22e3 - depth) / 5e3;
    }else{
      omeg = 0;
    }
    Pf = 9.8e3 * depth;
    T_global[2][2] = -2670*9.8*depth;
    if(par.freenode && gk1 == NZ) T_global[2][2] = -2670*9.8*DH/10.0;
    T_global[1][1] = omeg*(1.025837*(T_global[2][2]+Pf)-Pf) + (1-omeg)*T_global[2][2];
    T_global[0][0] = omeg*(0.974162*(T_global[2][2]+Pf)-Pf) + (1-omeg)*T_global[2][2];
    T_global[0][1] = omeg*(-0.158649*(T_global[2][2]+Pf));
    T_global[0][1] = -T_global[0][1];

    //T_global[2][2] += Pf;
    //T_global[1][1] += Pf;
    //T_global[0][0] += Pf;

    T_global[0][2] = 0;
    T_global[1][2] = 0;

    real_t C0;
    if(depth <= 4e3){
      C0 = 0.4 + 0.00020 * (4000 - depth);
    }else{
      C0 = 0.4;
    }
    F.C0[pos] = C0 * 1e6;

    T_global[1][0] = T_global[0][1];
    T_global[2][0] = T_global[0][2];
    T_global[2][1] = T_global[1][2];

    real_t temp_init_stress_xyz[3];
    matmul3x1(T_global, vec_n, temp_init_stress_xyz);

    real_t tn = dot_product(temp_init_stress_xyz, vec_n);
    real_t ts_vec[3], ts_vec_new[3];
    ts_vec[0] = temp_init_stress_xyz[0] - tn * vec_n[0];
    ts_vec[1] = temp_init_stress_xyz[1] - tn * vec_n[1];
    ts_vec[2] = temp_init_stress_xyz[2] - tn * vec_n[2];

    real_t ts = norm3(ts_vec);

    real_t y = (gj1-NY*0.5) * DH + 5e3;
    real_t z = (NZ-gk1) * DH - 10e3;
    real_t r = sqrt(y*y+z*z);
    real_t tau_nuke = 0;
    if (r <= 1.4e3){
      tau_nuke = 11.6e6;
    }else if (r <= 2.0e3){
      tau_nuke = 5.8e6 * (1.0 + cos(PI * (r - 1.4e3)/600.));
    }else{
      tau_nuke = 0.0;
    }

    real_t ts_new = ts + tau_nuke;
    //ts_new = ts;

    ts = MAX(ts, 1.0);

    ts_vec_new[0] = ts_new / ts * ts_vec[0] ;
    ts_vec_new[1] = ts_new / ts * ts_vec[1] ;
    ts_vec_new[2] = ts_new / ts * ts_vec[2] ;

    //ts_vec_new[0] = ts_vec[0];
    //ts_vec_new[1] = ts_vec[1];
    //ts_vec_new[2] = ts_vec[2];

    //temp_init_stress_xyz[0] = tn * vec_n[0] + ts_vec_new[0];
    //temp_init_stress_xyz[1] = tn * vec_n[1] + ts_vec_new[1];
    //temp_init_stress_xyz[2] = tn * vec_n[2] + ts_vec_new[2];

    F.str_init_x[pos] = temp_init_stress_xyz[0];
    F.str_init_y[pos] = temp_init_stress_xyz[1];
    F.str_init_z[pos] = temp_init_stress_xyz[2];

    F.Tn [pos] = F.str_init_x[pos] * vec_n [0]
               + F.str_init_y[pos] * vec_n [1]
               + F.str_init_z[pos] * vec_n [2];
    F.Ts1[pos] = F.str_init_x[pos] * vec_s1[0]
               + F.str_init_y[pos] * vec_s1[1]
               + F.str_init_z[pos] * vec_s1[2];
    F.Ts2[pos] = F.str_init_x[pos] * vec_s2[0]
               + F.str_init_y[pos] * vec_s2[1]
               + F.str_init_z[pos] * vec_s2[2];

    int j0 = NY/2; // reset j0 k0
    int k0 = NZ-int(10e3/DH);
    int widy = (int)(20e3/DH);
    int widz = (int)(10e3/DH);
#ifndef FreeSurface
    widz = widz - 50;
#endif
    if(
        gj1 >= j0 - widy &&
        gj1 <= j0 + widy &&
        gk1 >= k0 - widz &&
        gk1 <= k0 + widz
        ){
      F.str_peak[pos] = par.mu_s;
    }else{
      F.str_peak[pos] = 10000.;
    }
#endif
#if defined TPV101 || defined TPV102
    real_t a0 = 0.008;
    real_t b0 = 0.012;
    real_t da0 = 0.008;
    //real_t W = 15e3;
    real_t W = 15e3;
    real_t w = 3e3;
    //real_t RS_f0 = 0.6;
    real_t RS_V0 = 1e-6;
    //real_t RS_L = 0.02;
    real_t RS_Vini = 1e-12;
    real_t Tau_ini = -75e6;
    //real_t dTau0 = 25e6;
    real_t Tn_ini = -120e6;
    //real_t R = 3e3;

    real_t y = (gj-NY/2) * DH;
#ifdef FreeSurface
    real_t z = (NZ-1-gk) * DH - 7.5e3;
#else
    real_t z = (gk-NZ/2) * DH;
#endif
    real_t r = sqrt(y*y+z*z);
    F.a[pos] = a0 + da0*(1.0-Bfunc(y, W/1.0, w)*Bfunc(z, W/2.0, w));
    F.b[pos] = b0;
    real_t RS_a = F.a[pos];
    //real_t RS_b = F.b[pos];
    //RS_a = 0.008;
    //RS_b = 0.012;

    //real_t tmp = RS_a*log(2.0*sinh(Tau_ini/RS_a/Tn_ini))-RS_f0-RS_a*log(RS_Vini/RS_V0);
    //F.State[pos] = RS_L/RS_V0 * exp( tmp/RS_b);
    F.State[pos] = RS_a * log(2.0*sinh(Tau_ini/RS_a/Tn_ini)) - RS_a * log(RS_Vini/RS_V0);

    int stride = 2 * ny * nz;
    real_t *f_Vx = F.W + 0 * stride;
    real_t *f_Vy = F.W + 1 * stride;
    real_t *f_Vz = F.W + 2 * stride;

    //f_Vy[pos1 + 0*ny*nz] = +RS_Vini * 0.5;
    //f_Vy[pos1 + 1*ny*nz] = -RS_Vini * 0.5;

    F.Ts1[pos] = Tau_ini;
    F.Ts2[pos] = 0.0;
    F.Tn[pos] = -120e6;

    F.str_init_x[pos] = F.Tn [pos] * vec_n [0]
                      + F.Ts1[pos] * vec_s1[0]
                      + F.Ts2[pos] * vec_s2[0];
    F.str_init_y[pos] = F.Tn [pos] * vec_n [1]
                      + F.Ts1[pos] * vec_s1[1]
                      + F.Ts2[pos] * vec_s2[1];
    F.str_init_z[pos] = F.Tn [pos] * vec_n [2]
                      + F.Ts1[pos] * vec_s1[2]
                      + F.Ts2[pos] * vec_s2[2];

    //if(j==nj/2 && k==nk/2) printf("str_init = %e %e %e\n",
    //    F.str_init_x[pos],
    //    F.str_init_y[pos],
    //    F.str_init_z[pos]);
#endif
#ifdef TPV103
    real_t a0 = 0.01;
    real_t b0 = 0.014;
    real_t da0 = 0.01;
    //real_t W = 15e3;
    real_t W = 15e3;
    real_t w = 3e3;
    //real_t RS_f0 = 0.6;
    real_t RS_V0 = 1e-6;
    //real_t RS_L = 0.02;
    real_t RS_Vini = 1e-16;
    real_t Tau_ini = -40e6;
    //real_t dTau0 = 45e6;
    real_t Tn_ini = -120e6;
    //real_t R = 3e3;

    //real_t y = (j-nj/2) * DH;
    //real_t z = (nk-1-k) * DH - 7.5e3;
    real_t y = (gj-NY/2) * DH;
    real_t z = (gk-NZ/2) * DH;
    real_t r = sqrt(y*y+z*z);
    F.a[pos] = 0.01 + 0.01*(1.0-Bfunc(y, W/1.0, w)*Bfunc(z, W/2.0, w));
    F.b[pos] = b0;
    F.Vw[pos] = 0.1 + 0.9*(1.0-Bfunc(y, W/1.0, w)*Bfunc(z, W/2.0, w));
    real_t RS_a = F.a[pos];
    //real_t RS_b = F.b[pos];
    //RS_a = 0.008;
    //RS_b = 0.012;

    //real_t tmp = RS_a*log(2.0*sinh(Tau_ini/RS_a/Tn_ini))-RS_f0-RS_a*log(RS_Vini/RS_V0);
    //F.State[pos] = RS_L/RS_V0 * exp( tmp/RS_b);
    F.State[pos] = RS_a * log(2.0*RS_V0/RS_Vini*sinh(Tau_ini/RS_a/Tn_ini));

    int stride = 2 * ny * nz;
    real_t *f_Vx = F.W + 0 * stride;
    real_t *f_Vy = F.W + 1 * stride;
    real_t *f_Vz = F.W + 2 * stride;

    //f_Vy[pos1 + 0*ny*nz] = +RS_Vini * 0.5;
    //f_Vy[pos1 + 1*ny*nz] = -RS_Vini * 0.5;

    F.Ts1[pos] = Tau_ini;
    F.Ts2[pos] = 0.0;
    F.Tn[pos] = -120e6;

    F.str_init_x[pos] = F.Tn [pos] * vec_n [0]
                      + F.Ts1[pos] * vec_s1[0]
                      + F.Ts2[pos] * vec_s2[0];
    F.str_init_y[pos] = F.Tn [pos] * vec_n [1]
                      + F.Ts1[pos] * vec_s1[1]
                      + F.Ts2[pos] * vec_s2[1];
    F.str_init_z[pos] = F.Tn [pos] * vec_n [2]
                      + F.Ts1[pos] * vec_s1[2]
                      + F.Ts2[pos] * vec_s2[2];

    //if(j==nj/2 && k==nk/2) printf("str_init = %e %e %e\n",
    //    F.str_init_x[pos],
    //    F.str_init_y[pos],
    //    F.str_init_z[pos]);
#endif
#ifdef TPV104
    real_t a0 = 0.01;
    real_t b0 = 0.014;
    real_t da0 = 0.01;
    //real_t W = 15e3;
    real_t W = 15e3;
    real_t w = 3e3;
    //real_t RS_f0 = 0.6;
    real_t RS_V0 = 1e-6;
    //real_t RS_L = 0.02;
    real_t RS_Vini = 1e-16;
    real_t Tau_ini = -40e6;
    //real_t dTau0 = 45e6;
    real_t Tn_ini = -120e6;
    //real_t R = 3e3;

    //real_t y = (j-nj/2) * DH;
    real_t z = (NZ-1-gk) * DH - 7.5e3;
    real_t y = (gj-NY/2) * DH;
    //real_t z = (k-nk/2) * DH;
    real_t r = sqrt(y*y+z*z);
    F.a[pos] = 0.01 + 0.01*(1.0-Bfunc(y, W/1.0, w)*Bfunc(z, W/2.0, w));
    F.b[pos] = b0;
    F.Vw[pos] = 0.1 + 0.9*(1.0-Bfunc(y, W/1.0, w)*Bfunc(z, W/2.0, w));
    real_t RS_a = F.a[pos];
    //real_t RS_b = F.b[pos];
    //RS_a = 0.008;
    //RS_b = 0.012;

    //real_t tmp = RS_a*log(2.0*sinh(Tau_ini/RS_a/Tn_ini))-RS_f0-RS_a*log(RS_Vini/RS_V0);
    //F.State[pos] = RS_L/RS_V0 * exp( tmp/RS_b);
    F.State[pos] = RS_a * log(2.0*RS_V0/RS_Vini*sinh(Tau_ini/RS_a/Tn_ini));

    int stride = 2 * ny * nz;
    real_t *f_Vx = F.W + 0 * stride;
    real_t *f_Vy = F.W + 1 * stride;
    real_t *f_Vz = F.W + 2 * stride;

    //f_Vy[pos1 + 0*ny*nz] = +RS_Vini * 0.5;
    //f_Vy[pos1 + 1*ny*nz] = -RS_Vini * 0.5;

    F.Ts1[pos] = Tau_ini;
    F.Ts2[pos] = 0.0;
    F.Tn[pos] = -120e6;

    F.str_init_x[pos] = F.Tn [pos] * vec_n [0]
                      + F.Ts1[pos] * vec_s1[0]
                      + F.Ts2[pos] * vec_s2[0];
    F.str_init_y[pos] = F.Tn [pos] * vec_n [1]
                      + F.Ts1[pos] * vec_s1[1]
                      + F.Ts2[pos] * vec_s2[1];
    F.str_init_z[pos] = F.Tn [pos] * vec_n [2]
                      + F.Ts1[pos] * vec_s1[2]
                      + F.Ts2[pos] * vec_s2[2];

    //if(j==nj/2 && k==nk/2) printf("str_init = %e %e %e\n",
    //    F.str_init_x[pos],
    //    F.str_init_y[pos],
    //    F.str_init_z[pos]);
#endif

    if( gj1 <= Faultgrid[0 + 4*nfault]+3 || gj1 >= Faultgrid[1 + 4*nfault]-3 ){
      
    // if( gj1 <= par.Fault_grid[0]+3 || gj1 >= par.Fault_grid[1]-3 ){
      F.rup_index_y[pos] = 0;
      // F.rup_index_y[pos] = 1;
    }else{
      F.rup_index_y[pos] = 1;
      // F.rup_index_y[pos] = 0;
    }

    if( gk1 <= Faultgrid[2 + 4*nfault]+3 || gk1 >= Faultgrid[3 + 4*nfault]-3 ){
    // if( gk1 <= par.Fault_grid[2]+3 || gk1 >= par.Fault_grid[3]-3 ){
      F.rup_index_z[pos] = 0;
      // F.rup_index_z[pos] = 1;
    }else{
      F.rup_index_z[pos] = 1;
      // F.rup_index_z[pos] = 0;
    }
#ifdef Barrier
    if(gj1 >= par.Barrier_grid[0] - 3 && gj1 <= par.Barrier_grid[1] + 3
       && gk1 >= par.Barrier_grid[2] - 3 && gk1 <= par.Barrier_grid[3] +3
       ){
      F.rup_index_z[pos] = 0;
      F.rup_index_y[pos] = 0;
    }
#endif
    // use long stencil
    //F.rup_index_y[pos] = 0;
    //F.rup_index_z[pos] = 0;

    if(
        gj1 > par.PML_N && gj1 < par.NY - par.PML_N &&
        gk1 > par.PML_N
#ifndef FreeSurface
        && gk1 < par.NZ - par.PML_N
#endif
        ){
      F.united[pos] = 0;
    }else{
      F.united[pos] = 1;
    }
    if(
        gj1 > 30 && gj1 < par.NY - 30 &&
        gk1 > 30
#ifndef FreeSurface
        && gk1 < par.NZ - 30
#endif
        ){
      F.united[pos] = 0;
    }else{
      F.united[pos] = 1;
    }

    // if( gj1 >= Faultgrid[0 + 4*nfault] -30 && gj1 <= Faultgrid[1 + 4*nfault] + 30 &&
    //     gk1 >= Faultgrid[2 + 4*nfault] -30 && gk1 <= Faultgrid[3 + 4*nfault] + 30){
    //     F.united[pos] = 0;
    //   }else{
    //     F.united[pos] = 1;
    //   } 
    if( gj1 >= Faultgrid[0 + 4*nfault]+3 && gj1 <= Faultgrid[1 + 4*nfault]-3 &&
        gk1 >= Faultgrid[2 + 4*nfault]+3 && gk1 <= Faultgrid[3 + 4*nfault]){
    // if( gj1 >= par.Fault_grid[0]+3 && gj1 <= par.Fault_grid[1]-3 &&
        // gk1 >= par.Fault_grid[2]+3 && gk1 <= par.Fault_grid[3]-3 ){
      F.faultgrid[pos] = 1;
    }else{
      F.faultgrid[pos] = 0;
    }
#ifdef Barrier
    if(gj1 >= par.Barrier_grid[0] - 3 && gj1 <= par.Barrier_grid[1] +3
       && gk1 >= par.Barrier_grid[2] - 3 && gk1 <= par.Barrier_grid[3] +3
       ){
      F.faultgrid[pos] = 0;
    }
#endif
    if( (gj1 >= Faultgrid[0 + 4*nfault] - 17 && gj1 <= Faultgrid[0 + 4*nfault] + 6) || // 17  6
        (gj1 >= Faultgrid[1 + 4*nfault] - 6 && gj1 <= Faultgrid[1 + 4*nfault] + 17) ||
        (gk1 >= Faultgrid[2 + 4*nfault] - 17 && gk1 <= Faultgrid[2 + 4*nfault] + 6)
#ifndef FreeSurface
        || (gk1 >= Faultgrid[3 + 4*nfault]- 6 && gk1 <= Faultgrid[2 + 4*nfault] + 17)
#endif 
    ){
      F.smooth_flag[pos] = 1;
    }
    // if( (gj1 >= Faultgrid[0 + 4*nfault] - 18 && gj1 <= Faultgrid[0 + 4*nfault] + 9) &&
    //     nfault == 1)    
    // {
    //   F.smooth_flag[pos] = 1;
    // }
//     if( (gj1 >= Faultgrid[0 + 4*nfault] && gj1 < Faultgrid[0 + 4*nfault] + 1) ||
//         (gj1 > Faultgrid[1 + 4*nfault] - 1 && gj1 <= Faultgrid[1 + 4*nfault]) ||
//         (gk1 >= Faultgrid[2 + 4*nfault] && gk1 < Faultgrid[2 + 4*nfault] + 1)
// #ifndef FreeSurface
//         || (gk1 > Faultgrid[3 + 4*nfault] - 1 && gk1 <= Faultgrid[2 + 4*nfault])
// #endif 
//     ){
    if( (gj1 == Faultgrid[0 + 4*nfault]) ||
        (gj1 == Faultgrid[1 + 4*nfault]) ||
        (gk1 == Faultgrid[2 + 4*nfault])
#ifndef FreeSurface
        || (gk1 == Faultgrid[2 + 4*nfault])
#endif 
    ){
      F.smooth_flag[pos] = 0;
    }
//#ifdef RS
    if (
        par.Friction_type == 1 ||
        par.Friction_type == 2 ||
        par.Friction_type == 3
        ){
      real_t a_b = F.a[pos]-F.b[pos];
      // if(a_b < 0){
      //   F.faultgrid[pos] = 1;
      // }else{
      //   F.faultgrid[pos] = 0;
      // }
      if( gj1 >= Faultgrid[0 + 4*nfault] - 30 && gj1 <= Faultgrid[1 + 4*nfault] + 30 &&
          gk1 >= Faultgrid[2 + 4*nfault] - 30 && gk1 <= Faultgrid[3 + 4*nfault] + 30){
        F.united[pos] = 0;
      }else{
        F.united[pos] = 1;
      } 
    }
//#endif

    F.init_t0[pos] = -9999.9;
  }
}

void init_fault(Fault F, int nfault, int Faultgrid[]){
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);
  init_fault_cu <<<grid, block>>> (F, nfault, Faultgrid);
}

__global__ void init_stress_to_local_cu(Fault F){
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int pos, pos1;
  int pos_v;
  int NY = par.NY;
  int NZ = par.NZ;

  int nj = par.NY/par.PY;
  int nk = par.NZ/par.PZ;

  int ny = nj + 6;
  int nz = nk + 6;
  int num_fault = par.num_fault;         //*********wangzj

  for (int nfault = 0; nfault < num_fault; nfault++){
    if( j < nj && k < nk ){

      pos = j+k*nj + nfault * (nj*nk);
      pos_v = (k1 * ny + j1) * 3 + nfault * (ny*nz*3);

      real_t en[3], em[3], el[3];
      en[0] = F.vec_n [pos_v + 0];
      en[1] = F.vec_n [pos_v + 1];
      en[2] = F.vec_n [pos_v + 2];
      em[0] = F.vec_s1[pos_v + 0];
      em[1] = F.vec_s1[pos_v + 1];
      em[2] = F.vec_s1[pos_v + 2];
      el[0] = F.vec_s2[pos_v + 0];
      el[1] = F.vec_s2[pos_v + 1];
      el[2] = F.vec_s2[pos_v + 2];

      F.Tn [pos] = F.T0x[pos] * en[0]
                 + F.T0y[pos] * en[1]
                 + F.T0z[pos] * en[2];
      F.Ts1[pos] = F.T0x[pos] * em[0]
                 + F.T0y[pos] * em[1]
                 + F.T0z[pos] * em[2];
      F.Ts2[pos] = F.T0x[pos] * el[0]
                 + F.T0y[pos] * el[1]
                 + F.T0z[pos] * el[2];
    }
  }

  return;
}

void nc_read_init_stress(Fault F){

  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int num_fault = hostParams.num_fault;         //*********wangzj

  real_t *var = (real_t*) malloc(sizeof(real_t)*nj*nk*num_fault);
  // thisid dimension 0, 1, 2, thisid[2] vary first

  int err;
  int ncid;
  int varid;
  // static size_t start[] = {thisid[2]*nk+0, thisid[1]*nj+0};
  size_t* count;
  size_t* start;
  if (num_fault == 1){
    count = (size_t *) malloc(sizeof(size_t)*2);
    start = (size_t *) malloc(sizeof(size_t)*2);
    count[0] = nk;
    count[1] = nj;
    start[0] = thisid[2]*nk+0;
    start[1] = thisid[1]*nj+0;
  }
  else{
    count = (size_t *) malloc(sizeof(size_t)*3);
    start = (size_t *) malloc(sizeof(size_t)*3);
    count[0] = num_fault;
    count[1] = nk;
    count[2] = nj;
    start[0] = 0;
    start[1] = thisid[2]*nk+0;
    start[2] = thisid[1]*nj+0;  
  }
  // static size_t count[] = {num_fault, nk, nj}; // y vary first

  err = nc_open(Fault_init_stress, NC_NOWRITE, &ncid);
  handle_err(err);

  size_t size = nj*nk*num_fault*sizeof(real_t);

  err = nc_inq_varid(ncid, "Tx", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.T0x, var, size, cudaMemcpyHostToDevice);

  err = nc_inq_varid(ncid, "Ty", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.T0y, var, size, cudaMemcpyHostToDevice);

  err = nc_inq_varid(ncid, "Tz", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.T0z, var, size, cudaMemcpyHostToDevice);

  err = nc_inq_varid(ncid, "dTx", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.dT0x, var, size, cudaMemcpyHostToDevice);

  err = nc_inq_varid(ncid, "dTy", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.dT0y, var, size, cudaMemcpyHostToDevice);

  err = nc_inq_varid(ncid, "dTz", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  cudaMemcpy(F.dT0z, var, size, cudaMemcpyHostToDevice);
  //err = nc_inq_varid(ncid, "Tn", &varid); handle_err(err);
  //err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  //cudaMemcpy(F.Tn, var, size, cudaMemcpyHostToDevice);

  //err = nc_inq_varid(ncid, "Tm", &varid); handle_err(err);
  //err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  //cudaMemcpy(F.Ts1, var, size, cudaMemcpyHostToDevice);

  //err = nc_inq_varid(ncid, "Tl", &varid); handle_err(err);
  //err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
  //cudaMemcpy(F.Ts2, var, size, cudaMemcpyHostToDevice);

  if(hostParams.Friction_type == 0){
    err = nc_inq_varid(ncid, "mu_s", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.str_peak, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "mu_d", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.mu_d, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "Dc", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.Dc, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "C0", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.C0, var, size, cudaMemcpyHostToDevice);
  }else if (
      hostParams.Friction_type == 1 ||
      hostParams.Friction_type == 2 ||
      hostParams.Friction_type == 3
      ){
//#ifdef RS
    err = nc_inq_varid(ncid, "a", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.a, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "b", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.b, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "L", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.L, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "Vw", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.Vw, var, size, cudaMemcpyHostToDevice);

    err = nc_inq_varid(ncid, "State", &varid); handle_err(err);
    err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
    cudaMemcpy(F.State, var, size, cudaMemcpyHostToDevice);
    if(hostParams.Friction_type == 3){
      err = nc_inq_varid(ncid, "TP_hy", &varid); handle_err(err);
      err = nc_get_vara_real_t(ncid, varid, start, count, var); handle_err(err);
      cudaMemcpy(F.TP_hy, var, size, cudaMemcpyHostToDevice);
    }
//#endif
  }

// #ifdef Barrier
//   int *intvar = (int*) malloc(sizeof(int)*nj*nk*num_fault);
//   size_t intsize = nj*nk*num_fault*sizeof(int);
//   err = nc_inq_varid(ncid, "faultgrid", &varid); handle_err(err);
//   err = nc_get_vara_int(ncid, varid, start, count, intvar); handle_err(err);
//   cudaMemcpy(F.faultgrid, intvar, intsize, cudaMemcpyHostToDevice);
//   free(intvar);
// #endif

  err = nc_close(ncid);
  handle_err(err);

  free(var);

  // transform init stress to local coordinate, not nessessary,
  // just to export correct data

  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);
  init_stress_to_local_cu <<<grid, block>>> (F);

  return;
}

