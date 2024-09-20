#ifndef COMMON_H
#define COMMON_H

#define PI 3.141592653589793238463
#define FSIZE 9
#define WSIZE 9
#define MSIZE 13
#define CSIZE 3
#define MIN(a, b) (a < b) ? (a) : (b)
#define MAX(a, b) (a > b) ? (a) : (b)

//#define ENV_LOCAL_RANK "MV2_COMM_WORLD_LOCAL_RANK"
#define ENV_LOCAL_RANK "OMPI_COMM_WORLD_LOCAL_RANK"

#define CUDACHECK(call) {                                  \
  const cudaError_t error = call;                          \
  if (error != cudaSuccess) {                              \
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
    fprintf(stderr, "code: %d, reason: %s\n",              \
        error, cudaGetErrorString(error));                 \
  }                                                        \
}

#ifdef useNetCDF
#include "netcdf.h"
#define handle_err(e)                         \
{                                             \
  if (e != NC_NOERR) {                        \
    printf("nc error: %s\n", nc_strerror(e)); \
    exit(2);                                  \
  }                                           \
}
#endif

#ifdef DoublePrecision
typedef double real_t;
typedef double * realptr_t;
#define MPI_REAL_T MPI_DOUBLE
#define NC_REAL_T NC_DOUBLE
#else
typedef float real_t;
typedef float * realptr_t;
#define MPI_REAL_T MPI_FLOAT
#define NC_REAL_T NC_FLOAT
#endif

//#define RupThres 1.0e-3

typedef struct {
  int   NX, NY, NZ;
  int   PX, PY, PZ;
  int   ni, nj, nk;
  int   igpu;
  int   rankx, ranky, rankz;
  int   freenode;
  // int   faultnode;
  int   nx, ny, nz;
  int   NT;
  int   INPORT_GRID_TYPE;
  int   INPORT_STRESS_TYPE;
  int   Friction_type;
  int   EXPORT_TIME_SKIP;
  int   EXPORT_WAVE_SLICE_X;
  int   EXPORT_WAVE_SLICE_Y;
  int   EXPORT_WAVE_SLICE_Z;
  int   num_fault;                  // multi-fault    added by wangzj
  int* src_i;                       // multi-fault    added by wangzj
  real_t TMAX, DT, DH, rDH;
  int   PML_N;
  real_t PML_velocity;
  real_t PML_bmax;
  real_t PML_fc;
  int   DAMP_N;
  real_t mu_s, mu_d, mu_0, Dc, C0; // slip weakening
  real_t f0, V0, Vini, fw, L; // rate state
  real_t smooth_load_T;
  int* Fault_grid;           //wangzj
  // int   Fault_grid[4];
  int   Asp_grid[4];
  int   Barrier_grid[4];
  real_t vp1, vs1, rho1;
  real_t bi_vp1, bi_vs1, bi_rho1;
  real_t bi_vp2, bi_vs2, bi_rho2;
  real_t viscosity;
  real_t RupThres;
  int TP_n;
  int num_recv;
} Params;

typedef struct {
  realptr_t x, y, z;
} Coord;

typedef struct {
  realptr_t xix, xiy, xiz;
  realptr_t etx, ety, etz;
  realptr_t ztx, zty, ztz;
  realptr_t jac;
  realptr_t rho, lam, miu;
} Metric;

typedef struct {
  // W consists of Vx Vy Vz Txx Tyy Tzz Txy Txz Tyz
  // and each shape is Vx(ny, nz, nx), y vary first
  realptr_t W, mW, hW, tW;
  realptr_t matVx2Vz, matVy2Vz;
} Wave;

typedef struct {
  int n;
  realptr_t gx, gy, gz;
  realptr_t x, y, z;
  realptr_t i, j, k;
} Recv;

typedef struct {
  // Fault coefficients
  realptr_t lam_f; // bimaterial
  realptr_t mu_f;  // bimaterial
  realptr_t rho_f; // bimaterial
  realptr_t D11_1, D12_1, D13_1;
  realptr_t D21_1, D22_1, D23_1;
  realptr_t D31_1, D32_1, D33_1;
  realptr_t D11_2, D12_2, D13_2;
  realptr_t D21_2, D22_2, D23_2;
  realptr_t D31_2, D32_2, D33_2;
  realptr_t matPlus2Min1;
  realptr_t matPlus2Min2;
  realptr_t matPlus2Min3;
  realptr_t matPlus2Min4;
  realptr_t matPlus2Min5;
  realptr_t matMin2Plus1;
  realptr_t matMin2Plus2;
  realptr_t matMin2Plus3;
  realptr_t matMin2Plus4;
  realptr_t matMin2Plus5;
  realptr_t matT1toVxm;
  realptr_t matVytoVxm;
  realptr_t matVztoVxm;
  realptr_t matT1toVxp;
  realptr_t matVytoVxp;
  realptr_t matVztoVxp;
  realptr_t vec_n, vec_s1, vec_s2;
  // free surface
  realptr_t x_et, y_et, z_et;
  realptr_t x_zt, y_zt, z_zt;
  realptr_t matVx2Vz1, matVy2Vz1;
  realptr_t matVx2Vz2, matVy2Vz2;
  realptr_t matVx1_free, matVy1_free;
  realptr_t matVx2_free, matVy2_free;
  realptr_t matPlus2Min1f, matPlus2Min2f, matPlus2Min3f;
  realptr_t matMin2Plus1f, matMin2Plus2f, matMin2Plus3f;
  realptr_t matT1toVxfm;
  realptr_t matVytoVxfm;
  realptr_t matT1toVxfp;
  realptr_t matVytoVxfp;
  // Fault variables for updating at each time step
  // Split nodes: - side and + side
  // (2*ny*nz*fsize) Vx, Vy, Vz, T21, T21, T23, T31, T32, T33
  realptr_t W, mW, hW, tW;
  realptr_t Ws; // split nodes for smoothing
  realptr_t T11, T12, T13; // -1 0 1 continuous (3*ny*nz)
  realptr_t mT11, mT12, mT13; // -1 0 1 continuous (3*ny*nz)
  realptr_t hT11, hT12, hT13; // -1 0 1 continuous (3*ny*nz)
  // realptr_t tT11, tT12, tT13; // -1 0 1 continuous (3*ny*nz)
  realptr_t Tn, Ts1, Ts2; // ny*nz
  realptr_t tTn, tTs1, tTs2; // ny*nz
  realptr_t str_init_x, str_init_y, str_init_z; // ny*nz
  realptr_t T0x, T0y, T0z; // ny*nz
  realptr_t dT0x, dT0y, dT0z; // ny*nz
  realptr_t str_peak,mu_d,Dc,C0; // slip weakening friction law
  realptr_t a, b, L, Vw; // rate state friction law
  realptr_t hslip, slip, tslip, mslip; // ny*nz
  realptr_t slip1, slip2, rake; // ny*nz                           wangzj******
  realptr_t Vs1, Vs2; // 2*nj*nk
  realptr_t State, mState, hState, tState;
  realptr_t init_t0;
  realptr_t rup_sensor_Dy;
  realptr_t rup_sensor_Dz;
  realptr_t rup_sensor;
  realptr_t TP_T, TP_P, TP_hy;
  realptr_t TP_dT, TP_dP;
  realptr_t friction;
  int   *init_t0_flag;
  int   *smooth_flag;
  int   *rup_index_y, *rup_index_z;
  int   *united;
  int   *faultgrid;
  int   *flag_rup, *first_rup;
} Fault;

typedef struct {
  realptr_t Ax, Bx, Dx;
  realptr_t Ay, By, Dy;
  realptr_t Az, Bz, Dz;
  int isx1, isy1, isz1;
  int isx2, isy2, isz2;
  // 0~ND
  realptr_t Wx1, hWx1, mWx1, tWx1;
  realptr_t Wy1, hWy1, mWy1, tWy1;
  realptr_t Wz1, hWz1, mWz1, tWz1;
  // end-ND~end
  realptr_t Wx2, hWx2, mWx2, tWx2;
  realptr_t Wy2, hWy2, mWy2, tWy2;
  realptr_t Wz2, hWz2, mWz2, tWz2;
} PML;

typedef struct {
  int istart, jstart, kstart;
  int icount, jcount, kcount;
  int isx, isy, isz;
} Rectangle;

#endif
