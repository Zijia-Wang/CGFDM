// set globals here (only once)
//__device__ __constant__ int Flags[8][3];
//__device__ __constant__ int NX, NY, NZ, NT, PX, PY, PZ;
//__device__ __constant__ int nx, ny, nz, ni, nj, nk;
//__device__ __constant__ real_t DH, DT;// TMAX;
//__device__ __constant__ int NT;

/* device functions */
//#include <math.h>
#include "common.h"

__device__ void matcopy3x3(real_t A[][3], real_t B[][3]){
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
        B[i][j] = A[i][j];
    }
  return ;
}

__device__ void matadd3x3(real_t A[][3], real_t B[][3], real_t C[][3]){
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
        C[i][j] = A[i][j] + B[i][j];
    }
  return ;
}

__device__ void matsub3x3(real_t A[][3], real_t B[][3], real_t C[][3]){
  int i, j;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
        C[i][j] = A[i][j] - B[i][j];
    }
  return ;
}

__device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]){
  int i, j, k;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
      C[i][j] = 0.0;
      for (k = 0; k < 3; k++)
        C[i][j] += A[i][k] * B[k][j];
    }
  return ;
}
__device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]){
  C[0] = A[0][0]*B[0] + A[0][1]*B[1] + A[0][2]*B[2];
  C[1] = A[1][0]*B[0] + A[1][1]*B[1] + A[1][2]*B[2];
  C[2] = A[2][0]*B[0] + A[2][1]*B[1] + A[2][2]*B[2];
  return ;
}

__device__ void invert3x3(real_t m[][3]){
  real_t inv[3][3];
  real_t det;
  int i, j;

  inv[0][0] = m[1][1]*m[2][2] - m[2][1]*m[1][2];
  inv[0][1] = m[2][1]*m[0][2] - m[0][1]*m[2][2];
  inv[0][2] = m[0][1]*m[1][2] - m[0][2]*m[1][1];
  inv[1][0] = m[1][2]*m[2][0] - m[1][0]*m[2][2];
  inv[1][1] = m[0][0]*m[2][2] - m[2][0]*m[0][2];
  inv[1][2] = m[1][0]*m[0][2] - m[0][0]*m[1][2];
  inv[2][0] = m[1][0]*m[2][1] - m[1][1]*m[2][0];
  inv[2][1] = m[2][0]*m[0][1] - m[0][0]*m[2][1];
  inv[2][2] = m[0][0]*m[1][1] - m[0][1]*m[1][0];

  det = inv[0][0] * m[0][0]
      + inv[0][1] * m[1][0]
      + inv[0][2] * m[2][0];

  det = 1.0f / det;

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      m[i][j] = inv[i][j] * det;

  return ;
}

__device__ real_t norm3(real_t *A){
  real_t x2 = A[0]*A[0];
  real_t y2 = A[1]*A[1];
  real_t z2 = A[2]*A[2];
  return sqrtf( x2+y2+z2 );
}

__device__ void normalize(real_t *a){
  real_t a0=1.0f/sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
  a[0] *= a0;
  a[1] *= a0;
  a[2] *= a0;
  return;
}

__device__ void cross_product(real_t *A, real_t *B, real_t *C){
  C[0] = A[1] * B[2] - A[2] * B[1];
  C[1] = A[2] * B[0] - A[0] * B[2];
  C[2] = A[0] * B[1] - A[1] * B[0];
  return;
}

__device__ real_t dot_product(real_t *A, real_t *B){
  int i;
  real_t result = 0.0;
  for (i = 0; i < 3; i++)
    result += A[i] * B[i];
  return result;
}

__device__ real_t Bfunc(const real_t x, const real_t W, const real_t w){
  real_t xa = fabs(x);
  if(xa <= W){
    return 1.0;
  }else if(xa < W+w){
    real_t f = w/(xa-W-w) + w/(xa-W);
    return (0.5 * (1.0 + tanh(f)));
  }else{
    return 0.0;
  }
}
__device__ real_t Fr_func(const real_t r, const real_t R){
  if(r<R){
    return exp(r*r/(r*r-R*R));
    //return 1.0;
  }else{
    return 0.0;
  }
}

__device__ real_t Gt_func(const real_t t, const real_t T){
  //return 1.0;
  if(t<T){
    real_t t1 = t-T;
    real_t t2 = t-2.0*T;
    return exp(t1*t1/(t*t2));
  }else{
    return 1.0;
  }
}
