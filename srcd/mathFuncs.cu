#include <math.h>
#include <stdlib.h>
#include "common.h"

void invert3x3_h(real_t m[][3]){
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

void matmul3x3_h(real_t A[][3], real_t B[][3], real_t C[][3]){
  int i, j, k;
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++){
      C[i][j] = 0.0;
      for (k = 0; k < 3; k++)
        C[i][j] += A[i][k] * B[k][j];
    }

  return ;
}

void cross_product_h(real_t *A, real_t *B, real_t *C){
  C[0] = A[1] * B[2] - A[2] * B[1];
  C[1] = A[2] * B[0] - A[0] * B[2];
  C[2] = A[0] * B[1] - A[1] * B[0];
  return;
}

real_t dot_product_h(real_t *A, real_t *B){
  int i;
  real_t result = 0.0;
  for (i = 0; i < 3; i++)
    result += A[i] * B[i];

  return result;
}

real_t norm_h(real_t *a){
  return sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

int normalize_h(real_t *a){
  real_t a0=1.0f/sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
  a[0] *= a0;
  a[1] *= a0;
  a[2] *= a0;
  return 0;
}

real_t dist_point2plane_h(real_t x0[3], real_t x1[3], real_t x2[3], real_t x3[3]){

  double x12[3], x13[3], p[3];
  x12[0] = x2[0] - x1[0];
  x12[1] = x2[1] - x1[1];
  x12[2] = x2[2] - x1[2];
  x13[0] = x3[0] - x1[0];
  x13[1] = x3[1] - x1[1];
  x13[2] = x3[2] - x1[2];

  //cross_product_h(x12, x13, p);
  p[0] = x12[1] * x13[2] - x12[2] * x13[1];
  p[1] = x12[2] * x13[0] - x12[0] * x13[2];
  p[2] = x12[0] * x13[1] - x12[1] * x13[0];
  //double d = dot_product_h(p, x1);
  double d = p[0]*x1[0]+p[1]*x1[1]+p[2]*x1[2];
  double px0 = p[0]*x0[0]+p[1]*x0[1]+p[2]*x0[2];

  double L = fabs( px0 - d);

  double pp = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
  L = L/sqrt(pp);

  return L;
}

