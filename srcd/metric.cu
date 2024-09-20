#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

extern void invert3x3_h(real_t m[][3]);
extern void matmul3x3_h(real_t A[][3], real_t B[][3], real_t C[][3]);
extern void cross_product_h(real_t *A, real_t *B, real_t *C);
extern real_t dot_product_h(real_t *A, real_t *B);
extern real_t norm_h(real_t *a);
extern int normalize_h(real_t *a);
extern real_t dist_point2plane_h(real_t x0[3], real_t x1[3], real_t x2[3], real_t x3[3]);

void cal_metric(real_t *C, real_t *M) {

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  real_t rDH = 1.0/hostParams.DH;

  real_t x_xi, x_et, x_zt;
  real_t y_xi, y_et, y_zt;
  real_t z_xi, z_et, z_zt;
  real_t jac;
  real_t vec1[3], vec2[3], vec3[3], vecg[3];

  //int slice = nx * ny;
  int stride = nx * ny * nz;
  real_t *X = C;
  real_t *Y = X + stride;
  real_t *Z = Y + stride;
  real_t *XIX = M;
  real_t *XIY = XIX + stride;
  real_t *XIZ = XIY + stride;
  real_t *ETX = XIZ + stride;
  real_t *ETY = ETX + stride;
  real_t *ETZ = ETY + stride;
  real_t *ZTX = ETZ + stride;
  real_t *ZTY = ZTX + stride;
  real_t *ZTZ = ZTY + stride;
  real_t *JAC = ZTZ + stride;
  real_t *LAM = JAC + stride;
  real_t *MIU = LAM + stride;
  real_t *RHO = MIU + stride;
  for (int i = 3; i < nx - 3; i++){
    for (int k = 3; k < nz - 3; k++){
      for (int j = 3; j < ny - 3; j++){

        //int pos_c = (i * ny * nz + j * nz + k) * CSIZE;
        //int pos_m = (i * ny * nz + j * nz + k) * MSIZE;
        //int slice = ny * nz * CSIZE; int segment = nz * CSIZE;
        //int pos = k * ny * nx + j * nx + i;
        int pos = j + k * ny + i * ny * nz;

        x_xi = 0.0; x_et = 0.0; x_zt = 0.0;
        y_xi = 0.0; y_et = 0.0; y_zt = 0.0;
        z_xi = 0.0; z_et = 0.0; z_zt = 0.0;

        int nyz = ny * nz;
        //x_xi = L22F(X, pos, nyz) * rDH;
        //y_xi = L22F(Y, pos, nyz) * rDH;
        //z_xi = L22F(Z, pos, nyz) * rDH;

        //x_et = L22F(X, pos, 1) * rDH;
        //y_et = L22F(Y, pos, 1) * rDH;
        //z_et = L22F(Z, pos, 1) * rDH;

        //x_zt = L22F(X, pos, ny) * rDH;
        //y_zt = L22F(Y, pos, ny) * rDH;
        //z_zt = L22F(Z, pos, ny) * rDH;

        x_xi = 0.5 * rDH *( L22F(X, pos, nyz) + L22B(X, pos, nyz));
        y_xi = 0.5 * rDH *( L22F(Y, pos, nyz) + L22B(Y, pos, nyz));
        z_xi = 0.5 * rDH *( L22F(Z, pos, nyz) + L22B(Z, pos, nyz));

        x_et = 0.5 * rDH *( L22F(X, pos, 1  ) + L22B(X, pos, 1  ));
        y_et = 0.5 * rDH *( L22F(Y, pos, 1  ) + L22B(Y, pos, 1  ));
        z_et = 0.5 * rDH *( L22F(Z, pos, 1  ) + L22B(Z, pos, 1  ));

        x_zt = 0.5 * rDH *( L22F(X, pos, ny ) + L22B(X, pos, ny ));
        y_zt = 0.5 * rDH *( L22F(Y, pos, ny ) + L22B(Y, pos, ny ));
        z_zt = 0.5 * rDH *( L22F(Z, pos, ny ) + L22B(Z, pos, ny ));

        vec1[0] = x_xi; vec1[1] = y_xi; vec1[2] = z_xi;
        vec2[0] = x_et; vec2[1] = y_et; vec2[2] = z_et;
        vec3[0] = x_zt; vec3[1] = y_zt; vec3[2] = z_zt;

        cross_product_h(vec1, vec2, vecg);
        jac = dot_product_h(vecg, vec3);
        JAC[pos]  = jac;

        real_t rjac = 1.0/jac;

        cross_product_h(vec2, vec3, vecg);
        XIX[pos] = vecg[0] * rjac;
        XIY[pos] = vecg[1] * rjac;
        XIZ[pos] = vecg[2] * rjac;

        cross_product_h(vec3, vec1, vecg);
        ETX[pos] = vecg[0] * rjac;
        ETY[pos] = vecg[1] * rjac;
        ETZ[pos] = vecg[2] * rjac;

        cross_product_h(vec1, vec2, vecg);
        ZTX[pos] = vecg[0] * rjac;
        ZTY[pos] = vecg[1] * rjac;
        ZTZ[pos] = vecg[2] * rjac;


        //JAC[pos] = 1.0;
        //XIX[pos] = 1.0;
        //XIY[pos] = 0.0;
        //XIZ[pos] = 0.0;
        //ETX[pos] = 0.0;
        //ETY[pos] = 1.0;
        //ETZ[pos] = 0.0;
        //ZTX[pos] = 0.0;
        //ZTY[pos] = 0.0;
        //ZTZ[pos] = 1.0;

        real_t rho1 = hostParams.bi_rho1;
        real_t vs1  = hostParams.bi_vs1;
        real_t vp1  = hostParams.bi_vp1;

        real_t rho2 = hostParams.bi_rho2;
        real_t vs2  = hostParams.bi_vs2;
        real_t vp2  = hostParams.bi_vp2;

        real_t lam1 = rho1*(vp1*vp1-2*vs1*vs1);
        real_t miu1 = rho1*vs1*vs1;

        real_t lam2 = rho2*(vp2*vp2-2*vs2*vs2);
        real_t miu2 = rho2*vs2*vs2;

        //LAM[pos] = rho1*(vp1*vp1-2*vs1*vs1);
        //MIU[pos] = rho1*vs1*vs1; // miu
        //RHO[pos] = rho1; // rho
        int gi = hostParams.ni * hostParams.rankx + i;
        int i0 = hostParams.NX/2;
        if (gi < i0){
          LAM[pos] = lam1;
          MIU[pos] = miu1;
          RHO[pos] = rho1;
        }else{
          LAM[pos] = lam2;
          MIU[pos] = miu2;
          RHO[pos] = rho2;
        }


        // bimaterial


/*
        if (i == NX/2){
        printf("ijk = %03d %03d %03d metric = %f %f %f %f %f %f %f %f %f\n", 
            i, j, k,
            M[pos_m + 0], M[pos_m + 1], M[pos_m + 2],
            M[pos_m + 3], M[pos_m + 4], M[pos_m + 5],
            M[pos_m + 6], M[pos_m + 7], M[pos_m + 8]);
            //x_xi, y_xi, z_xi, x_et, y_et, z_et, x_zt, y_zt, z_zt);
            //
        }
 */
      }
    }
  }
  return;
}

void extend_Symm_array(real_t *W, int SIZE) {
  int i, j, k, n, l, pos_s_1, pos_d_1, pos_s_2, pos_d_2;
  //int i1 = ni1; int i2 = ni2 - 1;
  //int j1 = nj1; int j2 = nj2 - 1;
  //int k1 = nk1; int k2 = nk2 - 1;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int i1 = 3; int i2 = nx - 4;
  int j1 = 3; int j2 = ny - 4;
  int k1 = 3; int k2 = nz - 4;
  int stride = nx * ny * nz;
  //int nx = NX; int ny = NY; int nz = NZ;
  // extend x direction
  for (k = 0; k < nz; k++){
    for (j = 0; j < ny; j++){
      for (n = 1; n <= 3; n++){
        //pos_s_1 = ((i1-n) * ny * nz + j * nz + k) * SIZE;
        //pos_d_1 = ((i1+n) * ny * nz + j * nz + k) * SIZE;
        //pos_s_2 = ((i2+n) * ny * nz + j * nz + k) * SIZE;
        //pos_d_2 = ((i2-n) * ny * nz + j * nz + k) * SIZE;
        pos_s_1 = j + k * ny + (i1-n) * ny * nz;
        pos_d_1 = j + k * ny + (i1+n) * ny * nz;
        pos_s_2 = j + k * ny + (i2+n) * ny * nz;
        pos_d_2 = j + k * ny + (i2-n) * ny * nz;
        for (l = 0; l < SIZE; l++) {
          W[pos_s_1 + l*stride] = W[pos_d_1 + l*stride];
          W[pos_s_2 + l*stride] = W[pos_d_2 + l*stride];
        }
      }
    }
  }
  // extend y direction
  for (k = 0; k < nz; k++){
    for (i = 0; i < nx; i++){
      for (n = 1; n <= 3; n++){
        //pos_s_1 = k * ny * nx + (j1-n) * nx + i;
        //pos_d_1 = k * ny * nx + (j1+n) * nx + i;
        //pos_s_2 = k * ny * nx + (j2+n) * nx + i;
        //pos_d_2 = k * ny * nx + (j2-n) * nx + i;
        pos_s_1 = (j1-n) + k * ny + i * ny * nz;
        pos_d_1 = (j1+n) + k * ny + i * ny * nz;
        pos_s_2 = (j2+n) + k * ny + i * ny * nz;
        pos_d_2 = (j2-n) + k * ny + i * ny * nz;
        for (l = 0; l < SIZE; l++){
          W[pos_s_1 + l*stride] = W[pos_d_1 + l*stride];
          W[pos_s_2 + l*stride] = W[pos_d_2 + l*stride];
        }
      }
    }
  }
  // extend z direction
  for (j = 0; j < ny; j++){
    for (i = 0; i < nx; i++){
      for (n = 1; n <= 3; n++){
        //pos_s_1 = (k1-n) * ny * nx + j * nx + i;
        //pos_d_1 = (k1+n) * ny * nx + j * nx + i;
        //pos_s_2 = (k2+n) * ny * nx + j * nx + i;
        //pos_d_2 = (k2-n) * ny * nx + j * nx + i;
        pos_s_1 = j + (k1-n) * ny + i * ny * nz;
        pos_d_1 = j + (k1+n) * ny + i * ny * nz;
        pos_s_2 = j + (k2+n) * ny + i * ny * nz;
        pos_d_2 = j + (k2-n) * ny + i * ny * nz;
        for (l = 0; l < SIZE; l++){
          W[pos_s_1 + l*stride] = W[pos_d_1 + l*stride];
          W[pos_s_2 + l*stride] = W[pos_d_2 + l*stride];
        }
      }
    }
  }
  return;
}

void extend_crew_array(real_t *W, int SIZE) {
  //return;
  int i, j, k, n, l;
  long pos_s_1, pos_d_1, pos_s_2, pos_d_2;
  long pos_c_1, pos_c_2;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;
  real_t DH = hostParams.DH;

  int i1 = 3; int i2 = nx - 4;
  int j1 = 3; int j2 = ny - 4;
  int k1 = 3; int k2 = nz - 4;
  long stride = nx * ny * nz;
  for (l = 0; l < SIZE; l++) {
    // extend x direction
    for (n = 1; n <= 3; n++){
      for (k = 0; k < nz; k++){
        for (j = 0; j < ny; j++){
          pos_c_1 = j + k * ny +  i1    * ny * nz + l*stride;
          pos_s_1 = j + k * ny + (i1-n) * ny * nz + l*stride;
          pos_d_1 = j + k * ny + (i1+n) * ny * nz + l*stride;
          pos_c_2 = j + k * ny +  i2    * ny * nz + l*stride;
          pos_s_2 = j + k * ny + (i2+n) * ny * nz + l*stride;
          pos_d_2 = j + k * ny + (i2-n) * ny * nz + l*stride;
          W[pos_s_1] = 2.0*W[pos_c_1] - W[pos_d_1];
          W[pos_s_2] = 2.0*W[pos_c_2] - W[pos_d_2];
          //W[pos_s_1] = W[pos_c_1] - n*DH;
          //W[pos_s_2] = W[pos_c_2] + n*DH;
        }
      }
    }
    // extend y direction
    for (i = 0; i < nx; i++){
      for (k = 0; k < nz; k++){
        for (n = 1; n <= 3; n++){
          pos_c_1 =  j1    + k * ny + i * ny * nz + l*stride;
          pos_s_1 = (j1-n) + k * ny + i * ny * nz + l*stride;
          pos_d_1 = (j1+n) + k * ny + i * ny * nz + l*stride;
          pos_c_2 =  j2    + k * ny + i * ny * nz + l*stride;
          pos_s_2 = (j2+n) + k * ny + i * ny * nz + l*stride;
          pos_d_2 = (j2-n) + k * ny + i * ny * nz + l*stride;
          W[pos_s_1] = 2.0*W[pos_c_1] - W[pos_d_1];
          W[pos_s_2] = 2.0*W[pos_c_2] - W[pos_d_2];
        }
      }
    }
    // extend z direction
    for (i = 0; i < nx; i++){
      for (n = 1; n <= 3; n++){
        for (j = 0; j < ny; j++){
          pos_c_1 = j +  k1    * ny + i * ny * nz + l*stride;
          pos_s_1 = j + (k1-n) * ny + i * ny * nz + l*stride;
          pos_d_1 = j + (k1+n) * ny + i * ny * nz + l*stride;
          pos_c_2 = j +  k2    * ny + i * ny * nz + l*stride;
          pos_s_2 = j + (k2+n) * ny + i * ny * nz + l*stride;
          pos_d_2 = j + (k2-n) * ny + i * ny * nz + l*stride;
          W[pos_s_1] = 2.0*W[pos_c_1] - W[pos_d_1];
          W[pos_s_2] = 2.0*W[pos_c_2] - W[pos_d_2];
        }
      }
    }
  }
  return;
}

__global__ void check_metric(real_t *m, real_t *m_s, const int i, const int SIZE) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  int k = blockDim.y * blockIdx.y + threadIdx.y;
  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  int stride = nx * ny * nz;

  int pos, pos1;
  if (j < ny && k < nz) {
    //pos = k * ny * nx + j * nx + i;
    //pos1 = k * ny + j;
    pos = j + k * ny + i * ny * nz;
    pos1 = j + k * ny;
    for (int l = 0; l < SIZE; ++l) {
        m_s[l*ny*nz + pos1] = m[pos + l*stride];
    }
  }
  return;
}
