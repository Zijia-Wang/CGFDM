#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"

__device__ void print_mat3x3(real_t A[][3]){
  int i, j;
  printf("------------------------\n");
  for (i = 0; i < 3; i++){
    for (j = 0; j < 3; j++)
      printf("%10.2e", A[i][j]);
    printf("\n");
  }
  return;
}

extern __device__ void matcopy3x3(real_t A[][3], real_t B[][3]);
extern __device__ void matadd3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void matsub3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void invert3x3(real_t m[][3]);
extern __device__ real_t norm3(real_t *A);
extern __device__ void cross_product(real_t *A, real_t *B, real_t *C);
//extern __device__ real_t dot_product(real_t *A, real_t *B);

__global__ void init_fault_coef_cu(real_t *M, Fault FC, int i0, int nfault)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;
  //real_t DH = par.DH;
  //real_t DT = par.DT;

  // int i0 = nx/2;      wangzj
  int pos, pos_f;

  real_t e11, e12, e13, e21, e22, e23, e31, e32, e33;
  real_t g11, g12, g13, g21, g22, g23, g31, g32, g33;
  real_t lam, miu, lam2mu, jac;

  real_t D11_1[3][3], D12_1[3][3], D13_1[3][3];
  real_t D21_1[3][3], D22_1[3][3], D23_1[3][3];
  real_t D31_1[3][3], D32_1[3][3], D33_1[3][3];

  real_t D11_2[3][3], D12_2[3][3], D13_2[3][3];
  real_t D21_2[3][3], D22_2[3][3], D23_2[3][3];
  real_t D31_2[3][3], D32_2[3][3], D33_2[3][3];

  real_t K1_1[3][3], K2_1[3][3], K3_1[3][3];
  real_t K1_2[3][3], K2_2[3][3], K3_2[3][3];

  real_t mat[3][3];
  real_t D[3][3];
  real_t norm;

  real_t Q[3][3], A1[3][3], A2[3][3], A3[3][3];

  real_t B0[3][3];

  real_t B1_1[3][3];
  real_t B2_1[3][3];
  real_t B3_1[3][3];

  real_t B1_2[3][3];
  real_t B2_2[3][3];
  real_t B3_2[3][3];

  real_t x_xi,y_xi,z_xi;
  real_t x_et,y_et,z_et;
  real_t x_zt,y_zt,z_zt;

  real_t vec_n[3];
  real_t vec_s1[3];
  real_t vec_s2[3];
  real_t n1[3], n2[3], n3[3];
  real_t nn1[3], nn2[3], nn3[3];

  /* for free surface */
  real_t A[3][3], B[3][3], C[3][3];
  real_t matVx2Vz1[3][3], matVy2Vz1[3][3];
  real_t matVx2Vz2[3][3], matVy2Vz2[3][3];
  real_t matVx1_free[3][3], matVy1_free[3][3];
  real_t matVx2_free[3][3], matVy2_free[3][3];
  real_t matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  real_t matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  real_t g1_2, g2_2, g3_2;
  real_t h1_3, h2_3, h3_3;
  real_t vec1[3], vec2[3], vec3[3];
  real_t D11_12[3][3], D12_12[3][3], D13_12[3][3];
  real_t D11_13[3][3], D12_13[3][3], D13_13[3][3];
  real_t D11_1f[3][3], D12_1f[3][3], D13_1f[3][3];
  real_t D11_2f[3][3], D12_2f[3][3], D13_2f[3][3];

  real_t vec1_norm, Tovert[3][3];
  real_t mat1[3][3], mat2[3][3], mat3[3][3], mat4[3][3];

  int stride = nx*ny*nz;
  int nyz = ny * nz;
  int Dsize = nfault * nyz * 3*3;
  int MediaSize = nfault * nyz *2;
  int matsize1 = nfault * ny * 3*3;

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
  

  if (j < ny && k < nz)
  {
    //pos   = (i0*ny*nz + j*nz + k)*MSIZE;
    //pos   = k * ny * nx + j * nx + i0;
    pos   = j + k * ny + i0 * ny * nz;
    /* ny * nz * 9 */
    //pos_f = (j*nz + k)*3*3;
    //pos_f = (k * ny + j)*3*3;
    pos_f = (j + k * ny)*3*3 + Dsize;  //******wangzj

    e11 = XIX[pos];//M[pos + 0];
    e12 = XIY[pos];//M[pos + 1];
    e13 = XIZ[pos];//M[pos + 2];
    e21 = ETX[pos];//M[pos + 3];
    e22 = ETY[pos];//M[pos + 4];
    e23 = ETZ[pos];//M[pos + 5];
    e31 = ZTX[pos];//M[pos + 6];
    e32 = ZTY[pos];//M[pos + 7];
    e33 = ZTZ[pos];//M[pos + 8];
    jac = JAC[pos];//M[pos + 9];

    mat[0][0] = XIX[pos];//M[pos + 0];
    mat[0][1] = XIY[pos];//M[pos + 1];
    mat[0][2] = XIZ[pos];//M[pos + 2];
    mat[1][0] = ETX[pos];//M[pos + 3];
    mat[1][1] = ETY[pos];//M[pos + 4];
    mat[1][2] = ETZ[pos];//M[pos + 5];
    mat[2][0] = ZTX[pos];//M[pos + 6];
    mat[2][1] = ZTY[pos];//M[pos + 7];
    mat[2][2] = ZTZ[pos];//M[pos + 8];

    invert3x3(mat);

    x_xi = mat[0][0];
    y_xi = mat[1][0];
    z_xi = mat[2][0];
    x_et = mat[0][1];
    y_et = mat[1][1];
    z_et = mat[2][1];
    x_zt = mat[0][2];
    y_zt = mat[1][2];
    z_zt = mat[2][2];

    A1[0][0] = -1.0*( y_xi);
    A1[0][1] = -1.0*(-x_xi);
    A1[0][2] = -1.0*( 0.0 );
    A1[1][0] = -1.0*( z_xi);
    A1[1][1] = -1.0*( 0.0 );
    A1[1][2] = -1.0*(-x_xi);
    A1[2][0] = -1.0*( 0.0 );
    A1[2][1] = -1.0*( z_xi);
    A1[2][2] = -1.0*(-y_xi);

    A2[0][0] =  y_et;
    A2[0][1] = -x_et;
    A2[0][2] =  0.0;
    A2[1][0] =  z_et;
    A2[1][1] =  0.0;
    A2[1][2] = -x_et;
    A2[2][0] =  0.0;
    A2[2][1] =  z_et;
    A2[2][2] = -y_et;

    A3[0][0] =  y_zt;
    A3[0][1] = -x_zt;
    A3[0][2] =  0.0;
    A3[1][0] =  z_zt;
    A3[1][1] =  0.0;
    A3[1][2] = -x_zt;
    A3[2][0] =  0.0;
    A3[2][1] =  z_zt;
    A3[2][2] = -y_zt;

    Q[0][0] = e11;
    Q[0][1] = e12;
    Q[0][2] = e13;
    Q[1][0] = e11;
    Q[1][1] = e12;
    Q[1][2] = e13;
    Q[2][0] = e11;
    Q[2][1] = e12;
    Q[2][2] = e13;

   // pos = j + k * ny + i0 * ny * nz;

    //pos = j*nz + k;
    pos = j + k * ny  + nfault * (nyz);    //********wangzj
    FC.x_et[pos] = x_et;
    FC.y_et[pos] = y_et;
    FC.z_et[pos] = z_et;

    vec_s1[0] = x_et;//mat[0][1];
    vec_s1[1] = y_et;//mat[1][1];
    vec_s1[2] = z_et;//mat[2][1];

    norm = norm3(vec_s1);

#pragma unroll 3
    for (int ia = 0; ia < 3; ++ia)
    {
        vec_s1[ia] /= norm;
    }

    /* metric_xi_x: +0 */
    /* metric_xi_y: +1 */
    /* metric_xi_z: +2 */
    //pos   = (i0*ny*nz + j*nz + k)*MSIZE;
    //pos = k * ny * nx + j * nx + i0; // bug, missed index change
    pos = j + k * ny + i0 * ny * nz;
    vec_n[0] = XIX[pos];//M[pos + 0];
    vec_n[1] = XIY[pos];//M[pos + 1];
    vec_n[2] = XIZ[pos];//M[pos + 2];
    norm = norm3(vec_n);

    for (int ia = 0; ia < 3; ++ia)
    {
        vec_n[ia] /= norm;
    }

    cross_product(vec_n, vec_s1, vec_s2);

    //pos = (j*nz + k)*3;
    pos = (j + k * ny)*3 +  + nfault * (nyz*3);    //***********wangzj

    for (int ia = 0; ia < 3; ++ia)
    {
        FC.vec_n [pos+ia] = vec_n [ia];
        FC.vec_s1[pos+ia] = vec_s1[ia];
        FC.vec_s2[pos+ia] = vec_s2[ia];

        n1[ia] = vec_n [ia];
        n2[ia] = vec_s1[ia];
        n3[ia] = vec_s2[ia];
    }

    g11 = n1[0]*e11+n1[1]*e12+n1[2]*e13;
    g12 = n1[0]*e21+n1[1]*e22+n1[2]*e23;
    g13 = n1[0]*e31+n1[1]*e32+n1[2]*e33;

    g21 = n2[0]*e11+n2[1]*e12+n2[2]*e13;
    g22 = n2[0]*e21+n2[1]*e22+n2[2]*e23;
    g23 = n2[0]*e31+n2[1]*e32+n2[2]*e33;

    g31 = n3[0]*e11+n3[1]*e12+n3[2]*e13;
    g32 = n3[0]*e21+n3[1]*e22+n3[2]*e23;
    g33 = n3[0]*e31+n3[1]*e32+n3[2]*e33;
    //real_t vp1  = par.bi_vp1;
    //real_t vs1  = par.bi_vs1;
    //real_t rho1 = par.bi_rho1;
    //real_t vp2  = par.bi_vp2;
    //real_t vs2  = par.bi_vs2;
    //real_t rho2 = par.bi_rho2;
    //real_t lam1 = rho1*(vp1*vp1-2.0*vs1*vs1);
    //real_t lam2 = rho2*(vp2*vp2-2.0*vs2*vs2);
    //real_t miu1 = rho1*(vs1*vs1);
    //real_t miu2 = rho2*(vs2*vs2);

    //FC.rho_f[j + k * ny + 0 * ny * nz] = rho1;
    //FC.rho_f[j + k * ny + 1 * ny * nz] = rho2;

    //FC.lam_f[j + k * ny + 0 * ny * nz] = lam1;
    //FC.lam_f[j + k * ny + 1 * ny * nz] = lam2;

    //FC.mu_f [j + k * ny + 0 * ny * nz] = miu1;
    //FC.mu_f [j + k * ny + 1 * ny * nz] = miu2;

    pos = j + k * ny + i0 * ny * nz;

    FC.rho_f[j + k * ny + 0 * ny * nz + MediaSize] = RHO[pos-ny*nz];
    FC.rho_f[j + k * ny + 1 * ny * nz + MediaSize] = RHO[pos+ny*nz];

    FC.lam_f[j + k * ny + 0 * ny * nz + MediaSize] = LAM[pos-ny*nz];
    FC.lam_f[j + k * ny + 1 * ny * nz + MediaSize] = LAM[pos+ny*nz];

    FC.mu_f [j + k * ny + 0 * ny * nz + MediaSize] = MIU[pos-ny*nz];
    FC.mu_f [j + k * ny + 1 * ny * nz + MediaSize] = MIU[pos+ny*nz];

    /* minus */
    /* ! need to be modified as Fault's lam and mu */
    //lam = M[pos + 10];
    //miu = M[pos + 11];
    //lam = LAM[pos];
    //miu = MIU[pos];
    lam = FC.lam_f[j + k * ny + 0 * ny * nz + MediaSize];
    miu = FC.mu_f [j + k * ny + 0 * ny * nz + MediaSize];

    lam2mu = lam + 2.f*miu;
    nn1[0] = n1[0]*n1[0]*n1[0]
           + n1[1]*n1[1]*n1[0]
           + n1[2]*n1[2]*n1[0];
    nn1[1] = n1[0]*n1[0]*n1[1]
           + n1[1]*n1[1]*n1[1]
           + n1[2]*n1[2]*n1[1];
    nn1[2] = n1[0]*n1[0]*n1[2]
           + n1[1]*n1[1]*n1[2]
           + n1[2]*n1[2]*n1[2];

    nn2[0] = n2[0]*n2[0]*n2[0]
           + n2[1]*n2[1]*n2[0]
           + n2[2]*n2[2]*n2[0];
    nn2[1] = n2[0]*n2[0]*n2[1]
           + n2[1]*n2[1]*n2[1]
           + n2[2]*n2[2]*n2[1];
    nn2[2] = n2[0]*n2[0]*n2[2]
           + n2[1]*n2[1]*n2[2]
           + n2[2]*n2[2]*n2[2];

    nn3[0] = n3[0]*n3[0]*n3[0]
           + n3[1]*n3[1]*n3[0]
           + n3[2]*n3[2]*n3[0];
    nn3[1] = n3[0]*n3[0]*n3[1]
           + n3[1]*n3[1]*n3[1]
           + n3[2]*n3[2]*n3[1];
    nn3[2] = n3[0]*n3[0]*n3[2]
           + n3[1]*n3[1]*n3[2]
           + n3[2]*n3[2]*n3[2];

    K1_1[0][0]=lam2mu*g11*nn1[0]+lam*g21*nn2[0]+lam*g31*nn3[0];
    K1_1[0][1]=lam2mu*g11*nn1[1]+lam*g21*nn2[1]+lam*g31*nn3[1];
    K1_1[0][2]=lam2mu*g11*nn1[2]+lam*g21*nn2[2]+lam*g31*nn3[2];
    K1_1[1][0]=K1_1[0][0];
    K1_1[1][1]=K1_1[0][1];
    K1_1[1][2]=K1_1[0][2];
    K1_1[2][0]=K1_1[0][0];
    K1_1[2][1]=K1_1[0][1];
    K1_1[2][2]=K1_1[0][2];

    K2_1[0][0]=lam2mu*g12*nn1[0]+lam*g22*nn2[0]+lam*g32*nn3[0];
    K2_1[0][1]=lam2mu*g12*nn1[1]+lam*g22*nn2[1]+lam*g32*nn3[1];
    K2_1[0][2]=lam2mu*g12*nn1[2]+lam*g22*nn2[2]+lam*g32*nn3[2];
    K2_1[1][0]=K2_1[0][0];
    K2_1[1][1]=K2_1[0][1];
    K2_1[1][2]=K2_1[0][2];
    K2_1[2][0]=K2_1[0][0];
    K2_1[2][1]=K2_1[0][1];
    K2_1[2][2]=K2_1[0][2];

    K3_1[0][0]=lam2mu*g13*nn1[0]+lam*g23*nn2[0]+lam*g33*nn3[0];
    K3_1[0][1]=lam2mu*g13*nn1[1]+lam*g23*nn2[1]+lam*g33*nn3[1];
    K3_1[0][2]=lam2mu*g13*nn1[2]+lam*g23*nn2[2]+lam*g33*nn3[2];
    K3_1[1][0]=K3_1[0][0];
    K3_1[1][1]=K3_1[0][1];
    K3_1[1][2]=K3_1[0][2];
    K3_1[2][0]=K3_1[0][0];
    K3_1[2][1]=K3_1[0][1];
    K3_1[2][2]=K3_1[0][2];

    D11_1[0][0]=lam2mu*e11*e11+miu*(e12*e12+e13*e13);
    D11_1[0][1]=lam*e11*e12+miu*e12*e11;
    D11_1[0][2]=lam*e11*e13+miu*e13*e11;
    D11_1[1][0]=miu*e11*e12+lam*e12*e11;
    D11_1[1][1]=lam2mu*e12*e12+miu*(e11*e11+e13*e13);
    D11_1[1][2]=lam*e12*e13+miu*e13*e12;
    D11_1[2][0]=miu*e11*e13+lam*e13*e11;
    D11_1[2][1]=miu*e12*e13+lam*e13*e12;
    D11_1[2][2]=lam2mu*e13*e13+miu*(e11*e11+e12*e12);

    D12_1[0][0]=lam2mu*e11*e21+miu*(e12*e22+e13*e23);
    D12_1[0][1]=lam*e11*e22+miu*e12*e21;
    D12_1[0][2]=lam*e11*e23+miu*e13*e21;
    D12_1[1][0]=miu*e11*e22+lam*e12*e21;
    D12_1[1][1]=lam2mu*e12*e22+miu*(e11*e21+e13*e23);
    D12_1[1][2]=lam*e12*e23+miu*e13*e22;
    D12_1[2][0]=miu*e11*e23+lam*e13*e21;
    D12_1[2][1]=miu*e12*e23+lam*e13*e22;
    D12_1[2][2]=lam2mu*e13*e23+miu*(e11*e21+e12*e22);

    D13_1[0][0]=lam2mu*e11*e31+miu*(e12*e32+e13*e33);
    D13_1[0][1]=lam*e11*e32+miu*e12*e31;
    D13_1[0][2]=lam*e11*e33+miu*e13*e31;
    D13_1[1][0]=miu*e11*e32+lam*e12*e31;
    D13_1[1][1]=lam2mu*e12*e32+miu*(e11*e31+e13*e33);
    D13_1[1][2]=lam*e12*e33+miu*e13*e32;
    D13_1[2][0]=miu*e11*e33+lam*e13*e31;
    D13_1[2][1]=miu*e12*e33+lam*e13*e32;
    D13_1[2][2]=lam2mu*e13*e33+miu*(e11*e31+e12*e32);

    FC.D21_1[pos_f+3*0+0]=lam2mu*e21*e11+miu*(e22*e12+e23*e13);
    FC.D21_1[pos_f+3*0+1]=lam*e21*e12+miu*e22*e11;
    FC.D21_1[pos_f+3*0+2]=lam*e21*e13+miu*e23*e11;
    FC.D21_1[pos_f+3*1+0]=miu*e21*e12+lam*e22*e11;
    FC.D21_1[pos_f+3*1+1]=lam2mu*e22*e12+miu*(e21*e11+e23*e13);
    FC.D21_1[pos_f+3*1+2]=lam*e22*e13+miu*e23*e12;
    FC.D21_1[pos_f+3*2+0]=miu*e21*e13+lam*e23*e11;
    FC.D21_1[pos_f+3*2+1]=miu*e22*e13+lam*e23*e12;
    FC.D21_1[pos_f+3*2+2]=lam2mu*e23*e13+miu*(e21*e11+e22*e12);

    FC.D22_1[pos_f+3*0+0]=lam2mu*e21*e21+miu*(e22*e22+e23*e23);
    FC.D22_1[pos_f+3*0+1]=lam*e21*e22+miu*e22*e21;
    FC.D22_1[pos_f+3*0+2]=lam*e21*e23+miu*e23*e21;
    FC.D22_1[pos_f+3*1+0]=miu*e21*e22+lam*e22*e21;
    FC.D22_1[pos_f+3*1+1]=lam2mu*e22*e22+miu*(e21*e21+e23*e23);
    FC.D22_1[pos_f+3*1+2]=lam*e22*e23+miu*e23*e22;
    FC.D22_1[pos_f+3*2+0]=miu*e21*e23+lam*e23*e21;
    FC.D22_1[pos_f+3*2+1]=miu*e22*e23+lam*e23*e22;
    FC.D22_1[pos_f+3*2+2]=lam2mu*e23*e23+miu*(e21*e21+e22*e22);

    FC.D23_1[pos_f+3*0+0]=lam2mu*e21*e31+miu*(e22*e32+e23*e33);
    FC.D23_1[pos_f+3*0+1]=lam*e21*e32+miu*e22*e31;
    FC.D23_1[pos_f+3*0+2]=lam*e21*e33+miu*e23*e31;
    FC.D23_1[pos_f+3*1+0]=miu*e21*e32+lam*e22*e31;
    FC.D23_1[pos_f+3*1+1]=lam2mu*e22*e32+miu*(e21*e31+e23*e33);
    FC.D23_1[pos_f+3*1+2]=lam*e22*e33+miu*e23*e32;
    FC.D23_1[pos_f+3*2+0]=miu*e21*e33+lam*e23*e31;
    FC.D23_1[pos_f+3*2+1]=miu*e22*e33+lam*e23*e32;
    FC.D23_1[pos_f+3*2+2]=lam2mu*e23*e33+miu*(e21*e31+e22*e32);

    FC.D31_1[pos_f+3*0+0]=lam2mu*e31*e11+miu*(e32*e12+e33*e13);
    FC.D31_1[pos_f+3*0+1]=lam*e31*e12+miu*e32*e11;
    FC.D31_1[pos_f+3*0+2]=lam*e31*e13+miu*e33*e11;
    FC.D31_1[pos_f+3*1+0]=miu*e31*e12+lam*e32*e11;
    FC.D31_1[pos_f+3*1+1]=lam2mu*e32*e12+miu*(e31*e11+e33*e13);
    FC.D31_1[pos_f+3*1+2]=lam*e32*e13+miu*e33*e12;
    FC.D31_1[pos_f+3*2+0]=miu*e31*e13+lam*e33*e11;
    FC.D31_1[pos_f+3*2+1]=miu*e32*e13+lam*e33*e12;
    FC.D31_1[pos_f+3*2+2]=lam2mu*e33*e13+miu*(e31*e11+e32*e12);

    FC.D32_1[pos_f+3*0+0]=lam2mu*e31*e21+miu*(e32*e22+e33*e23);
    FC.D32_1[pos_f+3*0+1]=lam*e31*e22+miu*e32*e21;
    FC.D32_1[pos_f+3*0+2]=lam*e31*e23+miu*e33*e21;
    FC.D32_1[pos_f+3*1+0]=miu*e31*e22+lam*e32*e21;
    FC.D32_1[pos_f+3*1+1]=lam2mu*e32*e22+miu*(e31*e21+e33*e23);
    FC.D32_1[pos_f+3*1+2]=lam*e32*e23+miu*e33*e22;
    FC.D32_1[pos_f+3*2+0]=miu*e31*e23+lam*e33*e21;
    FC.D32_1[pos_f+3*2+1]=miu*e32*e23+lam*e33*e22;
    FC.D32_1[pos_f+3*2+2]=lam2mu*e33*e23+miu*(e31*e21+e32*e22);

    FC.D33_1[pos_f+3*0+0]=lam2mu*e31*e31+miu*(e32*e32+e33*e33);
    FC.D33_1[pos_f+3*0+1]=lam*e31*e32+miu*e32*e31;
    FC.D33_1[pos_f+3*0+2]=lam*e31*e33+miu*e33*e31;
    FC.D33_1[pos_f+3*1+0]=miu*e31*e32+lam*e32*e31;
    FC.D33_1[pos_f+3*1+1]=lam2mu*e32*e32+miu*(e31*e31+e33*e33);
    FC.D33_1[pos_f+3*1+2]=lam*e32*e33+miu*e33*e32;
    FC.D33_1[pos_f+3*2+0]=miu*e31*e33+lam*e33*e31;
    FC.D33_1[pos_f+3*2+1]=miu*e32*e33+lam*e33*e32;
    FC.D33_1[pos_f+3*2+2]=lam2mu*e33*e33+miu*(e31*e31+e32*e32);

    /* plus */
    /* ! need to be modified as Fault's lam and mu */
    //lam = M[pos + 10];
    //miu = M[pos + 11];
    //lam = LAM[pos];
    //miu = MIU[pos];
    lam = FC.lam_f[j + k * ny + 1 * ny * nz + MediaSize];
    miu = FC.mu_f [j + k * ny + 1 * ny * nz + MediaSize];

    lam2mu = lam + 2.f*miu;

    K1_2[0][0]=lam2mu*g11*nn1[0]+lam*g21*nn2[0]+lam*g31*nn3[0];
    K1_2[0][1]=lam2mu*g11*nn1[1]+lam*g21*nn2[1]+lam*g31*nn3[1];
    K1_2[0][2]=lam2mu*g11*nn1[2]+lam*g21*nn2[2]+lam*g31*nn3[2];
    K1_2[1][0]=K1_2[0][0];
    K1_2[1][1]=K1_2[0][1];
    K1_2[1][2]=K1_2[0][2];
    K1_2[2][0]=K1_2[0][0];
    K1_2[2][1]=K1_2[0][1];
    K1_2[2][2]=K1_2[0][2];

    K2_2[0][0]=lam2mu*g12*nn1[0]+lam*g22*nn2[0]+lam*g32*nn3[0];
    K2_2[0][1]=lam2mu*g12*nn1[1]+lam*g22*nn2[1]+lam*g32*nn3[1];
    K2_2[0][2]=lam2mu*g12*nn1[2]+lam*g22*nn2[2]+lam*g32*nn3[2];
    K2_2[1][0]=K2_2[0][0];
    K2_2[1][1]=K2_2[0][1];
    K2_2[1][2]=K2_2[0][2];
    K2_2[2][0]=K2_2[0][0];
    K2_2[2][1]=K2_2[0][1];
    K2_2[2][2]=K2_2[0][2];

    K3_2[0][0]=lam2mu*g13*nn1[0]+lam*g23*nn2[0]+lam*g33*nn3[0];
    K3_2[0][1]=lam2mu*g13*nn1[1]+lam*g23*nn2[1]+lam*g33*nn3[1];
    K3_2[0][2]=lam2mu*g13*nn1[2]+lam*g23*nn2[2]+lam*g33*nn3[2];
    K3_2[1][0]=K3_2[0][0];
    K3_2[1][1]=K3_2[0][1];
    K3_2[1][2]=K3_2[0][2];
    K3_2[2][0]=K3_2[0][0];
    K3_2[2][1]=K3_2[0][1];
    K3_2[2][2]=K3_2[0][2];

    D11_2[0][0]=lam2mu*e11*e11+miu*(e12*e12+e13*e13);
    D11_2[0][1]=lam*e11*e12+miu*e12*e11;
    D11_2[0][2]=lam*e11*e13+miu*e13*e11;
    D11_2[1][0]=miu*e11*e12+lam*e12*e11;
    D11_2[1][1]=lam2mu*e12*e12+miu*(e11*e11+e13*e13);
    D11_2[1][2]=lam*e12*e13+miu*e13*e12;
    D11_2[2][0]=miu*e11*e13+lam*e13*e11;
    D11_2[2][1]=miu*e12*e13+lam*e13*e12;
    D11_2[2][2]=lam2mu*e13*e13+miu*(e11*e11+e12*e12);

    D12_2[0][0]=lam2mu*e11*e21+miu*(e12*e22+e13*e23);
    D12_2[0][1]=lam*e11*e22+miu*e12*e21;
    D12_2[0][2]=lam*e11*e23+miu*e13*e21;
    D12_2[1][0]=miu*e11*e22+lam*e12*e21;
    D12_2[1][1]=lam2mu*e12*e22+miu*(e11*e21+e13*e23);
    D12_2[1][2]=lam*e12*e23+miu*e13*e22;
    D12_2[2][0]=miu*e11*e23+lam*e13*e21;
    D12_2[2][1]=miu*e12*e23+lam*e13*e22;
    D12_2[2][2]=lam2mu*e13*e23+miu*(e11*e21+e12*e22);

    D13_2[0][0]=lam2mu*e11*e31+miu*(e12*e32+e13*e33);
    D13_2[0][1]=lam*e11*e32+miu*e12*e31;
    D13_2[0][2]=lam*e11*e33+miu*e13*e31;
    D13_2[1][0]=miu*e11*e32+lam*e12*e31;
    D13_2[1][1]=lam2mu*e12*e32+miu*(e11*e31+e13*e33);
    D13_2[1][2]=lam*e12*e33+miu*e13*e32;
    D13_2[2][0]=miu*e11*e33+lam*e13*e31;
    D13_2[2][1]=miu*e12*e33+lam*e13*e32;
    D13_2[2][2]=lam2mu*e13*e33+miu*(e11*e31+e12*e32);

    FC.D21_2[pos_f+3*0+0]=lam2mu*e21*e11+miu*(e22*e12+e23*e13);
    FC.D21_2[pos_f+3*0+1]=lam*e21*e12+miu*e22*e11;
    FC.D21_2[pos_f+3*0+2]=lam*e21*e13+miu*e23*e11;
    FC.D21_2[pos_f+3*1+0]=miu*e21*e12+lam*e22*e11;
    FC.D21_2[pos_f+3*1+1]=lam2mu*e22*e12+miu*(e21*e11+e23*e13);
    FC.D21_2[pos_f+3*1+2]=lam*e22*e13+miu*e23*e12;
    FC.D21_2[pos_f+3*2+0]=miu*e21*e13+lam*e23*e11;
    FC.D21_2[pos_f+3*2+1]=miu*e22*e13+lam*e23*e12;
    FC.D21_2[pos_f+3*2+2]=lam2mu*e23*e13+miu*(e21*e11+e22*e12);

    FC.D22_2[pos_f+3*0+0]=lam2mu*e21*e21+miu*(e22*e22+e23*e23);
    FC.D22_2[pos_f+3*0+1]=lam*e21*e22+miu*e22*e21;
    FC.D22_2[pos_f+3*0+2]=lam*e21*e23+miu*e23*e21;
    FC.D22_2[pos_f+3*1+0]=miu*e21*e22+lam*e22*e21;
    FC.D22_2[pos_f+3*1+1]=lam2mu*e22*e22+miu*(e21*e21+e23*e23);
    FC.D22_2[pos_f+3*1+2]=lam*e22*e23+miu*e23*e22;
    FC.D22_2[pos_f+3*2+0]=miu*e21*e23+lam*e23*e21;
    FC.D22_2[pos_f+3*2+1]=miu*e22*e23+lam*e23*e22;
    FC.D22_2[pos_f+3*2+2]=lam2mu*e23*e23+miu*(e21*e21+e22*e22);

    FC.D23_2[pos_f+3*0+0]=lam2mu*e21*e31+miu*(e22*e32+e23*e33);
    FC.D23_2[pos_f+3*0+1]=lam*e21*e32+miu*e22*e31;
    FC.D23_2[pos_f+3*0+2]=lam*e21*e33+miu*e23*e31;
    FC.D23_2[pos_f+3*1+0]=miu*e21*e32+lam*e22*e31;
    FC.D23_2[pos_f+3*1+1]=lam2mu*e22*e32+miu*(e21*e31+e23*e33);
    FC.D23_2[pos_f+3*1+2]=lam*e22*e33+miu*e23*e32;
    FC.D23_2[pos_f+3*2+0]=miu*e21*e33+lam*e23*e31;
    FC.D23_2[pos_f+3*2+1]=miu*e22*e33+lam*e23*e32;
    FC.D23_2[pos_f+3*2+2]=lam2mu*e23*e33+miu*(e21*e31+e22*e32);

    FC.D31_2[pos_f+3*0+0]=lam2mu*e31*e11+miu*(e32*e12+e33*e13);
    FC.D31_2[pos_f+3*0+1]=lam*e31*e12+miu*e32*e11;
    FC.D31_2[pos_f+3*0+2]=lam*e31*e13+miu*e33*e11;
    FC.D31_2[pos_f+3*1+0]=miu*e31*e12+lam*e32*e11;
    FC.D31_2[pos_f+3*1+1]=lam2mu*e32*e12+miu*(e31*e11+e33*e13);
    FC.D31_2[pos_f+3*1+2]=lam*e32*e13+miu*e33*e12;
    FC.D31_2[pos_f+3*2+0]=miu*e31*e13+lam*e33*e11;
    FC.D31_2[pos_f+3*2+1]=miu*e32*e13+lam*e33*e12;
    FC.D31_2[pos_f+3*2+2]=lam2mu*e33*e13+miu*(e31*e11+e32*e12);

    FC.D32_2[pos_f+3*0+0]=lam2mu*e31*e21+miu*(e32*e22+e33*e23);
    FC.D32_2[pos_f+3*0+1]=lam*e31*e22+miu*e32*e21;
    FC.D32_2[pos_f+3*0+2]=lam*e31*e23+miu*e33*e21;
    FC.D32_2[pos_f+3*1+0]=miu*e31*e22+lam*e32*e21;
    FC.D32_2[pos_f+3*1+1]=lam2mu*e32*e22+miu*(e31*e21+e33*e23);
    FC.D32_2[pos_f+3*1+2]=lam*e32*e23+miu*e33*e22;
    FC.D32_2[pos_f+3*2+0]=miu*e31*e23+lam*e33*e21;
    FC.D32_2[pos_f+3*2+1]=miu*e32*e23+lam*e33*e22;
    FC.D32_2[pos_f+3*2+2]=lam2mu*e33*e23+miu*(e31*e21+e32*e22);

    FC.D33_2[pos_f+3*0+0]=lam2mu*e31*e31+miu*(e32*e32+e33*e33);
    FC.D33_2[pos_f+3*0+1]=lam*e31*e32+miu*e32*e31;
    FC.D33_2[pos_f+3*0+2]=lam*e31*e33+miu*e33*e31;
    FC.D33_2[pos_f+3*1+0]=miu*e31*e32+lam*e32*e31;
    FC.D33_2[pos_f+3*1+1]=lam2mu*e32*e32+miu*(e31*e31+e33*e33);
    FC.D33_2[pos_f+3*1+2]=lam*e32*e33+miu*e33*e32;
    FC.D33_2[pos_f+3*2+0]=miu*e31*e33+lam*e33*e31;
    FC.D33_2[pos_f+3*2+1]=miu*e32*e33+lam*e33*e32;
    FC.D33_2[pos_f+3*2+2]=lam2mu*e33*e33+miu*(e31*e31+e32*e32);

#pragma unroll 3
    for (int ii = 0; ii < 3; ii++){
#pragma unroll 3
      for (int jj = 0; jj < 3; jj++){
           D11_1[ii][jj] *= jac;
           D12_1[ii][jj] *= jac;
           D13_1[ii][jj] *= jac;
        FC.D21_1[pos_f+3*ii+jj] *= jac;
        FC.D22_1[pos_f+3*ii+jj] *= jac;
        FC.D23_1[pos_f+3*ii+jj] *= jac;
        FC.D31_1[pos_f+3*ii+jj] *= jac;
        FC.D32_1[pos_f+3*ii+jj] *= jac;
        FC.D33_1[pos_f+3*ii+jj] *= jac;
           D11_2[ii][jj] *= jac;
           D12_2[ii][jj] *= jac;
           D13_2[ii][jj] *= jac;
        FC.D21_2[pos_f+3*ii+jj] *= jac;
        FC.D22_2[pos_f+3*ii+jj] *= jac;
        FC.D23_2[pos_f+3*ii+jj] *= jac;
        FC.D31_2[pos_f+3*ii+jj] *= jac;
        FC.D32_2[pos_f+3*ii+jj] *= jac;
        FC.D33_2[pos_f+3*ii+jj] *= jac;

        D21_1[ii][jj] = FC.D21_1[pos_f+3*ii+jj];
        D22_1[ii][jj] = FC.D22_1[pos_f+3*ii+jj];
        D23_1[ii][jj] = FC.D23_1[pos_f+3*ii+jj];
        D31_1[ii][jj] = FC.D31_1[pos_f+3*ii+jj];
        D32_1[ii][jj] = FC.D32_1[pos_f+3*ii+jj];
        D33_1[ii][jj] = FC.D33_1[pos_f+3*ii+jj];
        D21_2[ii][jj] = FC.D21_2[pos_f+3*ii+jj];
        D22_2[ii][jj] = FC.D22_2[pos_f+3*ii+jj];
        D23_2[ii][jj] = FC.D23_2[pos_f+3*ii+jj];
        D31_2[ii][jj] = FC.D31_2[pos_f+3*ii+jj];
        D32_2[ii][jj] = FC.D32_2[pos_f+3*ii+jj];
        D33_2[ii][jj] = FC.D33_2[pos_f+3*ii+jj];

        FC.D11_1[pos_f+3*ii+jj] = D11_1[ii][jj];
        FC.D12_1[pos_f+3*ii+jj] = D12_1[ii][jj];
        FC.D13_1[pos_f+3*ii+jj] = D13_1[ii][jj];
        FC.D11_2[pos_f+3*ii+jj] = D11_2[ii][jj];
        FC.D12_2[pos_f+3*ii+jj] = D12_2[ii][jj];
        FC.D13_2[pos_f+3*ii+jj] = D13_2[ii][jj];
      }
    }

    matadd3x3(Q, A1, B0);
    // minus side
    // B1 = Q*D11 + A2*D21 + A3*D31
    matmul3x3(Q,  D11_1, mat1);
    matmul3x3(A2, D21_1, mat2);
    matmul3x3(A3, D31_1, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K1_1, mat4, B1_1);
    matadd3x3(mat1, mat4, B1_1);
    // B2 = Q*D12 + A2*D22 + A3*D32
    matmul3x3(Q,  D12_1, mat1);
    matmul3x3(A2, D22_1, mat2);
    matmul3x3(A3, D32_1, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K2_1, mat4, B2_1);
    matadd3x3(mat1, mat4, B2_1);
    // B3 = Q*D13 + A2*D23 + A3*D33
    matmul3x3(Q,  D13_1, mat1);
    matmul3x3(A2, D23_1, mat2);
    matmul3x3(A3, D33_1, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K3_1, mat4, B3_1);
    matadd3x3(mat1, mat4, B3_1);
    // plus side
    // B1 = Q*D11 + A2*D21 + A3*D31
    matmul3x3(Q,  D11_2, mat1);
    matmul3x3(A2, D21_2, mat2);
    matmul3x3(A3, D31_2, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K1_2, mat4, B1_2);
    matadd3x3(mat1, mat4, B1_2);
    // B2 = Q*D12 + A2*D22 + A3*D32
    matmul3x3(Q,  D12_2, mat1);
    matmul3x3(A2, D22_2, mat2);
    matmul3x3(A3, D32_2, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K2_2, mat4, B2_2);
    matadd3x3(mat1, mat4, B2_2);
    // B3 = Q*D13 + A2*D23 + A3*D33
    matmul3x3(Q,  D13_2, mat1);
    matmul3x3(A2, D23_2, mat2);
    matmul3x3(A3, D33_2, mat3);
    matadd3x3(mat2, mat3, mat4);
    matadd3x3(K3_2, mat4, B3_2);
    matadd3x3(mat1, mat4, B3_2);

    // Free surface
    // Minus side
    // invert B1-B3*(K31/K33)
    matcopy3x3(D33_1, D);
    invert3x3(D);
    //matmul3x3(D31_1, D, mat1); // bug fixed
    matmul3x3(D, D31_1, mat1);
    matmul3x3(B3_1, mat1, mat2);
    matsub3x3(B1_1, mat2, mat3);
    if(k==nz-4 && j==par.nj/2 && 0){
      printf("K11 = \n");
      print_mat3x3(D11_1);
      printf("K12 = \n");
      print_mat3x3(D12_1);
      printf("K13 = \n");
      print_mat3x3(D13_1);
      printf("K21 = \n");
      print_mat3x3(D21_1);
      printf("K22 = \n");
      print_mat3x3(D22_1);
      printf("K23 = \n");
      print_mat3x3(D23_1);
      printf("K31 = \n");
      print_mat3x3(D31_1);
      printf("K32 = \n");
      print_mat3x3(D32_1);
      printf("K33 = \n");
      print_mat3x3(D33_1);
      printf("B1 = \n");
      print_mat3x3(B1_1);
      printf("B3 = \n");
      print_mat3x3(B3_1);
      printf("inv(K33)*K31 = \n");
      print_mat3x3(mat1);
      printf("B3*inv(K33)*K31 = \n");
      print_mat3x3(mat2);
      printf("B1-B3*inv(K33)*K31 = \n");
      print_mat3x3(mat3);
    }
    invert3x3(mat3);
    // B2-B3*(K32/K33)
    //matmul3x3(D32_1, D, mat1);
    matmul3x3(D, D32_1, mat1);
    matmul3x3(B3_1, mat1, mat2);
    matsub3x3(B2_1, mat2, mat4);
    matmul3x3(mat3, mat4, mat);
    if(k == nz-4){
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matVytoVxfm[j*3*3+3*ii+jj + matsize1] = mat[ii][jj];  //wangzj
        }
      }
    }
    matmul3x3(mat3, B0, mat);
    if(k == nz-4){
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matT1toVxfm[j*3*3+3*ii+jj + matsize1] = mat[ii][jj];  //wangzj
        }
      }
    }
    // Plus side
    // invert B1-B3*(K31/K33)
    matcopy3x3(D33_2, D);
    invert3x3(D);
    //matmul3x3(D31_2, D, mat1);
    matmul3x3(D, D31_2, mat1);
    matmul3x3(B3_2, mat1, mat2);
    matsub3x3(B1_2, mat2, mat3);
    invert3x3(mat3);
    // B2-B3*(K32/K33)
    // matmul3x3(D32_2, D, mat1);
    matmul3x3(D, D32_2, mat1);
    matmul3x3(B3_2, mat1, mat2);
    matsub3x3(B2_2, mat2, mat4);
    matmul3x3(mat3, mat4, mat);
    if(k == nz-4){
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matVytoVxfp[j*3*3+3*ii+jj + matsize1] = mat[ii][jj];  //wangzj
        }
      }
    }
    matmul3x3(mat3, B0, mat);
    if(k == nz-4){
      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matT1toVxfp[j*3*3+3*ii+jj + matsize1] = mat[ii][jj];  //wangzj
        }
      }
    }

    // Minus side
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        D[ii][jj] = B1_1[ii][jj];
      }
    }
    invert3x3(D);
    // T1 Vy Vz -> Vx
    matmul3x3(D, B0, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matT1toVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, B2_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVytoVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, B3_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVztoVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    // Plus side
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        D[ii][jj] = B1_2[ii][jj];
      }
    }
    invert3x3(D);
    // T1 Vy Vz -> Vx
    matmul3x3(D, B0, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matT1toVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, B2_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVytoVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, B3_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVztoVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }

    // ===========================================================
    // replaced by:
    // K11*DxV+K12*DyV+K13*DzV=DtT1
    // Minus side
    matcopy3x3(D11_1, D);
    invert3x3(D);
    // T1 Vy Vz -> Vx
    matcopy3x3(D, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matT1toVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, D12_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVytoVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, D13_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVztoVxm[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    // Plus side
    matcopy3x3(D11_2, D);
    invert3x3(D);
    // T1 Vy Vz -> Vx
    matcopy3x3(D, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matT1toVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, D12_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVytoVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    matmul3x3(D, D13_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matVztoVxp[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    // ===========================================================

    /* minus -> plus */
    /* invert(D11_2) * D */
    /* D = D11_1, D12_1, D13_1, D12_2, D13_2 */
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        //D[ii][jj] = D11_2[ii][jj];
        D[ii][jj] = B1_2[ii][jj];
      }
    }
    invert3x3(D);
    /* invert(D11_2) * D11_1 */
    //matmul3x3(D, D11_1, mat);
    matmul3x3(D, B1_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matMin2Plus1[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_2) * D12_1 */
    //matmul3x3(D, D12_1, mat);
    matmul3x3(D, B2_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matMin2Plus2[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_2) * D13_1 */
    //matmul3x3(D, D13_1, mat);
    matmul3x3(D, B3_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matMin2Plus3[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_2) * D12_2 */
    //matmul3x3(D, D12_2, mat);
    matmul3x3(D, B2_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matMin2Plus4[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_2) * D13_2 */
    //matmul3x3(D, D13_2, mat);
    matmul3x3(D, B3_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matMin2Plus5[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }

    /* plus -> min */
    /* invert(D11_1) * D */
    /* D = D11_2, D12_2, D13_2, D12_1, D13_1 */
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        //D[ii][jj] = D11_1[ii][jj];
        D[ii][jj] = B1_1[ii][jj];
      }
    }
    invert3x3(D);
    /* invert(D11_1) * D11_2 */
    //matmul3x3(D, D11_2, mat);
    matmul3x3(D, B1_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matPlus2Min1[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_1) * D12_2 */
    //matmul3x3(D, D12_2, mat);
    matmul3x3(D, B2_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matPlus2Min2[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_1) * D13_2 */
    //matmul3x3(D, D13_2, mat);
    matmul3x3(D, B3_2, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matPlus2Min3[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_1) * D12_1 */
    //matmul3x3(D, D12_1, mat);
    matmul3x3(D, B2_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matPlus2Min4[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }
    /* invert(D11_1) * D13_1 */
    //matmul3x3(D, D13_1, mat);
    matmul3x3(D, B3_1, mat);
    for (int ii = 0; ii < 3; ++ii){
      for (int jj = 0; jj < 3; ++jj){
        FC.matPlus2Min5[pos_f+3*ii+jj] = mat[ii][jj];
      }
    }

    /* use mat[3][3] as metric[3][3] */
    //pos   = (i0*ny*nz + j*nz + k)*MSIZE;
    //pos = k * ny * nx + j * nx + i0;
    pos = j + k * ny + i0 * ny * nz;

    mat[0][0] = XIX[pos];//M[pos + 0];
    mat[0][1] = XIY[pos];//M[pos + 1];
    mat[0][2] = XIZ[pos];//M[pos + 2];
    mat[1][0] = ETX[pos];//M[pos + 3];
    mat[1][1] = ETY[pos];//M[pos + 4];
    mat[1][2] = ETZ[pos];//M[pos + 5];
    mat[2][0] = ZTX[pos];//M[pos + 6];
    mat[2][1] = ZTY[pos];//M[pos + 7];
    mat[2][2] = ZTZ[pos];//M[pos + 8];

    invert3x3(mat);

    //pos = j*nz + k;
    pos = j + k * ny + nfault * (nyz);       //********wangzj
    FC.x_et[pos] = mat[0][1];
    FC.y_et[pos] = mat[1][1];
    FC.z_et[pos] = mat[2][1];

    vec_s1[0] = mat[0][1];
    vec_s1[1] = mat[1][1];
    vec_s1[2] = mat[2][1];

    norm = norm3(vec_s1);

#pragma unroll 3
    for (int ia = 0; ia < 3; ++ia)
    {
        vec_s1[ia] /= norm;
    }

    /* metric_xi_x: +0 */
    /* metric_xi_y: +1 */
    /* metric_xi_z: +2 */
    //pos   = (i0*ny*nz + j*nz + k)*MSIZE;
    //pos = k * ny * nx + j * nx + i0; // bug, missed index change
    pos = j + k * ny + i0 * ny * nz;
    vec_n[0] = XIX[pos];//M[pos + 0];
    vec_n[1] = XIY[pos];//M[pos + 1];
    vec_n[2] = XIZ[pos];//M[pos + 2];
    norm = norm3(vec_n);

    for (int ia = 0; ia < 3; ++ia)
    {
        vec_n[ia] /= norm;
    }

    cross_product(vec_n, vec_s1, vec_s2);

    //pos = (j*nz + k)*3;
    pos = (j + k * ny)*3 + nfault * (nyz*3);          // *********wangzj

    for (int ia = 0; ia < 3; ++ia)
    {
        FC.vec_n [pos+ia] = vec_n [ia];
        FC.vec_s1[pos+ia] = vec_s1[ia];
        FC.vec_s2[pos+ia] = vec_s2[ia];
    }
//    if (nj/2 == j && nk/2 == k)
//    {
//     printf("vec_n : %e %e %e\n", FC.vec_n [pos+0], FC.vec_n [pos+1], FC.vec_n [pos+2]);
//     printf("vec_s1: %e %e %e\n", FC.vec_s1[pos+0], FC.vec_s1[pos+1], FC.vec_s1[pos+2]);
//     printf("vec_s2: %e %e %e\n", FC.vec_s2[pos+0], FC.vec_s2[pos+1], FC.vec_s2[pos+2]);
//    }

    /* (nz - 1) ------ p*/
    /* (nz - 2) ------ a*/
    /* (nz - 3) ------ d*/
    /* (nz - 4) ******  */
    /* (nz - 5) ++++++ m*/
    /* (nz - 6) ++++++ i*/
    /* (nz - 7) ++++++ r*/
    if ((nz - 4) == k)
    {
      //pos_f = (j*nz + k)*3*3;
      pos_f = (j + k * ny) * 3 * 3 + Dsize;    // *********wangzj
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          A[ii][jj] =  FC.D33_1[pos_f+ii*3+jj];
          B[ii][jj] = -FC.D31_1[pos_f+ii*3+jj];
          C[ii][jj] = -FC.D32_1[pos_f+ii*3+jj];
        }
      }

      invert3x3(A);
      matmul3x3(A, B, matVx2Vz1);
      matmul3x3(A, C, matVy2Vz1);

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          A[ii][jj] =  FC.D33_2[pos_f+ii*3+jj];
          B[ii][jj] = -FC.D31_2[pos_f+ii*3+jj];
          C[ii][jj] = -FC.D32_2[pos_f+ii*3+jj];
        }
      }
      invert3x3(A);
      matmul3x3(A, B, matVx2Vz2);
      matmul3x3(A, C, matVy2Vz2);


      //pos   = (i0*ny*nz + j*nz + k)*MSIZE;
      //pos = k * ny * nx + j * nx + i0;
      pos = j + k * ny + i0 * ny * nz;
      vec1[0] = XIX[pos];//M[pos+0];
      vec1[1] = XIY[pos];//M[pos+1];
      vec1[2] = XIZ[pos];//M[pos+2];

      //real_t vec1_norm = norm3(vec1);
      vec1_norm = norm3(vec1);

      //vec2[0] = FC.x_et[j*nz+k];
      //vec2[1] = FC.y_et[j*nz+k];
      //vec2[2] = FC.z_et[j*nz+k];
      vec2[0] = FC.x_et[j + k * ny + nfault * (nyz)];
      vec2[1] = FC.y_et[j + k * ny + nfault * (nyz)];
      vec2[2] = FC.z_et[j + k * ny + nfault * (nyz)];

      cross_product(vec1, vec2, vec3);
      g1_2 = 0.0f;
      g2_2 = vec2[0]*vec2[0] 
           + vec2[1]*vec2[1] 
           + vec2[2]*vec2[2];
      g2_2 = g2_2/vec1_norm;
      // this part can be removed
      g2_2 = 0.0f;
      g3_2 = 0.0f;
      //h1_3 = vec3[0]*M[pos+0]
      //     + vec3[1]*M[pos+1]
      //     + vec3[2]*M[pos+2];
      //h2_3 = vec3[0]*M[pos+3]
      //     + vec3[1]*M[pos+4]
      //     + vec3[2]*M[pos+5];
      //h3_3 = vec3[0]*M[pos+6]
      //     + vec3[1]*M[pos+7]
      //     + vec3[2]*M[pos+8];
      //pos = k * ny * nx + j * nx + i0;
      pos = j + k * ny + i0 * ny * nz;
      h1_3 = vec3[0]*XIX[pos]
           + vec3[1]*XIY[pos]
           + vec3[2]*XIZ[pos];
      h2_3 = vec3[0]*ETX[pos]
           + vec3[1]*ETY[pos]
           + vec3[2]*ETZ[pos];
      h3_3 = vec3[0]*ZTX[pos]
           + vec3[1]*ZTY[pos]
           + vec3[2]*ZTZ[pos];

      h1_3 = h1_3/vec1_norm;
      h2_3 = h2_3/vec1_norm;
      h3_3 = h3_3/vec1_norm;

      //miu = F->mu_f[0][j][k];
      //miu = M[pos + 11];
      miu = MIU[pos];
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = vec2[ii] * vec1[jj];
          D11_12[ii][jj] = miu * g1_2 * Tovert[ii][jj];
          D12_12[ii][jj] = miu * g2_2 * Tovert[ii][jj];
          D13_12[ii][jj] = miu * g3_2 * Tovert[ii][jj];
        }
      }

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = vec3[ii] * vec1[jj];
          D11_13[ii][jj] = miu * h1_3 * Tovert[ii][jj];
          D12_13[ii][jj] = miu * h2_3 * Tovert[ii][jj];
          D13_13[ii][jj] = miu * h3_3 * Tovert[ii][jj];
          D11_1f[ii][jj] = D11_1[ii][jj] + D11_12[ii][jj] + D11_13[ii][jj];
          D12_1f[ii][jj] = D12_1[ii][jj] + D12_12[ii][jj] + D12_13[ii][jj];
          D13_1f[ii][jj] = D13_1[ii][jj] + D13_12[ii][jj] + D13_13[ii][jj];
        }
      }

      //miu = F->mu_f[1][j][k];
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = vec2[ii] * vec1[jj];
          D11_12[ii][jj] = miu * g1_2 * Tovert[ii][jj];
          D12_12[ii][jj] = miu * g2_2 * Tovert[ii][jj];
          D13_12[ii][jj] = miu * g3_2 * Tovert[ii][jj];
        }
      }

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = vec3[ii] * vec1[jj];
          D11_13[ii][jj] = miu * h1_3 * Tovert[ii][jj];
          D12_13[ii][jj] = miu * h2_3 * Tovert[ii][jj];
          D13_13[ii][jj] = miu * h3_3 * Tovert[ii][jj];
          D11_2f[ii][jj] = D11_2[ii][jj] + D11_12[ii][jj] + D11_13[ii][jj];
          D12_2f[ii][jj] = D12_2[ii][jj] + D12_12[ii][jj] + D12_13[ii][jj];
          D13_2f[ii][jj] = D13_2[ii][jj] + D13_12[ii][jj] + D13_13[ii][jj];
        }
      }

      //real_t mat1[3][3], mat2[3][3], mat3[3][3], mat4[3][3];
      matmul3x3(D13_1f, matVx2Vz1, mat1);
      matmul3x3(D13_1f, matVy2Vz1, mat2);
      matmul3x3(D13_2f, matVx2Vz2, mat3);
      matmul3x3(D13_2f, matVy2Vz2, mat4);

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          matVx1_free[ii][jj] = D11_1f[ii][jj] + mat1[ii][jj];
          matVy1_free[ii][jj] = D12_1f[ii][jj] + mat2[ii][jj];
          matVx2_free[ii][jj] = D11_2f[ii][jj] + mat3[ii][jj];
          matVy2_free[ii][jj] = D12_2f[ii][jj] + mat4[ii][jj];
        }
      }

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = matVx1_free[ii][jj];
        }
      }
      invert3x3(Tovert);
      matmul3x3(Tovert, matVx2_free, matPlus2Min1f);
      matmul3x3(Tovert, matVy2_free, matPlus2Min2f);
      matmul3x3(Tovert, matVy1_free, matPlus2Min3f);

      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matT1toVxfm[j*9+3*ii+jj+ matsize1] = Tovert[ii][jj];         //******wangzj
          FC.matVytoVxfm[j*9+3*ii+jj+ matsize1] = matPlus2Min3f[ii][jj];  //******wangzj
        }
      }

      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          Tovert[ii][jj] = matVx2_free[ii][jj];
        }
      }
      invert3x3(Tovert);
      matmul3x3(Tovert, matVx1_free, matMin2Plus1f);
      matmul3x3(Tovert, matVy1_free, matMin2Plus2f);
      matmul3x3(Tovert, matVy2_free, matMin2Plus3f);

      for (int ii = 0; ii < 3; ++ii){
        for (int jj = 0; jj < 3; ++jj){
          FC.matT1toVxfp[j*9+3*ii+jj + matsize1] = Tovert[ii][jj];         //******wangzj
          FC.matVytoVxfp[j*9+3*ii+jj + matsize1] = matMin2Plus3f[ii][jj];  //******wangzj
        }
      }

      // save
      /* size = ny*3*3*/
      pos_f = j*3*3 + matsize1;         //********wangzj
      for (int ii = 0; ii < 3; ii++){
        for (int jj = 0; jj < 3; jj++){
          FC.matVx2Vz1    [pos_f+ii*3+jj] = matVx2Vz1    [ii][jj];
          FC.matVy2Vz1    [pos_f+ii*3+jj] = matVy2Vz1    [ii][jj];
          FC.matVx2Vz2    [pos_f+ii*3+jj] = matVx2Vz2    [ii][jj];
          FC.matVy2Vz2    [pos_f+ii*3+jj] = matVy2Vz2    [ii][jj];
          FC.matVx1_free  [pos_f+ii*3+jj] = matVx1_free  [ii][jj];
          FC.matVy1_free  [pos_f+ii*3+jj] = matVy1_free  [ii][jj];
          FC.matVx2_free  [pos_f+ii*3+jj] = matVx2_free  [ii][jj];
          FC.matVy2_free  [pos_f+ii*3+jj] = matVy2_free  [ii][jj];
          FC.matPlus2Min1f[pos_f+ii*3+jj] = matPlus2Min1f[ii][jj];
          FC.matPlus2Min2f[pos_f+ii*3+jj] = matPlus2Min2f[ii][jj];
          FC.matPlus2Min3f[pos_f+ii*3+jj] = matPlus2Min3f[ii][jj];
          FC.matMin2Plus1f[pos_f+ii*3+jj] = matMin2Plus1f[ii][jj];
          FC.matMin2Plus2f[pos_f+ii*3+jj] = matMin2Plus2f[ii][jj];
          FC.matMin2Plus3f[pos_f+ii*3+jj] = matMin2Plus3f[ii][jj];
        }
      }

//#define DEBUG
#ifdef DEBUG
      /* check */
      if (par.nj/2 == j){
          for (int ii = 0; ii < 3; ii++){
          for (int jj = 0; jj < 3; jj++){
            //mat[ii][jj] = F->matVx2Vz1[j][ii][jj];
            mat[ii][jj] = FC.matMin2Plus2f[j*3*3 + ii*3 +jj + matsize1];
          }
        }
        printf("lam = %10.2e; mu = %10.2e; lam2mu= %10.2e\n", lam, miu, lam2mu);
        printf("g^m_2 = %f %f %f\n", g1_2, g2_2, g3_2);
        printf("h^m_3 = %f %f %f\n", h1_3, h2_3, h3_3);

        printf("-inv(K3_xi) * K3_zt (-)\n");
        print_mat3x3(matVx2Vz1);
        printf("-inv(K3_et) * K3_zt (-)\n");
        print_mat3x3(matVy2Vz1);
        printf("-inv(K3_xi) * K3_zt (+)\n");
        print_mat3x3(matVx2Vz2);
        printf("-inv(K3_et) * K3_zt (+)\n");
        print_mat3x3(matVy2Vz2);
        printf("_3 K1_xi (+)\n");
        print_mat3x3(D11_13);
        printf("_3 K1_et (+)\n");
        print_mat3x3(D12_13);
        printf("_3 K1_zt (+)\n");
        print_mat3x3(D13_13);
        printf("_f K1_zt (-) = K1_zt + _2 K1_zt + _3 K1_zt\n");
        print_mat3x3(D13_1f);
        printf("_f K1_zt (+) = K1_zt + _2 K1_zt + _3 K1_zt\n");
        print_mat3x3(D13_2f);

        printf("_f K1_xi - K1_zt * inv(K3_zt)* K3_xi \n");
        print_mat3x3(matVx1_free);
        printf("_f K1_et - K1_zt * inv(K3_zt)* K3_et \n");
        print_mat3x3(matVy1_free);
        printf("_f K1_xi - K1_zt * inv(K3_zt)* K3_xi \n");
        print_mat3x3(matVx2_free);
        printf("_f K1_et - K1_zt * inv(K3_zt)* K3_et \n");
        print_mat3x3(matVy2_free);
        printf("matPlus2Min1f \n");
        print_mat3x3(matPlus2Min1f);
        printf("matPlus2Min2f \n");
        print_mat3x3(matPlus2Min2f);
        printf("matPlus2Min3f \n");
        print_mat3x3(matPlus2Min3f);
        printf("matMin2Plus1f \n");
        print_mat3x3(matMin2Plus1f);
        printf("matMin2Plus2f \n");
        print_mat3x3(matMin2Plus2f);
        printf("matMin2Plus3f \n");
        print_mat3x3(matMin2Plus3f);
        printf("mat \n");
        print_mat3x3(mat);
      }
#endif
    }
  }
  return;
}  /* init_fault_coef */

void init_fault_coef(realptr_t M, Fault FC, int i0, int nfault){
  int ny = hostParams.nj + 6;
  int nz = hostParams.nk + 6;
  dim3 block(16, 8, 1);
  dim3 grid(
      (ny+block.x-1)/block.x,
      (nz+block.y-1)/block.y, 1);
  init_fault_coef_cu <<<grid, block>>> (M, FC, i0, nfault);
}

/*
 **************************************************************************
 *  Free surface condtion for Velocity                                    *
 *  Transform parallel derivatives to normal, i.e. Vx2Vz                  *
 **************************************************************************
 */
__global__ void init_wave_free_cu(realptr_t M, PML P, Wave w){
  if(par.freenode==0) return;

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  //real_t DH = par.DH;
  //real_t DT = par.DT;
  int ni1 = 3;
  int nj1 = 3;
  int nk1 = 3;
  int ni2 = nx - 3;
  int nj2 = ny - 3;
  int nk2 = nz - 3;

  int stride = nx*ny*nz;

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
  //real_t *RHO = MIU + stride;


  int ii, jj;
  int k = nk2-1;
  real_t e11, e12, e13, e21, e22, e23, e31, e32, e33;
  real_t A[3][3], B[3][3], C[3][3];
  real_t AB[3][3], AC[3][3];
  real_t rb1, rb2, rb3;

  real_t lam, mu, lam2mu;

  int pos;

  if(i >= ni1 && i < ni2 && j >= nj1 && j < nj2){
  //for (i = ni1; i < ni2; i++)
  //  for (j = nj1; j < nj2; j++){

#ifdef usePML
      rb1 = 1.0f / P.Bx[i-ni1];
      rb2 = 1.0f / P.By[j-nj1];
      rb3 = 1.0f / P.Bz[k-nk1];
#else
      rb1 = 1.0f;
      rb2 = 1.0f;
      rb3 = 1.0f;
#endif

      //pos = (i*ny*nz + j*nz + k)*MSIZE;
      //pos = k * ny * nx + j * nx + i;
      pos = j + k * ny + i * ny * nz;
      //e11 = M[pos + 0];//e11 = M->xi_x[i][j][k];
      //e12 = M[pos + 1];//e12 = M->xi_y[i][j][k];
      //e13 = M[pos + 2];//e13 = M->xi_z[i][j][k];
      //e21 = M[pos + 3];//e21 = M->et_x[i][j][k];
      //e22 = M[pos + 4];//e22 = M->et_y[i][j][k];
      //e23 = M[pos + 5];//e23 = M->et_z[i][j][k];
      //e31 = M[pos + 6];//e31 = M->zt_x[i][j][k];
      //e32 = M[pos + 7];//e32 = M->zt_y[i][j][k];
      //e33 = M[pos + 8];//e33 = M->zt_z[i][j][k];
      e11 = XIX[pos];//e11 = M->xi_x[i][j][k];
      e12 = XIY[pos];//e12 = M->xi_y[i][j][k];
      e13 = XIZ[pos];//e13 = M->xi_z[i][j][k];
      e21 = ETX[pos];//e21 = M->et_x[i][j][k];
      e22 = ETY[pos];//e22 = M->et_y[i][j][k];
      e23 = ETZ[pos];//e23 = M->et_z[i][j][k];
      e31 = ZTX[pos];//e31 = M->zt_x[i][j][k];
      e32 = ZTY[pos];//e32 = M->zt_y[i][j][k];
      e33 = ZTZ[pos];//e33 = M->zt_z[i][j][k];
      lam = LAM[pos];
      mu  = MIU[pos];

      ////lam = D->lam[i][j][k]; mu = D->mu[i][j][k]; lam2mu = lam + 2.0f*mu;
      //lam = M[pos + 10]; mu = M[pos + 11];
      lam2mu = lam + 2.0f*mu;

      A[0][0] = rb3*(lam2mu*e31*e31 + mu*(e32*e32+e33*e33));
      A[0][1] = rb3*(lam*e31*e32 + mu*e32*e31);
      A[0][2] = rb3*(lam*e31*e33 + mu*e33*e31);
      A[1][0] = rb3*(lam*e32*e31 + mu*e31*e32);
      A[1][1] = rb3*(lam2mu*e32*e32 + mu*(e31*e31+e33*e33));
      A[1][2] = rb3*(lam*e32*e33 + mu*e33*e32);
      A[2][0] = rb3*(lam*e33*e31 + mu*e31*e33);
      A[2][1] = rb3*(lam*e33*e32 + mu*e32*e33);
      A[2][2] = rb3*(lam2mu*e33*e33 + mu*(e31*e31+e32*e32));
      invert3x3(A);

      B[0][0] = -rb1*(lam2mu*e31*e11 + mu*(e32*e12+e33*e13));
      B[0][1] = -rb1*(lam*e31*e12 + mu*e32*e11);
      B[0][2] = -rb1*(lam*e31*e13 + mu*e33*e11);
      B[1][0] = -rb1*(lam*e32*e11 + mu*e31*e12);
      B[1][1] = -rb1*(lam2mu*e32*e12 + mu*(e31*e11+e33*e13));
      B[1][2] = -rb1*(lam*e32*e13 + mu*e33*e12);
      B[2][0] = -rb1*(lam*e33*e11 + mu*e31*e13);
      B[2][1] = -rb1*(lam*e33*e12 + mu*e32*e13);
      B[2][2] = -rb1*(lam2mu*e33*e13 + mu*(e31*e11+e32*e12));

      C[0][0] = -rb2*(lam2mu*e31*e21 + mu*(e32*e22+e33*e23));
      C[0][1] = -rb2*(lam*e31*e22 + mu*e32*e21);
      C[0][2] = -rb2*(lam*e31*e23 + mu*e33*e21);
      C[1][0] = -rb2*(lam*e32*e21 + mu*e31*e22);
      C[1][1] = -rb2*(lam2mu*e32*e22 + mu*(e31*e21+e33*e23));
      C[1][2] = -rb2*(lam*e32*e23 + mu*e33*e22);
      C[2][0] = -rb2*(lam*e33*e21 + mu*e31*e23);
      C[2][1] = -rb2*(lam*e33*e22 + mu*e32*e23);
      C[2][2] = -rb2*(lam2mu*e33*e23 + mu*(e31*e21+e32*e22));
      //A[0][0] = lam2mu*e31*e31 + mu*(e32*e32+e33*e33);
      //A[0][1] = lam*e31*e32 + mu*e32*e31;
      //A[0][2] = lam*e31*e33 + mu*e33*e31;
      //A[1][0] = lam*e32*e31 + mu*e31*e32;
      //A[1][1] = lam2mu*e32*e32 + mu*(e31*e31+e33*e33);
      //A[1][2] = lam*e32*e33 + mu*e33*e32;
      //A[2][0] = lam*e33*e31 + mu*e31*e33;
      //A[2][1] = lam*e33*e32 + mu*e32*e33;
      //A[2][2] = lam2mu*e33*e33 + mu*(e31*e31+e32*e32);
      //invert3x3(A);

      //B[0][0] = -lam2mu*e31*e11 - mu*(e32*e12+e33*e13);
      //B[0][1] = -lam*e31*e12 - mu*e32*e11;
      //B[0][2] = -lam*e31*e13 - mu*e33*e11;
      //B[1][0] = -lam*e32*e11 - mu*e31*e12;
      //B[1][1] = -lam2mu*e32*e12 - mu*(e31*e11+e33*e13);
      //B[1][2] = -lam*e32*e13 - mu*e33*e12;
      //B[2][0] = -lam*e33*e11 - mu*e31*e13;
      //B[2][1] = -lam*e33*e12 - mu*e32*e13;
      //B[2][2] = -lam2mu*e33*e13 - mu*(e31*e11+e32*e12);

      //C[0][0] = -lam2mu*e31*e21 - mu*(e32*e22+e33*e23);
      //C[0][1] = -lam*e31*e22 - mu*e32*e21;
      //C[0][2] = -lam*e31*e23 - mu*e33*e21;
      //C[1][0] = -lam*e32*e21 - mu*e31*e22;
      //C[1][1] = -lam2mu*e32*e22 - mu*(e31*e21+e33*e23);
      //C[1][2] = -lam*e32*e23 - mu*e33*e22;
      //C[2][0] = -lam*e33*e21 - mu*e31*e23;
      //C[2][1] = -lam*e33*e22 - mu*e32*e23;
      //C[2][2] = -lam2mu*e33*e23 - mu*(e31*e21+e32*e22);

      matmul3x3(A, B, AB);
      matmul3x3(A, C, AC);

      for(ii = 0; ii < 3; ii++)
        for(jj = 0; jj < 3; jj++){
          w.matVx2Vz[(j + i * ny)*9 + ii*3 + jj] = AB[ii][jj];
          w.matVy2Vz[(j + i * ny)*9 + ii*3 + jj] = AC[ii][jj];
          //w.matVx2Vz[(j*nx + i)*9 + ii*3 + jj] = AB[ii][jj];
          //w.matVy2Vz[(j*nx + i)*9 + ii*3 + jj] = AC[ii][jj];
          //W->matF2Vz [i*ny + j][ii][jj] =  A[ii][jj];
        }

  }
  return;
}

void init_wave_free(realptr_t M, PML P, Wave W)
{
  int ny = hostParams.nj + 6;
  int nx = hostParams.ni + 6;
  dim3 block(256, 1, 1);
  dim3 grid;
  grid.x = (ny + block.x-1)/block.x;
  grid.y = (nx + block.y-1)/block.y;
  grid.z = 1;
  init_wave_free_cu <<<grid, block>>> (M, P, W);
}
