#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>
//#include <mpi.h>
#include "common.h"
#include "macdrp.h"
#include "params.h"

//#define DEBUG
#define CONSPD 2.0f // power for d
#define CONSPB 2.0f // power for beta
#define CONSPA 1.0f // power for alpha
#define AbsVzero
//#define PI 3.1415926535898

inline real_t cal_pml_R(int N){
  return (real_t) (pow(10, -( (log10((real_t)N)-1.0)/log10(2.0) + 3.0)));
}

inline real_t cal_pml_dmax(real_t L, real_t Vp, real_t Rpp){
  return (real_t) (-Vp / (2.0 * L) * log(Rpp) * (CONSPD + 1.0));
}

inline real_t cal_pml_amax(real_t fc){return PI*fc;}

//inline real_t cal_pml_bmax(){return 3.0;}

inline real_t cal_pml_d(real_t x, real_t L, real_t dmax){
  return (x<0) ? 0.0f : (real_t) (dmax * pow(x/L, CONSPD));
}

inline real_t cal_pml_a(real_t x, real_t L, real_t amax){
  return (x<0) ? 0.0f : (real_t) (amax * (1.0 - pow(x/L, CONSPA)));
}

inline real_t cal_pml_b(real_t x, real_t L, real_t bmax){
  return (x<0) ? 1.0f : (real_t) (1.0 + (bmax-1.0) * pow(x/L, CONSPB));
}

void abs_init(PML P){

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  real_t DH = hostParams.DH;

  int PML_N = hostParams.PML_N;
  real_t Vp = hostParams.PML_velocity;
  real_t PML_fc = hostParams.PML_fc;
  real_t bmax = hostParams.PML_bmax;

  real_t Rpp = cal_pml_R(PML_N);
  real_t amax = cal_pml_amax(PML_fc);
  //real_t bmax = 3.0;
  real_t L0 = DH*(PML_N-1);
  real_t dmax = cal_pml_dmax(L0, Vp, Rpp);

  for (int i = 0; i < ni; i++){
    P.Ax[i] = 0.0f;
    P.Bx[i] = 1.0f;
    P.Dx[i] = 0.0f;
  }

  for (int i = 0; i < nj; i++){
    P.Ay[i] = 0.0f;
    P.By[i] = 1.0f;
    P.Dy[i] = 0.0f;
  }

  for (int i = 0; i < nk; i++){
    P.Az[i] = 0.0f;
    P.Bz[i] = 1.0f;
    P.Dz[i] = 0.0f;
  }

  real_t *Lx = (real_t *) malloc(sizeof(real_t)*ni);
  real_t *Ly = (real_t *) malloc(sizeof(real_t)*nj);
  real_t *Lz = (real_t *) malloc(sizeof(real_t)*nk);

  for (int i = 0; i < ni; i++) Lx[i] = -1.0f;
  for (int i = 0; i < nj; i++) Ly[i] = -1.0f;
  for (int i = 0; i < nk; i++) Lz[i] = -1.0f;
#ifdef usePML
  if(P.isx1){
    for (int i = 0; i < PML_N; i++)
      Lx[i+0] = (PML_N-1-i) * DH;
  }
  if(P.isx2){
    for (int i = 0; i < PML_N; i++)
      Lx[ni-PML_N+i-0] = i * DH;
  }
  if(P.isy1){
    for (int i = 0; i < PML_N; i++)
      Ly[i+0] = (PML_N-1-i) * DH;
  }
  // BUG
  if(P.isy2){
    for (int i = 0; i < PML_N; i++)
      Ly[nj-PML_N+i-0] = i * DH;
  }
  if(P.isz1){
    for (int i = 0; i < PML_N; i++)
      Lz[i+0] = (PML_N-1-i) * DH;
  }
  if(P.isz2){
    for (int i = 0; i < PML_N; i++)
      Lz[nk-PML_N+i-0] = i * DH;
  }
#endif

  for (int i = 0; i < ni; i++){
    P.Ax[i] = cal_pml_a(Lx[i], L0, amax);
    P.Bx[i] = cal_pml_b(Lx[i], L0, bmax);
    P.Dx[i] = cal_pml_d(Lx[i], L0, dmax);
  }
  for (int i = 0; i < nj; i++){
    P.Ay[i] = cal_pml_a(Ly[i], L0, amax);
    P.By[i] = cal_pml_b(Ly[i], L0, bmax);
    P.Dy[i] = cal_pml_d(Ly[i], L0, dmax);
  }
  for (int i = 0; i < nk; i++){
    P.Az[i] = cal_pml_a(Lz[i], L0, amax);
    P.Bz[i] = cal_pml_b(Lz[i], L0, bmax);
    P.Dz[i] = cal_pml_d(Lz[i], L0, dmax);
  }

  // convert d_x to d_x/beta_x since only d_x/beta_x needed
  for (int i = 0; i < ni; i++) P.Dx[i] /= P.Bx[i];
  for (int i = 0; i < nj; i++) P.Dy[i] /= P.By[i];
  for (int i = 0; i < nk; i++) P.Dz[i] /= P.Bz[i];

#ifdef DEBUG
  FILE *fp;
  char fnm[1000];
  sprintf(fnm, "%s/ABD%03d%03d%03d.txt", OUT, thisid[0], thisid[1], thisid[2]);
  //sprintf(fnm, "output/ABD.txt");
  fp = fopen(fnm, "w");
  for (int i = 0; i < ni; i++)
    fprintf(fp, "ABDx[%05d] = %f %f %f\n", i, P.Ax[i], P.Bx[i], P.Dx[i]);
  for (int i = 0; i < nj; i++)
    fprintf(fp, "ABDy[%05d] = %f %f %f\n", i, P.Ay[i], P.By[i], P.Dy[i]);
  for (int i = 0; i < nk; i++)
    fprintf(fp, "ABDz[%05d] = %f %f %f\n", i, P.Az[i], P.Bz[i], P.Dz[i]);
  fclose(fp);
#endif


  free(Lx);
  free(Ly);
  free(Lz);
  return;
}
