#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "params.h"
#include "common.h"
#include "macdrp.h"

#define POW2(x) ( (x) * (x) )
#define GAUSS_FUN(t,a,t0) (exp(-POW2(((t)-(t0))/(a)))/(a*1.772453850905516))

__global__ void add_source_ricker_cu(Wave W, real_t *M, int it, int irk,
    int isrc, int jsrc, int ksrc)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  long stride = nx * ny * nz;

  real_t *hTxx = W.hW + 3 * stride;
  real_t *hTyy = W.hW + 4 * stride;
  real_t *hTzz = W.hW + 5 * stride;
  real_t *hTxy = W.hW + 6 * stride;
  real_t *hTxz = W.hW + 7 * stride;
  real_t *hTyz = W.hW + 8 * stride;

  real_t *JAC = M + 9 * stride;

  if (idx < 1) {

    real_t t;
    real_t rickerfc = 1.5;
    real_t tdelay = 1.2/rickerfc;
    tdelay = 1.0;
    real_t DT = par.DT;
    real_t DH = par.DH;

    if(irk == 0){ // RK_begin
      t = (it + 0.0f) * DT;
    }else if(irk == 1 || irk == 2){
      t = (it + 0.5f) * DT;
    }else{
      t = (it + 1.0f) * DT;
    }

    real_t f0 = sqrt(PI)*0.5f;
    real_t r = PI * rickerfc * (t-tdelay);
    real_t rr = r*r;
    real_t s = r*(3.0f - 2.0f*rr) * exp(-rr) * f0 * PI * rickerfc;

    if(it == 0 && irk == 0) s = 0.0;

    real_t Mxx, Myy, Mzz, Mxy, Mxz, Myz;
    real_t M0 = 1e16;
    s *= M0;
    // explosive source
    Mxx = 1.0; Myy = 1.0; Mzz = 1.0;
    Mxy = 0.0; Mxz = 0.0; Myz = 0.0;
    // double couple source
    //Mxx = 0.0f; Myy = 0.0f; Mzz = 0.0f;
    //Mxy = 1.0f; Mxz = 0.0f; Myz = 0.0f;

    int NSRCEXT = 3;
    for (int i = -NSRCEXT; i <= NSRCEXT; i++){
      for (int j = -NSRCEXT; j <= NSRCEXT; j++){
        for (int k = -NSRCEXT; k <= NSRCEXT; k++){
          real_t ra = 0.5*NSRCEXT;
          real_t D1 = GAUSS_FUN(i, ra, 0.0);
          real_t D2 = GAUSS_FUN(j, ra, 0.0);
          real_t D3 = GAUSS_FUN(k, ra, 0.0);
          real_t amp = D1*D2*D3;

          amp /= 0.998125703461425; // # 3
          //amp /= 0.9951563131100551; // # 5

          if ( NSRCEXT == 0 ) amp = 1.0;

          long pos = (isrc + 3 + i) * ny * nz + (ksrc + 3 + k) * ny + (jsrc + 3 + j);

          real_t V = s * amp / (JAC[pos] * DH * DH * DH);
          //printf("it = %d, rk = %d, t = %f, SOURCE = %e\n", it, irk, t, V);

          hTxx[pos] -= Mxx * V;
          hTyy[pos] -= Myy * V;
          hTzz[pos] -= Mzz * V;
          hTxy[pos] -= Mxy * V;
          hTxz[pos] -= Mxz * V;
          hTyz[pos] -= Myz * V;
        }
      }
    } // end i, j, k
  }

  return;
}

//extern "C"
void add_source_ricker(Wave W, real_t *M, int it, int irk)
{
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int isrc = hostParams.NX / 2;
  int jsrc = hostParams.NY / 2;
  int ksrc = hostParams.NZ / 2;

  // transform to local index

  int isrc_local = isrc % ni;
  int jsrc_local = jsrc % nj;
  int ksrc_local = ksrc % nk;

  // get mpi index
  int isrc_mpi = isrc / ni;
  int jsrc_mpi = jsrc / nj;
  int ksrc_mpi = ksrc / nk;

  if (
      hostParams.rankx == isrc_mpi &&
      hostParams.ranky == jsrc_mpi &&
      hostParams.rankz == ksrc_mpi )
  {
    add_source_ricker_cu <<<1, 1>>> (W, M, it, irk,
        isrc_local, jsrc_local, ksrc_local);
  }

  return;
}
