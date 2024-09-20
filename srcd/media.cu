#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "params.h"
#include "common.h"

#ifdef DoublePrecision
#define nc_get_vara_real_t nc_get_vara_double
#else
#define nc_get_vara_real_t nc_get_vara_float
#endif

void cal_range_media(realptr_t M, real_t *range){

  real_t rho_min = 1.0e30 , rho_max = -1.0e30;
  real_t vp_min = 1.0e30 , vp_max = -1.0e30;
  real_t vs_min = 1.0e30 , vs_max = -1.0e30;
  real_t rho, vp, vs, lam, mu;
  int pos;
  int i, j, k;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  real_t *LAM = M + 10*nx*ny*nz;
  real_t *MIU = M + 11*nx*ny*nz;
  real_t *RHO = M + 12*nx*ny*nz;

  for ( i = 3; i < nx-3; i++){
    for ( j = 3; j < ny-3; j++){
      for ( k = 3; k < nz-3; k++){

        pos = i * ny * nz + k * ny + j;

        lam = LAM[pos];
        mu  = MIU[pos];
        rho = RHO[pos];

        vp = sqrtf((lam + 2.0f * mu)/rho);
        vs = sqrtf(mu/rho);

        rho_min = MIN(rho_min, rho);
        rho_max = MAX(rho_max, rho);
        vp_min = MIN(vp_min, vp);
        vp_max = MAX(vp_max, vp);
        vs_min = MIN(vs_min, vs);
        vs_max = MAX(vs_max, vs);
      }
    }
  }

  range[0] = vp_min;
  range[1] = vp_max;
  range[2] = vs_min;
  range[3] = vs_max;
  range[4] = rho_min;
  range[5] = rho_max;
  return;
}

void init_media1d(real_t *C, real_t *M){
#define MAXLAY 100
  char str[1000];
  int layernum;
  real_t depth[MAXLAY],vp[MAXLAY],vs[MAXLAY],rho[MAXLAY];
  FILE *fp;
  if(NULL == (fp = fopen(Media1D, "r"))){
    if(masternode){
      printf("can not open input media dat\n");
      exit(-1);
    }
  }else{
    fgets(str, sizeof(str), fp);
    //sscanf(str, "%d", &layernum);
    fscanf(fp, "%d\n", &layernum);
    if(masternode) printf("layernum = %d\n", layernum);
    fgets(str, sizeof(str), fp);
    if(masternode) puts(str);
    for (int i = 0; i < layernum; i++){
      fscanf(fp, "%g %g %g %g\n", &depth[i],&rho[i],&vp[i],&vs[i]);
      if(masternode) printf("%g %g %g %g\n", depth[i],rho[i],vp[i],vs[i]);
    }
    fclose(fp);
  }

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  int stride = nx * ny * nz;
  //real_t *X = C + stride * 0;
  //real_t *Y = C + stride * 1;
  real_t *Z = C + stride * 2;
  real_t *LAM = M + stride * 10;
  real_t *MIU = M + stride * 11;
  real_t *RHO = M + stride * 12;

  for (int i = 3; i < nx - 3; i++){
    for (int k = 3; k < nz - 3; k++){
      for (int j = 3; j < ny - 3; j++){

        int pos = j + k * ny + i * ny * nz;

        real_t z = Z[pos];

        real_t depth1 = -z;

        // half space
        real_t rho1 = rho[layernum-1];
        real_t  vp1 =  vp[layernum-1];
        real_t  vs1 =  vs[layernum-1];

        int flag = 1;
        for (int l = layernum-1; l >= 0; l--){
          if(depth1 > depth[l] && flag){
            flag = 0;
            vp1 = vp[l];
            vs1 = vs[l];
            rho1 = rho[l];
          }
        }

        LAM[pos] = rho1*(vp1*vp1-2*vs1*vs1);
        MIU[pos] = rho1*vs1*vs1;
        RHO[pos] = rho1;

      }
    }
  }

  return;
}

void init_media3d(real_t *C, real_t *M){

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  int err;
  int ncid;
  int varid;

  // Attention !!!
  // j vary first, then k, i
  //static size_t start[] = {thisid[0]*ni+3, thisid[2]*nk+3, thisid[1]*nj+3};
  static size_t start[] = {thisid[0]*ni, thisid[2]*nk, thisid[1]*nj};
  static size_t count[] = {ni, nk, nj};

  err = nc_open(Media3D, NC_NOWRITE, &ncid); handle_err(err);

  real_t *vp  = (real_t*) malloc(sizeof(real_t)*ni*nj*nk);
  real_t *vs  = (real_t*) malloc(sizeof(real_t)*ni*nj*nk);
  real_t *rho = (real_t*) malloc(sizeof(real_t)*ni*nj*nk);
  printf("read meida3d\n");
  err = nc_inq_varid(ncid, "vp", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, vp); handle_err(err);
  printf("read vp ok!\n");
  err = nc_inq_varid(ncid, "vs", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, vs); handle_err(err);

  err = nc_inq_varid(ncid, "rho", &varid); handle_err(err);
  err = nc_get_vara_real_t(ncid, varid, start, count, rho); handle_err(err);

  real_t *LAM = M + nx * ny * nz * 10;
  real_t *MIU = M + nx * ny * nz * 11;
  real_t *RHO = M + nx * ny * nz * 12;

  for (int i = 0; i < ni; i++){
    for (int k = 0; k < nk; k++){
      for (int j = 0; j < nj; j++){

        int i1 = i + 3;
        int j1 = j + 3;
        int k1 = k + 3;

        int pos1 = j1 + k1 * ny + i1 * ny * nz; // with ghost points
        int pos  = j + k * nj + i * nj * nk; // without ghost points

        real_t rho1 = rho[pos];
        real_t vp1  = vp [pos];
        real_t vs1  = vs [pos];

        LAM[pos1] = rho1*(vp1*vp1-2*vs1*vs1);
        MIU[pos1] = rho1*vs1*vs1;
        RHO[pos1] = rho1;

      }
    }
  }

  free(vp);free(vs);free(rho);
  return;
}
