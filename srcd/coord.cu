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

void get_coord_x_h(real_t *C, real_t *hC, const int i) {
  int pos;
  int pos_s;
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      pos = j + k * ny + i * ny * nz;
      pos_s = j + k * ny;

      hC[pos_s + 0 * ny * nz] = C[pos + 0 * nx * ny * nz];
      hC[pos_s + 1 * ny * nz] = C[pos + 1 * nx * ny * nz];
      hC[pos_s + 2 * ny * nz] = C[pos + 2 * nx * ny * nz];
    }
  }
  return;
}

void get_coord_y_h(real_t *C, real_t *hC, const int j) {
  int pos;
  int pos_s;
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  //real_t DH = par.DH;

  for (int i = 0; i < nx; ++i) {
    for (int k = 0; k < nz; ++k) {
      //pos = (i * ny * nz + j * nz + k) * CSIZE;
      ///pos_s = (i * nz + k) * CSIZE;
      pos = j + k * ny + i * ny * nz;
      pos_s = k + i * nz;

      hC[pos_s + 0 * nx * nz] = C[pos + 0 * nx * ny * nz];
      hC[pos_s + 1 * nx * nz] = C[pos + 1 * nx * ny * nz];
      hC[pos_s + 2 * nx * nz] = C[pos + 2 * nx * ny * nz];

      //for (int l = 0; l < 3; ++l) {
      //hC[pos_s + l] = C[pos + l];

      //}
    }
  }
  return;
}

void nc_read_fault_geometry(real_t *fault_x, real_t *fault_y, real_t *fault_z){

  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int ny = nj + 6;
  int nz = nk + 6;

  // thisid dimension 0, 1, 2, thisid[2] vary first

  int err;
  int ncid;
  int xid, yid, zid;
  static size_t start[] = {thisid[2]*nk, thisid[1]*nj};
  static size_t count[] = {nk, nj}; // y vary first
  
  err = nc_open(Fault_geometry, NC_NOWRITE, &ncid);
  handle_err(err);

  err = nc_inq_varid(ncid, "x", &xid);
  handle_err(err);
  err = nc_inq_varid(ncid, "y", &yid);
  handle_err(err);
  err = nc_inq_varid(ncid, "z", &zid);
  handle_err(err);
  

  //               Y axis
  //               .
  //              /|\
  //               |
  //           \   |__________
  //            \  |          |
  //             \ |  Fault   |
  //              \|__________|_____\  X axis
  //             O \                /
  //                \
  //                 \
  //                 _\|
  //                   Z axis

  //               Z' axis
  //               .
  //              /|\
  //    (+X') axis |
  //           \   |__________
  //            \  |          |
  //             \ |  Fault   |
  //              \|__________|_____\  Y' axis
  //             O \                /
  //                \
  //                 \
  //                 _\|
  //                  (-X') axis

  //  Y' = X
  //  Z' = Y
  //  X' = -Z

  // X'-Y'-Z' is the coordinate in this code

  //err = nc_get_vara_real_t(ncid, xid, start, count, fault_y);
  //handle_err(err);
  //err = nc_get_vara_real_t(ncid, yid, start, count, fault_z);
  //err = nc_get_vara_real_t(ncid, zid, start, count, fault_x);
  // remember to use -X in this code!

  // NOW I have changed the input, you must use X'-Y'-Z' coord
#ifdef DoublePrecision
  err = nc_get_vara_double(ncid, xid, start, count, fault_x);handle_err(err);
  err = nc_get_vara_double(ncid, yid, start, count, fault_y);handle_err(err);
  err = nc_get_vara_double(ncid, zid, start, count, fault_z);handle_err(err);
#else
  err = nc_get_vara_float(ncid, xid, start, count, fault_x);handle_err(err);
  err = nc_get_vara_float(ncid, yid, start, count, fault_y);handle_err(err);
  err = nc_get_vara_float(ncid, zid, start, count, fault_z);handle_err(err);
#endif

  err = nc_close(ncid); handle_err(err);

  return;
}

void nc_read_fault_geometry3D(real_t *fault_x, real_t *fault_y, real_t *fault_z){

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  // thisid dimension 0, 1, 2, thisid[2] vary first

  int err;
  int ncid;
  int xid, yid, zid;
  static size_t start[] = {thisid[0]*ni, thisid[2]*nk, thisid[1]*nj};
  static size_t count[] = {ni, nk, nj}; // y vary first
  
  err = nc_open(Fault_geometry, NC_NOWRITE, &ncid);
  handle_err(err);

  err = nc_inq_varid(ncid, "x", &xid);
  handle_err(err);
  err = nc_inq_varid(ncid, "y", &yid);
  handle_err(err);
  err = nc_inq_varid(ncid, "z", &zid);
  handle_err(err);
  

#ifdef DoublePrecision
  err = nc_get_vara_double(ncid, xid, start, count, fault_x);handle_err(err);
  err = nc_get_vara_double(ncid, yid, start, count, fault_y);handle_err(err);
  err = nc_get_vara_double(ncid, zid, start, count, fault_z);handle_err(err);
#else
  err = nc_get_vara_float(ncid, xid, start, count, fault_x);handle_err(err);
  err = nc_get_vara_float(ncid, yid, start, count, fault_y);handle_err(err);
  err = nc_get_vara_float(ncid, zid, start, count, fault_z);handle_err(err);
#endif

  err = nc_close(ncid); handle_err(err);

  return;
}

void construct_coord(real_t *C){

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int NX = hostParams.NX;
  int NZ = hostParams.NZ;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  real_t DH = hostParams.DH;

  int pos;
  int pos1 = 0;
  int stride = nx * ny * nz;
  real_t *X = C;
  real_t *Y = X + stride;
  real_t *Z = Y + stride;

  real_t *xline = (real_t *) malloc(sizeof(real_t)*(NX + 6));

  int i0 = NX/2 + 3; // nucleate location
  xline[i0] = 0;
  int width1 = 10;
  int width2 = 55;
  real_t compr = 1.0;

  for (int i = i0+1; i < NX + 6; i++){
    int dist = abs(i - i0);

    if (dist < width1){
      compr = 0;
    }else if(dist < width2){
      compr = 1.0 - cos(PI *(i - (i0 + width1))/(real_t)(width2-width1));
    }else{
      compr = 2.0;
    }
    compr = 0.5 + 0.25 * compr;

    xline[i] = xline[i-1] + DH * compr;
  }

  for (int i = i0-1; i >= 0; i--){
    int dist = abs(i - i0);

    if (dist < width1){
      compr = 0;
    }else if(dist < width2){
      compr = 1.0 - cos(PI *(i - (i0 - width1))/(real_t)(width2-width1));
    }else{
      compr = 2.0;
    }
    compr = 0.5 + 0.25 * compr;

    xline[i] = xline[i+1] - DH * compr;
  }

  // real_t *fault_x = (real_t *) malloc(sizeof(real_t)*nj*nk);
  // real_t *fault_y = (real_t *) malloc(sizeof(real_t)*nj*nk);
  // real_t *fault_z = (real_t *) malloc(sizeof(real_t)*nj*nk);

  // memset(fault_x, 0, sizeof(real_t)*nj*nk);
  // memset(fault_y, 0, sizeof(real_t)*nj*nk);
  // memset(fault_z, 0, sizeof(real_t)*nj*nk);
  real_t *fault_x;
  real_t *fault_y;
  real_t *fault_z;

  if (1 == hostParams.INPORT_GRID_TYPE) {
    fault_x = (real_t *) malloc(sizeof(real_t)*nj*nk);
    fault_y = (real_t *) malloc(sizeof(real_t)*nj*nk);
    fault_z = (real_t *) malloc(sizeof(real_t)*nj*nk);

    memset(fault_x, 0, sizeof(real_t)*nj*nk);
    memset(fault_y, 0, sizeof(real_t)*nj*nk);
    memset(fault_z, 0, sizeof(real_t)*nj*nk);
    nc_read_fault_geometry(fault_x, fault_y, fault_z);
  }

  if (2 == hostParams.INPORT_GRID_TYPE) {
    fault_x = (real_t *) malloc(sizeof(real_t)*nj*nk*ni);
    fault_y = (real_t *) malloc(sizeof(real_t)*nj*nk*ni);
    fault_z = (real_t *) malloc(sizeof(real_t)*nj*nk*ni);

    memset(fault_x, 0, sizeof(real_t)*nj*nk*ni);
    memset(fault_y, 0, sizeof(real_t)*nj*nk*ni);
    memset(fault_z, 0, sizeof(real_t)*nj*nk*ni);
    nc_read_fault_geometry3D(fault_x, fault_y, fault_z);
  }

  for (int i = 3; i < nx-3; i++){
    for (int k = 3; k < nz-3; k++){
      for (int j = 3; j < ny-3; j++){
        int gi = thisid[0]*ni+i-3;
        int gj = thisid[1]*nj+j-3;
        int gk = thisid[2]*nk+k-3;
        //int gi = i-3;
        //int gj = j-3;
        //int gk = k-3;
        //int gi = i;
        //int gj = j;
        //int gk = k;
        int NX = hostParams.NX;
        int NY = hostParams.NY;
        int NZ = hostParams.NZ;

        real_t x0 = -hostParams.NX/2*DH;
        real_t y0 = -hostParams.NY/2*DH;
        real_t z0 = (1-hostParams.NZ)*DH;

        real_t gauss_height = 0.0e3;
        real_t gauss_width = 2.0e3;

        int gj0=NY/2;
        int gi0=NX/2;
        z0 += gauss_height *
          exp(-(pow(gj-gj0,2)+pow(gi-gi0,2))/(pow(gauss_width/DH,2)+1e-30));

        real_t x = gi * DH + x0;
        real_t y = gj * DH + y0;
        real_t z = gk * DH + z0;
#ifdef NormalRefine
        x = xline[i + thisid[0] * ni];
#endif
#ifdef TPV10
        x = xline[gi+3] + (NZ-1-gk)*DH*cos(PI/3.0);
        z = (gk-NZ+1) * DH * sin(PI/3.0);
#endif
#ifdef TPV28
        real_t r1, r2;
        r1 = pow(y + 10.5e3, 2) + pow(z + 7.5e3, 2);
        r2 = pow(y - 10.5e3, 2) + pow(z + 7.5e3, 2);
#ifndef FreeSurface
        r1 = pow(y + 10.5e3, 2) + pow((gk-NZ/2)*DH, 2);
        r2 = pow(y - 10.5e3, 2) + pow((gk-NZ/2)*DH, 2);
#endif
        r1 = sqrt(r1);
        r2 = sqrt(r2);

        real_t fxy = 0;
        if(r1 < 3.0e3){
          fxy = 300. * (1. + cos(PI * r1 / 3.0e3));
        }
        if(r2 < 3.0e3){
          fxy = 300. * (1. + cos(PI * r2 / 3.0e3));
        }
        x += fxy;
#endif
//#define TPV3_hill
#ifdef TPV3_hill
        real_t r1, r2;
        r1 = pow(y + 7.5e3, 2) + pow((gk-NZ/2)*DH, 2);
        r2 = pow(y - 7.5e3, 2) + pow((gk-NZ/2)*DH, 2);
        r1 = sqrt(r1);
        r2 = sqrt(r2);

        real_t fxy = 0;
        if(r1 < 3.0e3){
          fxy = 300. * (1. + cos(PI * r1 / 3.0e3));
        }
        if(r2 < 3.0e3){
          fxy = 300. * (1. + cos(PI * r2 / 3.0e3));
        }
        x += fxy;
#endif

        if (1 == hostParams.INPORT_GRID_TYPE) {
          x = fault_x[j-3 + (k-3) * nj] + gi * DH + x0;
#ifdef TPV29
        //x = -(fault_x[j+k*ny]-41.23)*damp - 41.23 + xline[i+thisid[0]*ni];
        //x = -(fault_x[j+k*ny]-41.23) - 41.23 + xline[i+thisid[0]*ni];
        //  x = (fault_x[j-3+(k-3)*nj]-41.23) - 41.23 + xline[i+thisid[0]*ni];
#endif
#ifdef NormalRefine
          x = fault_x[j-3 + (k-3) * nj] + xline[i + thisid[0] * ni];
#endif
          y = fault_y[j-3 + (k-3) * nj];
          z = fault_z[j-3 + (k-3) * nj];
        }

        if (2 == hostParams.INPORT_GRID_TYPE) {
          
          pos1  = (j - 3) + (k - 3) * nj + (i - 3) * nj * nk; // without ghost points
          x = fault_x[pos1];
          y = fault_y[pos1];
          z = fault_z[pos1];
        }

        pos = j + k * ny + i * ny * nz;
        X[pos] = x;
        Y[pos] = y;
        Z[pos] = z;
      }
    }
  }
  free(xline);

  if (1 <= hostParams.INPORT_GRID_TYPE) {
    free(fault_x);
    free(fault_y);
    free(fault_z);
  }
  return;
}

void cal_range_steph(real_t *C, real_t *range){

  real_t hmin = 1.0e38 , hmax = -1.0e38;
  int pos, pos1, pos2, pos3;
  real_t L;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *X = C;
  real_t *Y = C + nx * ny * nz;
  real_t *Z = C + 2 * nx * ny * nz;

  int i, j, k, ii, jj, kk;
  for ( i = 3; i < nx-3; i++)
    for ( j = 3; j < ny-3; j++)
      for ( k = 3; k < nz-3; k++){

        pos = j + k * ny + i * ny * nz;

        real_t x0[3], x1[3], x2[3], x3[3];
        x0[0] = X[pos];
        x0[1] = Y[pos];
        x0[2] = Z[pos];

        for ( ii = -1; ii <= 1; ii = ii+2)
          for ( jj = -1; jj <= 1; jj = jj+2)
            for ( kk = -1; kk <= 1; kk = kk+2){

              //pos1 = ((i+ii)*ny*lnz+j*lnz+k)*CSIZE;
              //pos2 = (i*ny*lnz+(j+jj)*lnz+k)*CSIZE;
              //pos3 = (i*ny*lnz+j*lnz+(k+kk))*CSIZE;

              pos1 = (i+ii) * ny * nz + k * ny + j;
              pos2 = i * ny * nz + k * ny + (j+jj);
              pos3 = i * ny * nz + (k+kk) * ny + j;

              //x1[0] = C[pos1 + 0]; x2[0] = C[pos2 + 0]; x3[0] = C[pos3 + 0];
              //x1[1] = C[pos1 + 1]; x2[1] = C[pos2 + 1]; x3[1] = C[pos3 + 1];
              //x1[2] = C[pos1 + 2]; x2[2] = C[pos2 + 2]; x3[2] = C[pos3 + 2];
              x1[0] = X[pos1]; x2[0] = X[pos2]; x3[0] = X[pos3];
              x1[1] = Y[pos1]; x2[1] = Y[pos2]; x3[1] = Y[pos3];
              x1[2] = Z[pos1]; x2[2] = Z[pos2]; x3[2] = Z[pos3];

              L = dist_point2plane_h(x0, x1, x2, x3);
              hmin = MIN(hmin, L);
              hmax = MAX(hmax, L);

            }
      }

  //printf("rank %4d %4d %4d distance of point2plane = %10.2e ~ %10.2e\n",
  //    thisid[0], thisid[1], thisid[2], hmin, hmax);
  range[0] = hmin;
  range[1] = hmax;
  return;
}
