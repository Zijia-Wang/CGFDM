#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "netcdf.h"

#define handle_err(e)                         \
{                                             \
  if (e != NC_NOERR) {                        \
    printf("nc error: %s\n", nc_strerror(e)); \
    exit(2);                                  \
  }                                           \
}

// MacCormack DRP finite difference scheme
// backward and forward (LB or LF)
#define c_1 0.30874
#define c_2 0.63260
#define c_3 1.23300
#define c_4 0.33340
#define c_5 0.04168

#define LF(var,idx,stride) \
  (-c_1*var[idx-stride  ] \
   -c_2*var[idx         ] \
   +c_3*var[idx+stride  ] \
   -c_4*var[idx+stride*2] \
   +c_5*var[idx+stride*3])

#define LB(var,idx,stride) \
  (-c_5*var[idx-stride*3] \
   +c_4*var[idx-stride*2] \
   -c_3*var[idx-stride  ] \
   +c_2*var[idx         ] \
   +c_1*var[idx+stride  ])


struct Coord {
  float *x;
  float *y;
  float *z;
};

struct Metric {
  float *xix;
  float *xiy;
  float *xiz;
  float *etx;
  float *ety;
  float *etz;
  float *ztx;
  float *zty;
  float *ztz;
  float *jac;
  float *vec_n;
  float *vec_s1;
  float *vec_s2;
};

float norm_h(float *a){
  return sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}

void normalize_h(float *a){
  float a0=1.0f/sqrtf(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
  a[0] *= a0;
  a[1] *= a0;
  a[2] *= a0;
  return;
}

void get_geometry_size(char filename[], size_t *ny, size_t *nz){
  // Get ny, nz
  int err, ncid;
  int ydimid, zdimid;

  err = nc_open(filename, NC_NOWRITE, &ncid);
  handle_err(err);

  err = nc_inq_dimid(ncid, "ny", &ydimid);
  handle_err(err);

  err = nc_inq_dimid(ncid, "nz", &zdimid);
  handle_err(err);

  err = nc_inq_dimlen(ncid, ydimid, ny);
  handle_err(err);

  err = nc_inq_dimlen(ncid, zdimid, nz);
  handle_err(err);

  err = nc_close(ncid);
  handle_err(err);

  return;
}

void nc_read_fault_geometry(char filename[],
    struct Coord *fault_coord){

  int err;
  int ncid;
  int xid, yid, zid;

  err = nc_open(filename, NC_NOWRITE, &ncid);
  handle_err(err);

  err = nc_inq_varid(ncid, "x", &xid);
  handle_err(err);

  err = nc_inq_varid(ncid, "y", &yid);
  handle_err(err);

  err = nc_inq_varid(ncid, "z", &zid);
  handle_err(err);

  err = nc_get_var_float(ncid, xid, fault_coord->x);
  handle_err(err);

  err = nc_get_var_float(ncid, yid, fault_coord->y);
  handle_err(err);

  err = nc_get_var_float(ncid, zid, fault_coord->z);
  handle_err(err);

  err = nc_close(ncid);
  handle_err(err);

  return;
}

void nc_write_fault_metric(char filename[], struct Metric fault_metric,
    size_t ny, size_t nz){
  int err;
  int ncid;
  int ydimid, zdimid, Threedimid;
  int dimids[2];
  int dimid3[3];
  int varid[20];

  err = nc_create(filename, NC_CLOBBER, &ncid);
  handle_err(err);

  err = nc_def_dim(ncid, "ny", ny, &ydimid);
  handle_err(err);

  err = nc_def_dim(ncid, "nz", nz, &zdimid);
  handle_err(err);

  err = nc_def_dim(ncid, "three", 3, &Threedimid);
  handle_err(err);

  dimids[0] = zdimid;
  dimids[1] = ydimid; // y vary first

  dimid3[0] = zdimid;
  dimid3[1] = ydimid; // y vary first
  dimid3[2] = Threedimid; // y vary first

  err = nc_def_var(ncid, "xix", NC_FLOAT, 2, dimids, &varid[0]);
  err = nc_def_var(ncid, "xiy", NC_FLOAT, 2, dimids, &varid[1]);
  err = nc_def_var(ncid, "xiz", NC_FLOAT, 2, dimids, &varid[2]);
  err = nc_def_var(ncid, "etx", NC_FLOAT, 2, dimids, &varid[3]);
  err = nc_def_var(ncid, "ety", NC_FLOAT, 2, dimids, &varid[4]);
  err = nc_def_var(ncid, "etz", NC_FLOAT, 2, dimids, &varid[5]);
  err = nc_def_var(ncid, "ztx", NC_FLOAT, 2, dimids, &varid[6]);
  err = nc_def_var(ncid, "zty", NC_FLOAT, 2, dimids, &varid[7]);
  err = nc_def_var(ncid, "ztz", NC_FLOAT, 2, dimids, &varid[8]);
  err = nc_def_var(ncid, "jac", NC_FLOAT, 2, dimids, &varid[9]);

  err = nc_def_var(ncid, "vec_n",  NC_FLOAT, 3, dimid3, &varid[10]);
  err = nc_def_var(ncid, "vec_s1", NC_FLOAT, 3, dimid3, &varid[11]);
  err = nc_def_var(ncid, "vec_s2", NC_FLOAT, 3, dimid3, &varid[12]);

  err = nc_enddef(ncid);
  handle_err(err);

  err = nc_put_var_float(ncid, varid[0], fault_metric.xix);
  err = nc_put_var_float(ncid, varid[1], fault_metric.xiy);
  err = nc_put_var_float(ncid, varid[2], fault_metric.xiz);
  err = nc_put_var_float(ncid, varid[3], fault_metric.etx);
  err = nc_put_var_float(ncid, varid[4], fault_metric.ety);
  err = nc_put_var_float(ncid, varid[5], fault_metric.etz);
  err = nc_put_var_float(ncid, varid[6], fault_metric.ztx);
  err = nc_put_var_float(ncid, varid[7], fault_metric.zty);
  err = nc_put_var_float(ncid, varid[8], fault_metric.ztz);
  err = nc_put_var_float(ncid, varid[9], fault_metric.jac);

  err = nc_put_var_float(ncid, varid[10], fault_metric.vec_n );
  err = nc_put_var_float(ncid, varid[11], fault_metric.vec_s1);
  err = nc_put_var_float(ncid, varid[12], fault_metric.vec_s2);

  err = nc_close(ncid);
  handle_err(err);
}

void get_range(float *arr, int arrsize, float range[]){
  float vmin = 1.0e38;
  float vmax = -1.0e38;

  for (int i = 0; i < arrsize; i++){
    float v = arr[i];
    vmin = (vmin < v) ? vmin : v;
    vmax = (vmax > v) ? vmax : v;
  }

  range[0] = vmin;
  range[1] = vmax;

  return;
}

inline float dot_product_h(float *A, float *B){
  float result = 0.0;
  for (int i = 0; i < 3; i++)
    result += A[i] * B[i];
  return result;
}

inline void cross_product_h(float *A, float *B, float *C){
  C[0] = A[1] * B[2] - A[2] * B[1];
  C[1] = A[2] * B[0] - A[0] * B[2];
  C[2] = A[0] * B[1] - A[1] * B[0];
  return;
}

void cal_metric(struct Coord fault_coord, struct Metric *fault_metric,
    size_t ny, size_t nz, float DH){

  float rDH = 1.0/DH;

  float x_xi, x_et, x_zt;
  float y_xi, y_et, y_zt;
  float z_xi, z_et, z_zt;
  float jac;
  float vec1[3], vec2[3], vec3[3], vecg[3];

  float *X = fault_coord.x;
  float *Y = fault_coord.y;
  float *Z = fault_coord.z;

  float vec_n[3];
  float vec_s1[3];
  float vec_s2[3];

  for (size_t i = 3; i <= 3; i++){
    for (size_t k = 3; k < nz - 3; k++){
      for (size_t j = 3; j < ny - 3; j++){

        int pos = j + k * ny + i * ny * nz;
        int pos_m = j + k * ny;

        x_xi = 0.0; x_et = 0.0; x_zt = 0.0;
        y_xi = 0.0; y_et = 0.0; y_zt = 0.0;
        z_xi = 0.0; z_et = 0.0; z_zt = 0.0;

        int nyz = ny * nz;
        //if(i==3 && j==3 && k==3)
        //  printf("nyz = %d, pos = %d, rDH = %f\n", nyz, pos, rDH);

        x_xi = 0.5 * rDH *( LF(X, pos, nyz) + LB(X, pos, nyz));
        y_xi = 0.5 * rDH *( LF(Y, pos, nyz) + LB(Y, pos, nyz));
        z_xi = 0.5 * rDH *( LF(Z, pos, nyz) + LB(Z, pos, nyz));

        x_et = 0.5 * rDH *( LF(X, pos, 1  ) + LB(X, pos, 1  ));
        y_et = 0.5 * rDH *( LF(Y, pos, 1  ) + LB(Y, pos, 1  ));
        z_et = 0.5 * rDH *( LF(Z, pos, 1  ) + LB(Z, pos, 1  ));

        x_zt = 0.5 * rDH *( LF(X, pos, ny ) + LB(X, pos, ny ));
        y_zt = 0.5 * rDH *( LF(Y, pos, ny ) + LB(Y, pos, ny ));
        z_zt = 0.5 * rDH *( LF(Z, pos, ny ) + LB(Z, pos, ny ));

        vec1[0] = x_xi; vec1[1] = y_xi; vec1[2] = z_xi;
        vec2[0] = x_et; vec2[1] = y_et; vec2[2] = z_et;
        vec3[0] = x_zt; vec3[1] = y_zt; vec3[2] = z_zt;

        cross_product_h(vec1, vec2, vecg);
        jac = dot_product_h(vecg, vec3);
        fault_metric->jac[pos_m] = jac;

        float rjac = 1.0/jac;

        cross_product_h(vec2, vec3, vecg);
        fault_metric->xix[pos_m] = vecg[0] * rjac;
        fault_metric->xiy[pos_m] = vecg[1] * rjac;
        fault_metric->xiz[pos_m] = vecg[2] * rjac;

        cross_product_h(vec3, vec1, vecg);
        fault_metric->etx[pos_m] = vecg[0] * rjac;
        fault_metric->ety[pos_m] = vecg[1] * rjac;
        fault_metric->etz[pos_m] = vecg[2] * rjac;

        cross_product_h(vec1, vec2, vecg);
        fault_metric->ztx[pos_m] = vecg[0] * rjac;
        fault_metric->zty[pos_m] = vecg[1] * rjac;
        fault_metric->ztz[pos_m] = vecg[2] * rjac;

        //fault_metric->xix[pos_m] = x_xi;
        //fault_metric->xiy[pos_m] = y_xi;
        //fault_metric->xiz[pos_m] = z_xi;
        //fault_metric->etx[pos_m] = x_et;
        //fault_metric->ety[pos_m] = y_et;
        //fault_metric->etz[pos_m] = z_et;
        //fault_metric->ztx[pos_m] = x_zt;
        //fault_metric->zty[pos_m] = y_zt;
        //fault_metric->ztz[pos_m] = z_zt;

        vec_n[0] = fault_metric->xix[pos_m];
        vec_n[1] = fault_metric->xiy[pos_m];
        vec_n[2] = fault_metric->xiz[pos_m];

        normalize_h(vec_n);

        vec_s1[0] = x_et;
        vec_s1[1] = y_et;
        vec_s1[2] = z_et;

        normalize_h(vec_s1);

        cross_product_h(vec_n, vec_s1, vec_s2);

        for (int l = 0; l < 3; l++){
          fault_metric->vec_n [pos_m * 3 + l] = vec_n [l];
          fault_metric->vec_s1[pos_m * 3 + l] = vec_s1[l];
          fault_metric->vec_s2[pos_m * 3 + l] = vec_s2[l];
        }

      }
    }
  }
  return;
}

int main(int argc, char *argv[]){

  if (argc != 4){
    printf("Calculate metric from fault geometry data\n"
        "Usage: %s input(Fault geometry) output(Fault metric) DH\n",
        argv[0]);
    exit(-1);
  }

  size_t ny, nz;
  // Get ny, nz
  get_geometry_size(argv[1], &ny, &nz);
  printf("ny = %ld, nz = %ld\n", ny, nz);

  struct Coord fault_coord;
  fault_coord.x = (float *) malloc(sizeof(float)*ny*nz);
  fault_coord.y = (float *) malloc(sizeof(float)*ny*nz);
  fault_coord.z = (float *) malloc(sizeof(float)*ny*nz);

  nc_read_fault_geometry(argv[1], &fault_coord);

  float fault_x_range[2];
  float fault_y_range[2];
  float fault_z_range[2];
  get_range(fault_coord.x, ny*nz, fault_x_range);
  get_range(fault_coord.y, ny*nz, fault_y_range);
  get_range(fault_coord.z, ny*nz, fault_z_range);
  printf(
      "range of fault x = %f ~ %f\n"
      "range of fault y = %f ~ %f\n"
      "range of fault z = %f ~ %f\n" ,
      fault_x_range[0], fault_x_range[1],
      fault_y_range[0], fault_y_range[1],
      fault_z_range[0], fault_z_range[1]);

  struct Coord coord;
  struct Metric fault_metric;

  coord.x = (float *) malloc(sizeof(float)*ny*nz*7);
  coord.y = (float *) malloc(sizeof(float)*ny*nz*7);
  coord.z = (float *) malloc(sizeof(float)*ny*nz*7);

  fault_metric.xix = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.xiy = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.xiz = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.etx = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.ety = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.etz = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.ztx = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.zty = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.ztz = (float *) malloc(sizeof(float)*ny*nz);
  fault_metric.jac = (float *) malloc(sizeof(float)*ny*nz);

  fault_metric.vec_n  = (float *) malloc(sizeof(float)*ny*nz*3);
  fault_metric.vec_s1 = (float *) malloc(sizeof(float)*ny*nz*3);
  fault_metric.vec_s2 = (float *) malloc(sizeof(float)*ny*nz*3);

  //float DH = 50;
  float DH = atof(argv[3]);
  printf("DH = %f\n", DH);
  for (size_t i = 0; i < 7; i++){
    for (size_t k = 0; k < nz; k++){
      for (size_t j = 0; j < ny; j++){
        size_t pos = j + k * ny + i * ny * nz;
        size_t pos_f = j + k * ny;
        coord.x[pos] = fault_coord.x[pos_f] + ((int)i-3) * DH/2.0;
        coord.y[pos] = fault_coord.y[pos_f];
        coord.z[pos] = fault_coord.z[pos_f];
      }
    }
  }

  // free fault coord
  free(fault_coord.x);
  free(fault_coord.y);
  free(fault_coord.z);

  cal_metric(coord, &fault_metric, ny, nz, DH);

  // free coord (extended coord)

  // export the calculated fault metric (ny by nz, y vary first)
  nc_write_fault_metric(argv[2], fault_metric, ny, nz);

  float metric_range[10][2];
  get_range(fault_metric.xix, ny*nz, metric_range[0]);
  get_range(fault_metric.xiy, ny*nz, metric_range[1]);
  get_range(fault_metric.xiz, ny*nz, metric_range[2]);
  get_range(fault_metric.etx, ny*nz, metric_range[3]);
  get_range(fault_metric.ety, ny*nz, metric_range[4]);
  get_range(fault_metric.etz, ny*nz, metric_range[5]);
  get_range(fault_metric.ztx, ny*nz, metric_range[6]);
  get_range(fault_metric.zty, ny*nz, metric_range[7]);
  get_range(fault_metric.ztz, ny*nz, metric_range[8]);
  get_range(fault_metric.jac, ny*nz, metric_range[9]);
  printf(
      "range of fault xix = %f ~ %f\n"
      "range of fault xiy = %f ~ %f\n"
      "range of fault xiz = %f ~ %f\n"
      "range of fault etx = %f ~ %f\n"
      "range of fault ety = %f ~ %f\n"
      "range of fault etz = %f ~ %f\n"
      "range of fault ztx = %f ~ %f\n"
      "range of fault zty = %f ~ %f\n"
      "range of fault ztz = %f ~ %f\n"
      "range of fault jac = %f ~ %f\n" ,
      metric_range[0][0], metric_range[0][1],
      metric_range[1][0], metric_range[1][1],
      metric_range[2][0], metric_range[2][1],
      metric_range[3][0], metric_range[3][1],
      metric_range[4][0], metric_range[4][1],
      metric_range[5][0], metric_range[5][1],
      metric_range[6][0], metric_range[6][1],
      metric_range[7][0], metric_range[7][1],
      metric_range[8][0], metric_range[8][1],
      metric_range[9][0], metric_range[9][1]);


  // free fault metric

  return 0;
}
