#ifdef useNetCDF
#include <stdio.h>
#include <stdlib.h>
#include "params.h"
#include "common.h"
#include "io.h"

void nc_def_wave_yz(int global_i, ncFile *nc)
{
  int err;
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int dimx = (int) (global_i / ni);

  if (thisid[0] != dimx) return;

  char filename[1000];
  sprintf(filename, "%s/wave_yz_mpi%02d%02d%02d.nc", OUT,
      thisid[0], thisid[1], thisid[2]);
  err = nc_create(filename, NC_CLOBBER, &(nc->ncid)); handle_err(err);

  // define dimensions
  err = nc_def_dim(nc->ncid, "nt", NC_UNLIMITED, &(nc->dimid[0]));
  err = nc_def_dim(nc->ncid, "nz", nk,           &(nc->dimid[1]));
  err = nc_def_dim(nc->ncid, "ny", nj,           &(nc->dimid[2]));
  handle_err(err);

  const int dimid2[2] = {nc->dimid[1], nc->dimid[2]};

  // define variables
#ifdef DoublePrecision
  err = nc_def_var(nc->ncid, "x", NC_DOUBLE, 2, dimid2, &(nc->varid[20]));
  err = nc_def_var(nc->ncid, "y", NC_DOUBLE, 2, dimid2, &(nc->varid[21]));
  err = nc_def_var(nc->ncid, "z", NC_DOUBLE, 2, dimid2, &(nc->varid[22]));

  err = nc_def_var(nc->ncid, "Vx" , NC_DOUBLE, 3, nc->dimid, &nc->varid[0]);
  err = nc_def_var(nc->ncid, "Vy" , NC_DOUBLE, 3, nc->dimid, &nc->varid[1]);
  err = nc_def_var(nc->ncid, "Vz" , NC_DOUBLE, 3, nc->dimid, &nc->varid[2]);
#else
  err = nc_def_var(nc->ncid, "x", NC_FLOAT, 2, dimid2, &(nc->varid[20]));
  err = nc_def_var(nc->ncid, "y", NC_FLOAT, 2, dimid2, &(nc->varid[21]));
  err = nc_def_var(nc->ncid, "z", NC_FLOAT, 2, dimid2, &(nc->varid[22]));

  err = nc_def_var(nc->ncid, "Vx" , NC_FLOAT, 3, nc->dimid, &nc->varid[0]);
  err = nc_def_var(nc->ncid, "Vy" , NC_FLOAT, 3, nc->dimid, &nc->varid[1]);
  err = nc_def_var(nc->ncid, "Vz" , NC_FLOAT, 3, nc->dimid, &nc->varid[2]);
#endif

  // end define
  err = nc_enddef(nc->ncid);
  handle_err(err);

  return;
}

void nc_put_wave_yz_coord(real_t *C, int global_i, ncFile nc)
{
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int dimx = (int) (global_i / ni);
  int i = global_i - dimx * ni;

  if (thisid[0] != dimx) return;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *x = (real_t *) malloc(sizeof(real_t)*nk*nj);
  real_t *y = (real_t *) malloc(sizeof(real_t)*nk*nj);
  real_t *z = (real_t *) malloc(sizeof(real_t)*nk*nj);

  for (int k = 0; k < nk; k++){
    for (int j = 0; j < nj; j++){
      int j1 = j + 3;
      int k1 = k + 3;
      int i1 = i + 3;
      int pos = j1 + k1 * ny + i1 * ny * nz;
      int nxyz = nx*ny*nz;
      x[j + k * nj] = C[pos + 0 * nxyz];
      y[j + k * nj] = C[pos + 1 * nxyz];
      z[j + k * nj] = C[pos + 2 * nxyz];
    }
  }

  int err;
#ifdef DoublePrecision
  err = nc_put_var_double(nc.ncid, nc.varid[20], x);
  err = nc_put_var_double(nc.ncid, nc.varid[21], y);
  err = nc_put_var_double(nc.ncid, nc.varid[22], z);
#else
  err = nc_put_var_float(nc.ncid, nc.varid[20], x);
  err = nc_put_var_float(nc.ncid, nc.varid[21], y);
  err = nc_put_var_float(nc.ncid, nc.varid[22], z);
#endif
  handle_err(err);

  free(x);free(y);free(z);
  return;
}

__global__ void get_wave_slice_x(real_t *W, int i, real_t *W1)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;

  int ny = nj + 6;
  int nz = nk + 6;

  if (k < nk && j < nj){
    int i1 = i+3;
    int j1 = j+3;
    int k1 = k+3;
    int pos = j1 + k1 * ny + i1 * ny * nz;
    W1[j + k * nj] = W[pos];
  }
  return;
}

void nc_put_wave_yz(real_t *W, int global_i, int it, ncFile nc)
{
  int err;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  long nxyz = nx*ny*nz;

  int dimx = (int) (global_i / ni);
  int i = global_i - dimx * ni;

  if (thisid[0] != dimx) return;

  dim3 block(16, 16, 1);
  dim3 grid( (nj + block.x - 1 )/block.x, (nk + block.y - 1 )/block.y, 1);

  real_t *hostData, *deviceData;
  size_t size = sizeof(real_t)*nk*nj;

  cudaMalloc((real_t **) &deviceData, size);
  hostData = (real_t *) malloc(size);

  size_t start[3] = {it, 0, 0};
  size_t count[3] = {1, nk, nj};
  for (int ivar = 0; ivar < 3; ivar ++){
    get_wave_slice_x <<< grid, block >>> (W + ivar*nxyz, i, deviceData);
    cudaMemcpy(hostData, deviceData, size, cudaMemcpyDeviceToHost);
#ifdef DoublePrecision
    err = nc_put_vara_double(nc.ncid, nc.varid[ivar], start, count, hostData);
#else
    err = nc_put_vara_float(nc.ncid, nc.varid[ivar], start, count, hostData);
#endif
    handle_err(err);
  }

  nc_sync(nc.ncid);

  free(hostData);
  cudaFree(deviceData);
  return;
}

void nc_end_wave_yz(int global_i, ncFile nc)
{
  int ni = hostParams.ni;
  int dimx = (int) (global_i / ni);
  if (thisid[0] != dimx) return;

  int err;
  nc_sync(nc.ncid);
  err = nc_close(nc.ncid);
  handle_err(err);

  return;
}
#endif