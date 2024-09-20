#ifdef useNetCDF
#include <stdio.h>
#include <stdlib.h>
//#include "netcdf.h"
#include "params.h"
#include "common.h"
#include "io.h"

void nc_def_wave_xz(int global_j, ncFile *nc)
{
  int err;
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int dimy = (int) (global_j / nj);

  if (thisid[1] != dimy) return;

  char filename[1000];
  sprintf(filename, "%s/wave_xz_mpi%02d%02d%02d.nc", OUT,
      thisid[0], thisid[1], thisid[2]);
  err = nc_create(filename, NC_CLOBBER, &(nc->ncid)); handle_err(err);

  // define dimensions
  err = nc_def_dim(nc->ncid, "nt", NC_UNLIMITED, &(nc->dimid[0]));
  err = nc_def_dim(nc->ncid, "nx", ni,           &(nc->dimid[1]));
  err = nc_def_dim(nc->ncid, "nz", nk,           &(nc->dimid[2]));
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

void nc_put_wave_xz_coord(real_t *C, int global_j, ncFile nc)
{
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int dimy = (int) (global_j / nj);
  int j = global_j - dimy * nj;

  if (thisid[1] != dimy) return;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *x = (real_t *) malloc(sizeof(real_t)*ni*nk);
  real_t *y = (real_t *) malloc(sizeof(real_t)*ni*nk);
  real_t *z = (real_t *) malloc(sizeof(real_t)*ni*nk);

  for (int i = 0; i < ni; i++){
    for (int k = 0; k < nk; k++){
      int j1 = j + 3;
      int k1 = k + 3;
      int i1 = i + 3;
      int pos = j1 + k1 * ny + i1 * ny * nz;
      int nxyz = nx*ny*nz;
      x[k + i * nk] = C[pos + 0 * nxyz];
      y[k + i * nk] = C[pos + 1 * nxyz];
      z[k + i * nk] = C[pos + 2 * nxyz];
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

__global__ void get_wave_slice_y(real_t *W, int j, real_t *W1)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int ni = par.ni;
  int nk = par.nk;

  int ny = par.ny;
  int nz = par.nz;

  if (i < ni && k < nk){
    int i1 = i+3;
    int j1 = j+3;
    int k1 = k+3;
    int pos = j1 + k1 * ny + i1 * ny * nz;
    W1[k + i * nk] = W[pos];
  }
  return;
}

void nc_put_wave_xz(real_t *W, int global_j, int it, ncFile nc)
{
  int err;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  long nxyz = nx*ny*nz;

  int dimy = (int) (global_j / nj);
  int j = global_j - dimy * nj;

  if (thisid[1] != dimy) return;

  dim3 block(16, 16, 1);
  dim3 grid( (nk + block.x - 1 )/block.x, (ni + block.y - 1 )/block.y, 1);

  real_t *hostData, *deviceData;
  size_t size = sizeof(real_t)*ni*nk;

  cudaMalloc((real_t **) &deviceData, size);
  hostData = (real_t *) malloc(size);

  size_t start[3] = {it, 0, 0};
  size_t count[3] = {1, ni, nk};
  for (int ivar = 0; ivar < 3; ivar ++){
    get_wave_slice_y <<< grid, block >>> (W + ivar*nxyz, j, deviceData);
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

void nc_end_wave_xz(int global_j, ncFile nc)
{
  int nj = hostParams.nj;
  int dimy = (int) (global_j / nj);
  if (thisid[1] != dimy) return;

  int err;
  nc_sync(nc.ncid);
  err = nc_close(nc.ncid);
  handle_err(err);

  return;
}
#endif
