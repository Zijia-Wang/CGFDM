#ifdef useNetCDF
#include <stdio.h>
#include <stdlib.h>
#include "params.h"
#include "common.h"
#include "io.h"

#ifdef DoublePrecision
#define nc_get_vara_real_t nc_get_vara_double
#define nc_put_vara_real_t nc_put_vara_double
#else
#define nc_get_vara_real_t nc_get_vara_float
#define nc_put_vara_real_t nc_put_vara_float
#endif

void locate_recv(Recv *R, realptr_t C){
  int n;
  real_t x, y, z;
  FILE *fid = fopen("recv_xyz.txt", "r");
  fscanf(fid, "%d\n", &n);
  hostParams.num_recv = n;
  printf("n = %d\n", n);
  R->gx = (real_t *) malloc(sizeof(real_t)*n);
  R->gy = (real_t *) malloc(sizeof(real_t)*n);
  R->gz = (real_t *) malloc(sizeof(real_t)*n);
  R->x = (real_t *) malloc(sizeof(real_t)*n);
  R->y = (real_t *) malloc(sizeof(real_t)*n);
  R->z = (real_t *) malloc(sizeof(real_t)*n);
  R->i = (real_t *) malloc(sizeof(real_t)*n);
  R->j = (real_t *) malloc(sizeof(real_t)*n);
  R->k = (real_t *) malloc(sizeof(real_t)*n);
  for (int i = 0; i < n; i++){
    fscanf(fid, "%f %f %f\n", &x, &y, &z);
    printf("recv coord = %g %g %g\n", x, y, z);
    R->gx[i] = x;
    R->gy[i] = y;
    R->gz[i] = z;
  }
  fclose(fid);

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int npt = 0;
  for (int l = 0; l < hostParams.num_recv; l++){
    int i0, j0, k0; 
    real_t x0,y0,z0;
    real_t rmin = 1e30;
    for (int i = 0; i < nx-0; i++){
      for (int j = 0; j < ny-0; j++){
        for (int k = 0; k < nz-0; k++){
          long pos = j + k * ny + i * ny * nz;
          x = C[pos + 0 * nx * ny * nz];
          y = C[pos + 1 * nx * ny * nz];
          z = C[pos + 2 * nx * ny * nz];

          real_t r = pow(x-R->gx[l],2) + pow(y-R->gy[l],2) + pow(z-R->gz[l],2);
          if(r<rmin){
            i0 = i;
            j0 = j;
            k0 = k;
            rmin = r;
            // save the recv coord in this MPI domain
            x0 = R->gx[l];
            y0 = R->gy[l];
            z0 = R->gz[l];
          }
        }
      }
    } // end i,j,k
    if (
        i0 >= 3 && i0 < nx-3 &&
        j0 >= 3 && j0 < ny-3 &&
        k0 >= 3 && k0 < nz-3 ){
      npt += 1;
      R->i[npt-1] = i0;
      R->j[npt-1] = j0;
      R->k[npt-1] = k0;
      R->x[npt-1] = x0;
      R->y[npt-1] = y0;
      R->z[npt-1] = z0;
    }
  }
  R->n = npt;

  return;
}

void nc_def_recv(Recv R, Wave W, ncFile *nc)
{
  //locate_recv(R);
  int err;

  int nj = hostParams.nj;
  int nk = hostParams.nk;
  
  int npt = R.n;
  if(npt==0) return;

  char filename[1000];
  sprintf(filename, "%s/recv_mpi%02d%02d%02d.nc", OUT,
      thisid[0], thisid[1], thisid[2]);

  err = nc_create(filename, NC_CLOBBER, &(nc->ncid)); handle_err(err);

  // define dimensions
  err = nc_def_dim(nc->ncid, "nt", NC_UNLIMITED, &(nc->dimid[0]));
  err = nc_def_dim(nc->ncid, "npt", npt,         &(nc->dimid[1]));
  handle_err(err);

  //const int dimid2[2] = {nc->dimid[0], nc->dimid[1]};

  err = nc_def_var(nc->ncid, "x", NC_REAL_T, 1, &nc->dimid[1], &(nc->varid[20]));
  err = nc_def_var(nc->ncid, "y", NC_REAL_T, 1, &nc->dimid[1], &(nc->varid[21]));
  err = nc_def_var(nc->ncid, "z", NC_REAL_T, 1, &nc->dimid[1], &(nc->varid[22]));
  err = nc_def_var(nc->ncid, "i", NC_INT, 1, &nc->dimid[1], &(nc->varid[23]));
  err = nc_def_var(nc->ncid, "j", NC_INT, 1, &nc->dimid[1], &(nc->varid[24]));
  err = nc_def_var(nc->ncid, "k", NC_INT, 1, &nc->dimid[1], &(nc->varid[25]));

  handle_err(err);

  // define variables
  err = nc_def_var(nc->ncid, "Vx" , NC_REAL_T, 2, nc->dimid, &(nc->varid[0]));
  err = nc_def_var(nc->ncid, "Vy" , NC_REAL_T, 2, nc->dimid, &(nc->varid[1]));
  err = nc_def_var(nc->ncid, "Vz" , NC_REAL_T, 2, nc->dimid, &(nc->varid[2]));

  handle_err(err);

  err = nc_enddef(nc->ncid); handle_err(err);
  // end define ===============================================================

  return;
}

void nc_put_recv_coord(real_t *C, ncFile nc, int i0, int faultnode)
{
  if(!faultnode) return;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *x = (real_t *) malloc(sizeof(real_t)*nj*nk);
  real_t *y = (real_t *) malloc(sizeof(real_t)*nj*nk);
  real_t *z = (real_t *) malloc(sizeof(real_t)*nj*nk);

  // int srci = hostParams.NX/2;

  // int i = srci % ni; // local i
  int i = i0 % ni; // local i

  for (int j = 0; j < nj; j++){
    for (int k = 0; k < nk; k++){
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

  free(x);
  free(y);
  free(z);

  return;
}

void nc_put_recv(Fault F, int it, ncFile nc)
{
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  size_t start[3] = {it, 0, 0};
  size_t count[3] = {1, nk, nj};
  int err;

  size_t ibytes = sizeof(real_t)*nj*nk;
  real_t *data = (real_t *) malloc(ibytes);

#ifdef DoublePrecision
  cudaMemcpy(data, F.Vs1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[0], start, count, data);

  cudaMemcpy(data, F.Vs2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[1], start, count, data);

  cudaMemcpy(data, F.Tn,  ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[2], start, count, data);

  cudaMemcpy(data, F.Ts1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[3], start, count, data);

  cudaMemcpy(data, F.Ts2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[4], start, count, data);

  cudaMemcpy(data, F.slip, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[5], start, count, data);

  cudaMemcpy(data, F.State, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[6], start, count, data);

  cudaMemcpy(data, F.init_t0, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc.ncid, nc.varid[10], data);
  handle_err(err);
#else
  cudaMemcpy(data, F.Vs1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[0], start, count, data);

  cudaMemcpy(data, F.Vs2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[1], start, count, data);

  cudaMemcpy(data, F.Tn,  ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[2], start, count, data);

  cudaMemcpy(data, F.Ts1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[3], start, count, data);

  cudaMemcpy(data, F.Ts2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[4], start, count, data);

  cudaMemcpy(data, F.slip, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[5], start, count, data);

  cudaMemcpy(data, F.State, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[6], start, count, data);

  cudaMemcpy(data, F.init_t0, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc.ncid, nc.varid[10], data);
  handle_err(err);

#endif

  int *dataint = (int *) malloc(sizeof(int)*nj*nk);

  cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_int(nc.ncid, nc.varid[14], start, count, dataint);

  cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_int(nc.ncid, nc.varid[15], start, count, dataint);

  cudaMemcpy(data, F.rup_sensor, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[30], start, count, data);

  cudaMemcpy(data, F.TP_T, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[31], start, count, data);
  cudaMemcpy(data, F.TP_P, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[32], start, count, data);

  handle_err(err);

  nc_sync(nc.ncid);
  free(data); data = NULL;
  free(dataint);

  return;
}

void nc_end_recv(Recv R, ncFile nc)
{
  if(R.n==0) return;
  int err;
  nc_sync(nc.ncid);
  err = nc_close(nc.ncid);
  handle_err(err);
  return;
}
#endif
