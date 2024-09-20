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

void nc_def_fault(Fault F, ncFile *nc)
{
  int err;

  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int num_fault = hostParams.num_fault;         //*********wangzj
                                                   //*********wangzj 
  // int* dimid2;
  int dimsize = 3;
  int dimsizeT = dimsize + 1;

 
  char filename[1000];
  sprintf(filename, "%s/fault_mpi%02d%02d%02d.nc", OUT,
      thisid[0], thisid[1], thisid[2]);

  err = nc_create(filename, NC_CLOBBER, &(nc->ncid)); handle_err(err);

  // define dimensions
  err = nc_def_dim(nc->ncid, "nt", NC_UNLIMITED, &(nc->dimid[0]));
  err = nc_def_dim(nc->ncid, "nfault", num_fault, &(nc->dimid[1]));
  err = nc_def_dim(nc->ncid, "nz", nk,           &(nc->dimid[2]));
  err = nc_def_dim(nc->ncid, "ny", nj,           &(nc->dimid[3]));

  const int dimid2[3] = {nc->dimid[1], nc->dimid[2], nc->dimid[3]};

  handle_err(err);

  err = nc_def_var(nc->ncid, "x", NC_REAL_T, dimsize, dimid2, &(nc->varid[20]));
  err = nc_def_var(nc->ncid, "y", NC_REAL_T, dimsize, dimid2, &(nc->varid[21]));
  err = nc_def_var(nc->ncid, "z", NC_REAL_T, dimsize, dimid2, &(nc->varid[22]));

  err = nc_def_var(nc->ncid, "init_t0",  NC_REAL_T, dimsize, dimid2, &(nc->varid[10]));
  if (hostParams.Friction_type == 0){
    err = nc_def_var(nc->ncid, "C0",       NC_REAL_T, dimsize, dimid2, &(nc->varid[11]));
    err = nc_def_var(nc->ncid, "str_peak", NC_REAL_T, dimsize, dimid2, &(nc->varid[12]));
  }
  err = nc_def_var(nc->ncid, "united",      NC_INT, dimsize, dimid2, &(nc->varid[9]));
  // err = nc_def_var(nc->ncid, "faultgrid",      NC_INT, dimsize, dimid2, &(nc->varid[13]));
  //err = nc_def_var(nc->ncid, "rup_index_y", NC_INT, 2, dimid2, &(nc->varid[14]));
  //err = nc_def_var(nc->ncid, "rup_index_z", NC_INT, 2, dimid2, &(nc->varid[15]));
  if (hostParams.Friction_type > 0){
    err = nc_def_var(nc->ncid, "a", NC_REAL_T, dimsize, dimid2, &(nc->varid[13]));
    err = nc_def_var(nc->ncid, "b", NC_REAL_T, dimsize, dimid2, &(nc->varid[14]));
    err = nc_def_var(nc->ncid, "Vw", NC_REAL_T, dimsize, dimid2, &(nc->varid[15]));
  }
  handle_err(err);

  // define variables
  err = nc_def_var(nc->ncid, "Vs1"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[0]));
  err = nc_def_var(nc->ncid, "Vs2"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[1]));
  err = nc_def_var(nc->ncid, "tn"   , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[2]));
  err = nc_def_var(nc->ncid, "ts1"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[3]));
  err = nc_def_var(nc->ncid, "ts2"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[4]));
  err = nc_def_var(nc->ncid, "Us0"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[5]));
  
  err = nc_def_var(nc->ncid, "Us1"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[6]));   // wangzj  **********
  err = nc_def_var(nc->ncid, "Us2"  , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[7]));   // ************
  err = nc_def_var(nc->ncid, "rake" , NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[8]));   // ************
  // err = nc_def_var(nc->ncid, "rup_index_y", NC_INT, dimsizeT, nc->dimid, &(nc->varid[14]));
  // err = nc_def_var(nc->ncid, "rup_index_z", NC_INT, dimsizeT, nc->dimid, &(nc->varid[15]));
  #ifdef RupSensor
    err = nc_def_var(nc->ncid, "rup_sensor", NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[16]));
  #endif
  if (hostParams.Friction_type > 0){
    err = nc_def_var(nc->ncid, "State", NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[17]));
    
    if (hostParams.Friction_type == 3){
    err = nc_def_var(nc->ncid, "TP_T", NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[18]));
    err = nc_def_var(nc->ncid, "TP_P", NC_REAL_T, dimsizeT, nc->dimid, &(nc->varid[19]));
    }
  }

  handle_err(err);

  err = nc_enddef(nc->ncid); handle_err(err);
  // end define ===============================================================

  // put some initial values
  real_t *data = (real_t *) malloc(sizeof(real_t)*nj*nk*num_fault);
  int *dataint = (int *) malloc(sizeof(int)*nj*nk*num_fault);

#ifdef DoublePrecision

  cudaMemcpy(dataint, F.united, sizeof(int)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
  // cudaMemcpy(dataint, F.faultgrid, sizeof(int)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
  err = nc_put_var_int(nc->ncid, nc->varid[9], dataint);handle_err(err);

  if (hostParams.Friction_type == 0){
    cudaMemcpy(data, F.C0, sizeof(double)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_double(nc->ncid, nc->varid[11], data);handle_err(err);

    cudaMemcpy(data, F.str_peak, sizeof(double)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_double(nc->ncid, nc->varid[12], data);handle_err(err);
  }

  //cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[14], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[15], dataint);handle_err(err);
  if (hostParams.Friction_type > 0){
    cudaMemcpy(data, F.a, sizeof(double)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_double(nc->ncid, nc->varid[13], data);handle_err(err);

    cudaMemcpy(data, F.b, sizeof(double)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_double(nc->ncid, nc->varid[14], data);handle_err(err);

    cudaMemcpy(data, F.Vw, sizeof(double)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_double(nc->ncid, nc->varid[15], data);handle_err(err);
  }
#else
  cudaMemcpy(dataint, F.united, sizeof(int)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
  //cudaMemcpy(dataint, F.faultgrid, sizeof(int)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
  err = nc_put_var_int(nc->ncid, nc->varid[9], dataint);handle_err(err);
  if (hostParams.Friction_type == 0){
    cudaMemcpy(data, F.C0, sizeof(float)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_float(nc->ncid, nc->varid[11], data);handle_err(err);

    cudaMemcpy(data, F.str_peak, sizeof(float)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_float(nc->ncid, nc->varid[12], data);handle_err(err);
  }

  //cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[14], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[15], dataint);handle_err(err);
  if (hostParams.Friction_type > 0){
    cudaMemcpy(data, F.a, sizeof(float)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_float(nc->ncid, nc->varid[13], data);handle_err(err);

    cudaMemcpy(data, F.b, sizeof(float)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_float(nc->ncid, nc->varid[14], data);handle_err(err);

    cudaMemcpy(data, F.Vw, sizeof(float)*nj*nk*num_fault, cudaMemcpyDeviceToHost);
    err = nc_put_var_float(nc->ncid, nc->varid[15], data);handle_err(err);
  }
#endif

  free(data);
  free(dataint);
  return;
}

void nc_put_fault_coord(real_t *C, ncFile nc, int faultnode)
{
  // if(!hostParams.faultnode) return;
  if(!faultnode) return;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int num_fault = hostParams.num_fault;         //*********wangzj

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *x = (real_t *) malloc(sizeof(real_t)*nj*nk*num_fault);
  real_t *y = (real_t *) malloc(sizeof(real_t)*nj*nk*num_fault);
  real_t *z = (real_t *) malloc(sizeof(real_t)*nj*nk*num_fault);

 // int srci = hostParams.NX/2;

 
  for (int nfault = 0; nfault < num_fault; nfault++){
    int srci = hostParams.src_i[nfault];
    int i = srci % ni; // local i
    for (int j = 0; j < nj; j++){
      for (int k = 0; k < nk; k++){
        int j1 = j + 3;
        int k1 = k + 3;
        int i1 = i + 3;
        int pos = j1 + k1 * ny + i1 * ny * nz;
        int nxyz = nx*ny*nz;
        x[j + k * nj + nfault*(nj*nk)] = C[pos + 0 * nxyz];
        y[j + k * nj + nfault*(nj*nk)] = C[pos + 1 * nxyz];
        z[j + k * nj + nfault*(nj*nk)] = C[pos + 2 * nxyz];
      }
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

void* ncput_fault(void* ptr)
{
  pthreadFault* pthdF = (pthreadFault*)ptr;
  real_t* hostFault = pthdF->hostFault;
  int nj = pthdF->nj;
  int nk = pthdF->nk;
  int it = pthdF->it;
  int num_fault = hostParams.num_fault;         //*********wangzj
  ncFile nc = pthdF->nc;
  
  int njk = nj * nk;
  int err;

  int num_output = 10;
  int sensor_pos = 0;
#ifdef RupSensor
    num_output++;
    sensor_pos = 1;
#endif
  if (hostParams.Friction_type > 0){
    num_output++;
    if (hostParams.Friction_type == 3){
      num_output = num_output + 2;
    }
  }
  // size_t* start;
  // size_t* count;

  for (int nfault = 0; nfault < num_fault; nfault++){
    
    size_t start[4] = {it, nfault, 0, 0};
    size_t count[4] = {1, 1, nk, nj};
    // }
    int Fdatasize = nfault * njk * num_output;
                                                 //*******wangzj
#ifdef DoublePrecision
  err = nc_put_vara_double(nc.ncid, nc.varid[0], start, count, hostFault+0*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[1], start, count, hostFault+1*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[2], start, count, hostFault+2*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[3], start, count, hostFault+3*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[4], start, count, hostFault+4*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[5], start, count, hostFault+5*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[6], start, count, hostFault+6*njk + Fdatasize);
  err = nc_put_vara_double(nc.ncid, nc.varid[7], start, count, hostFault+7*njk + Fdatasize);        // ******** 
  err = nc_put_vara_double(nc.ncid, nc.varid[8], start, count, hostFault+8*njk + Fdatasize);       
  size_t start_2[3] = {nfault, 0, 0};
  size_t count_2[3] = {1, nk, nj}; 
  err = nc_put_vara_double(nc.ncid, nc.varid[10], start_2, count_2, hostFault+9*njk + Fdatasize);
#ifdef RupSensor
  err = nc_put_vara_double(nc.ncid, nc.varid[16], start, count, hostFault+10*njk + Fdatasize);
#endif
  if (hostParams.Friction_type > 0){
    err = nc_put_vara_double(nc.ncid, nc.varid[17], start, count, hostFault+(9 + sensor_pos + 1)*njk + Fdatasize);
    if (hostParams.Friction_type == 3){ 
      err = nc_put_vara_double(nc.ncid, nc.varid[18], start, count, hostFault+(9 + sensor_pos + 2)*njk + Fdatasize);
      err = nc_put_vara_double(nc.ncid, nc.varid[19], start, count, hostFault+(9 + sensor_pos + 3)*njk + Fdatasize);
    }
  }
#else
  err = nc_put_vara_float(nc.ncid, nc.varid[0], start, count, hostFault+0*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[1], start, count, hostFault+1*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[2], start, count, hostFault+2*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[3], start, count, hostFault+3*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[4], start, count, hostFault+4*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[5], start, count, hostFault+5*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[6], start, count, hostFault+6*njk + Fdatasize);
  err = nc_put_vara_float(nc.ncid, nc.varid[7], start, count, hostFault+7*njk + Fdatasize);        // ******** 
  err = nc_put_vara_float(nc.ncid, nc.varid[8], start, count, hostFault+8*njk + Fdatasize);
  size_t start_2[3] = {nfault, 0, 0};
  size_t count_2[3] = {1, nk, nj}; 
  err = nc_put_vara_float(nc.ncid, nc.varid[10], start_2, count_2, hostFault+9*njk + Fdatasize);
#ifdef RupSensor  
  err = nc_put_vara_float(nc.ncid, nc.varid[16], start, count, hostFault+10*njk + Fdatasize);
#endif
  if (hostParams.Friction_type > 0){
    err = nc_put_vara_float(nc.ncid, nc.varid[17], start, count, hostFault+(9 + sensor_pos + 1)*njk + Fdatasize);
    if (hostParams.Friction_type == 3){ 
      err = nc_put_vara_float(nc.ncid, nc.varid[18], start, count, hostFault+(9 + sensor_pos + 2)*njk + Fdatasize);
      err = nc_put_vara_float(nc.ncid, nc.varid[19], start, count, hostFault+(9 + sensor_pos + 3)*njk + Fdatasize);
    }
  }
#endif
  

  // int jj, kk;
  // real_t* rup_index_y = hostFault+14*njk + Fdatasize;
  // real_t* rup_index_z = hostFault+15*njk + Fdatasize;
  // int* rup_index = (int *)malloc(sizeof(int)*2*njk);
  // for (kk = 0; kk < nk; kk++)
  // {
  //   for (jj = 0; jj < nj; jj++)
  //   {
  //     int pos = jj + kk * nj;
  //     rup_index[pos] = (int)rup_index_y[pos];
  //     rup_index[pos+njk] = (int)rup_index_z[pos];
  //   }
  // }
  // err = nc_put_vara_int(nc.ncid, nc.varid[14], start, count, rup_index);
  // err = nc_put_vara_int(nc.ncid, nc.varid[15], start, count, rup_index+njk);
  // handle_err(err);
  // free(rup_index);
  }
  nc_sync(nc.ncid);
  // free(rup_index);
  // free(start);    //wangzj
  // free(count);    //wangzj
  return NULL;
}

void* ncput_wave(void* ptr)
{
  pthreadWave* pthdW = (pthreadWave*)ptr;
  real_t* hostWave = pthdW->hostData;
  int ni = pthdW->ni;
  int nj = pthdW->nj;
  int nk = pthdW->nk;
  int it = pthdW->it;
  ncFile ncX = pthdW->ncX;
  ncFile ncY = pthdW->ncY;
  ncFile ncZ = pthdW->ncZ;
  int global_i = pthdW->global_i;
  int global_j = pthdW->global_j;
  int global_k = pthdW->global_k;
  int id[3] = {pthdW->id[0], pthdW->id[1], pthdW->id[2]};
  
  
  int dimz = (int) (global_k / nk);
  int dimy = (int) (global_j / nj);
  int dimx = (int) (global_i / ni);
  int skip = 0;
  int err;

  if (id[2] == dimz)
  {
    size_t start_Z[3] = {it, 0, 0};
    size_t count_Z[3] = {1, ni, nj};
#ifdef DoublePrecision
    err = nc_put_vara_double(ncZ.ncid, ncZ.varid[0], start_Z, count_Z, hostWave);
    err = nc_put_vara_double(ncZ.ncid, ncZ.varid[1], start_Z, count_Z, hostWave + 1*ni*nj);
    err = nc_put_vara_double(ncZ.ncid, ncZ.varid[2], start_Z, count_Z, hostWave + 2*ni*nj);
#else
    err = nc_put_vara_float(ncZ.ncid, ncZ.varid[0], start_Z, count_Z, hostWave);
    err = nc_put_vara_float(ncZ.ncid, ncZ.varid[1], start_Z, count_Z, hostWave + 1*ni*nj);
    err = nc_put_vara_float(ncZ.ncid, ncZ.varid[2], start_Z, count_Z, hostWave + 2*ni*nj);
#endif
    handle_err(err);
    nc_sync(ncZ.ncid);
    skip += 3 * ni * nj;
  }
  if (id[1] == dimy)
  {
    size_t start_Y[3] = {it, 0, 0};
    size_t count_Y[3] = {1, ni, nk};
#ifdef DoublePrecision
    err = nc_put_vara_double(ncY.ncid, ncY.varid[0], start_Y, count_Y, hostWave+skip);
    err = nc_put_vara_double(ncY.ncid, ncY.varid[1], start_Y, count_Y, hostWave+skip + 1*ni*nk);
    err = nc_put_vara_double(ncY.ncid, ncY.varid[2], start_Y, count_Y, hostWave+skip + 2*ni*nk);
#else
    err = nc_put_vara_float(ncY.ncid, ncY.varid[0], start_Y, count_Y, hostWave+skip);
    err = nc_put_vara_float(ncY.ncid, ncY.varid[1], start_Y, count_Y, hostWave+skip + 1*ni*nk);
    err = nc_put_vara_float(ncY.ncid, ncY.varid[2], start_Y, count_Y, hostWave+skip + 2*ni*nk);
#endif
    handle_err(err);
    nc_sync(ncY.ncid);
    skip += 3 * ni * nk;
  }
  if (id[0] == dimx)
  {
    size_t start_X[3] = {it, 0, 0};
    size_t count_X[3] = {1, nk, nj};
#ifdef DoublePrecision
    err = nc_put_vara_double(ncX.ncid, ncX.varid[0], start_X, count_X, hostWave+skip);
    err = nc_put_vara_double(ncX.ncid, ncX.varid[1], start_X, count_X, hostWave+skip + 1*nk*nj);
    err = nc_put_vara_double(ncX.ncid, ncX.varid[2], start_X, count_X, hostWave+skip + 2*nk*nj);
#else
    err = nc_put_vara_float(ncX.ncid, ncX.varid[0], start_X, count_X, hostWave+skip);
    err = nc_put_vara_float(ncX.ncid, ncX.varid[1], start_X, count_X, hostWave+skip + 1*nk*nj);
    err = nc_put_vara_float(ncX.ncid, ncX.varid[2], start_X, count_X, hostWave+skip + 2*nk*nj);
#endif
    handle_err(err);
    nc_sync(ncX.ncid);
  }
  return NULL;
}


__global__ void get_wave_slice_z(real_t *W, long nxyz, int k, real_t *W1)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int ni = par.ni;
  int nj = par.nj;

  int ny = par.ny;
  int nz = par.nz;

  if (i < ni && j < nj){
    int i1 = i+3;
    int j1 = j+3;
    int k1 = k+3;
    int pos = j1 + k1 * ny + i1 * ny * nz;
    W1[j + i * nj] = W[pos];
    W1[j + i * nj + 1 * ni * nj] = W[pos + 1 * nxyz];
    W1[j + i * nj + 2 * ni * nj] = W[pos + 2 * nxyz];
  }
  return;
}

__global__ void get_wave_slice_y(real_t *W, long nxyz, int j, real_t *W1)
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
    W1[k + i * nk + 1 * ni * nk] = W[pos + 1 * nxyz];
    W1[k + i * nk + 2 * ni * nk] = W[pos + 2 * nxyz];
  }
  return;
}

__global__ void get_wave_slice_x(real_t *W, long nxyz, int i, real_t *W1)
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
    W1[j + k * nj + 1 * nj * nk] = W[pos + 1 * nxyz];
    W1[j + k * nj + 2 * nj * nk] = W[pos + 2 * nxyz];
  }
  return;
}


__global__ void pack_fault(Fault F, real_t *deviceFault, const int nj, const int nk, int num_output)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int num_fault = par.num_fault;         //*********wangzj
  int sensor_pos = 0;
  int njk = nj * nk;

  if (j < nj && k < nk)
  {
    for (int nfault = 0; nfault < num_fault; nfault++){
      int pos = j + k * nj + nfault * (njk*num_output);
      int pos_f = j + k * nj + nfault * njk;                //wangzj

      deviceFault[pos + 0 * njk] = F.Vs1[pos_f];
      deviceFault[pos + 1 * njk] = F.Vs2[pos_f];
      deviceFault[pos + 2 * njk] = F.Tn[pos_f];
      deviceFault[pos + 3 * njk] = F.Ts1[pos_f];
      deviceFault[pos + 4 * njk] = F.Ts2[pos_f];
      deviceFault[pos + 5 * njk] = F.slip[pos_f];
      
      deviceFault[pos + 6 * njk] = F.slip1[pos_f];
      deviceFault[pos + 7 * njk] = F.slip2[pos_f];
      deviceFault[pos + 8 * njk] = F.rake[pos_f];
      deviceFault[pos + 9 * njk] = F.init_t0[pos_f];

      
#ifdef RupSensor
      sensor_pos = 1;
      deviceFault[pos + (9 + sensor_pos) * njk] = F.rup_sensor[pos_f];
#endif
      if (par.Friction_type > 0){
        deviceFault[pos + (9 + sensor_pos + 1) * njk] = F.State[pos_f];
        if (par.Friction_type == 3){
          deviceFault[pos + (9 + sensor_pos + 2) * njk] = F.TP_T[pos_f];
          deviceFault[pos + (9 + sensor_pos + 3) * njk] = F.TP_P[pos_f];
        }
      }
      // deviceFault[pos + 14 * njk] = (real_t)F.rup_index_y[pos_f];
      // deviceFault[pos + 15 * njk] = (real_t)F.rup_index_z[pos_f];
    }

  }
  return;

}

void nc_put_faultwave(Fault F, real_t *W, int global_k, int global_j, int global_i, real_t** hostptr)
{
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int num_fault = hostParams.num_fault;         //*********wangzj

  long nxyz = nx*ny*nz;

  int dimz = (int) (global_k / nk);
  int k = global_k - dimz * nk;

  int dimy = (int) (global_j / nj);
  int j = global_j - dimy * nj;

  int dimx = (int) (global_i / ni);
  int i = global_i - dimx * ni;

  int countwave = 0; 
  if (thisid[2] == dimz)  countwave += 3*sizeof(real_t)*ni*nj;
  if (thisid[1] == dimy)  countwave += 3*sizeof(real_t)*ni*nk;
  if (thisid[0] == dimx)  countwave += 3*sizeof(real_t)*nk*nj;

  // Allocate and initialize an array of stream handles
  // int n_streams = 2;
  // cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(cudaStream_t));

  // for (int i = 0 ; i < n_streams ; i++)
  // {
  //   CUDACHECK(cudaStreamCreate(&(streams[i])));
  // }
  int num_output = 10;
#ifdef RupSensor
    num_output++;
#endif
  if (hostParams.Friction_type > 0){
    num_output++;
    if (hostParams.Friction_type == 3){
      num_output = num_output + 2;
    }
  }
  dim3 block_fault(32, 16, 1);
  dim3 grid_fault((nj + block_fault .x - 1 )/block_fault.x, (nk + block_fault.y - 1 )/block_fault.y, 1);
  pack_fault <<<grid_fault, block_fault>>>(F, hostptr[2], nj, nk, num_output);
  // printf("==================================1\n");
  cudaMemcpy(hostptr[0], hostptr[2], num_output*sizeof(real_t)*nj*nk*num_fault, cudaMemcpyDeviceToHost);


  int skip = 0;
  if (thisid[2] == dimz)
  {
    dim3 block(32, 32, 1);
    dim3 grid( (nj + block.x - 1 )/block.x, (ni + block.y - 1 )/block.y, 1);

    get_wave_slice_z <<< grid, block>>> (W, nxyz, k, hostptr[3]);
    skip += 3 * ni * nj;
  }


  if (thisid[1] == dimy)
  {
    dim3 block(32, 32, 1);
    dim3 grid( (nk + block.x - 1 )/block.x, (ni + block.y - 1 )/block.y, 1);

    get_wave_slice_y <<< grid, block>>> (W, nxyz, j, hostptr[3]+skip);
    skip += 3 * ni * nk;
  }


  if (thisid[0] == dimx)
  {
    dim3 block(32, 32, 1);
    dim3 grid( (nj + block.x - 1 )/block.x, (nk + block.y - 1 )/block.y, 1);

    get_wave_slice_x <<< grid, block>>> (W, nxyz, i, hostptr[3]+skip);
    skip += 3 * nk * nj;
  }
  
  cudaMemcpy(hostptr[1], hostptr[3], countwave, cudaMemcpyDeviceToHost);


  // cudaStreamSynchronize(streams[0]);
  // cudaStreamSynchronize(streams[1]);
  // CUDACHECK(cudaStreamDestroy(streams[0]));
  // CUDACHECK(cudaStreamDestroy(streams[1]));

  return;
}

void nc_end_fault(ncFile nc)
{
  int err;
  nc_sync(nc.ncid);
  err = nc_close(nc.ncid);
  handle_err(err);
  return;
}
#endif
