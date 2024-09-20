/*
 *******************************************************************************
 *                                                                             *
 *  modules for Seismic Wave Message Passing Interface (SWMPI)                 *
 *                                                                             *
 *******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
//#include <mpi.h>


//  packed into continuous data
//
//  OOOOOOOOOOOOOOO_____OOOOOOOOOOOOOOO_____OOOOOOOOOOOOOOO____OOO......
//  |\____________/    |
//  |      |           |
//   \   blocklen     /
//    \______________/
//            |
//         stride

void __global__ pack(real_t *wp, real_t *w, int count, int blocklen, int stride){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int ncb = count * blocklen;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  int nxyz = nx * ny * nz;
  if(j < count && i < blocklen){
    wp[i + j * blocklen + 0 * ncb] = w[i + j * stride + 0 * nxyz];
    wp[i + j * blocklen + 1 * ncb] = w[i + j * stride + 1 * nxyz];
    wp[i + j * blocklen + 2 * ncb] = w[i + j * stride + 2 * nxyz];
    wp[i + j * blocklen + 3 * ncb] = w[i + j * stride + 3 * nxyz];
    wp[i + j * blocklen + 4 * ncb] = w[i + j * stride + 4 * nxyz];
    wp[i + j * blocklen + 5 * ncb] = w[i + j * stride + 5 * nxyz];
    wp[i + j * blocklen + 6 * ncb] = w[i + j * stride + 6 * nxyz];
    wp[i + j * blocklen + 7 * ncb] = w[i + j * stride + 7 * nxyz];
    wp[i + j * blocklen + 8 * ncb] = w[i + j * stride + 8 * nxyz];
  }
}

void pack_host(real_t *wp, real_t *w, int count, int blocklen, int stride, int arrsize){
  int ncb = count * blocklen;
  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;
  int nxyz = nx * ny * nz;
  for (int l = 0; l < arrsize; l++){
    for (int j = 0; j < count; j++){
      for (int i = 0; i < blocklen; i++){
        wp[i + j * blocklen + l * ncb] = w[i + j * stride + l * nxyz];
      }
    }
  }
  return;
}

void __global__ unpack(real_t *w, real_t *wp, int count, int blocklen, int stride){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int ncb = count * blocklen;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  int nxyz = nx * ny * nz;
  if(j < count && i < blocklen){
    w[i + j * stride + 0 * nxyz] = wp[i + j * blocklen + 0 * ncb];
    w[i + j * stride + 1 * nxyz] = wp[i + j * blocklen + 1 * ncb];
    w[i + j * stride + 2 * nxyz] = wp[i + j * blocklen + 2 * ncb];
    w[i + j * stride + 3 * nxyz] = wp[i + j * blocklen + 3 * ncb];
    w[i + j * stride + 4 * nxyz] = wp[i + j * blocklen + 4 * ncb];
    w[i + j * stride + 5 * nxyz] = wp[i + j * blocklen + 5 * ncb];
    w[i + j * stride + 6 * nxyz] = wp[i + j * blocklen + 6 * ncb];
    w[i + j * stride + 7 * nxyz] = wp[i + j * blocklen + 7 * ncb];
    w[i + j * stride + 8 * nxyz] = wp[i + j * blocklen + 8 * ncb];
  }
}

void unpack_host(real_t *w, real_t *wp, int count, int blocklen, int stride, int arrsize){
  int ncb = count * blocklen;
  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;
  int nxyz = nx * ny * nz;
  for (int l = 0; l < arrsize; l++){
    for (int j = 0; j < count; j++){
      for (int i = 0; i < blocklen; i++){
        w[i + j * stride + l * nxyz] = wp[i + j * blocklen + l * ncb];
      }
    }
  }
  return;
}

void exchange_wave(real_t *wave)
{
  //return ;
  if(hostParams.PX * hostParams.PY * hostParams.PZ < 2) return;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int nj1 = 3; int nj2 = nj + 3;
  int nk1 = 3; int nk2 = nk + 3;
  int ni1 = 3; int ni2 = ni + 3;
  int count, stride, blocklen, size;
  int pos_s, pos_d;
  MPI_Status stat;

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at x plane (Y-O-Z)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  size = 3*ny*nz*sizeof(real_t);

  //>>>>>>>>>>>>>>>>>>>>>  send x0 and recv from x0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = ni1 * ny * nz; pos_d = ni2 * ny * nz;
  // pack data
  if(MPI_PROC_NULL != neigxid[0]){
    cudaMemcpy(wave_yz_send0 + 0*3*ny*nz, wave + pos_s + 0*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 1*3*ny*nz, wave + pos_s + 1*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 2*3*ny*nz, wave + pos_s + 2*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 3*3*ny*nz, wave + pos_s + 3*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 4*3*ny*nz, wave + pos_s + 4*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 5*3*ny*nz, wave + pos_s + 5*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 6*3*ny*nz, wave + pos_s + 6*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 7*3*ny*nz, wave + pos_s + 7*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send0 + 8*3*ny*nz, wave + pos_s + 8*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
  }
  // send and recv data
  MPI_Sendrecv(wave_yz_send0, 3*nz*ny*WSIZE, MPI_REAL_T, neigxid[0], 111,
               wave_yz_recv0, 3*nz*ny*WSIZE, MPI_REAL_T, neigxid[1], 111,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigxid[1]){
    cudaMemcpy(wave + pos_d + 0*nx*ny*nz, wave_yz_recv0 + 0*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 1*nx*ny*nz, wave_yz_recv0 + 1*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 2*nx*ny*nz, wave_yz_recv0 + 2*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 3*nx*ny*nz, wave_yz_recv0 + 3*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 4*nx*ny*nz, wave_yz_recv0 + 4*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 5*nx*ny*nz, wave_yz_recv0 + 5*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 6*nx*ny*nz, wave_yz_recv0 + 6*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 7*nx*ny*nz, wave_yz_recv0 + 7*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 8*nx*ny*nz, wave_yz_recv0 + 8*3*ny*nz, size, cudaMemcpyDeviceToDevice);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send y1 and recv from y1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = (ni2-3)*ny*nz; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigxid[1]){
    cudaMemcpy(wave_yz_send1 + 0*3*ny*nz, wave + pos_s + 0*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 1*3*ny*nz, wave + pos_s + 1*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 2*3*ny*nz, wave + pos_s + 2*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 3*3*ny*nz, wave + pos_s + 3*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 4*3*ny*nz, wave + pos_s + 4*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 5*3*ny*nz, wave + pos_s + 5*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 6*3*ny*nz, wave + pos_s + 6*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 7*3*ny*nz, wave + pos_s + 7*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave_yz_send1 + 8*3*ny*nz, wave + pos_s + 8*nx*ny*nz, size, cudaMemcpyDeviceToDevice);
  }
  // send and recv data
  MPI_Sendrecv(wave_yz_send1, 3*nz*ny*WSIZE, MPI_REAL_T, neigxid[1], 112,
               wave_yz_recv1, 3*nz*ny*WSIZE, MPI_REAL_T, neigxid[0], 112,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigxid[0]){
    cudaMemcpy(wave + pos_d + 0*nx*ny*nz, wave_yz_recv1 + 0*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 1*nx*ny*nz, wave_yz_recv1 + 1*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 2*nx*ny*nz, wave_yz_recv1 + 2*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 3*nx*ny*nz, wave_yz_recv1 + 3*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 4*nx*ny*nz, wave_yz_recv1 + 4*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 5*nx*ny*nz, wave_yz_recv1 + 5*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 6*nx*ny*nz, wave_yz_recv1 + 6*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 7*nx*ny*nz, wave_yz_recv1 + 7*3*ny*nz, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(wave + pos_d + 8*nx*ny*nz, wave_yz_recv1 + 8*3*ny*nz, size, cudaMemcpyDeviceToDevice);
  }
  // unpack data

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at y plane (X-O-Z)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  blocklen = 3;
  count = nx * nz;
  stride = ny;
  size = blocklen * count * WSIZE;
  dim3 blk2(3, 128, 1);
  dim3 grd2((blocklen+blk2.x-1)/blk2.x, (count+blk2.y-1)/blk2.y, 1);

  //>>>>>>>>>>>>>>>>>>>>>  send y0 and recv from y0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nj1; pos_d = nj2;
  // pack data
  if(MPI_PROC_NULL != neigyid[0]){
    pack<<<grd2,blk2>>>(wave_xz_send0, wave + pos_s, count, blocklen, stride);
  }

  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());
  // send and recv data
  MPI_Sendrecv(wave_xz_send0, size, MPI_REAL_T, neigyid[0], 121,
               wave_xz_recv0, size, MPI_REAL_T, neigyid[1], 121,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigyid[1]){
    unpack<<<grd2, blk2>>>(wave + pos_d, wave_xz_recv0, count, blocklen, stride);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send y1 and recv from y1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nj2-3; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigyid[1]){
    pack<<<grd2, blk2>>>(wave_xz_send1, wave + pos_s, count, blocklen, stride);
  }
  cudaDeviceSynchronize();
  // send and recv data
  MPI_Sendrecv(wave_xz_send1, size, MPI_REAL_T, neigyid[1], 122,
               wave_xz_recv1, size, MPI_REAL_T, neigyid[0], 122,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigyid[0]){
    unpack<<<grd2, blk2>>>(wave + pos_d, wave_xz_recv1, count, blocklen, stride);
  }

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at z plane (X-O-Y)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  blocklen = ny * 3;
  count = nx;
  stride = ny * nz;
  size = blocklen * count * WSIZE;
  dim3 blk3(16, 16, 1);
  dim3 grd3((blocklen+blk3.x-1)/blk3.x, (count+blk3.y-1)/blk3.y, 1);

  //>>>>>>>>>>>>>>>>>>>>>  send z0 and recv from z0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nk1 * ny; pos_d = nk2 * ny;
  // pack data
  if(MPI_PROC_NULL != neigzid[0]){
    pack<<<grd3,blk3>>>(wave_xy_send0, wave + pos_s, count, blocklen, stride);
  }
  cudaDeviceSynchronize();
  // send and recv data
  MPI_Sendrecv(wave_xy_send0, size, MPI_REAL_T, neigzid[0], 131,
               wave_xy_recv0, size, MPI_REAL_T, neigzid[1], 131,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigzid[1]){
    unpack<<<grd3, blk3>>>(wave + pos_d, wave_xy_recv0, count, blocklen, stride);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send z1 and recv from z1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = (nk2-3) * ny; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigzid[1]){
    pack<<<grd3, blk3>>>(wave_xy_send1, wave + pos_s, count, blocklen, stride);
  }
  cudaDeviceSynchronize();
  // send and recv data
  MPI_Sendrecv(wave_xy_send1, size, MPI_REAL_T, neigzid[1], 132,
               wave_xy_recv1, size, MPI_REAL_T, neigzid[0], 132,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigzid[0]){
    unpack<<<grd3, blk3>>>(wave + pos_d, wave_xy_recv1, count, blocklen, stride);
  }

  return;
}

/* this module is used to exchange the boundary points of coord and metric,
   and will only be excuted once
   e.g.
   exchange_array(coord, coordsize);
   exchange_metric(metric, metricsize);
   */
void exchange_array(real_t *arr, int arrsize)
{
  //return ;
  if(hostParams.PX * hostParams.PY * hostParams.PZ < 2) return;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int nj1 = 3; int nj2 = nj + 3;
  int nk1 = 3; int nk2 = nk + 3;
  int ni1 = 3; int ni2 = ni + 3;
  int count, stride, blocklen, size;
  int pos_s, pos_d;
  MPI_Status stat;

  real_t * arr_yz_send0 = (real_t *) malloc(sizeof(real_t)*ny*nz*3*arrsize);
  real_t * arr_yz_send1 = (real_t *) malloc(sizeof(real_t)*ny*nz*3*arrsize);
  real_t * arr_yz_recv0 = (real_t *) malloc(sizeof(real_t)*ny*nz*3*arrsize);
  real_t * arr_yz_recv1 = (real_t *) malloc(sizeof(real_t)*ny*nz*3*arrsize);
  real_t * arr_xz_send0 = (real_t *) malloc(sizeof(real_t)*nx*nz*3*arrsize);
  real_t * arr_xz_send1 = (real_t *) malloc(sizeof(real_t)*nx*nz*3*arrsize);
  real_t * arr_xz_recv0 = (real_t *) malloc(sizeof(real_t)*nx*nz*3*arrsize);
  real_t * arr_xz_recv1 = (real_t *) malloc(sizeof(real_t)*nx*nz*3*arrsize);
  real_t * arr_xy_send0 = (real_t *) malloc(sizeof(real_t)*nx*ny*3*arrsize);
  real_t * arr_xy_send1 = (real_t *) malloc(sizeof(real_t)*nx*ny*3*arrsize);
  real_t * arr_xy_recv0 = (real_t *) malloc(sizeof(real_t)*nx*ny*3*arrsize);
  real_t * arr_xy_recv1 = (real_t *) malloc(sizeof(real_t)*nx*ny*3*arrsize);

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at x plane (Y-O-Z)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  size = 3*ny*nz*sizeof(real_t);

  //>>>>>>>>>>>>>>>>>>>>>  send x0 and recv from x0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = ni1 * ny * nz; pos_d = ni2 * ny * nz;
  // pack data
  if(MPI_PROC_NULL != neigxid[0]){
    for (int i = 0; i < arrsize; i++){
      memcpy(arr_yz_send0 + i*3*ny*nz, arr + pos_s + i*nx*ny*nz, size);
    }
  }
  // send and recv data
  MPI_Sendrecv(arr_yz_send0, 3*nz*ny*arrsize, MPI_REAL_T, neigxid[0], 111,
               arr_yz_recv0, 3*nz*ny*arrsize, MPI_REAL_T, neigxid[1], 111,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigxid[1]){
    for (int i = 0; i < arrsize; i++){
      memcpy(arr + pos_d + i*nx*ny*nz, arr_yz_recv0 + i*3*ny*nz, size);
    }
  }
  //>>>>>>>>>>>>>>>>>>>>>  send y1 and recv from y1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = (ni2-3)*ny*nz; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigxid[1]){
    for (int i = 0; i < arrsize; i++){
      memcpy(arr_yz_send1 + i*3*ny*nz, arr + pos_s + i*nx*ny*nz, size);
    }
  }
  // send and recv data
  MPI_Sendrecv(arr_yz_send1, 3*nz*ny*arrsize, MPI_REAL_T, neigxid[1], 112,
               arr_yz_recv1, 3*nz*ny*arrsize, MPI_REAL_T, neigxid[0], 112,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigxid[0]){
    for (int i = 0; i < arrsize; i++){
      memcpy(arr + pos_d + i*nx*ny*nz, arr_yz_recv1 + i*3*ny*nz, size);
    }
  }

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at y plane (X-O-Z)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  blocklen = 3;
  count = nx * nz;
  stride = ny;
  size = blocklen * count * arrsize;

  //>>>>>>>>>>>>>>>>>>>>>  send y0 and recv from y0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nj1; pos_d = nj2;
  // pack data
  if(MPI_PROC_NULL != neigyid[0]){
    pack_host(arr_xz_send0, arr + pos_s, count, blocklen, stride, arrsize);
  }

  // send and recv data
  MPI_Sendrecv(arr_xz_send0, size, MPI_REAL_T, neigyid[0], 121,
               arr_xz_recv0, size, MPI_REAL_T, neigyid[1], 121,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigyid[1]){
    unpack_host(arr + pos_d, arr_xz_recv0, count, blocklen, stride, arrsize);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send y1 and recv from y1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nj2-3; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigyid[1]){
    pack_host(arr_xz_send1, arr + pos_s, count, blocklen, stride, arrsize);
  }
  // send and recv data
  MPI_Sendrecv(arr_xz_send1, size, MPI_REAL_T, neigyid[1], 122,
               arr_xz_recv1, size, MPI_REAL_T, neigyid[0], 122,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigyid[0]){
    unpack_host(arr + pos_d, arr_xz_recv1, count, blocklen, stride, arrsize);
  }

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Exchange at z plane (X-O-Y)
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  blocklen = ny * 3;
  count = nx;
  stride = ny * nz;
  size = blocklen * count * arrsize;

  //>>>>>>>>>>>>>>>>>>>>>  send z0 and recv from z0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = nk1 * ny; pos_d = nk2 * ny;
  // pack data
  if(MPI_PROC_NULL != neigzid[0]){
    pack_host(arr_xy_send0, arr + pos_s, count, blocklen, stride, arrsize);
  }
  // send and recv data
  MPI_Sendrecv(arr_xy_send0, size, MPI_REAL_T, neigzid[0], 131,
               arr_xy_recv0, size, MPI_REAL_T, neigzid[1], 131,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigzid[1]){
    unpack_host(arr + pos_d, arr_xy_recv0, count, blocklen, stride, arrsize);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send z1 and recv from z1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  pos_s = (nk2-3) * ny; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigzid[1]){
    pack_host(arr_xy_send1, arr + pos_s, count, blocklen, stride, arrsize);
  }
  // send and recv data
  MPI_Sendrecv(arr_xy_send1, size, MPI_REAL_T, neigzid[1], 132,
               arr_xy_recv1, size, MPI_REAL_T, neigzid[0], 132,
               SWMPI_COMM, &stat);
  // unpack data
  if(MPI_PROC_NULL != neigzid[0]){
    unpack_host(arr + pos_d, arr_xy_recv1, count, blocklen, stride, arrsize);
  }

  return;
}
