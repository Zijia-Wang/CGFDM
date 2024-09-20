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

void __global__ pack_y(real_t *pack, Fault F, int j0, int nfault){
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  int ny = par.ny;
  int nz = par.nz;

  if(j < 3 && k < nz){
    pack[j + k * 3 + 0 * 3 * nz] = F.T11[(j0 + j) + k * ny + nfault * (7*ny*nz)];
    pack[j + k * 3 + 1 * 3 * nz] = F.T12[(j0 + j) + k * ny + nfault * (7*ny*nz)];
    pack[j + k * 3 + 2 * 3 * nz] = F.T13[(j0 + j) + k * ny + nfault * (7*ny*nz)];

    // F.W shape (ny, nz, 2, FSIZE) 1st dimension vary first
    for (int m = 0; m < 2*FSIZE; m++)
      pack[j + k * 3 + (m + 3) * 3 * nz] = F.W[(j0 + j) + k * ny + m * ny * nz + nfault * (ny*nz*FSIZE*2)];
  }
}

void __global__ pack_z(real_t *pack, Fault F, int k0, int nfault){
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  int ny = par.ny;
  int nz = par.nz;

  if(j < ny && k < 3){
    pack[j + k * ny + 0 * 3 * ny] = F.T11[j + (k + k0) * ny + nfault * (7*ny*nz)];
    pack[j + k * ny + 1 * 3 * ny] = F.T12[j + (k + k0) * ny + nfault * (7*ny*nz)];
    pack[j + k * ny + 2 * 3 * ny] = F.T13[j + (k + k0) * ny + nfault * (7*ny*nz)];

    // F.W shape (ny, nz, 2, FSIZE) 1st dimension vary first
    for (int m = 0; m < 2*FSIZE; m++)
      pack[j + k * ny + (m + 3) * 3 * ny] = F.W[j + (k + k0) * ny + m * ny * nz + nfault * (ny*nz*FSIZE*2)];
  }
}

void __global__ unpack_y(Fault F, real_t *pack, int j0, int nfault){
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  int ny = par.ny;
  int nz = par.nz;

  if(j < 3 && k < nz){
    F.T11[(j0 + j) + k * ny + nfault * (7*ny*nz)] = pack[j + k * 3 + 0  * 3 * nz];
    F.T12[(j0 + j) + k * ny + nfault * (7*ny*nz)] = pack[j + k * 3 + 1  * 3 * nz];
    F.T13[(j0 + j) + k * ny + nfault * (7*ny*nz)] = pack[j + k * 3 + 2  * 3 * nz];
    // F.W shape (ny, nz, 2, FSIZE) 1st dimension vary first
    for (int m = 0; m < 2*FSIZE; m++)
      F.W[(j0 + j) + k * ny + m * ny * nz + nfault * (ny*nz*FSIZE*2)] = pack[j + k * 3 + (m + 3) * 3 * nz];
  }
}

void __global__ unpack_z(Fault F, real_t *pack, int k0, int nfault){
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int k = threadIdx.y + blockIdx.y * blockDim.y;

  int nj = par.NY/par.PY;
  int nk = par.NZ/par.PZ;
  int ny = nj + 6;
  int nz = nk + 6;

  if(j < ny && k < 3){
    F.T11[j + (k + k0) * ny + nfault * (7*ny*nz)]              = pack[j + k * ny + 0  * 3 * ny]; 
    F.T12[j + (k + k0) * ny + nfault * (7*ny*nz)]              = pack[j + k * ny + 1  * 3 * ny]; 
    F.T13[j + (k + k0) * ny + nfault * (7*ny*nz)]              = pack[j + k * ny + 2  * 3 * ny]; 
    // F.W shape (ny, nz, 2, FSIZE) 1st dimension vary first
    for (int m = 0; m < 2*FSIZE; m++)
      F.W[j + (k + k0) * ny + m * ny * nz + nfault * (ny*nz*FSIZE*2)] = pack[j + k * ny + (m + 3) * 3 * ny];
  }
}

void exchange_fault(Fault F, int nfault){
  if(hostParams.PY * hostParams.PZ < 2) return;

  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int nj1 = 3; int nj2 = nj + 3;
  int nk1 = 3; int nk2 = nk + 3;
  //int ni1 = 3; int ni2 = ni + 3;
  //int count, stride, blocklen, size;
  //int pos_s, pos_d;
  int size;
  MPI_Status stat;

  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Y direction
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  size = (3 + 2 * FSIZE) * 3 * nz;
  dim3 blk2(3, 128, 1);
  dim3 grd2((3+blk2.x-1)/blk2.x, (nz+blk2.y-1)/blk2.y, 1);

  //>>>>>>>>>>>>>>>>>>>>>  send y0 and recv from y0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //pos_s = nj1; pos_d = nj2;
  // pack data
  if(MPI_PROC_NULL != neigyid[0]){
    pack_y<<<grd2,blk2>>>(fault_y_send0, F, nj1, nfault);
  }

  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());
  // send and recv data
  MPI_Sendrecv(fault_y_send0, size, MPI_REAL_T, neigyid[0], 221,
               fault_y_recv0, size, MPI_REAL_T, neigyid[1], 221,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigyid[1]){
    unpack_y<<<grd2, blk2>>>(F, fault_y_recv0, nj2, nfault);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send y1 and recv from y1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //pos_s = nj2-3; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigyid[1]){
    pack_y<<<grd2,blk2>>>(fault_y_send1, F, nj2-3, nfault);
  }

  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());
  // send and recv data
  MPI_Sendrecv(fault_y_send1, size, MPI_REAL_T, neigyid[1], 222,
               fault_y_recv1, size, MPI_REAL_T, neigyid[0], 222,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigyid[0]){
    unpack_y<<<grd2, blk2>>>(F, fault_y_recv1, 0, nfault);
  }

  cudaDeviceSynchronize();
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
  // Z direction
  //cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

  size = (3 + 2 * FSIZE) * ny * 3;
  dim3 blk3(128, 3, 1);
  dim3 grd3((ny+blk3.x-1)/blk3.x, (3+blk3.y-1)/blk3.y, 1);

  //>>>>>>>>>>>>>>>>>>>>>  send z0 and recv from z0 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //pos_s = nk1; pos_d = nk2;
  // pack data
  if(MPI_PROC_NULL != neigzid[0]){
    pack_z<<<grd3,blk3>>>(fault_z_send0, F, nk1, nfault);
  }

  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());
  // send and recv data
  MPI_Sendrecv(fault_z_send0, size, MPI_REAL_T, neigzid[0], 231,
               fault_z_recv0, size, MPI_REAL_T, neigzid[1], 231,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigzid[1]){
    unpack_z<<<grd3, blk3>>>(F, fault_z_recv0, nk2, nfault);
  }
  //>>>>>>>>>>>>>>>>>>>>>  send z1 and recv from z1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //pos_s = nk2-3; pos_d = 0;
  // pack data
  if(MPI_PROC_NULL != neigzid[1]){
    pack_z<<<grd3,blk3>>>(fault_z_send1, F, nk2-3, nfault);
  }

  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());
  // send and recv data
  MPI_Sendrecv(fault_z_send1, size, MPI_REAL_T, neigzid[1], 232,
               fault_z_recv1, size, MPI_REAL_T, neigzid[0], 232,
               SWMPI_COMM, &stat);
  cudaDeviceSynchronize();
  // unpack data
  if(MPI_PROC_NULL != neigzid[0]){
    unpack_z<<<grd3, blk3>>>(F, fault_z_recv1, 0, nfault);
  }

/*
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
*/
  return;
}
