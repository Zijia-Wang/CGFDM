#ifndef PARAMS_H
#define PARAMS_H

#include "common.h"
#include <mpi.h>

// 引用性声明
//extern __device__ __constant__ int Flags[8][3];
extern Params hostParams;
extern __device__ __constant__ Params par;
// extern int * Device_Faultgrid;
extern int masternode, freenode, absnode;
// extern int faultnode;
extern int thisid[3];
extern int neigxid[2], neigyid[2], neigzid[2];

extern char PARAMS_FILE[1000];
extern char OUT[1000];
extern char Fault_geometry[1000];
extern char Fault_init_stress[1000];
extern char Media1D[1000];
extern char Media3D[1000];
extern int flag_media1d;
extern int flag_media3d;

extern MPI_Comm SWMPI_COMM;
extern realptr_t wave_yz_send0, wave_yz_recv0;
extern realptr_t wave_yz_send1, wave_yz_recv1;
extern realptr_t wave_xz_send0, wave_xz_recv0;
extern realptr_t wave_xz_send1, wave_xz_recv1;
extern realptr_t wave_xy_send0, wave_xy_recv0;
extern realptr_t wave_xy_send1, wave_xy_recv1;

extern realptr_t fault_z_send0, fault_z_recv0;
extern realptr_t fault_z_send1, fault_z_recv1;
extern realptr_t fault_y_send0, fault_y_recv0;
extern realptr_t fault_y_send1, fault_y_recv1;

#endif
