#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"

void *cuda_malloc(size_t len){
  void *p;
  const cudaError_t err = cudaMalloc(&p, len);
  if (cudaSuccess == err) return p;
  //fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
  fprintf(stderr, "Error @ %s, ", __FILE__);
  fprintf(stderr, "code: %d, reson: %s\n", err, cudaGetErrorString(err));
  return 0;
}

void alloc_wave(Wave *W){
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  W-> W = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*nz*WSIZE);
  W->hW = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*nz*WSIZE);
  W->mW = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*nz*WSIZE);
  W->tW = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*nz*WSIZE);
  W->matVx2Vz = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*3*3);
  W->matVy2Vz = (real_t *) cuda_malloc(sizeof(real_t)*nx*ny*3*3);

  cudaMemset(W-> W, 0, sizeof(real_t)*nx*ny*nz*WSIZE);
  cudaMemset(W->hW, 0, sizeof(real_t)*nx*ny*nz*WSIZE);
  cudaMemset(W->mW, 0, sizeof(real_t)*nx*ny*nz*WSIZE);
  cudaMemset(W->tW, 0, sizeof(real_t)*nx*ny*nz*WSIZE);

  return;
}

void dealloc_wave(Wave W){
  CUDACHECK(cudaFree(W.W));
  CUDACHECK(cudaFree(W.hW));
  CUDACHECK(cudaFree(W.mW));
  CUDACHECK(cudaFree(W.tW));
  CUDACHECK(cudaFree(W.matVx2Vz));
  CUDACHECK(cudaFree(W.matVy2Vz));
  return;
}

void alloc_wave_halo(){
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  CUDACHECK(cudaMalloc((real_t **) &wave_yz_send0, sizeof(real_t)*ny*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_yz_send1, sizeof(real_t)*ny*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_yz_recv0, sizeof(real_t)*ny*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_yz_recv1, sizeof(real_t)*ny*nz*3*WSIZE));

  CUDACHECK(cudaMalloc((real_t **) &wave_xz_send0, sizeof(real_t)*nx*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xz_send1, sizeof(real_t)*nx*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xz_recv0, sizeof(real_t)*nx*nz*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xz_recv1, sizeof(real_t)*nx*nz*3*WSIZE));

  CUDACHECK(cudaMalloc((real_t **) &wave_xy_send0, sizeof(real_t)*nx*ny*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xy_send1, sizeof(real_t)*nx*ny*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xy_recv0, sizeof(real_t)*nx*ny*3*WSIZE));
  CUDACHECK(cudaMalloc((real_t **) &wave_xy_recv1, sizeof(real_t)*nx*ny*3*WSIZE));

  //int size = 3 + 2 * FSIZE;
  int size = 7 + 2 * FSIZE;
  CUDACHECK(cudaMalloc((real_t **) &fault_z_send0, sizeof(real_t)*ny*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_z_send1, sizeof(real_t)*ny*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_z_recv0, sizeof(real_t)*ny*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_z_recv1, sizeof(real_t)*ny*3*size));

  CUDACHECK(cudaMalloc((real_t **) &fault_y_send0, sizeof(real_t)*nz*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_y_send1, sizeof(real_t)*nz*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_y_recv0, sizeof(real_t)*nz*3*size));
  CUDACHECK(cudaMalloc((real_t **) &fault_y_recv1, sizeof(real_t)*nz*3*size));

  return;
}

void alloc_fault(Fault *F){
  //int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int num_fault = hostParams.num_fault;         //*********wangzj
  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int TP_n = hostParams.TP_n;

  F-> W = (real_t *) cuda_malloc(sizeof(real_t)*2*ny*nz*FSIZE*num_fault);
  F->Ws = (real_t *) cuda_malloc(sizeof(real_t)*2*ny*nz*FSIZE*num_fault);
  F->mW = (real_t *) cuda_malloc(sizeof(real_t)*2*ny*nz*FSIZE*num_fault);
  F->hW = (real_t *) cuda_malloc(sizeof(real_t)*2*ny*nz*FSIZE*num_fault);
  F->tW = (real_t *) cuda_malloc(sizeof(real_t)*2*ny*nz*FSIZE*num_fault);
  F->T11 = (real_t *) cuda_malloc(sizeof(real_t)*7*ny*nz*num_fault);
  F->T12 = (real_t *) cuda_malloc(sizeof(real_t)*7*ny*nz*num_fault);
  F->T13 = (real_t *) cuda_malloc(sizeof(real_t)*7*ny*nz*num_fault);
  F->mT11 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->mT12 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->mT13 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->hT11 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->hT12 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->hT13 = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  // F->T11S = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  // F->T12S = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  // F->T13S = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);


  //if (hostParams.Friction_type == 3){
    F->TP_T  = (real_t *) cuda_malloc(sizeof(real_t)*TP_n*nj*nk*num_fault);
    F->TP_P  = (real_t *) cuda_malloc(sizeof(real_t)*TP_n*nj*nk*num_fault);
    F->TP_dT = (real_t *) cuda_malloc(sizeof(real_t)*TP_n*nj*nk*num_fault);
    F->TP_dP = (real_t *) cuda_malloc(sizeof(real_t)*TP_n*nj*nk*num_fault);
    F->TP_hy = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
    cudaMemset(F->TP_T, 0, sizeof(real_t)*TP_n*nj*nk*num_fault);
    cudaMemset(F->TP_P, 0, sizeof(real_t)*TP_n*nj*nk*num_fault);
  //}
  F->friction = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);

  // for output
  F->str_init_x = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->str_init_y = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->str_init_z = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->T0x        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->T0y        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->T0z        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->dT0x       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->dT0y       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->dT0z       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Tn         = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Ts1        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Ts2        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->tTn        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->tTs1       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->tTs2       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->a          = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->b          = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->L          = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Vw         = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->str_peak   = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->mu_d       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Dc         = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->C0         = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->slip       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->hslip      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->mslip      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->tslip      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->slip1      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault); // wangzj ****
  F->slip2      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);  // ***
  F->Vs1        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->Vs2        = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->rake       = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault); // wangzj ****
  F->State      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->mState     = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->hState     = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->tState     = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);

  F->rup_sensor_Dy    = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->rup_sensor_Dz    = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);
  F->rup_sensor      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);

  F->united   = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->faultgrid   = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->rup_index_y  = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->rup_index_z  = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->flag_rup     = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->first_rup    = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->init_t0_flag = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->smooth_flag = (int *)   cuda_malloc(sizeof(int)*nj*nk*num_fault);
  F->init_t0      = (real_t *) cuda_malloc(sizeof(real_t)*nj*nk*num_fault);

  cudaMemset(F->init_t0_flag, 0, sizeof(int)  *nj*nk*num_fault);
  cudaMemset(F->init_t0,      0, sizeof(real_t)*nj*nk*num_fault);
  cudaMemset(F->slip,         0, sizeof(real_t)*nj*nk*num_fault);
  cudaMemset(F->hslip,        0, sizeof(real_t)*nj*nk*num_fault);
  cudaMemset(F->slip1,        0, sizeof(real_t)*nj*nk*num_fault); // wzj********
  cudaMemset(F->slip2,        0, sizeof(real_t)*nj*nk*num_fault); 
  cudaMemset(F->rake,         0, sizeof(real_t)*nj*nk*num_fault); // ******

  return;
}

void dealloc_fault(Fault F){
  CUDACHECK(cudaFree(F.W           ));
  CUDACHECK(cudaFree(F.Ws          ));
  CUDACHECK(cudaFree(F.mW          ));
  CUDACHECK(cudaFree(F.hW          ));
  CUDACHECK(cudaFree(F.tW          ));
  CUDACHECK(cudaFree(F.T11         ));
  CUDACHECK(cudaFree(F.T12         ));
  CUDACHECK(cudaFree(F.T13         ));
  CUDACHECK(cudaFree(F.str_init_x  ));
  CUDACHECK(cudaFree(F.str_init_y  ));
  CUDACHECK(cudaFree(F.str_init_z  ));
  CUDACHECK(cudaFree(F.Tn          ));
  CUDACHECK(cudaFree(F.Ts1         ));
  CUDACHECK(cudaFree(F.Ts2         ));
  CUDACHECK(cudaFree(F.tTn         ));
  CUDACHECK(cudaFree(F.tTs1        ));
  CUDACHECK(cudaFree(F.tTs2        ));
  CUDACHECK(cudaFree(F.a           ));
  CUDACHECK(cudaFree(F.b           ));
  CUDACHECK(cudaFree(F.str_peak    ));
  CUDACHECK(cudaFree(F.C0          ));
  CUDACHECK(cudaFree(F.slip        ));
  CUDACHECK(cudaFree(F.hslip       ));
  CUDACHECK(cudaFree(F.slip1       )); // wzj******
  CUDACHECK(cudaFree(F.slip2       )); // ****
  CUDACHECK(cudaFree(F.Vs1         ));
  CUDACHECK(cudaFree(F.Vs2         ));
  CUDACHECK(cudaFree(F.rake        )); // wzj******
  CUDACHECK(cudaFree(F.State       ));
  CUDACHECK(cudaFree(F.mState      ));
  CUDACHECK(cudaFree(F.hState      ));
  CUDACHECK(cudaFree(F.tState      ));
  CUDACHECK(cudaFree(F.rup_index_y ));
  CUDACHECK(cudaFree(F.rup_index_z ));
  CUDACHECK(cudaFree(F.flag_rup    ));
  CUDACHECK(cudaFree(F.first_rup   ));
  CUDACHECK(cudaFree(F.init_t0_flag));
  CUDACHECK(cudaFree(F.init_t0     ));
  return;
}

void alloc_fault_coef(Fault *FC){
  //int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int num_fault = hostParams.num_fault;         //*********wangzj
  //int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  FC->rho_f = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*2*num_fault);
  FC->mu_f  = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*2*num_fault);
  FC->lam_f = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*2*num_fault);

  FC->D11_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D12_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D13_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D21_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D22_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D23_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D31_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D32_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D33_1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);

  FC->D11_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D12_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D13_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D21_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D22_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D23_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D31_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D32_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->D33_2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);

  FC->matMin2Plus1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matMin2Plus2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matMin2Plus3 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matMin2Plus4 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matMin2Plus5 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);

  FC->matPlus2Min1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matPlus2Min2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matPlus2Min3 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matPlus2Min4 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matPlus2Min5 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);

  FC->matT1toVxm = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matVytoVxm = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matVztoVxm = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matT1toVxp = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matVytoVxp = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);
  FC->matVztoVxp = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*3*num_fault);

  FC->vec_n  = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*num_fault);
  FC->vec_s1 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*num_fault);
  FC->vec_s2 = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*3*num_fault);
  FC->x_et   = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*num_fault);
  FC->y_et   = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*num_fault);
  FC->z_et   = (real_t *) cuda_malloc(sizeof(real_t)*ny*nz*num_fault);

  FC->matVx2Vz1     = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVy2Vz1     = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVx2Vz2     = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVy2Vz2     = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVx1_free   = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVy1_free   = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVx2_free   = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVy2_free   = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matPlus2Min1f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matPlus2Min2f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matPlus2Min3f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matMin2Plus1f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matMin2Plus2f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matMin2Plus3f = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);

  FC->matT1toVxfm = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVytoVxfm = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matT1toVxfp = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);
  FC->matVytoVxfp = (real_t *) cuda_malloc(sizeof(real_t)*ny*3*3*num_fault);

  return;
}

void dealloc_fault_coef(Fault FC){
  CUDACHECK(cudaFree( FC.rho_f         ));
  CUDACHECK(cudaFree( FC.mu_f          ));
  CUDACHECK(cudaFree( FC.lam_f         ));
  CUDACHECK(cudaFree( FC.D21_1         ));
  CUDACHECK(cudaFree( FC.D22_1         ));
  CUDACHECK(cudaFree( FC.D23_1         ));
  CUDACHECK(cudaFree( FC.D31_1         ));
  CUDACHECK(cudaFree( FC.D32_1         ));
  CUDACHECK(cudaFree( FC.D33_1         ));
  CUDACHECK(cudaFree( FC.D21_2         ));
  CUDACHECK(cudaFree( FC.D22_2         ));
  CUDACHECK(cudaFree( FC.D23_2         ));
  CUDACHECK(cudaFree( FC.D31_2         ));
  CUDACHECK(cudaFree( FC.D32_2         ));
  CUDACHECK(cudaFree( FC.D33_2         ));
  CUDACHECK(cudaFree( FC.matMin2Plus1  ));
  CUDACHECK(cudaFree( FC.matMin2Plus2  ));
  CUDACHECK(cudaFree( FC.matMin2Plus3  ));
  CUDACHECK(cudaFree( FC.matMin2Plus4  ));
  CUDACHECK(cudaFree( FC.matMin2Plus5  ));
  CUDACHECK(cudaFree( FC.matPlus2Min1  ));
  CUDACHECK(cudaFree( FC.matPlus2Min2  ));
  CUDACHECK(cudaFree( FC.matPlus2Min3  ));
  CUDACHECK(cudaFree( FC.matPlus2Min4  ));
  CUDACHECK(cudaFree( FC.matPlus2Min5  ));
  CUDACHECK(cudaFree( FC.vec_n         ));
  CUDACHECK(cudaFree( FC.vec_s1        ));
  CUDACHECK(cudaFree( FC.vec_s2        ));
  CUDACHECK(cudaFree( FC.x_et          ));
  CUDACHECK(cudaFree( FC.y_et          ));
  CUDACHECK(cudaFree( FC.z_et          ));
  CUDACHECK(cudaFree( FC.matVx2Vz1     ));
  CUDACHECK(cudaFree( FC.matVy2Vz1     ));
  CUDACHECK(cudaFree( FC.matVx2Vz2     ));
  CUDACHECK(cudaFree( FC.matVy2Vz2     ));
  CUDACHECK(cudaFree( FC.matVx1_free   ));
  CUDACHECK(cudaFree( FC.matVy1_free   ));
  CUDACHECK(cudaFree( FC.matVx2_free   ));
  CUDACHECK(cudaFree( FC.matVy2_free   ));
  CUDACHECK(cudaFree( FC.matPlus2Min1f ));
  CUDACHECK(cudaFree( FC.matPlus2Min2f ));
  CUDACHECK(cudaFree( FC.matPlus2Min3f ));
  CUDACHECK(cudaFree( FC.matMin2Plus1f ));
  CUDACHECK(cudaFree( FC.matMin2Plus2f ));
  CUDACHECK(cudaFree( FC.matMin2Plus3f ));
  return;
}

void alloc_pml_host (PML *P){
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  P->Ax = (real_t*)malloc(sizeof(real_t)*ni);
  P->Bx = (real_t*)malloc(sizeof(real_t)*ni);
  P->Dx = (real_t*)malloc(sizeof(real_t)*ni);
  P->Ay = (real_t*)malloc(sizeof(real_t)*nj);
  P->By = (real_t*)malloc(sizeof(real_t)*nj);
  P->Dy = (real_t*)malloc(sizeof(real_t)*nj);
  P->Az = (real_t*)malloc(sizeof(real_t)*nk);
  P->Bz = (real_t*)malloc(sizeof(real_t)*nk);
  P->Dz = (real_t*)malloc(sizeof(real_t)*nk);
  return;
}

void dealloc_pml_host (PML P){
  free(P.Ax);
  free(P.Bx);
  free(P.Dx);
  free(P.Ay);
  free(P.By);
  free(P.Dy);
  free(P.Az);
  free(P.Bz);
  free(P.Dz);
  return;
}

void alloc_pml (PML *P){
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  int N = hostParams.PML_N;
  P->Ax = (real_t *) cuda_malloc(sizeof(real_t)*ni);
  P->Bx = (real_t *) cuda_malloc(sizeof(real_t)*ni);
  P->Dx = (real_t *) cuda_malloc(sizeof(real_t)*ni);
  P->Ay = (real_t *) cuda_malloc(sizeof(real_t)*nj);
  P->By = (real_t *) cuda_malloc(sizeof(real_t)*nj);
  P->Dy = (real_t *) cuda_malloc(sizeof(real_t)*nj);
  P->Az = (real_t *) cuda_malloc(sizeof(real_t)*nk);
  P->Bz = (real_t *) cuda_malloc(sizeof(real_t)*nk);
  P->Dz = (real_t *) cuda_malloc(sizeof(real_t)*nk);

  if(P->isx1) {
    size_t ibytes = N*nj*nk*WSIZE*sizeof(real_t);
    P-> Wx1 = (real_t *) cuda_malloc(ibytes);
    P->hWx1 = (real_t *) cuda_malloc(ibytes);
    P->mWx1 = (real_t *) cuda_malloc(ibytes);
    P->tWx1 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wx1, 0, ibytes);
    cudaMemset(P->hWx1, 0, ibytes);
    cudaMemset(P->mWx1, 0, ibytes);
    cudaMemset(P->tWx1, 0, ibytes);
  }
  if(P->isx2){
    size_t ibytes = N*nj*nk*WSIZE*sizeof(real_t);
    P-> Wx2 = (real_t *) cuda_malloc(ibytes);
    P->hWx2 = (real_t *) cuda_malloc(ibytes);
    P->mWx2 = (real_t *) cuda_malloc(ibytes);
    P->tWx2 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wx2, 0, ibytes);
    cudaMemset(P->hWx2, 0, ibytes);
    cudaMemset(P->mWx2, 0, ibytes);
    cudaMemset(P->tWx2, 0, ibytes);
  }
  if(P->isy1) {
    size_t ibytes = N*nk*ni*WSIZE*sizeof(real_t);
    P-> Wy1 = (real_t *) cuda_malloc(ibytes);
    P->hWy1 = (real_t *) cuda_malloc(ibytes);
    P->mWy1 = (real_t *) cuda_malloc(ibytes);
    P->tWy1 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wy1, 0, ibytes);
    cudaMemset(P->hWy1, 0, ibytes);
    cudaMemset(P->mWy1, 0, ibytes);
    cudaMemset(P->tWy1, 0, ibytes);
  }
  if(P->isy2){
    size_t ibytes = N*nk*ni*WSIZE*sizeof(real_t);
    P-> Wy2 = (real_t *) cuda_malloc(ibytes);
    P->hWy2 = (real_t *) cuda_malloc(ibytes);
    P->mWy2 = (real_t *) cuda_malloc(ibytes);
    P->tWy2 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wy2, 0, ibytes);
    cudaMemset(P->hWy2, 0, ibytes);
    cudaMemset(P->mWy2, 0, ibytes);
    cudaMemset(P->tWy2, 0, ibytes);
  }
  if(P->isz1) {
    size_t ibytes = N*ni*nj*WSIZE*sizeof(real_t);
    P-> Wz1 = (real_t *) cuda_malloc(ibytes);
    P->hWz1 = (real_t *) cuda_malloc(ibytes);
    P->mWz1 = (real_t *) cuda_malloc(ibytes);
    P->tWz1 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wz1, 0, ibytes);
    cudaMemset(P->hWz1, 0, ibytes);
    cudaMemset(P->mWz1, 0, ibytes);
    cudaMemset(P->tWz1, 0, ibytes);
  }
  if(P->isz2){
    size_t ibytes = N*ni*nj*WSIZE*sizeof(real_t);
    P-> Wz2 = (real_t *) cuda_malloc(ibytes);
    P->hWz2 = (real_t *) cuda_malloc(ibytes);
    P->mWz2 = (real_t *) cuda_malloc(ibytes);
    P->tWz2 = (real_t *) cuda_malloc(ibytes);
    cudaMemset(P-> Wz2, 0, ibytes);
    cudaMemset(P->hWz2, 0, ibytes);
    cudaMemset(P->mWz2, 0, ibytes);
    cudaMemset(P->tWz2, 0, ibytes);
  }
  return;
}

void dealloc_pml (PML P){
  CUDACHECK(cudaFree( P.Ax ));
  CUDACHECK(cudaFree( P.Ay ));
  CUDACHECK(cudaFree( P.Az ));
  CUDACHECK(cudaFree( P.Bx ));
  CUDACHECK(cudaFree( P.By ));
  CUDACHECK(cudaFree( P.Bz ));
  CUDACHECK(cudaFree( P.Dx ));
  CUDACHECK(cudaFree( P.Dy ));
  CUDACHECK(cudaFree( P.Dz ));

  if(P.isx1) {
    CUDACHECK(cudaFree( P. Wx1 ));
    CUDACHECK(cudaFree( P.hWx1 ));
    CUDACHECK(cudaFree( P.mWx1 ));
    CUDACHECK(cudaFree( P.tWx1 ));
  }
  if(P.isx2){
    CUDACHECK(cudaFree( P. Wx2 ));
    CUDACHECK(cudaFree( P.hWx2 ));
    CUDACHECK(cudaFree( P.mWx2 ));
    CUDACHECK(cudaFree( P.tWx2 ));
  }
  if(P.isy1) {
    CUDACHECK(cudaFree( P. Wy1 ));
    CUDACHECK(cudaFree( P.hWy1 ));
    CUDACHECK(cudaFree( P.mWy1 ));
    CUDACHECK(cudaFree( P.tWy1 ));
  }
  if(P.isy2){
    CUDACHECK(cudaFree( P. Wy2 ));
    CUDACHECK(cudaFree( P.hWy2 ));
    CUDACHECK(cudaFree( P.mWy2 ));
    CUDACHECK(cudaFree( P.tWy2 ));
  }
  if(P.isz1) {
    CUDACHECK(cudaFree( P. Wz1 ));
    CUDACHECK(cudaFree( P.hWz1 ));
    CUDACHECK(cudaFree( P.mWz1 ));
    CUDACHECK(cudaFree( P.tWz1 ));
  }
  if(P.isz2){
    CUDACHECK(cudaFree( P. Wz2 ));
    CUDACHECK(cudaFree( P.hWz2 ));
    CUDACHECK(cudaFree( P.mWz2 ));
    CUDACHECK(cudaFree( P.tWz2 ));
  }
  return;
}

void cpy_host2device_pml(PML P, const PML h_P){
  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;
  cudaMemcpy(P.Ax, h_P.Ax, sizeof(real_t)*ni, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Bx, h_P.Bx, sizeof(real_t)*ni, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Dx, h_P.Dx, sizeof(real_t)*ni, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Ay, h_P.Ay, sizeof(real_t)*nj, cudaMemcpyHostToDevice);
  cudaMemcpy(P.By, h_P.By, sizeof(real_t)*nj, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Dy, h_P.Dy, sizeof(real_t)*nj, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Az, h_P.Az, sizeof(real_t)*nk, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Bz, h_P.Bz, sizeof(real_t)*nk, cudaMemcpyHostToDevice);
  cudaMemcpy(P.Dz, h_P.Dz, sizeof(real_t)*nk, cudaMemcpyHostToDevice);
  return;
}
