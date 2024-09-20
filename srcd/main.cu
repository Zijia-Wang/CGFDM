#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "io.h"
#include <mpi.h>
#include <pthread.h>

/* for timing */
#include <sys/time.h>
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
/* for timing */

extern void get_params(int argc, char **argv);
extern void print_params();
extern void set_device_params();

//extern void alloc_metric(double *M);
extern void alloc_wave(Wave *);
extern void alloc_wave_halo();
extern void alloc_fault(Fault *);                  
extern void alloc_fault_coef(Fault *);             
extern void alloc_pml(PML *);
extern void alloc_pml_host(PML *);
extern void dealloc_wave(Wave);
extern void dealloc_fault(Fault);
extern void dealloc_fault_coef(Fault);
extern void dealloc_pml_host(PML);
extern void dealloc_pml(PML);
extern void cpy_host2device_pml(PML, const PML);

extern void get_coord_y_h(realptr_t C, realptr_t hC, const int j);
extern void get_coord_x_h(realptr_t C, realptr_t hC, const int i);
extern void construct_coord(realptr_t C);
extern void cal_metric(realptr_t C, realptr_t M);
extern void extend_Symm_array(realptr_t W, int SIZE);
extern void extend_crew_array(realptr_t W, int SIZE);
extern void abs_init(PML);
extern void init_cerjan_host(realptr_t damp);

extern void init_media1d(realptr_t C, realptr_t M);
extern void init_media3d(realptr_t C, realptr_t M);

extern void exchange_array(realptr_t, int arrsize);
extern void exchange_wave(realptr_t);
extern void exchange_fault(Fault, int);               //*******

extern void init_wave_3d(realptr_t);
extern void init_fault(Fault, int, int Faultgrid[]);                   //*******
extern void trial_sw(Wave, Fault, realptr_t M, int it, int irk, int, int, int, int, int, int Faultgrid[]);       //
extern void trial_rs(Wave, Fault, realptr_t M, int it, int irk, int, int, int, int, int, int Faultgrid[]);       //
extern void thermpress(Wave, Fault, realptr_t M, int it, int irk, int, int, int, int);          //
extern void wave2fault(Wave, Fault, realptr_t M, int, int);         //    
extern void fault2wave(Wave, Fault, realptr_t M, int, int);         //
extern void wave_deriv (Wave, realptr_t M, PML, int, int, int);
extern void abs_deriv_x(Wave, realptr_t M, PML, int, int);
extern void abs_deriv_y(Wave, realptr_t M, PML, int, int);
extern void abs_deriv_z(Wave, realptr_t M, PML, int, int);
//extern void fault_deriv(Wave, Fault, realptr_t M, int, int, int);      
extern void fault_dvelo(Wave, Fault, realptr_t M, int, int, int, int, int);          //
extern void fault_dstrs_f(Wave, Fault, realptr_t M, int, int, int, int, int);         //
extern void fault_dstrs_b(Wave, Fault, realptr_t M, int, int, int, int, int);         //
extern void wave_rk(Wave, PML, int irk);
extern void fault_rk(Fault, int irk, int);                         //
extern void state_rk(Fault, int irk, int, int Faultgrid[]);                         //
extern void init_fault_coef(realptr_t M, Fault, int, int);            //************
extern void init_wave_free(realptr_t M, PML, Wave);
extern void smooth_gauss_volume(Wave, Fault, realptr_t M, int, int, int Faultgrid[], int);   //
extern void smooth_T1(Fault, int, int Faultgrid[]);
//extern void fault_filter(Wave, Fault, realptr_t M);
extern void nc_read_init_stress(Fault F);                         

extern void cal_range_steph(realptr_t C, real_t *range);
extern void cal_range_media(realptr_t M, real_t *range);

extern void add_source_ricker(Wave W, realptr_t M, int it, int irk);

extern void apply_cerjan(Wave, realptr_t);

extern void cal_rup_sensor(Fault, int, int Faultgrid[]);                //*********

int FD_Flags[8][3] = {
{-1, -1, -1},
{ 1,  1, -1},
{ 1,  1,  1},
{-1, -1,  1},
{-1,  1, -1},
{ 1, -1, -1},
{ 1, -1,  1},
{-1,  1,  1}}; // -1(B) 1(F)
//int FD_Flags[8][3] = {
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1},
//{ 1,  1,  1}}; // -1(B) 1(F)

void setDeviceBeforeInit(){
  char *localRankStr = NULL;
  int rank = 0, devCount = 0;
  if (NULL != (localRankStr = getenv(ENV_LOCAL_RANK))){
    rank = atoi(localRankStr);
  }
  CUDACHECK(cudaGetDeviceCount( &devCount));
  if(masternode) printf("There are %d GPUs on each node\n", devCount);
  CUDACHECK(cudaSetDevice( (rank + 0) % devCount));
}

void Initialize(int *argc, char ***argv, int *rank, int *size){
  setDeviceBeforeInit();
  MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, size);
}

int main(int argc, char **argv){

  int rank, size, dims[3];
  int reorder = 1; // true
  int periods[3] = {0, 0, 0}; // false
  int ilen = 0;
  int* Faultgrid;
  
  char processor[MPI_MAX_PROCESSOR_NAME];
  
  Initialize(&argc, &argv, &rank, &size);  
  MPI_Get_processor_name(processor, &ilen);

  get_params(argc, argv);

  if(hostParams.PX*hostParams.PY*hostParams.PZ < 2){
    cudaSetDevice(hostParams.igpu);
  }

  int oldRank = rank;

  dims[0] = hostParams.PX;
  dims[1] = hostParams.PY;
  dims[2] = hostParams.PZ;
  
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &SWMPI_COMM);
  MPI_Comm_rank(SWMPI_COMM, &rank);

  if(rank != oldRank){
    printf("Rank change: from %d to %d\n", oldRank, rank);
  }

  MPI_Cart_coords(SWMPI_COMM, rank, 3, thisid);
  MPI_Cart_shift(SWMPI_COMM, 0, 1, &neigxid[0], &neigxid[1]);
  MPI_Cart_shift(SWMPI_COMM, 1, 1, &neigyid[0], &neigyid[1]);
  MPI_Cart_shift(SWMPI_COMM, 2, 1, &neigzid[0], &neigzid[1]);

  if(0 == rank) masternode = 1;
  if(masternode) print_params();

  int absnode[6] = {0, 0, 0, 0, 0, 0};

  if(neigxid[0] == MPI_PROC_NULL) absnode[0] = 1;
  if(neigxid[1] == MPI_PROC_NULL) absnode[1] = 1;
  if(neigyid[0] == MPI_PROC_NULL) absnode[2] = 1;
  if(neigyid[1] == MPI_PROC_NULL) absnode[3] = 1;
  if(neigzid[0] == MPI_PROC_NULL) absnode[4] = 1;
  if(neigzid[1] == MPI_PROC_NULL){
#ifdef FreeSurface
    absnode[5] = 0; freenode = 1;
#else
    absnode[5] = 1; freenode = 0;
#endif
  }

  hostParams.freenode = freenode;
  // hostParams.faultnode = faultnode;

  hostParams.rankx = thisid[0];
  hostParams.ranky = thisid[1];
  hostParams.rankz = thisid[2];

  set_device_params(); // copy parameters from host to device
  cudaDeviceSynchronize();

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = hostParams.nx;
  int ny = hostParams.ny;
  int nz = hostParams.nz;

  int i0 = 0;
  int num_fault = hostParams.num_fault;
  int faultnode = 0;
  // int* faultnode_Dev;
  //int srci = hostParams.NX / 2; // fault plane index X
  for(int nfault = 0; nfault < num_fault; nfault++){
    i0 = hostParams.src_i[nfault];
    if(i0 / ni == thisid[0]) faultnode = 1;
    // hostParams.faultnode = faultnode;
  }

  // cudaMalloc(&faultnode_Dev, sizeof(int)*1);
  // cudaMemcpy(faultnode_Dev, faultnode, sizeof(int)*1, cudaMemcpyHostToDevice);

  MPI_Barrier(MPI_COMM_WORLD);

  realptr_t M;  // device
  Wave  W;   // device
  Fault F;   // device
  PML   P;   // device
  PML   h_P; // host
  Recv R; // host


  h_P.isx1 = absnode[0];
  h_P.isx2 = absnode[1];
  h_P.isy1 = absnode[2];
  h_P.isy2 = absnode[3];
  h_P.isz1 = absnode[4];
  h_P.isz2 = absnode[5];

  P.isx1 = absnode[0];
  P.isx2 = absnode[1];
  P.isy1 = absnode[2];
  P.isy2 = absnode[3];
  P.isz1 = absnode[4];
  P.isz2 = absnode[5];

  MPI_Barrier(MPI_COMM_WORLD);
  printf("(%s): %d %d %d masternode = %d, freenode = %d\n", processor,
      thisid[0],
      thisid[1],
      thisid[2],
      masternode, freenode);
 // MPI_Barrier(MPI_COMM_WORLD);
  printf("(%s): %d %d %d absnode = (%d %d) (%d %d) (%d %d)\n", processor,
      thisid[0],
      thisid[1],
      thisid[2],
      absnode[0],
      absnode[1],
      absnode[2],
      absnode[3],
      absnode[4],
      absnode[5]);
  MPI_Barrier(MPI_COMM_WORLD);

  cudaMalloc((void**) &M, sizeof(real_t)*nx*ny*nz*MSIZE);
  alloc_wave(&W);
  alloc_fault(&F);
  alloc_fault_coef(&F);

  alloc_wave_halo();

#ifdef usePML
  alloc_pml_host(&h_P);
  abs_init(h_P);
  //coef_surface(P, M);
  alloc_pml(&P);
  cpy_host2device_pml(P, h_P);
#endif

#ifdef useCerjan
  if (masternode) printf("Using Cerjan (1985) for absorbing ...\n");

  real_t *damp_h = (real_t *) malloc(sizeof(real_t)*nx*ny*nz);
  memset(damp_h, 0, sizeof(real_t)*nx*ny*nz);
  real_t *damp;
  cudaMalloc((real_t **) &damp, sizeof(real_t)*nx*ny*nz);

  init_cerjan_host(damp_h);
  cudaMemcpy(damp, damp_h, sizeof(real_t)*nx*ny*nz, cudaMemcpyHostToDevice);
#endif

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaGetLastError());

  realptr_t hostCoord, hostMetric;
  hostCoord  = (real_t *)malloc(sizeof(real_t)*nx*ny*nz*CSIZE);
  hostMetric = (real_t *)malloc(sizeof(real_t)*nx*ny*nz*MSIZE);
  memset(hostCoord,  0, sizeof(real_t)*nx*ny*nz*CSIZE);
  memset(hostMetric, 0, sizeof(real_t)*nx*ny*nz*MSIZE);

  construct_coord(hostCoord);
  extend_crew_array(hostCoord, CSIZE);
  MPI_Barrier(MPI_COMM_WORLD);
  exchange_array(hostCoord, CSIZE);
  MPI_Barrier(MPI_COMM_WORLD);

  //locate_recv(&R,hostCoord);

  real_t range[2];
  cal_range_steph(hostCoord, range);
  real_t hmin = range[0];
  real_t hmax = range[1];
  real_t hmin_global, hmax_global;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&hmin, &hmin_global, 1, MPI_REAL_T, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&hmax, &hmax_global, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);

  cal_metric(hostCoord, hostMetric);
  if (flag_media1d) init_media1d(hostCoord, hostMetric);
  if (flag_media3d) init_media3d(hostCoord, hostMetric);

  extend_Symm_array(hostMetric, MSIZE);
  MPI_Barrier(MPI_COMM_WORLD);
  exchange_array(hostMetric, MSIZE);
  cudaMemcpy(M, hostMetric, sizeof(real_t)*nx*ny*nz*MSIZE, cudaMemcpyHostToDevice);
  //free(hostCoord);
  //free(hostMetric);

  real_t media_range[6];
  cal_range_media(hostMetric, media_range);
  real_t  vp_min = media_range[0];
  real_t  vp_max = media_range[1];
  real_t  vs_min = media_range[2];
  real_t  vs_max = media_range[3];
  real_t rho_min = media_range[4];
  real_t rho_max = media_range[5];
  real_t vp_min_global, vs_min_global, rho_min_global;
  real_t vp_max_global, vs_max_global, rho_max_global;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&vp_min, &vp_min_global, 1, MPI_REAL_T, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&vp_max, &vp_max_global, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&vs_min, &vs_min_global, 1, MPI_REAL_T, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&vs_max, &vs_max_global, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rho_min, &rho_min_global, 1, MPI_REAL_T, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rho_max, &rho_max_global, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);

  //printf("range h = %10.2e ~ %10.2e, in rank %d %d %d\n",
  //hmin, hmax, thisid[0], thisid[1], thi    sid[2]);

  float dtmax = 1.3 * hmin_global / vp_max_global;

  if(masternode){
    printf("global range of h   = %10.2e ~ %10.2e m\n",
        hmin_global, hmax_global);
    printf("global range of vp  = %10.2e ~ %10.2e m/s\n",
        vp_min_global, vp_max_global);
    printf("global range of vs  = %10.2e ~ %10.2e m/s\n",
        vs_min_global, vs_max_global);
    printf("global range of rho = %10.2e ~ %10.2e kg/m^3\n",
        rho_min_global, rho_max_global);
    if(hostParams.DT < dtmax){
      printf("DT = %10.2e < dtmax = %10.2e (sec)\n"
             "satisfy stability condition, OK\n",
          hostParams.DT, dtmax);
    }else{
      printf("Serious Error: DT = %10.2e > dtmax = %10.2e (sec)\n"
             "               do not satisfy stability condition, ABORT!\n",
          hostParams.DT, dtmax);
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 110);
    }
  }

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaGetLastError());
  
  cudaMalloc((int**) &Faultgrid, sizeof(int)*4*hostParams.num_fault);
  cudaMemcpy(Faultgrid, hostParams.Fault_grid, sizeof(int)*4*hostParams.num_fault, cudaMemcpyHostToDevice);
  // for (int i = 0; i < 4*hostParams.num_fault; i++){
  //   Faultgrid[i] = hostParams.Fault_grid[i];
  // }
#ifdef Rupture
  for(int nfault = 0; nfault < num_fault; nfault++){
    i0 = hostParams.src_i[nfault];
    init_fault_coef(M, F, i0, nfault);
    init_fault(F, nfault, Faultgrid);
  }
  if (1 == hostParams.INPORT_STRESS_TYPE){
    // overwrite by input init stress
    nc_read_init_stress(F);
  }
#endif
#ifdef FreeSurface
  init_wave_free(M, P, W);
#endif

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaGetLastError());

  /* for timing */
  double t_start;
  double t_elapsed;
  double elapsed_time_s;
  elapsed_time_s = 0.f;

  cudaEvent_t e_start, e_end;

  cudaEventCreate(&e_start);
  cudaEventCreate(&e_end  );

#ifdef useNetCDF
#ifdef Rupture
  ncFile ncFault;
#endif
  ncFile ncSliceX;
  ncFile ncSliceY;
  ncFile ncSliceZ;
 // ncFile ncRecv;
#endif

  int nt = hostParams.NT;
  init_wave_3d(W.W);
#ifdef usePML
  int ND = hostParams.PML_N;
  if(P.isx1) cudaMemset(P.Wx1, 0, sizeof(real_t)*ND*nj*nk);
  if(P.isx2) cudaMemset(P.Wx2, 0, sizeof(real_t)*ND*nj*nk);
  if(P.isy1) cudaMemset(P.Wy1, 0, sizeof(real_t)*ND*nk*ni);
  if(P.isy2) cudaMemset(P.Wy2, 0, sizeof(real_t)*ND*nk*ni);
  if(P.isz1) cudaMemset(P.Wz1, 0, sizeof(real_t)*ND*ni*nj);
  if(P.isz2) cudaMemset(P.Wz2, 0, sizeof(real_t)*ND*ni*nj);
#endif
  cudaDeviceSynchronize();
  CUDACHECK(cudaGetLastError());

  int slice_x_index = hostParams.EXPORT_WAVE_SLICE_X;
  int slice_y_index = hostParams.EXPORT_WAVE_SLICE_Y;
  int slice_z_index = hostParams.EXPORT_WAVE_SLICE_Z;
#ifdef useNetCDF
#ifdef Rupture
//  printf("aaaaaa\n");
  nc_def_fault(F, &ncFault);
//  printf("Netcdf def OK\n");
  nc_put_fault_coord(hostCoord, ncFault, faultnode);
  //CUDACHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0;

  int thread_count = 2;
  pthread_t* thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));

  size_t ibytes = sizeof(real_t)*nj*nk*num_fault;
  real_t *hostFault = NULL;
  cudaMallocHost((void**)&hostFault, 16*ibytes);
  // hostFault = (real_t*)malloc(16*ibytes);
  real_t *deviceFault = NULL;
  cudaMalloc((void **)&deviceFault, 16*ibytes);

  real_t *hostData = NULL;
  real_t *deviceData = NULL;
  int countwave = 3*sizeof(real_t)*(ni*nj+ni*nk+nk*nj);
  cudaMalloc((void **) &deviceData, countwave);
  cudaMallocHost((void**)&hostData, countwave);
  real_t* hostptr[4] = {hostFault, hostData, deviceFault, deviceData};

#endif
  nc_def_wave_xy(slice_z_index, &ncSliceZ);
  nc_put_wave_xy_coord(hostCoord, slice_z_index, ncSliceZ);

  nc_def_wave_xz(slice_y_index, &ncSliceY);
  nc_put_wave_xz_coord(hostCoord, slice_y_index, ncSliceY);

  nc_def_wave_yz(slice_x_index, &ncSliceX);
  nc_put_wave_yz_coord(hostCoord, slice_x_index, ncSliceX);

//  nc_def_recv(R, W, &ncRecv);
#endif
  for (int it = 0; it < nt; it++){

    /* for timing */
    t_start = seconds();
    /* for timing */
#ifdef useNetCDF
    int time_skip = hostParams.EXPORT_TIME_SKIP;
    int it_skip;
    if ((it % time_skip) == 0){
      it_skip = (int)(it/time_skip);
#ifdef Rupture
    if (it_skip > 0)
      {
        pthread_join(thread_handles[0], NULL);
        pthread_join(thread_handles[1], NULL);
      }

    nc_put_faultwave(F, W.W, slice_z_index, slice_y_index, slice_x_index, hostptr);
    pthreadFault pthdFault = {hostptr[0], it_skip, hostParams.nj, hostParams.nk, ncFault};
    pthreadWave pthdWave = {hostptr[1], it_skip, hostParams.ni, hostParams.nj, hostParams.nk,\
                            ncSliceX, ncSliceY, ncSliceZ, slice_x_index, slice_y_index, slice_z_index,\
                            {thisid[0], thisid[1], thisid[2]}};

    pthread_create(&thread_handles[0], NULL, ncput_fault, &pthdFault);
    pthread_create(&thread_handles[1], NULL, ncput_wave, &pthdWave);
//      printf("put OK \n");
#else
// #endif
    nc_put_wave_xy(W.W, slice_z_index, it_skip, ncSliceZ);
    nc_put_wave_xz(W.W, slice_y_index, it_skip, ncSliceY);
    nc_put_wave_yz(W.W, slice_x_index, it_skip, ncSliceX);
#endif
    }
#ifdef Rupture
    if ( (it % time_skip) == 0 && (it == (nt-1) || it == (nt-2)) )
    {
      pthread_join(thread_handles[0], NULL);
      pthread_join(thread_handles[1], NULL);
    }
#endif
#endif

    // select flags for F(orward) or B(ackward) operator
    int FlagX = FD_Flags[it % 8][0];
    int FlagY = FD_Flags[it % 8][1];
    int FlagZ = FD_Flags[it % 8][2];

    for(int nfault = 0; nfault < num_fault; nfault++){
      cal_rup_sensor(F, nfault, Faultgrid);
      if (hostParams.Friction_type == 3){
        thermpress(W, F, M, it, 0, FlagX, FlagY, FlagZ, nfault);
        }
    }
    for (int irk = 0; irk < 4; irk ++){

      //nc_put_fault(F, it*4+irk, ncFault);
      //nc_put_fault(F, it, ncFault);
      //if(irk % 2){ FlagX *= -1; FlagY *= -1; FlagZ *= -1;}

      exchange_wave(W.W);
#ifdef Rupture
    for(int nfault = 0; nfault < num_fault; nfault++){
      i0 = hostParams.src_i[nfault];
      exchange_fault(F, nfault);
      wave2fault(W, F, M, i0, nfault);
      state_rk(F, irk, nfault, Faultgrid);
      if (hostParams.Friction_type == 0){
        trial_sw(W, F, M, it, irk, FlagX, FlagY, FlagZ, i0, nfault, Faultgrid);
      }else if (
          hostParams.Friction_type == 1 ||
          hostParams.Friction_type == 2 ||
          hostParams.Friction_type == 3 ){
        trial_rs(W, F, M, it, irk, FlagX, FlagY, FlagZ, i0, nfault, Faultgrid);
      }
      
      fault2wave(W, F, M, i0, nfault);
    }
#endif
      wave_deriv(W, M, P, FlagX, FlagY, FlagZ);
#ifndef Rupture
      add_source_ricker(W, M, it, irk);
#endif
#ifdef DynTrigger
      add_source_ricker(W, M, it, irk);
#endif
#ifdef usePML
      if(h_P.isx1) abs_deriv_x(W, M, P, FlagX, 0);
      if(h_P.isx2) abs_deriv_x(W, M, P, FlagX, ni-ND);

      if(h_P.isy1) abs_deriv_y(W, M, P, FlagY, 0);
      if(h_P.isy2) abs_deriv_y(W, M, P, FlagY, nj-ND);

      if(h_P.isz1) abs_deriv_z(W, M, P, FlagZ, 0);
      if(h_P.isz2) abs_deriv_z(W, M, P, FlagZ, nk-ND);
#endif

#ifdef Rupture
      for(int nfault = 0; nfault < num_fault; nfault++){
        i0 = hostParams.src_i[nfault];
        //fault_deriv(W, F, M, FlagX, FlagY, FlagZ);
        fault_dvelo(W, F, M, FlagX, FlagY, FlagZ, i0, nfault);
        //fault_dstrs(W, F, M, FlagX, FlagY, FlagZ);
        if(FlagX == 1){ // Forward
          fault_dstrs_f(W, F, M, FlagX, FlagY, FlagZ, i0, nfault);
        }else{
          fault_dstrs_b(W, F, M, FlagX, FlagY, FlagZ, i0, nfault);
        }
        // smoothT1(F, nfault, Faultgrid);
      }

#endif
      wave_rk(W, P, irk);

#ifdef Rupture
  for(int nfault = 0; nfault < num_fault; nfault++){
      i0 = hostParams.src_i[nfault];
      fault_rk(F, irk, nfault);
//#if defined(Rupture) && defined(FaultSmooth)
//      exchange_wave(W.W);
//      exchange_fault(F);
//      smooth_gauss_volume(W, F, M);
//#endif
      fault2wave(W, F, M, i0, nfault);
  }
#endif

#ifdef useCerjan
      apply_cerjan(W, damp);
#endif

      cudaDeviceSynchronize();
      //CUDACHECK(cudaGetLastError());

      // Reverse the FD Flags for next substep of Runge-Kutta
      FlagX *= -1; FlagY *= -1; FlagZ *= -1;
    } // end irk
#if defined(Rupture) && defined(FaultSmooth)
    if (it % 1 == 0){
    exchange_wave(W.W);
    for(int nfault = 0; nfault < num_fault; nfault++){
      i0 = hostParams.src_i[nfault];
      exchange_fault(F, nfault);
      smooth_gauss_volume(W, F, M, i0, nfault, Faultgrid, it);
    }
    //fault_filter(W, F, M);
    }
#endif
    // if (it % 1 == 0){
    //   exchange_wave(W.W);
    //   for(int nfault = 0; nfault < num_fault; nfault++){
    //     i0 = hostParams.src_i[nfault];
    //     exchange_fault(F, nfault);
    //     smooth_T1(F, nfault, Faultgrid);
    // }
    // }
   
    /* for timing */
    cudaDeviceSynchronize();
    t_elapsed = seconds() - t_start;
    elapsed_time_s += t_elapsed;
    if(0 == it % 10 && masternode){
      printf("> %6d, total %d, %8.4lf s\n", it, nt, t_elapsed);
      fflush(stdout);
    }

  } // end it

#ifdef useNetCDF
#ifdef Rupture
  nc_end_fault(ncFault);
#endif
  nc_end_wave_xz(slice_y_index, ncSliceY);
  nc_end_wave_xy(slice_z_index, ncSliceZ);
  nc_end_wave_yz(slice_x_index, ncSliceX);

 // nc_end_recv(R, ncRecv);
#endif

  /* for timing */
  float elapsed_time_ms_cuda;
  cudaEventRecord(e_end, 0);
  cudaEventElapsedTime(&elapsed_time_ms_cuda, e_start, e_end);
  cudaDeviceSynchronize();
  cudaGetLastError();
  fflush(stdout);
  if(masternode){
    printf("----------------------------------\n");
    printf("> Total %6d: CPU: %e s | GPU: %e s\n",
        nt, elapsed_time_s, elapsed_time_ms_cuda/1e3);
  }
  //printf("> Performance: %8.2f MCells/s\n",
  //    (double) nx * ny * nz / (elapsed_time_ms * 1e3f));
  fflush(stdout);

  cudaEventDestroy(e_start);
  cudaEventDestroy(e_end);
  /* for timing */

  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaGetLastError());

  dealloc_wave(W);
  //dealloc_fault(F);
  //dealloc_fault_coef(F);
  ////dealloc_pml_host(h_P);
  //dealloc_pml(P);
  cudaFree(Faultgrid);
  cudaFree(deviceData);
  cudaFree(deviceFault);
  cudaFreeHost(hostFault);
  cudaFreeHost(hostData);

  MPI_Finalize();

  return 0;
}
