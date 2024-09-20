#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "common.h"
#include "params.h"
#include "cJSON.h"

// 定义性声明
//__device__ __constant__ int Flags[8][3];
Params hostParams;
__device__ __constant__ Params par;
// int * Device_Faultgrid;
int masternode = 0;
int freenode = 0;
// int faultnode = 0;
int neigxid[2], neigyid[2], neigzid[2];
int thisid[3];
MPI_Comm SWMPI_COMM;
real_t *wave_yz_send0, *wave_yz_recv0;
real_t *wave_yz_send1, *wave_yz_recv1;
real_t *wave_xz_send0, *wave_xz_recv0;
real_t *wave_xz_send1, *wave_xz_recv1;
real_t *wave_xy_send0, *wave_xy_recv0;
real_t *wave_xy_send1, *wave_xy_recv1;

real_t *fault_z_send0, *fault_z_recv0;
real_t *fault_z_send1, *fault_z_recv1;
real_t *fault_y_send0, *fault_y_recv0;
real_t *fault_y_send1, *fault_y_recv1;

char PARAMS_FILE[1000];
char OUT[1000];
char Fault_geometry[1000];
char Fault_init_stress[1000];
char Media1D[1000];
char Media3D[1000];
int flag_media1d;
int flag_media3d;

void set_device_params(){

  //int h_Flags[8][3] = {
  //{-1, -1, -1},
  //{ 1,  1, -1},
  //{ 1,  1,  1},
  //{-1, -1,  1},
  //{-1,  1, -1},
  //{ 1, -1, -1},
  //{ 1, -1,  1},
  //{-1,  1,  1}};

  //cudaMemcpyToSymbol(Flags, h_Flags, sizeof(int)*8*3);
  cudaMemcpyToSymbol(par, &hostParams, sizeof(Params));
  // cudaMalloc((int**) &Device_Faultgrid, sizeof(int)*4*hostParams.num_fault);
  // cudaMemcpy(Device_Faultgrid, hostParams.Fault_grid, 4*hostParams.num_fault, cudaMemcpyHostToDevice);  //wangzj
  return;
}

void get_params(int argc, char **argv)
{
  // ==========================================================================
  // First, we set parameters by defaults
  sprintf(OUT, "output");
  sprintf(Fault_geometry, "Fault_geometry.nc");
  sprintf(Fault_init_stress, "Fault_init_stress.nc");

  hostParams.EXPORT_TIME_SKIP = 1;

  hostParams.TMAX = 1.0;
  hostParams.DT = 0.01;
  hostParams.DH = 100.0;
  hostParams.NX = 100;
  hostParams.NY = 90;
  hostParams.NZ = 80;
  hostParams.PX = 1;
  hostParams.PY = 1;
  hostParams.PZ = 1;
  hostParams.igpu = 0;

  hostParams.num_fault = 1;                  // for multi-fault ****wangzj
  //hostParams.src_i = hostParams.NX/2;       

  //hostParams.Fault_grid = {51, 100, 51, 100};
  //hostParams.Asp_grid = {0, 0, 0, 0};
  // hostParams.Barrier_grid = {0, 0, 0, 0};
  hostParams.mu_s = 0.677;
  hostParams.mu_d = 0.525;
  hostParams.Dc = 0.4;
  hostParams.C0 = 0.0;

  hostParams.PML_N = 12;
  hostParams.PML_velocity = 6000.0;
  hostParams.PML_bmax = 3.0;
  hostParams.PML_fc = 1.0;

  hostParams.DAMP_N = 20;

  hostParams.vp1 = 6000.0;
  hostParams.vs1 = 3464.0;
  hostParams.rho1 = 2670.0;
  hostParams.bi_vp1 = 6000.0;
  hostParams.bi_vs1 = 3464.0;
  hostParams.bi_rho1 = 2670.0;
  hostParams.bi_vp2 = 6000.0;
  hostParams.bi_vs2 = 3464.0;
  hostParams.bi_rho2 = 2670.0;

  hostParams.viscosity = 0.;
  hostParams.RupThres = 0.;

  hostParams.TP_n = 1;
  // ==========================================================================

  // Then, we read new parameters from configuration file
  // To determine which file to read
  strcpy(PARAMS_FILE, "params.json");
  if (argc > 1) {
    memcpy(PARAMS_FILE, argv[1], strlen(argv[1]));
  }
  if(masternode) printf("PARAMS_FILE = %s\n", PARAMS_FILE);

  FILE *fp;
  if (NULL == (fp = fopen(PARAMS_FILE, "r"))){
      printf("Error at opening the configuration file %s\n", PARAMS_FILE);
      exit(-1);
  }
  fseek(fp, 0, SEEK_END);
  long len = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *str = (char*)malloc(len+1);
  fread(str, 1, len, fp);
  fclose(fp);
  //printf("configure json file:\n%s\n", str);

  cJSON *root, *item;
  root = cJSON_Parse(str);
  if (NULL == root) {
    printf("Error at parsing json!\n");
    exit(-1);
  }

  //memcpy(OUT, item->valuestring, strlen(item->valuestring)+1);
  //strcpy(OUT, item->valuestring);

  if (item = cJSON_GetObjectItem(root, "OUT"))
    strcpy(OUT, item->valuestring);

  if (item = cJSON_GetObjectItem(root, "Fault_geometry"))
    strcpy(Fault_geometry, item->valuestring);

  if (item = cJSON_GetObjectItem(root, "Fault_init_stress"))
    strcpy(Fault_init_stress, item->valuestring);

  flag_media1d = 0;
  if (item = cJSON_GetObjectItem(root, "Media1D")){
    strcpy(Media1D, item->valuestring);
    flag_media1d = 1;
  }

  flag_media3d = 0;
  if (item = cJSON_GetObjectItem(root, "Media3D")){
    strcpy(Media3D, item->valuestring);
    flag_media3d = 1;
  }

  if (item = cJSON_GetObjectItem(root, "Asp_grid")){
    for (int i = 0; i < 4; i++){
      hostParams.Asp_grid[i] = cJSON_GetArrayItem(item, i)->valueint;
    }
  }

  if (item = cJSON_GetObjectItem(root, "Barrier_grid")){
    for (int i = 0; i < 4; i++){
      hostParams.Barrier_grid[i] = cJSON_GetArrayItem(item, i)->valueint;
    }
  }

  if (item = cJSON_GetObjectItem(root, "TMAX"))
    hostParams.TMAX  = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "DT"))
    hostParams.DT = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "DH"))
    hostParams.DH = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "NX"))
    hostParams.NX = item->valueint;
  if (item = cJSON_GetObjectItem(root, "NY"))
    hostParams.NY = item->valueint;
  if (item = cJSON_GetObjectItem(root, "NZ"))
    hostParams.NZ = item->valueint;
  if (item = cJSON_GetObjectItem(root, "PX"))
    hostParams.PX = item->valueint;
  if (item = cJSON_GetObjectItem(root, "PY"))
    hostParams.PY = item->valueint;
  if (item = cJSON_GetObjectItem(root, "PZ"))
    hostParams.PZ = item->valueint;

  if (item = cJSON_GetObjectItem(root, "igpu"))
    hostParams.igpu = item->valueint;

  if (item = cJSON_GetObjectItem(root, "viscosity"))
    hostParams.viscosity = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "RupThres"))
    hostParams.RupThres = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "INPORT_GRID_TYPE"))
    hostParams.INPORT_GRID_TYPE = item->valueint;

  if (item = cJSON_GetObjectItem(root, "num_fault"))
    hostParams.num_fault = item->valueint;

  if (item = cJSON_GetObjectItem(root, "src_i")){
    hostParams.src_i = (int*) malloc(sizeof(int) *hostParams.num_fault);
    for (int i = 0; i < hostParams.num_fault; i++){
      hostParams.src_i[i] = cJSON_GetArrayItem(item, i)->valueint;
      if (hostParams.num_fault > 1){
        hostParams.src_i[i]--; 
      }
    }
  }
  // int FGsize = 4 * hostParams.num_fault;
  if (item = cJSON_GetObjectItem(root, "Fault_grid")){
  //int array_size = cJSON_GetArraySize(item);
    hostParams.Fault_grid = (int*) malloc(sizeof(int)*4*hostParams.num_fault);
    for (int i = 0; i < 4*hostParams.num_fault; i++){
      hostParams.Fault_grid[i] = cJSON_GetArrayItem(item, i)->valueint;
      // printf("faultgrid=%d\n", hostParams.Fault_grid[i]);
    }
  }

  if (item = cJSON_GetObjectItem(root, "INPORT_STRESS_TYPE"))
    hostParams.INPORT_STRESS_TYPE = item->valueint;
  if (item = cJSON_GetObjectItem(root, "EXPORT_TIME_SKIP"))
    hostParams.EXPORT_TIME_SKIP = item->valueint;
  if (item = cJSON_GetObjectItem(root, "EXPORT_WAVE_SLICE_X"))
    hostParams.EXPORT_WAVE_SLICE_X = item->valueint;
  if (item = cJSON_GetObjectItem(root, "EXPORT_WAVE_SLICE_Y"))
    hostParams.EXPORT_WAVE_SLICE_Y = item->valueint;
  if (item = cJSON_GetObjectItem(root, "EXPORT_WAVE_SLICE_Z"))
    hostParams.EXPORT_WAVE_SLICE_Z = item->valueint;

  // friction 0 : slip weakening
  if (item = cJSON_GetObjectItem(root, "Friction_type"))
    hostParams.Friction_type  = item->valueint;

  // friction 0 : slip weakening
  if (item = cJSON_GetObjectItem(root, "mu_s"))
    hostParams.mu_s  = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "mu_0"))
    hostParams.mu_0  = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "mu_d"))
    hostParams.mu_d = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "Dc"))
    hostParams.Dc = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "C0"))
    hostParams.C0 = item->valuedouble;

  // friction 1 or 2 : rate state
  if (item = cJSON_GetObjectItem(root, "RS_V0"))
    hostParams.V0  = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "RS_Vini"))
    hostParams.Vini  = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "RS_f0"))
    hostParams.f0 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "RS_fw"))
    hostParams.fw = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "RS_L"))
    hostParams.L = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "smooth_load_T"))
    hostParams.smooth_load_T = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "vs1"))
    hostParams.vs1 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "vp1"))
    hostParams.vp1 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "rho1"))
    hostParams.bi_rho1 = item->valuedouble;

  if (item = cJSON_GetObjectItem(root, "bi_vs1"))
    hostParams.bi_vs1 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "bi_vp1"))
    hostParams.bi_vp1 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "bi_rho1"))
    hostParams.bi_rho1 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "bi_vs2"))
    hostParams.bi_vs2 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "bi_vp2"))
    hostParams.bi_vp2 = item->valuedouble;
  if (item = cJSON_GetObjectItem(root, "bi_rho2"))
    hostParams.bi_rho2 = item->valuedouble;

  // PML parameters
  hostParams.PML_N = 20; //
  hostParams.PML_velocity = 3000.0; // m/s
  hostParams.PML_bmax = 3.0; //
  hostParams.PML_fc = 1.0; // Hz

  if(item = cJSON_GetObjectItem(root, "PML_N"))
    hostParams.PML_N = item->valueint;
  if(item = cJSON_GetObjectItem(root, "PML_velocity"))
    hostParams.PML_velocity = item->valuedouble;
  if(item = cJSON_GetObjectItem(root, "PML_bmax"))
    hostParams.PML_bmax = item->valuedouble;
  if(item = cJSON_GetObjectItem(root, "PML_fc"))
    hostParams.PML_fc = item->valuedouble;

  if(item = cJSON_GetObjectItem(root, "DAMP_N"))
    hostParams.DAMP_N = item->valueint;

  if(item = cJSON_GetObjectItem(root, "TP_n"))
    hostParams.TP_n = item->valueint;

  cJSON_Delete(root); // read json parameters end

  hostParams.ni = hostParams.NX / hostParams.PX;
  hostParams.nj = hostParams.NY / hostParams.PY;
  hostParams.nk = hostParams.NZ / hostParams.PZ;

  hostParams.nx = hostParams.ni + 6;
  hostParams.ny = hostParams.nj + 6;
  hostParams.nz = hostParams.nk + 6;

  hostParams.NT = (int)(hostParams.TMAX/hostParams.DT);
  hostParams.rDH = 1.0/hostParams.DH;

  return;
}

void print_params()
{
  printf( "# Parameters setting:\n"
          "* Tmax           = %g (sec)\n"
          "* DT             = %g (sec)\n"
          "* DH             = %g (m)\n"
          "* NX, NY, NZ, NT = %d %d %d %d\n"
          "* PX, PY, PZ     = %d %d %d\n"
          "* Fault_grid     = %d %d %d %d\n"
          "* Asp_grid       = %d %d %d %d\n"
          "* Barrier_grid   = %d %d %d %d\n"
          "* vp             = %g/%g\n"
          "* vs             = %g/%g\n"
          "* rho            = %g/%g\n"
          "* PML_N          = %d\n"
          "* PML_velocity   = %g\n"
          "* PML_fc         = %g\n"
          "* PML_bmax       = %g\n"
          "* RupThres       = %g\n"
          "\n"
          ,
          hostParams.TMAX,
          hostParams.DT,
          hostParams.DH,
          hostParams.NX,
          hostParams.NY,
          hostParams.NZ,
          hostParams.NT,
          hostParams.PX,
          hostParams.PY,
          hostParams.PZ,
          hostParams.Fault_grid[0],
          hostParams.Fault_grid[1],
          hostParams.Fault_grid[2],
          hostParams.Fault_grid[3],
          // hostParams.Fault_grid[4],
          // hostParams.Fault_grid[5],
          // hostParams.Fault_grid[6],
          // hostParams.Fault_grid[7],
          hostParams.Asp_grid[0],
          hostParams.Asp_grid[1],
          hostParams.Asp_grid[2],
          hostParams.Asp_grid[3],
          hostParams.Barrier_grid[0],
          hostParams.Barrier_grid[1],
          hostParams.Barrier_grid[2],
          hostParams.Barrier_grid[3],
          hostParams.bi_vp1,
          hostParams.bi_vp2,
          hostParams.bi_vs1,
          hostParams.bi_vs2,
          hostParams.bi_rho1,
          hostParams.bi_rho2,
          hostParams.PML_N,
          hostParams.PML_velocity,
          hostParams.PML_fc,
          hostParams.PML_bmax,
          hostParams.RupThres
          );
  printf( "# Friction type: %d\n", hostParams.Friction_type );
  if(hostParams.Friction_type == 0){
    printf( "* Dc             = %g (m)\n"
            "* C0             = %g (Pa)\n"
            "* mu_s           = %g\n"
            "* mu_d           = %g\n"
            "\n"
            ,
            hostParams.Dc,
            hostParams.C0,
            hostParams.mu_s,
            hostParams.mu_d
           );
  }else if (
      hostParams.Friction_type == 1 ||
      hostParams.Friction_type == 2 ||
      hostParams.Friction_type == 3
      ){
    printf( "* RS_V0          = %g (m/s)\n"
            "* RS_Vini        = %g (m/s)\n"
            "* RS_f0          = %g\n"
            "* RS_fw          = %g\n"
            "* RS_L           = %g (m)\n"
            "* TP_n           = %d\n"
            "\n"
            ,
            hostParams.V0,
            hostParams.Vini,
            hostParams.f0,
            hostParams.fw,
            hostParams.L,
            hostParams.TP_n
           );
  }
  printf( "* smooth_load_T   = %g (s)\n", hostParams.smooth_load_T);
  if(flag_media1d) printf("Media1D = %s\n", Media1D);
  if(flag_media3d) printf("Media3D = %s\n", Media3D);
  return;
}
