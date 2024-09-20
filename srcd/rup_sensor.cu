#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

__global__
void cal_rup_sensor_D_cu(Fault F, int nfault, int Faultgrid[])
{

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int mpifaultsize = nfault * (nj*nk);        //wangzj

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  int i0 = nx/2;

  int gj = par.ranky * nj + j;
  int gk = par.rankz * nk + k;
  int gj1 = gj + 1;
  int gk1 = gk + 1;

  real_t *p = F.Ts1;

  //if(j >= 3 && j < nj-3 && k >= 3 && k < nk-3){
  if( gj1 >= Faultgrid[0 + 4*nfault]+3 && gj1 <= Faultgrid[1 + 4*nfault]-3 && 
      gk1 >= Faultgrid[2 + 4*nfault]+3 && gk1 <= Faultgrid[3 + 4*nfault]-3 )
  // if( gj1 >= par.Fault_grid[0 + 4*nfault]+3 && gj1 <= par.Fault_grid[1 + 4*nfault]-3 && 
      // gk1 >= par.Fault_grid[2 + 4*nfault]+3 && gk1 <= par.Fault_grid[3 + 4*nfault]-3 )
  {
    int pos = j + nj * k + mpifaultsize;
    if(F.united[pos]) return;
    F.rup_sensor_Dy[pos] = 0.25 * (-p[pos+1] + 2.0*p[pos] - p[pos-1]);
    F.rup_sensor_Dz[pos] = 0.25 * (-p[pos+nj] + 2.0*p[pos] - p[pos-nj]);
  }
}

__global__
void cal_rup_sensor_magn_cu(Fault F, int nfault, int Faultgrid[])
{

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int ni = par.NX / par.PX;
  int nj = par.NY / par.PY;
  int nk = par.NZ / par.PZ;
  int mpifaultsize = nfault * (nj*nk);        //wangzj

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;
  //int i0 = nx/2;

  real_t *p = F.Ts1 + mpifaultsize;
  real_t *Dy = F.rup_sensor_Dy + mpifaultsize;
  real_t *Dz = F.rup_sensor_Dz + mpifaultsize;

  int gj = par.ranky * nj + j;
  int gk = par.rankz * nk + k;
  int gj1 = gj + 1;
  int gk1 = gk + 1;

  if( gj1 >= Faultgrid[0 + 4*nfault]+3 && gj1 <= Faultgrid[1 + 4*nfault]-3 && 
      gk1 >= Faultgrid[2 + 4*nfault]+3 && gk1 <= Faultgrid[3 + 4*nfault]-3 )
  // if( gj1 >= par.Fault_grid[0]+3 && gj1 <= par.Fault_grid[1]-3 && 
      // gk1 >= par.Fault_grid[2]+3 && gk1 <= par.Fault_grid[3]-3 )
  {
    int pos = j + nj * k;
    real_t D1 = Dy[pos] - Dy[pos+1];
    real_t D2 = Dy[pos] - Dy[pos-1];
    real_t Dp_magn = 0.5 * ( D1*D1 + D2*D2 );

    D1 = Dz[pos] - Dy[pos+nj];
    D2 = Dz[pos] - Dy[pos-nj];
    Dp_magn += 0.5 * ( D1*D1 + D2*D2 );

    real_t p0 = p[pos]*p[pos];

    F.rup_sensor[pos + mpifaultsize] = Dp_magn/(p0+1e-16) + 1e-16;

    if (F.united[pos + mpifaultsize]) F.rup_sensor[pos + mpifaultsize] = 0; 
  }
}

void cal_rup_sensor(Fault F, int nfault, int Faultgrid[])
{
  dim3 block(8, 8, 1);
  dim3 grid(
      (hostParams.nj+block.x-1)/block.x,
      (hostParams.nk+block.y-1)/block.y, 1);

  cal_rup_sensor_D_cu <<<grid, block>>> (F, nfault, Faultgrid);
  cal_rup_sensor_magn_cu <<<grid, block>>> (F, nfault, Faultgrid);
}
