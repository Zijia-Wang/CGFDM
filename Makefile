##############################################################################
#  Makefile
##############################################################################

# configure CUDA and MPI path
#CUDAHOME   := /home/wqzhang/cuda-10.1
#MPIHOME    := /home/wqzhang/install/openmpi-gnu
CUDAHOME   := /public/software/cuda-10.0
MPIHOME    := /public/software/openmpi-4.1.1-cuda.10
NETCDFHOME := /public/software/netcdf-c-4.8.0

CC    := $(MPIHOME)/bin/mpicxx
#CC    := $(MPIHOME)/bin/mpicc
#CC    := g++ -O2
#GC    := nvcc -arch=sm_61 #--use_fast_math
#GC    := nvcc -O3 -arch=sm_61 --ptxas-options=-v -maxrregcount=64
#GC    := nvcc -O3 -arch=sm_61 -Xptxas=-v -maxrregcount=127
#GC    := nvcc -O3 -arch=sm_61 -Xptxas=-v
GC    := $(CUDAHOME)/bin/nvcc -O2 -arch=sm_61 -rdc=true #
#GC     += -Xptxas=-v
#GC     += -maxrregcount=127
#GC     += -Xcompiler -march=native -ccbin g++ -m64

LIB    := -L$(CUDAHOME)/lib64 -lcudart -L$(MPIHOME)/lib -lmpi
#LIB        := -L$(CUDAHOME)/lib64  -L$(MPIHOME)/lib -lmpi
INC    := -I$(CUDAHOME)/include -I$(MPIHOME)/include

DIR_BIN    := ./bin
DIR_SRC    := ./srcd
DIR_OBJ    := ./obj

#=============================================================================
# Macros
#=============================================================================

# use double precison or not [optional]
DoublePrecision := ON

# use PML or use Cerjan(1985) for absorbing [optional]
useCerjan :=
usePML    := ON

# add traction free boundary condition [optional]
FreeSurface := ON

# choose data format (netCDF or binary data)
# Notice! useBin is not working at this stage,
# it will be completed in the future if necessary
useNetCDF := ON
useBin    :=

# Dynamic rupture simulation [optional]
# elasitc wave simulation can be performed if the Rupture macro is not defined
Rupture := ON
Barrier :=
# if the Rupture is ON, then select one of fault friction laws
# SW: slip weakening
# RS: rate-state
# [must choose one of them if Rupture is ON]
#SW  := ON
#RS  := ON
#-----------------------------------------------------------------------------

# benchmark problems of SECE/USGS website (http://scecdata.usc.edu/cvws/)
# these macros are defined to easily perform TPV problems
# [optional]
TPV6   := 
TPV10  := 
TPV28  := 
TPV29  := 
TPV101 := 
TPV102 := 
TPV103 := 
TPV104 :=
TPV22 :=
TPV23 := 
#-----------------------------------------------------------------------------

# use conservative form to solve moment equations [optional]
Conservative := 
#-----------------------------------------------------------------------------

# use traction image method (Zhang, 2006) to deal with the derivative of
# tractions on the fault, or just use low order FD schemes
# [must choose one of them]
TractionLow := 
TractionImg := ON
#-----------------------------------------------------------------------------

# use High-Order Compact FD scheme for calculating velocity derivatives
# near the fault surface
# [optional]
VHOC := 
#-----------------------------------------------------------------------------

DxV_OSD :=
DxV_OSD1 := 
DxV_NCFD := 
DxV_T1 := ON
DxV_T1f := 
DxV_24 := 
DyzV_center := 
DxV_hT1 := 

# smooth near strong boundary (gauss_width = 0.4)
# or perform gaussian smooth for the curved fault (gauss_width = 0.7) 
FaultSmooth := ON 

#-----------------------------------------------------------------------------

# adpatively select long and short FD stencil on the fault
# [optional]
SelectStencil := 
RupSensor :=
#-----------------------------------------------------------------------------

# Prakash-Clifton Regularization for Bimaterial case
# [optional]
Regularize := 
#-----------------------------------------------------------------------------

# Grid Refinement in the Normal (x) direction of the fault
# [optional]
NormalRefine := ON

# Trial Traction Implementation in Runge-Kutte scheme
RKtrial := 

# thermpress
#Thermpress := ON
#-----------------------------------------------------------------------------

DFLAG_LIST := DoublePrecision FreeSurface useCerjan usePML \
    useNetCDF useBin \
    Rupture Regularize NormalRefine Barrier \
    FaultSmooth SelectStencil \
    Conservative \
    TractionLow TractionImg \
    VHOC DxV_OSD DxV_OSD1 DxV_NCFD DxV_T1 DxV_T1f DxV_hT1 DxV_24 DyzV_center \
    RKtrial TP \
    TPV6 TPV10 TPV28 TPV29 TPV101 TPV102 TPV103 TPV104 TPV22 TPV23
#SW RS \

DFLAGS := $(foreach flag,$(DFLAG_LIST),$(if $($(flag)),-D$(flag),)) $(DFLAGS)
DFLAGS := $(strip $(DFLAGS))
#=============================================================================

# use NetCDF or not
INC := $(INC) $(if $(useNetCDF), -I$(NETCDFHOME)/include)
LIB := $(LIB) $(if $(useNetCDF), -L$(NETCDFHOME)/lib -lnetcdf -lpthread)

OBJS := cjson.o set_params.o alloc.o device.o init.o cerjan.o \
    coord.o wave.o  media.o source.o \
    fault_coef.o \
    rk4.o mathFuncs.o metric.o \
    mod_mpi.o  \
    io_wave_xy.o io_wave_xz.o io_wave_yz.o  \
    rup_sensor.o \
    main.o io_recv.o 

OBJS += $(if $(usePML),init_pml.o pmlx.o pmly.o pmlz.o)

# OBJS += $(if $(Rupture),\
#     transform.o fault_dvelo.o fault_dstrs_f.o fault_dstrs_b.o \
#     mod_mpi_fault.o wjl_io_faultandwave_0929.o)

OBJS += $(if $(Rupture),\
    transform.o fault_dvelo.o fault_dstrs_f.o fault_dstrs_b.o \
    mod_mpi_fault.o io_faultandwaveNotstream.o)

#OBJS += $(if $(Rupture)$(SW),trial_slipweakening.o)
#OBJS += $(if $(Rupture)$(RS),trial_ratestate.o)
OBJS += $(if $(SW),trial_slipweakening.o)
OBJS += $(if $(RS),trial_ratestate.o)
OBJS += trial_slipweakening.o
OBJS += trial_ratestate.o
OBJS += $(if $(FaultSmooth),smooth.o)
#OBJS += $(if $(Thermpress),thermpress.o)
OBJS += thermpress.o
#OBJS += $(if $(TP),thermpress.o)
#OBJS += $(if $(FaultSmooth),filter.o)


OBJS := $(addprefix $(DIR_OBJ)/,$(OBJS))

vpath  %.c .
vpath  %.cu .
#
$(DIR_BIN)/a.out: $(OBJS)
	$(GC) $^ $(LIB) -o $@
$(DIR_OBJ)/%.o : $(DIR_SRC)/%.c
	$(CC) $(DFLAGS) $(INC) -c $^ -o $@
$(DIR_OBJ)/%.o : $(DIR_SRC)/%.cu
	$(GC) $(DFLAGS) $(INC) -c $^ -o $@

tool:
	g++ -Wall $(DIR_SRC)/tool_cal_fault_metric.c \
	-o $(DIR_BIN)/tool_cal_fault_metric \
	-I$(NETCDFHOME)/include -L$(NETCDFHOME)/lib -lnetcdf

clean:
	rm -rf $(OBJS)
	rm -rf $(DIR_BIN)/a.out
