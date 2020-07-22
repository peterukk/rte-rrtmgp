! This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
!
! Contacts: Robert Pincus and Eli Mlawer
! email:  rrtmgp@aer.com
!
! Copyright 2015-2018,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
! -------------------------------------------------------------------------------------------------
!
! Example program to demonstrate the calculation of longwave radiative fluxes in clear, aerosol-free skies.
!   The example files come from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
!   The large problem (1800 profiles) is divided into blocks
!
! Program is invoked as rrtmgp_rfmip_lw [block_size inp_outp_file  coefficient_file upflux_file downflux_file]
!   All arguments are optional but need to be specified in order.
!
! -------------------------------------------------------------------------------------------------
!
! Error checking: Procedures in rte+rrtmgp return strings which are empty if no errors occured
!   Check the incoming string, print it out and stop execution if non-empty
!
subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  use iso_c_binding
  character(len=*), intent(in) :: error_msg

  if(error_msg /= "") then
    write (error_unit,*) trim(error_msg)
    write (error_unit,*) "rrtmgp_rfmip_lw stopping"
    stop
  end if
end subroutine stop_on_err
! -------------------------------------------------------------------------------------------------
!
! Main program
!
! -------------------------------------------------------------------------------------------------
program rrtmgp_rfmip_lw
  ! --------------------------------------------------
  !
  ! Modules for working with rte and rrtmgp
  !
#ifdef USE_OPENMP
  use omp_lib
#endif
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp, wl, i4
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !   Shortwave calculations would use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  use mo_optical_props,      only: ty_optical_props_1scl
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly...
  !
  use mo_gas_concentrations, only: ty_gas_concs
  !
  ! ... and another type to encapsulate the longwave source functions.
  !
  use mo_source_functions,   only: ty_source_func_lw
  !
  ! RTE longwave driver
  !
  use mo_rte_lw,             only: rte_lw
  !
  ! RTE driver uses a derived type to reduce spectral fluxes to whatever the user wants
  !   Here we're just reporting broadband fluxes
  !
  use mo_fluxes,             only: ty_fluxes_broadband
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_rfmip_io,           only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   unblock_and_write_3D, unblock_and_write_3D_sp, unblock, &
                                   read_and_block_lw_bc, determine_gas_names
  use mo_simple_netcdf,      only: read_field, write_field, get_dim_size
  use netcdf
  use mod_network
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif
  implicit none

#ifdef USE_PAPI  
#include "f90papi.h"
#endif  
  ! --------------------------------------------------
  !
  ! Local variables
  !
  character(len=132) :: rfmip_file,kdist_file
  character(len=132) :: flxdn_file, flxup_file, inp_outp_file, flx_file, flx_file_ref, flx_file_lbl
  integer            :: nargs, ncol, nlay, nbnd, ngas, ngpt, nexp, nblocks, block_size, forcing_index, physics_index, n_quad_angles = 1
  logical             :: top_at_1
  integer            :: b, icol, ibnd, igpt, count_rate, iTime1, iTime2, ncid, ninputs
  character(len=4)   :: block_size_char, forcing_index_char = '1', physics_index_char = '1'
  
  character(len=32 ), &
            dimension(:),             allocatable :: kdist_gas_names, rfmip_gas_names
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev ! block_size, nlay, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn
  real(wp), dimension(:,:,:),         allocatable :: rlu_ref, rld_ref, rlu_nn, rld_nn, rlu_lbl, rld_lbl, rldu_ref, rldu_nn, rldu_lbl
  real(sp), dimension(:,:,:,:),       allocatable :: tau_lw, planck_frac
  real(sp), dimension(:,:,:,:),       allocatable :: nn_input
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis, sfc_t  ! block_size, nblocks (emissivity is spectrally constant)
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis_spec    ! nbands, block_size (spectrally-resolved emissivity)
  real(wp), dimension(:),             allocatable :: means,stdevs ,temparray

  character (len = 80)                :: modelfile_tau, modelfile_source
  type(network_type), dimension(2)    :: neural_nets ! One model for predicting planck fractions, one for optical depths

  logical 		:: use_nn, save_output, save_input, compare_flux, save_flux, tlev_provided
  integer(i4)                               :: check          ! PAPI's error flag WHICH SHOULD BE CHECKED!
  !
  ! Classes used by rte+rrtmgp
  !
  type(ty_gas_optics_rrtmgp)  :: k_dist
  type(ty_source_func_lw)     :: source

  type(ty_optical_props_1scl) :: optical_props
  type(ty_fluxes_broadband)   :: fluxes
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array

#ifdef USE_TIMING
  integer :: ret, i
  print *, "using GPTL timing library"
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
#ifdef USE_PAPI  
  ret = GPTLsetoption (PAPI_SP_OPS, 1);
#endif  
  ret = gptlinitialize()
#endif

  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional

  !  ------------ I/O and settings -----------------
  ! Use neural networks for gas optics? 
  use_nn      = .true.
  ! Save outputs (tau, planck fracs) and inputs (scaled gases)
  save_input  = .false.
  save_output = .false.
  ! Save fluxes
  save_flux    = .false.
  ! compare fluxes to reference code as well as line-by-line (RFMIP only)
  compare_flux = .true.

  ! ------------ Neural network model weights -----------------
  ! Planck model
  ! modelfile_source        = "../../neural/data/pfrac-19-16-16-pow2-minmax2.txt" 
  modelfile_source        = "../../neural/data/pfrac-18-16-16_new2.txt" 
  ! Optical depth model
  ! modelfile_tau           = "../../neural/data/tau-lw-19-64-64-ynorm-pow8-minmax3.txt"  
  modelfile_tau           = "../../neural/data/tau-lw-18-58-58_new.txt" 

  ! modelfile_tau           = "../../neural/data/tau-lw-tmp.txt"

  ! Save upwelling and downwelling fluxes in the same file
  flx_file = 'output_fluxes/rlud_Efx_RTE-RRTMGP-NN-181204_rad-irf_r1i1p1f1_gn.nc'
  ! flx_file = 'output_fluxes/rlud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc'
  

  ! FOR NN MODEL DEVELOPMENT 
  ! Where to save neural network inputs (using NN code) and target outputs (planck fractions and optical depths, using reference code)

  ninputs = 18
  ! inp_outp_file =  "../../../rrtmgp_dev/inputs_outputs/inp_outp_lw_CKDMIP-MM-2.0_1f1.nc"
  ! inp_outp_file =  "../../../rrtmgp_dev/inputs_outputs/inp_outp_lw_RFMIP-Halton-rnd-0.4-2.0_1f1.nc"

  ! The coefficients for scaling the INPUTS are currently still hard-coded in mo_gas_optics_rrtmgp.F90

  if (use_nn) then
	  print *, 'loading planck fraction model from ', modelfile_source
    call neural_nets(1) % load(modelfile_source)

	  print *, 'loading tau model from ', modelfile_tau
    call neural_nets(2) % load(modelfile_tau)
        
    ! Here we can change the neural network computational kernels (SGEMM is fastest, but requires a BLAS library)
    ! If BLAS is not available, output_opt_flatmodel is the fastest kernel

#ifndef USE_OPENACC
    call change_kernel(neural_nets(1),  output_opt_flatmodel, output_sgemm_pfrac) ! custom pfrac kernel (outputs are post-processed inside kernel)
    call change_kernel(neural_nets(2),  output_opt_flatmodel, output_sgemm_tau)   ! custom tau kernel 
#endif
  end if  

  !
  print *, "Usage: rrtmgp_rfmip_lw [block_size] [rfmip_file] [k-distribution_file] [forcing_index (1,2,3)] [physics_index (1,2)]"
  nargs = command_argument_count()

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size
  if(nargs >= 2) call get_command_argument(2, rfmip_file)
  if(nargs >= 3) call get_command_argument(3, kdist_file)
  if(nargs >= 4) call get_command_argument(4, forcing_index_char)
  if(nargs >= 5) call get_command_argument(5, physics_index_char)

  ! How big is the problem? Does it fit into blocks of the size we've specified?
  !
  call read_size(rfmip_file, ncol, nlay, nexp)
  print *, "input file:", rfmip_file
  print *, "ncol:", ncol
  print *, "nexp:", nexp
  print *, "nlay:", nlay

  if(mod(ncol*nexp, block_size) /= 0 ) call stop_on_err("rrtmgp_rfmip_lw: number of columns doesn't fit evenly into blocks.")
  nblocks = (ncol*nexp)/block_size
  print *, "Doing ",  nblocks, "blocks of size ", block_size

  read(forcing_index_char, '(i4)') forcing_index
  if(forcing_index < 1 .or. forcing_index > 4) &
    stop "Forcing index is invalid (must be 1,2 or 3)"

  read(physics_index_char, '(i4)') physics_index
  if(physics_index < 1 .or. physics_index > 2) &
    stop "Physics index is invalid (must be 1 or 2)"
  if(physics_index == 2) n_quad_angles = 3

  n_quad_angles = 1
                
  !
  ! Identify the set of gases used in the calculation based on the forcing index
  !   A gas might have a different name in the k-distribution than in the files
  !   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  !
  call determine_gas_names(rfmip_file, kdist_file, forcing_index, kdist_gas_names, rfmip_gas_names)
  print *, "Calculation uses RFMIP gases: ", (trim(rfmip_gas_names(b)) // " ", b = 1, size(rfmip_gas_names))
  print *, "-----------------"
  print *, "Calculation uses RFMIP gases: ", (trim(kdist_gas_names(b)) // " ", b = 1, size(kdist_gas_names))

  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  !
  ! Allocation on assignment within reading routines
  !
  
  call read_and_block_pt(rfmip_file, block_size, p_lay, p_lev, t_lay, t_lev)
  print *, "shape t_lay, min, max", shape(t_lay), maxval(t_lay), minval(t_lay)

  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)

  !
  ! Read the gas concentrations and surface properties
  !

  call read_and_block_gases_ty(rfmip_file, block_size, kdist_gas_names, rfmip_gas_names, gas_conc_array)

  call read_and_block_lw_bc(rfmip_file, block_size, sfc_emis, sfc_t)
  
  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))

  print *, "min of play", minval(p_lay), "p_lay = k_dist%get_press_min()", k_dist%get_press_min() 

  where(p_lay < k_dist%get_press_min()) p_lay = k_dist%get_press_min() + spacing (k_dist%get_press_min())

  print *, "min of play", minval(p_lay)

  if(.not. k_dist%source_is_internal()) &
    stop "rrtmgp_rfmip_lw: k-distribution file isn't LW"

  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()
  ngas = k_dist%get_ngas()
  print *, "in total: ", ngas, " input gases"
  !
  ! RRTMGP won't run with pressure less than its minimum. The top level in the RFMIP file
  !   is set to 10^-3 Pa. Here we pretend the layer is just a bit less deep.
  !   This introduces an error but shows input sanitizing.
  !
  if(top_at_1) then
    p_lev(1,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lev(nlay+1,:,:) &
                 = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  end if

  !
  ! Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  !   gas optical properties, and source functions. The %alloc() routines carry along
  !   the spectral discretization from the k-distribution.
  !

  allocate(flux_up(    	nlay+1, block_size, nblocks), &
           flux_dn(    	nlay+1, block_size, nblocks))
  allocate(sfc_emis_spec(nbnd, block_size))

  if (save_output) then 
    allocate(tau_lw(    	ngpt, nlay, block_size, nblocks))
    !tau_lw = 0.0_sp
    allocate(planck_frac(	ngpt, nlay, block_size, nblocks))
    !planck_frac = 0.0_sp
  end if
  if (save_input) then
    allocate(nn_input( 	ninputs, nlay, block_size, nblocks)) ! dry air + gases + temperature + pressure
  end if

  call stop_on_err(source%alloc            (block_size, nlay, k_dist))   
  call stop_on_err(optical_props%alloc_1scl(block_size, nlay, k_dist))

  !
  ! OpenACC directives put data on the GPU where it can be reused with communication
  ! NOTE: these are causing problems right now, most likely due to a compiler
  ! bug related to the use of Fortran classes on the GPU.
  !
  !$acc enter data create(sfc_emis_spec)
  !$acc enter data create(optical_props, optical_props%tau)

  ! already created in constructor?
  ! ! $acc enter data create(source, source%lay_source, source%lev_source_inc, source%lev_source_dec, source%sfc_source)
  ! --------------------------------------------------

#ifdef USE_OPENMP
    !print *, "OpenMP processes available:", omp_get_num_procs()
#endif
  if (use_nn) then
    print *, "starting computations, using neural networks for RRTMGP kernel"
  else
    print *, "starting computations"
  end if
  call system_clock(count_rate=count_rate)
  call system_clock(iTime1)
  !
  ! Loop over blocks
  !

!bo $OMP PARALLEL shared(neural_nets, k_dist) firstprivate(sfc_emis_spec,fluxes,optical_props,source)
!bo $OMP DO 

#ifdef USE_TIMING
do i = 1, 10
#endif
  do b = 1, nblocks
    
    ! print *, b, "/", nblocks
#ifdef USE_OPENMP
    ! PRINT *, "Hello from process: ", OMP_GET_THREAD_NUM()
    ! print *, "my t_lay(5,5,b) for b:",b,"  is", t_lay(5,5,b)
#endif

    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)    

    !
    ! Expand the spectrally-constant surface emissivity to a per-band emissivity for each column
    !   (This is partly to show how to keep work on GPUs using OpenACC)
    !
    !$acc parallel loop collapse(2) copyin(sfc_emis)
    do icol = 1, block_size
      do ibnd = 1, nbnd
        sfc_emis_spec(ibnd,icol) = sfc_emis(icol,b)
      end do
    end do

    !
    ! Compute the optical properties of the atmosphere and the Planck source functions
    !    from pressures, temperatures, and gas concentrations...
    !

#ifdef USE_TIMING
    ret =  gptlstart('gas_optics (LW)')
#endif
optical_props%tau = 0.0
    if (use_nn) then
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b),          &
                                          p_lev(:,:,b),         &
                                          t_lay(:,:,b),         &
                                          sfc_t(:  ,b),         &
                                          gas_conc_array(b),    &
                                          optical_props,        &
                                          source,               &
                                          neural_nets,          & !net_pfrac, net_tau
                                         tlev = t_lev(:,:,b)))

    else 
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b),      &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        sfc_t(:  ,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        source,             &
                                        tlev = t_lev(:,:,b)))!, nn_inputs= nn_input(:,:,:,b)))
    end if

  !  !$acc update self(optical_props%tau)
  !   print *," max, min (tau)",   maxval(optical_props%tau), minval(optical_props%tau)

  !! !$acc update self(source%planck_frac)  
    ! print *," max, min (pfrac)", maxval(source%planck_frac), minval(source%planck_frac)


#ifdef USE_TIMING
    ret =  gptlstop('gas_optics (LW)')
#endif
    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
#ifdef USE_TIMING
    ret =  gptlstart('rte_lw')
#endif
   ! !$acc enter data create(fluxes)
   ! !$acc enter data create(fluxes%flux_up,fluxes%flux_dn)
    call stop_on_err(rte_lw(optical_props,   &
                            top_at_1,        &
                            source,          &
                            sfc_emis_spec,   &
                            fluxes, n_gauss_angles = n_quad_angles, use_2stream = .false., compute_gpoint_fluxes = .false.))

#ifdef USE_TIMING
    ret =  gptlstop('rte_lw')
#endif
    ! Save optical depths
    if (save_output) then
      tau_lw(:,:,:,b)      = optical_props%tau(:,:,:)
      planck_frac(:,:,:,b) = source%planck_frac(:,:,:)
    end if

  end do ! blocks

#ifdef USE_TIMING
end do
#endif

!bo $OMP END DO
!bo $OMP END PARALLEL
!bo $OMP barrier

call source%finalize()
!$acc exit data delete(source)

  !$acc exit data delete(sfc_emis_spec)
  !$acc exit data delete(optical_props%tau, optical_props)

  ! deallocate(sfc_emis_spec, gas_conc_array, optical_props%tau, source%planck_frac, source%lay_source_bnd, source%lev_source_bnd,source%sfc_source_bnd)

#ifdef USE_TIMING
  ! End timers
  !
  ret = gptlpr(block_size)
  ret = gptlfinalize()
#endif
  call system_clock(iTime2)
  print *,'Elapsed time on everything ',real(iTime2-iTime1)/real(count_rate)

  ! --------------------------------------------------m
  
  allocate(temparray(   block_size*(nlay+1)*nblocks)) 
  temparray = pack(flux_dn(:,:,:),.true.)
  print *, "mean of flux_down is:", sum(temparray, dim=1)/size(temparray, dim=1)  !  mean of flux_down is:   103.2458
  temparray = pack(flux_up(:,:,:),.true.)
  print *, "mean of flux_up is:", sum(temparray, dim=1)/size(temparray, dim=1)
  deallocate(temparray)

  if (save_output) then 
    allocate(temparray(   block_size*nlay*ngpt*nblocks)) 
    temparray = pack(tau_lw(:,:,:,:),.true.)
    print *, "mean of tau is", sum(temparray, dim=1)/size(temparray, dim=1)
    print *, "max of tau is", maxval(tau_lw)
    print *, "min of tau is", minval(tau_lw)
      ! Save optical depths and planck fractions
    print *, "Attempting to save outputs to" , inp_outp_file
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'tau_lw',tau_lw)
    temparray = pack(planck_frac(:,:,:,:),.true.)
    print *, "mean of pfrac is", sum(temparray, dim=1)/size(temparray, dim=1)
    print *, "max of pfrac is", maxval(planck_frac)
    print *, "min of pfrac is", minval(planck_frac)
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'planck_frac',planck_frac)
    print *, "outputs saved to ", inp_outp_file
    deallocate(temparray)
  end if

  ! Save neural network inputs ?
  if (save_input) then
    print *, "Attempting to save neural network inputs to ", inp_outp_file
    ! This function also deallocates the input 
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'nn_input',nn_input)
    print *, "inputs saved to ", inp_outp_file
  end if

  ! Save fluxes ?
  if (save_flux) then
    print *, "Attempting to save fluxes to ", flx_file
    call unblock_and_write(trim(flx_file), 'rlu', flux_up)
    call unblock_and_write(trim(flx_file), 'rld', flux_dn)
    print *, "Fluxes saved to ", flx_file
  end if 

  if (compare_flux) then
    print *, "-----------------------------------------------------------------------------------------"
    print *, "-----COMPARING ERRORS (W.R.T LINE-BY-LINE) OF NEURAL NETWORK AND ORIGINAL SCHEME --------"
    print *, "-----------------------------------------------------------------------------------------"

    allocate(rld_ref( nlay+1, ncol, nexp))
    allocate(rlu_ref( nlay+1, ncol, nexp))  
    allocate(rldu_ref( nlay+1, ncol, nexp))  
    allocate(rld_nn( nlay+1, ncol, nexp))
    allocate(rlu_nn( nlay+1, ncol, nexp))
    allocate(rldu_nn( nlay+1, ncol, nexp))
    allocate(rld_lbl( nlay+1, ncol, nexp))
    allocate(rlu_lbl( nlay+1, ncol, nexp))
    allocate(rldu_lbl( nlay+1, ncol, nexp))

    flx_file_ref = 'output_fluxes/rlud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc'
    flx_file_lbl = 'output_fluxes/rlud_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'

    call unblock(flux_up, rlu_nn)
    call unblock(flux_dn, rld_nn)

    rldu_nn = rld_nn - rlu_nn

    if(nf90_open(trim(flx_file_ref), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_ref))

    rlu_ref = read_field(ncid, "rlu", nlay+1, ncol, nexp)
    rld_ref = read_field(ncid, "rld", nlay+1, ncol, nexp)
    rldu_ref = rld_ref - rlu_ref

    if(nf90_open(trim(flx_file_lbl), NF90_NOWRITE, ncid) /= NF90_NOERR) &
    call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_lbl))

    rlu_lbl = read_field(ncid, "rlu", nlay+1, ncol, nexp)
    rld_lbl = read_field(ncid, "rld", nlay+1, ncol, nexp)
    rldu_lbl = rld_lbl - rlu_lbl

    print *, "------------- UPWELLING -------------- "

    print *, "MAE in upwelling fluxes of NN and RRTMGP, present-day:            ", &
     mae(reshape(rlu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rlu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in upwelling fluxes of NN and RRTMGP, future:                 ", &
     mae(reshape(rlu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rlu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rlu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "bias in upwelling flux of NN and RRTMGP, present-day, top-of-atm.:", &
      bias(reshape(rlu_lbl(1,:,1), shape = [1*ncol]),    reshape(rlu_nn(1,:,1), shape = [1*ncol])), &
      bias(reshape(rlu_lbl(1,:,1), shape = [1*ncol]),    reshape(rlu_ref(1,:,1), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, future, top-of-atm.:     ", &
      bias(reshape(rlu_lbl(1,:,4), shape = [1*ncol]),    reshape(rlu_nn(1,:,4), shape = [1*ncol])), &
      bias(reshape(rlu_lbl(1,:,4), shape = [1*ncol]),    reshape(rlu_ref(1,:,4), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, future-all, top-of-atm.: ", &
      bias(reshape(rlu_lbl(1,:,17), shape = [1*ncol]),    reshape(rlu_nn(1,:,17), shape = [1*ncol])), &
      bias(reshape(rlu_lbl(1,:,17), shape = [1*ncol]),    reshape(rlu_ref(1,:,17), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, ALL EXPS, top-of-atm.:   ", &
      bias(reshape(rlu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rlu_nn(1,:,:), shape = [nexp*ncol])), &
      bias(reshape(rlu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rlu_ref(1,:,:), shape = [nexp*ncol])) 


    print *, "-------------- DOWNWELLING --------------"

    print *, "MAE in downwelling fluxes of NN and RRTMGP, present-day:          ", &
     mae(reshape(rld_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rld_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN and RRTMGP, future:               ", &
     mae(reshape(rld_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
    mae(reshape(rld_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rld_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "-------------- NET FLUX --------------"

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, pres.day:  ", &
     maxval(rldu_lbl(:,:,1)-rldu_nn(:,:,1)), maxval(rldu_lbl(:,:,1)-rldu_ref(:,:,1))

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, future:    ", &
     maxval(rldu_lbl(:,:,4)-rldu_nn(:,:,4)), maxval(rldu_lbl(:,:,4)-rldu_ref(:,:,4))

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, future-all:", &
     maxval(rldu_lbl(:,:,17)-rldu_nn(:,:,17)), maxval(rldu_lbl(:,:,17)-rldu_ref(:,:,17))

     print *, "---------"

     print *, "MAE in net fluxes of NN and RRTMGP, present-day:               ", &
     mae(reshape(rldu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
     mae(reshape(rldu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    print *, "MAE in net fluxes of NN and RRTMGP, future:                    ", &
     mae(reshape(rldu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
     mae(reshape(rldu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in net fluxes of NN and RRTMGP, future-all:                ", &
     mae(reshape(rldu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,17), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rldu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

     print *, "MAE in net fluxes of NN and RRTMGP, ALL EXPS:                  ", &
     mae(reshape(rldu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rldu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)])), &
     mae(reshape(rldu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rldu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)])) 

    print *, "---------"

    print *, "RMSE in net fluxes of NN and RRTMGP, present-day, PBL:         ", &
     rmse(reshape(rldu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rldu_nn(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)])), &
     rmse(reshape(rldu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rldu_ref(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]))

    print *, "RMSE in net fluxes of NN and RRTMGP, present-day, SURFACE:     ", &
     rmse(reshape(rldu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rldu_nn(nlay+1,:,1), shape = [1*ncol])), &
     rmse(reshape(rldu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rldu_ref(nlay+1,:,1), shape = [1*ncol]))

    print *, "RMSE in net fluxes of NN and RRTMGP, future-all, SURFACE:     ", &
     rmse(reshape(rldu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rldu_nn(nlay+1,:,17), shape = [1*ncol])), &
     rmse(reshape(rldu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rldu_ref(nlay+1,:,17), shape = [1*ncol]))

    print *, "RMSE in net fluxes of NN and RRTMGP, pre-industrial, SURFACE: ", &
     rmse(reshape(rldu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rldu_nn(nlay+1,:,2), shape = [1*ncol])), &
     rmse(reshape(rldu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rldu_ref(nlay+1,:,2), shape = [1*ncol]))

    print *, "---------"

    print *, "bias in net fluxes of NN and RRTMGP, present-day:              ", &
     bias(reshape(rldu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rldu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    print *, "bias in net fluxes of NN and RRTMGP, present-day, SURFACE:     ", &
     bias(reshape(rldu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rldu_nn(nlay+1,:,1), shape = [1*ncol])), &
     bias(reshape(rldu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rldu_ref(nlay+1,:,1), shape = [1*ncol])) 

    print *, "bias in net fluxes of NN and RRTMGP, future:                   ", &
     bias(reshape(rldu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rldu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "bias in net fluxes of NN and RRTMGP, future-all:               ", &
     bias(reshape(rldu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rldu_nn(:,:,17), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rldu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rldu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

    print *, "bias in net fluxes of NN and RRTMGP, future-all, SURFACE:      ", &
     bias(reshape(rldu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rldu_nn(nlay+1,:,17), shape = [1*ncol])), &
     bias(reshape(rldu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rldu_ref(nlay+1,:,17), shape = [1*ncol])) 

    print *, "---------"

    print *, "MAE in upwelling fluxes of NN w.r.t RRTMGP, present-day:       ", &
     mae(reshape(rlu_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN w.r.t RRTMGP, present-day:     ", &
     mae(reshape(rld_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "Max-diff in d.w. flux w.r.t RRTMGP ", &
     maxval(rld_ref(:,:,:)-rld_nn(:,:,:))
 
    print *, "Max-diff in u.w. flux w.r.t RRTMGP ", &
     maxval(rlu_ref(:,:,:)-rlu_nn(:,:,:))

    print *, "Max-diff in net flux w.r.t RRTMGP  ", &
     maxval(rldu_ref(:,:,:)-rldu_nn(:,:,:)) 

    deallocate(rld_ref,rlu_ref,rld_nn,rlu_nn,rld_lbl,rlu_lbl,rldu_ref,rldu_nn,rldu_lbl)

  end if

  deallocate(flux_up, flux_dn)
  print *, "SUCCESS!"

  contains
  subroutine standardscaler(x,means,stdevs)
    implicit none
    real(wp), dimension(:,:,:,:), intent(inout) :: x 
    real(wp), dimension(:),       intent(in   ) :: means,stdevs

    integer :: i

    do i=1,ngas
      x(:,:,i,:) = x(:,:,i,:) - means(i) 
      x(:,:,i,:) = x(:,:,i,:) / stdevs(i)
    end do
  end subroutine standardscaler

  function rmse(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = x1 - x2
    res = sqrt( sum(diff**2)/size(diff) )
  end function rmse

  function mae(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff, dim=1)/size(diff, dim=1)
  end function mae

  function bias(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: mean1,mean2, res
    
    mean1 = sum(x1, dim=1)/size(x1, dim=1)
    mean2 = sum(x2, dim=1)/size(x2, dim=1)
    res = mean1 - mean2

  end function bias

  function mean(x1) result(mean1)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1
    real(wp) :: mean1
    
    mean1 = sum(x1, dim=1)/size(x1, dim=1)

  end function mean

end program rrtmgp_rfmip_lw

