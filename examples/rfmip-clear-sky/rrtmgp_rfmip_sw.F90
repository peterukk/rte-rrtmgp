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
! Example program to demonstrate the calculation of shortwave radiative fluxes in clear, aerosol-free skies.
!   The example files come from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
!   The large problem (1800 profiles) is divided into blocks
!
! Program is invoked as rrtmgp_rfmip_sw [block_size input_file  coefficient_file upflux_file downflux_file]
!   All arguments are optional but need to be specified in order.
!
! -------------------------------------------------------------------------------------------------
!
! Error checking: Procedures in rte+rrtmgp return strings which are empty if no errors occured
!   Check the incoming string, print it out and stop execution if non-empty
!
subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  character(len=*), intent(in) :: error_msg

  if(error_msg /= "") then
    write (error_unit,*) trim(error_msg)
    write (error_unit,*) "rrtmgp_rfmip_sw stopping"
    stop
  end if
end subroutine stop_on_err
! -------------------------------------------------------------------------------------------------
!
! Main program
!
! -------------------------------------------------------------------------------------------------
program rrtmgp_rfmip_sw
  ! --------------------------------------------------
  !
  ! Modules for working with rte and rrtmgp
  !
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp
  !
  ! Array utilities
  !
  use mo_rte_util_array,     only: zero_array
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  use mo_optical_props,      only: ty_optical_props_2str
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly
  !
  use mo_gas_concentrations, only: ty_gas_concs
  !
  ! RTE shortwave driver
  !
  use mo_rte_sw,             only: rte_sw
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
                                   read_and_block_sw_bc, determine_gas_names, unblock_and_write2                             
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
  character(len=132) :: rfmip_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                        kdist_file = 'coefficients_sw.nc'
  character(len=132) :: flx_file, flx_file_ref, flx_file_lbl, inp_outp_file
  integer            :: nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size, forcing_index
  logical 		       :: top_at_1, use_nn, save_input_output, compare_flux, save_flux
  integer            :: b, icol, ibnd, igpt, igas, ncid, ngas, ninputs
  character(len=4)   :: block_size_char, forcing_index_char = '1'
  character(len=32 ), &
            dimension(:),             allocatable :: kdist_gas_names, rfmip_gas_games
    character (len = 80)                :: modelfile_tau, modelfile_ray
  
  type(network_type), dimension(2)    :: neural_nets ! First model for predicting optical depths, second for planck fractions          
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev ! block_size, nlay, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:),         allocatable :: rsu_ref, rsd_ref, rsu_nn, rsd_nn, rsu_lbl, rsd_lbl, rsdu_ref, rsdu_nn, rsdu_lbl, col_dry
  real(sp), dimension(:,:,:,:),       allocatable :: nn_input, tau_sw, ssa
  real(wp), dimension(:,:  ),         allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
                                                     ! block_size, nblocks
  real(wp), dimension(:,:  ),         allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  real(wp), dimension(:),             allocatable :: temparray

  !
  ! Classes used by rte+rrtmgp
  !
  type(ty_gas_optics_rrtmgp)                     :: k_dist
  type(ty_optical_props_2str)                    :: optical_props
  type(ty_fluxes_broadband)                      :: fluxes

  real(wp), dimension(:,:), allocatable          :: toa_flux ! block_size, ngpt
  real(wp), dimension(:  ), allocatable          :: def_tsi, mu0    ! block_size
  logical , dimension(:,:), allocatable          :: usecol ! block_size, nblocks
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array
  real(wp), parameter :: deg_to_rad = acos(-1._wp)/180._wp
#ifdef USE_TIMING
  integer :: ret, i
#endif


  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
  !
  !  ------------ I/O and settings -----------------
  ! Use neural networks for gas optics? 
  use_nn      = .true.
  ninputs     =  7
  ! Save neural network inputs (gas concentrations) and target outputs (tau, ssa)
  save_input_output  = .false.
  inp_outp_file =  "../../../../rrtmgp_dev/inputs_outputs/inp_outp_sw_Garand-big_1f1.nc"

  ! Save fluxes
  save_flux    = .true.
  ! compare fluxes to reference code as well as line-by-line (RFMIP only)
  compare_flux = .true.


  modelfile_tau           = "../../neural/data/tau-sw-abs-7-16-16-mae_2.txt" 
  modelfile_ray           = "../../neural/data/tau-sw-ray-7-16-16_2.txt" 

  if (use_nn) then
	  print *, 'loading tau model from ', modelfile_tau
    call neural_nets(1) % load(modelfile_tau)
    print *, 'loading rayleigh model from ', modelfile_ray
    call neural_nets(2) % load(modelfile_ray)
  end if  

  print *, "Usage: rrtmgp_rfmip_lw [block_size] [rfmip_file] [k-distribution_file] [forcing_index (1,2,3)] [physics_index (1,2)]"
  nargs = command_argument_count()

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size
  if(nargs >= 2) call get_command_argument(2, rfmip_file)
  if(nargs >= 3) call get_command_argument(3, kdist_file)
  if(nargs >= 4) call get_command_argument(4, forcing_index_char)

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

  if (use_nn) then
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-NN-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  else
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  end if
  !
  ! Identify the set of gases used in the calculation based on the forcing index
  !   A gas might have a different name in the k-distribution than in the files
  !   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  !
  call determine_gas_names(rfmip_file, kdist_file, forcing_index, kdist_gas_names, rfmip_gas_games)
  print *, "Calculation uses RFMIP gases: ", (trim(rfmip_gas_games(b)) // " ", b = 1, size(rfmip_gas_games))

  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  !
  ! Allocation on assignment within reading routines
  !
  call read_and_block_pt(rfmip_file, block_size, p_lay, p_lev, t_lay, t_lev)
  print *, "shape t_lay, min, max", shape(t_lay), maxval(t_lay), minval(t_lay)
  !
  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)

  !
  ! Read the gas concentrations and surface properties
  !
  call read_and_block_gases_ty(rfmip_file, block_size, kdist_gas_names, rfmip_gas_games, gas_conc_array)

  call read_and_block_sw_bc(rfmip_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)
  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))
  if(.not. k_dist%source_is_external()) &
    stop "rrtmgp_rfmip_sw: k-distribution file isn't SW"
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()

  allocate(toa_flux(k_dist%get_ngpt(), block_size), &
           def_tsi(block_size), usecol(block_size,nblocks))
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
  ! RTE will fail if passed solar zenith angles greater than 90 degree. We replace any with
  !   nighttime columns with a default solar zenith angle. We'll mask these out later, of
  !   course, but this gives us more work and so a better measure of timing.
  !
  do b = 1, nblocks
    usecol(1:block_size,b)  = solar_zenith_angle(1:block_size,b) < 90._wp - 2._wp * spacing(90._wp)
  end do

  !
  ! Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  !   gas optical properties, and source functions. The %alloc() routines carry along
  !   the spectral discretization from the k-distribution.
  !

  allocate(flux_up(    	nlay+1, block_size, nblocks), &
           flux_dn(    	nlay+1, block_size, nblocks))

  allocate(flux_dn_dir(    	nlay+1, block_size, nblocks))

  allocate(mu0(block_size), sfc_alb_spec(nbnd,block_size))
  call stop_on_err(optical_props%alloc_2str(block_size, nlay, k_dist))
  !$acc enter data create(optical_props, optical_props%tau, optical_props%ssa, optical_props%g)
  !$acc enter data create (toa_flux, def_tsi)
  !$acc enter data create (sfc_alb_spec, mu0)

  if (save_input_output) then
    allocate(nn_input( 	ninputs,  nlay, block_size, nblocks)) ! temperature + pressure + gases
    allocate(col_dry(             nlay, block_size, nblocks)) ! number of dry air molecules
    allocate(tau_sw(    	ngpt,   nlay, block_size, nblocks))
    allocate(ssa(	        ngpt,   nlay, block_size, nblocks))
  end if

  ! --------------------------------------------------
#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
#ifdef USE_PAPI  
  ret = GPTLsetoption (PAPI_SP_OPS, 1);
#endif  
  ret =  gptlinitialize()
#endif
  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
do i = 1, 10
#endif
  do b = 1, nblocks
    ! print *, b, "/", nblocks


    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)
    fluxes%flux_dn_dir => flux_dn_dir(:,:,b)
    !
    ! Compute the optical properties of the atmosphere and the Planck source functions
    !    from pressures, temperatures, and gas concentrations...
    !
#ifdef USE_TIMING
    ret =  gptlstart('gas_optics (SW)')
#endif


      ! print *,  "min col_dry, max input", minval(col_dry), maxval(nn_input(:,:,:,b))

    if (use_nn) then
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
      p_lev(:,:,b),       &
      t_lay(:,:,b),       &
      gas_conc_array(b),  &
      optical_props,      &
      toa_flux,neural_nets=neural_nets))
    else
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        toa_flux))
    end if
#ifdef USE_TIMING
    ret =  gptlstop('gas_optics (SW)')
#endif
    ! Boundary conditions
    !   (This is partly to show how to keep work on GPUs using OpenACC in a host application)
    ! What's the total solar irradiance assumed by RRTMGP?
    !
#ifdef _OPENACC
    call zero_array(block_size, def_tsi)
    !$acc parallel loop collapse(2) copy(def_tsi) copyin(toa_flux)
    do icol = 1, block_size
      do igpt = 1, ngpt
        !$acc atomic update
        def_tsi(icol) = def_tsi(icol) + toa_flux(igpt, icol)
      end do
    end do
#else
    !
    ! More compactly...
    !
    def_tsi(1:block_size) = sum(toa_flux, dim=1)
#endif
    !
    ! Normalize incoming solar flux to match RFMIP specification
    !
    !$acc parallel loop collapse(2) copyin(total_solar_irradiance, def_tsi) copy(toa_flux)
    do icol = 1, block_size
      do igpt = 1, ngpt
        toa_flux(igpt,icol) = toa_flux(igpt,icol) * total_solar_irradiance(icol,b)/def_tsi(icol)
      end do
    end do
    !
    ! Expand the spectrally-constant surface albedo to a per-band albedo for each column
    !
    !$acc parallel loop collapse(2) copyin(surface_albedo)
    do icol = 1, block_size
      do ibnd = 1, nbnd
        sfc_alb_spec(ibnd,icol) = surface_albedo(icol,b)
      end do
    end do
    !
    ! Cosine of the solar zenith angle
    !
    !$acc parallel loop copyin(solar_zenith_angle, usecol)
    do icol = 1, block_size
      mu0(icol) = merge(cos(solar_zenith_angle(icol,b)*deg_to_rad), 1._wp, usecol(icol,b))
    end do

    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
#ifdef USE_TIMING
    ret =  gptlstart('rte_sw')
#endif

    call stop_on_err(rte_sw(optical_props,   &
                            top_at_1,        &
                            mu0,             &
                            toa_flux,        &
                            sfc_alb_spec,    &
                            sfc_alb_spec,    &
                            fluxes, compute_gpoint_fluxes = .false.))
                       
#ifdef USE_TIMING
    ret =  gptlstop('rte_sw')
#endif
    ! Save RRTMGP inputs and outputs
    if (save_input_output) then
      tau_sw(:,:,:,b)     = optical_props%tau(:,:,:)
      ssa(:,:,:,b)        = optical_props%ssa(:,:,:)
    end if
    !
    ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
    !

    do icol = 1, block_size
      if(.not. usecol(icol,b)) then
        flux_up(:,icol,b)  = 0._wp
        flux_dn(:,icol,b)  = 0._wp
      end if
    end do

  end do

  !
  ! End timers
  !
#ifdef USE_TIMING
 end do
  ret = gptlpr(block_size)
  ret = gptlfinalize()
#endif
  ! print *, "max flux_up, flux_dn:", maxval(flux_up(:,:,:)), maxval(flux_dn(:,:,:))

  allocate(temparray(   block_size*(nlay+1)*nblocks)) 
  temparray = pack(flux_dn(:,:,:),.true.)
  print *, "mean of flux_down is:", sum(temparray, dim=1)/size(temparray, dim=1)
  temparray = pack(flux_up(:,:,:),.true.)
  print *, "mean of flux_up is:", sum(temparray, dim=1)/size(temparray, dim=1)
  deallocate(temparray)

  !$acc exit data delete(optical_props%tau, optical_props%ssa, optical_props%g, optical_props)
  !$acc exit data delete(sfc_alb_spec, mu0)
  !$acc exit data delete(toa_flux, def_tsi)
  ! --------------------------------------------------


  if (save_input_output) then 
    print *, "Attempting to save neural network inputs to ", inp_outp_file
    ! This function also deallocates the input 
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'nn_input',nn_input)
    call unblock_and_write2(trim(inp_outp_file),       'col_dry', col_dry)
    print *, "Inputs were saved to ", inp_outp_file

    allocate(temparray(   block_size*nlay*ngpt*nblocks)) 
    temparray = pack(tau_sw(:,:,:,:),.true.)
    print *, "mean of tau is", sum(temparray, dim=1)/size(temparray, dim=1)
    print *, "max, min of tau is", maxval(tau_sw), minval(tau_sw)
    temparray = pack(ssa(:,:,:,:),.true.)
    print *, "mean of ssa is", sum(temparray, dim=1)/size(temparray, dim=1)
    deallocate(temparray)

    print *, "Attempting to save outputs to" , inp_outp_file
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'tau_sw', tau_sw)
    call unblock_and_write_3D_sp(trim(inp_outp_file), 'ssa',    ssa)
    print *, "Outputs were saved to ", inp_outp_file
  end if


    ! Save fluxes ?
  if (save_flux) then
    print *, "Attempting to save fluxes to ", flx_file
    call unblock_and_write(trim(flx_file), 'rsu', flux_up)
    call unblock_and_write(trim(flx_file), 'rsd', flux_dn)
    print *, "Fluxes saved to ", flx_file
  end if 


  if (compare_flux) then
    print *, "-----------------------------------------------------------------------------------------"
    print *, "-----COMPARING ERRORS (W.R.T LINE-BY-LINE) OF NEURAL NETWORK AND ORIGINAL SCHEME --------"
    print *, "-----------------------------------------------------------------------------------------"

    allocate(rsd_ref( nlay+1, ncol, nexp))
    allocate(rsu_ref( nlay+1, ncol, nexp))  
    allocate(rsdu_ref( nlay+1, ncol, nexp))  
    allocate(rsd_nn( nlay+1, ncol, nexp))
    allocate(rsu_nn( nlay+1, ncol, nexp))
    allocate(rsdu_nn( nlay+1, ncol, nexp))
    allocate(rsd_lbl( nlay+1, ncol, nexp))
    allocate(rsu_lbl( nlay+1, ncol, nexp))
    allocate(rsdu_lbl( nlay+1, ncol, nexp))

    flx_file_ref = 'output_fluxes/rsud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc'
    flx_file_lbl = 'output_fluxes/rsud_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'

    call unblock(flux_up, rsu_nn)
    call unblock(flux_dn, rsd_nn)

    rsdu_nn = rsd_nn - rsu_nn

    if(nf90_open(trim(flx_file_ref), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_ref))

    rsu_ref = read_field(ncid, "rsu", nlay+1, ncol, nexp)
    rsd_ref = read_field(ncid, "rsd", nlay+1, ncol, nexp)
    rsdu_ref = rsd_ref - rsu_ref

    if(nf90_open(trim(flx_file_lbl), NF90_NOWRITE, ncid) /= NF90_NOERR) &
    call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_lbl))

    rsu_lbl = read_field(ncid, "rsu", nlay+1, ncol, nexp)
    rsd_lbl = read_field(ncid, "rsd", nlay+1, ncol, nexp)
    rsdu_lbl = rsd_lbl - rsu_lbl

    print *, "------------- UPWELLING -------------- "

    print *, "MAE in upwelling fluxes of NN and RRTMGP, present-day:            ", &
     mae(reshape(rsu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in upwelling fluxes of NN and RRTMGP, future:                 ", &
     mae(reshape(rsu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "bias in upwelling flux of NN and RRTMGP, present-day, top-of-atm.:", &
      bias(reshape(rsu_lbl(1,:,1), shape = [1*ncol]),    reshape(rsu_nn(1,:,1), shape = [1*ncol])), &
      bias(reshape(rsu_lbl(1,:,1), shape = [1*ncol]),    reshape(rsu_ref(1,:,1), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, future, top-of-atm.:     ", &
      bias(reshape(rsu_lbl(1,:,4), shape = [1*ncol]),    reshape(rsu_nn(1,:,4), shape = [1*ncol])), &
      bias(reshape(rsu_lbl(1,:,4), shape = [1*ncol]),    reshape(rsu_ref(1,:,4), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, future-all, top-of-atm.: ", &
      bias(reshape(rsu_lbl(1,:,17), shape = [1*ncol]),    reshape(rsu_nn(1,:,17), shape = [1*ncol])), &
      bias(reshape(rsu_lbl(1,:,17), shape = [1*ncol]),    reshape(rsu_ref(1,:,17), shape = [1*ncol])) 

    print *, "bias in upwelling flux of NN and RRTMGP, ALL EXPS, top-of-atm.:   ", &
      bias(reshape(rsu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rsu_nn(1,:,:), shape = [nexp*ncol])), &
      bias(reshape(rsu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rsu_ref(1,:,:), shape = [nexp*ncol])) 


    print *, "-------------- DOWNWELLING --------------"

    print *, "MAE in downwelling fluxes of NN and RRTMGP, present-day:          ", &
     mae(reshape(rsd_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,1), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsd_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN and RRTMGP, future:               ", &
     mae(reshape(rsd_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,4), shape = [1*ncol*(nlay+1)])),&
    mae(reshape(rsd_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsd_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "-------------- NET FLUX --------------"

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, pres.day:  ", &
     maxval(rsdu_lbl(:,:,1)-rsdu_nn(:,:,1)), maxval(rsdu_lbl(:,:,1)-rsdu_ref(:,:,1))

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, future:    ", &
     maxval(rsdu_lbl(:,:,4)-rsdu_nn(:,:,4)), maxval(rsdu_lbl(:,:,4)-rsdu_ref(:,:,4))

     print *, "Max-vertical-error in net fluxes of NN and RRTMGP, future-all:", &
     maxval(rsdu_lbl(:,:,17)-rsdu_nn(:,:,17)), maxval(rsdu_lbl(:,:,17)-rsdu_ref(:,:,17))

     print *, "---------"

     print *, "MAE in net fluxes of NN and RRTMGP, present-day:               ", &
     mae(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
     mae(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    print *, "MAE in net fluxes of NN and RRTMGP, future:                    ", &
     mae(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
     mae(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in net fluxes of NN and RRTMGP, future-all:                ", &
     mae(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,17), shape = [1*ncol*(nlay+1)])),&
     mae(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

     print *, "MAE in net fluxes of NN and RRTMGP, ALL EXPS:                  ", &
     mae(reshape(rsdu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)])), &
     mae(reshape(rsdu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)])) 

    print *, "---------"

    print *, "MAE in net fluxes at TOA of NN and RRTMGP, future:             ", &
     mae(reshape(rsdu_lbl(1,:,4), shape = [1*ncol*(1)]), reshape(rsdu_nn(1,:,4), shape = [1*ncol*(1)])), &
     mae(reshape(rsdu_lbl(1,:,4), shape = [1*ncol*(1)]), reshape(rsdu_ref(1,:,4), shape = [1*ncol*(1)]))

     print *, "MAE in net fluxes at TOA of NN and RRTMGP, present-day:        ", &
     mae(reshape(rsdu_lbl(1,:,1), shape = [1*ncol*(1)]), reshape(rsdu_nn(1,:,1), shape = [1*ncol*(1)])), &
     mae(reshape(rsdu_lbl(1,:,1), shape = [1*ncol*(1)]), reshape(rsdu_ref(1,:,1), shape = [1*ncol*(1)]))

     print *, "---------"


    print *, "RMSE in net fluxes of NN and RRTMGP, present-day, PBL:         ", &
     rmse(reshape(rsdu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rsdu_nn(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)])), &
     rmse(reshape(rsdu_lbl(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]),    reshape(rsdu_ref(33:nlay+1,:,1), shape = [1*ncol*(nlay+1-33)]))

    print *, "RMSE in net fluxes of NN and RRTMGP, present-day, SURFACE:     ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol]))

    print *, "RMSE in net fluxes of NN and RRTMGP, future-all, SURFACE:     ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,17), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,17), shape = [1*ncol]))

    print *, "RMSE in net fluxes of NN and RRTMGP, pre-industrial, SURFACE: ", &
     rmse(reshape(rsdu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,2), shape = [1*ncol])), &
     rmse(reshape(rsdu_lbl(nlay+1,:,2), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,2), shape = [1*ncol]))

    print *, "---------"

    print *, "bias in net fluxes of NN and RRTMGP, present-day:              ", &
     bias(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,1), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,1), shape = [1*ncol*(nlay+1)])) 

    print *, "bias in net fluxes of NN and RRTMGP, present-day, SURFACE:     ", &
     bias(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol])), &
     bias(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol])) 

    print *, "bias in net fluxes of NN and RRTMGP, future:                   ", &
     bias(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,4), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rsdu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "bias in net fluxes of NN and RRTMGP, future-all:               ", &
     bias(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_nn(:,:,17), shape = [1*ncol*(nlay+1)])), &
     bias(reshape(rsdu_lbl(:,:,17), shape = [1*ncol*(nlay+1)]), reshape(rsdu_ref(:,:,17), shape = [1*ncol*(nlay+1)]))

    print *, "bias in net fluxes of NN and RRTMGP, future-all, SURFACE:      ", &
     bias(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,17), shape = [1*ncol])), &
     bias(reshape(rsdu_lbl(nlay+1,:,17), shape = [1*ncol]),    reshape(rsdu_ref(nlay+1,:,17), shape = [1*ncol])) 

    print *, "---------"

    print *, "MAE in upwelling fluxes of NN w.r.t RRTMGP, present-day:       ", &
     mae(reshape(rsu_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN w.r.t RRTMGP, present-day:     ", &
     mae(reshape(rsd_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "Max-diff in d.w. flux w.r.t RRTMGP ", &
     maxval(rsd_ref(:,:,:)-rsd_nn(:,:,:))
 
    print *, "Max-diff in u.w. flux w.r.t RRTMGP ", &
     maxval(rsu_ref(:,:,:)-rsu_nn(:,:,:))

    print *, "Max-diff in net flux w.r.t RRTMGP  ", &
     maxval(rsdu_ref(:,:,:)-rsdu_nn(:,:,:)) 

    deallocate(rsd_ref,rsu_ref,rsd_nn,rsu_nn,rsd_lbl,rsu_lbl,rsdu_ref,rsdu_nn,rsdu_lbl)

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

end program rrtmgp_rfmip_sw
