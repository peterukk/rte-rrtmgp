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
  use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
                                            stdout=>output_unit, &
                                            stderr=>error_unit
  ! --------------------------------------------------
  !
  ! Modules for working with rte and rrtmgp
  !
  use omp_lib
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp
  !
  ! Optical properties of the atmosphere as array of values
  !   In the longwave we include only absorption optical depth (_1scl)
  !   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  ! use mo_optical_props,      only: ty_optical_props_arry, make_2str, make_1scl
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
  use mo_fluxes,             only: ty_fluxes_broadband, ty_fluxes_flexible
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_io_rfmipstyle_generic,           only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   unblock, read_and_block_sw_bc, determine_gas_names                        
  use mo_simple_netcdf,      only: read_field, write_field, get_dim_size
  use netcdf
  use mod_network_rrtmgp   
#ifdef USE_OPENACC  
  use cublas
  use openacc
#endif          
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr_file, gptlfinalize, gptlsetoption, &
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
  character(len=132)  ::  rfmip_file = '../rfmip-clear-sky/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                          kdist_file = 'coefficients_sw.nc'
  ! Neural networks for gas optics (optional) - netCDF files which describe the models and pre-processing coefficients
  ! (The two models predict SW absorption and Rayleigh scattering, respectively)
  character(len=80)   ::  modelfile_tau= "../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.nc", &
                          modelfile_ray= "../../neural/data/BEST_tau-sw-ray-7-16-16_2.nc"
  character(len=132)  ::  flx_file, flx_file_lbl, timing_file
  integer             ::  nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size, forcing_index, iref
  integer             ::  b, icol, ilay,ibnd, igpt, igas, ncid, ninputs, ret, i, istat, num_metrics, iexp
  character(len=4)    ::  block_size_char, forcing_index_char = '1'
  character(len=32 ), dimension(:),  allocatable   ::  kdist_gas_names, rfmip_gas_names
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, plev, t_lay, t_lev ! nlay,blocksize, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:),         allocatable :: rsu_new, rsd_new, rsu_lbl, rsd_lbl, rsdu_new, rsdu_lbl
  real(wp), dimension(:,:  ),         allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
  real(wp), dimension(:,:  ),         allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  real(wp), dimension(:),             allocatable :: temparray
  real(wp), allocatable :: errors(:), hr_nn(:,:,:),  hr_lbl(:,:,:)
  character(len=80), allocatable :: metric_names(:)
  real(wp), parameter :: deg_to_rad = acos(-1._wp)/180._wp
  real(wp) :: def_tsi_s, val

  logical 		        :: use_rrtmgp_nn = .false., save_flux = .false., top_at_1
  !
  ! Classes used by rte+rrtmgp
  !
  type(ty_gas_optics_rrtmgp)                    :: k_dist
  type(ty_optical_props_2str)                   :: optical_props
  type(ty_fluxes_flexible)                      :: fluxes
  type(rrtmgp_network_type), dimension(2)        :: neural_nets ! First model is absorption, second is Rayleigh

  real(wp), dimension(:,:), allocatable         :: toa_flux ! block_size, ngpt
  real(wp), dimension(:  ), allocatable         :: def_tsi, mu0    ! block_size
  logical , dimension(:,:), allocatable         :: usecol ! block_size, nblocks
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array

  ! Initialize GPU kernel
#ifdef USE_OPENACC  
  type(cublasHandle) :: h
  istat = cublasCreate(h) 
  ! istat = cublasSetStream(h, acc_get_cuda_stream(acc_async_sync))
#endif

  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
  !
  !  ------------ I/O and settings -----------------
  ! Save fluxes
  ! save_flux    = .false.

  print *, "Usage: rrtmgp_rfmip_sw [block_size] [k-distribution_file] [forcing_index] [NN_sw_abs_file] [NN_sw_ray_file]"

  nargs = command_argument_count()

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size
  ! if(nargs >= 2) call get_command_argument(2, rfmip_file)
  if(nargs >= 2) call get_command_argument(2, kdist_file)
  if(nargs >= 3) call get_command_argument(3, forcing_index_char)

  if(nargs == 4) stop "provide 1-3 or 5 arguments"
  if(nargs >= 5) then
    use_rrtmgp_nn = .true.
    call get_command_argument(4, modelfile_tau)
    call get_command_argument(5, modelfile_ray)
  end if
  ! How big is the problem? Does it fit into blocks of the size we've specified?
  !
  call read_size(rfmip_file, ncol, nlay, nexp)

  if(mod(ncol*nexp, block_size) /= 0 ) call stop_on_err("rrtmgp_rfmip_lw: number of columns doesn't fit evenly into blocks.")
  nblocks = (ncol*nexp)/block_size
  ! print *, "Doing ",  nblocks, "blocks of size ", block_size

  read(forcing_index_char, '(i4)') forcing_index
  if(forcing_index < 1 .or. forcing_index > 4) &
    stop "Forcing index is invalid (must be 1,2 or 3)"

  ! Save upwelling and downwelling fluxes in the same file
  if (use_rrtmgp_nn) then
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-NN-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  else
    flx_file = 'output_fluxes/rsud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f' // trim(forcing_index_char) // '_gn.nc'
  end if
  !
  ! Identify the set of gases used in the calculation based on the forcing index
  !   A gas might have a different name in the k-distribution than in the files
  !   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  !
  call determine_gas_names(rfmip_file, kdist_file, forcing_index, kdist_gas_names, rfmip_gas_names)
  ! print *, "Input file gas names: ", (trim(rfmip_gas_names(b)) // " ", b = 1, size(rfmip_gas_names))
  ! print *, "K-dist gas names: ", (trim(kdist_gas_names(b)) // " ", b = 1, size(kdist_gas_names))

  ! Load Neural Network models
  if (use_rrtmgp_nn) then
	  ! print *, 'loading shortwave absorption model from ', modelfile_tau
    call neural_nets(1) % load_netcdf(modelfile_tau)
    ! print *, 'loading rayleigh model from ', modelfile_ray
    call neural_nets(2) % load_netcdf(modelfile_ray)
    ninputs = size(neural_nets(1) % layers(1) % w_transposed, 2)
    ! print *, "NN supports gases: ", (trim(neural_nets(1)%input_names(b)) // " ", b = 3, size(neural_nets(1)%input_names))
  end if  
  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  !
  ! Allocation on assignment within reading routines
  !
  call read_and_block_pt(rfmip_file, block_size, p_lay, p_lev, t_lay, t_lev)
  !
  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)

  !
  ! Read the gas concentrations and surface properties
  !
  call read_and_block_gases_ty(rfmip_file, block_size, kdist_gas_names, rfmip_gas_names, gas_conc_array)
  ! do b = 1, size(gas_conc_array(1)%concs)
  !   print *, "max of gas ", gas_conc_array(1)%gas_name(b), ":", maxval(gas_conc_array(1)%concs(b)%conc)
  ! end do

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
    p_lay(1,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lay(nlay+1,:,:) &
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

  ! allocate(mu0(block_size), sfc_alb_spec(nbnd,block_size))
  allocate(mu0(block_size), sfc_alb_spec(ngpt,block_size))

  ! Alloc is generic and calls appropriate allocation routine for the specific type of optical_props
  ! call stop_on_err(optical_props%alloc(block_size, nlay, k_dist))
  call stop_on_err(optical_props%alloc_2str(block_size, nlay, k_dist))
  
  !$acc enter data create (toa_flux, def_tsi)
  !$acc enter data create (sfc_alb_spec, mu0) 
  !$acc enter data copyin(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)

#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
#ifdef USE_PAPI  
#ifdef DOUBLE_PRECISION
  ret = GPTLsetoption (PAPI_DP_OPS, 1);         ! Turn on FLOPS estimate (DP)
#else
  ret = GPTLsetoption (PAPI_SP_OPS, 1);         ! Turn on FLOPS estimate (SP)
#endif
#endif  
  ret = gptlinitialize()
#endif

  ! --------------------------------------------------

  if (use_rrtmgp_nn) then
    print *, "starting clear-sky SW computations, using neural networks as RRTMGP kernel"
  else
    print *, "starting clear-sky SW computations, using lookup-table as RRTMGP kernel"
  end if
  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
  ret =  gptlstart('clear_sky_total (SW)')
#endif
  !$OMP PARALLEL firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0,fluxes,optical_props) default(shared)
  !$OMP DO 
  do b = 1, nblocks

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
    
    if (use_rrtmgp_nn) then
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        toa_flux, neural_nets=neural_nets))
    else
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        toa_flux))
    end if
!     PRINT *, "process: ", OMP_GET_THREAD_NUM(), "b: ",b," mean tau ", mean_3d(optical_props%tau)
    ! print *," max, min (tau)",   maxval(optical_props%tau), minval(optical_props%tau)
#ifdef USE_TIMING
    ret =  gptlstop('gas_optics (SW)')
#endif
    !
    ! Boundary conditions
    !
    ! What's the total solar irradiance assumed by RRTMGP?
    ! 
    !$acc parallel loop gang default(present)
    do icol = 1, block_size
      def_tsi_s = 0.0_wp
      !$acc loop vector reduction(+:def_tsi_s)
      do igpt = 1, ngpt
        def_tsi_s = def_tsi_s + toa_flux(igpt, icol)
      end do
      def_tsi(icol) = def_tsi_s
    end do

    !$acc parallel default(present)    
    !$acc loop collapse(2)
    do icol = 1, block_size
      do igpt = 1, ngpt
        ! Normalize incoming solar flux to match RFMIP specification
        toa_flux(igpt,icol) = toa_flux(igpt,icol) * total_solar_irradiance(icol,b)/def_tsi(icol)
        ! Expand the spectrally-constant surface albedo to a per-g-point albedo for each column
        sfc_alb_spec(igpt,icol) = surface_albedo(icol,b)
      end do
    end do
    !
    ! Cosine of the solar zenith angle
    !
    !$acc loop
    do icol = 1, block_size
      mu0(icol) = merge(cos(solar_zenith_angle(icol,b)*deg_to_rad), 1._wp, usecol(icol,b))
    end do
    !$acc end parallel

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
                            fluxes))!,          &
#ifdef USE_TIMING
    ret =  gptlstop('rte_sw')
#endif
    !
    ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
    !
    do icol = 1, block_size
      if(.not. usecol(icol,b)) then
        flux_up(:,icol,b)  = 0._wp
        flux_dn(:,icol,b)  = 0._wp
      end if
    end do
    
  end do !blocks
  !$OMP END DO
  !$OMP END PARALLEL

  ! End timers
#ifdef USE_TIMING
  ret =  gptlstop('clear_sky_total (SW)')
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  !$acc exit data delete(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)
  !$acc exit data delete(sfc_alb_spec, mu0)
  !$acc exit data delete(toa_flux, def_tsi)
  call optical_props%finalize() ! Also deallocates arrays on device

#ifdef USE_OPENACC  
  istat = cublasDestroy(h) 
#endif

  ! --------------------------------------------------
  ! Save fluxes ?
  if (save_flux) then
    call unblock_and_write(trim(flx_file), 'rsu', flux_up)
    call unblock_and_write(trim(flx_file), 'rsd', flux_dn)
    print *, "Fluxes saved to ", flx_file
  end if 

  ! Compare fluxes to benchmark line-by-line results, alongside reference RTE+RRTMGP computations?
  print *, "-----COMPARING ERRORS (W.R.T. LINE-BY-LINE) -------"

  allocate(rsd_new( nlay+1, ncol, nexp))
  allocate(rsu_new( nlay+1, ncol, nexp))
  allocate(rsdu_new( nlay+1, ncol, nexp))
  allocate(rsd_lbl( nlay+1, ncol, nexp))
  allocate(rsu_lbl( nlay+1, ncol, nexp))
  allocate(rsdu_lbl( nlay+1, ncol, nexp))

  allocate(plev( nlay+1, ncol, nexp))


  flx_file_lbl = 'output_fluxes/rsud_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'

  call unblock(flux_up, rsu_new)
  call unblock(flux_dn, rsd_new)
  call unblock(p_lev, plev)


  rsdu_new = rsd_new - rsu_new

  if(nf90_open(trim(flx_file_lbl), NF90_NOWRITE, ncid) /= NF90_NOERR) &
  call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_lbl))

  rsu_lbl = read_field(ncid, "rsu", nlay+1, ncol, nexp)
  rsd_lbl = read_field(ncid, "rsd", nlay+1, ncol, nexp)
  rsdu_lbl = rsd_lbl - rsu_lbl

  ! Error metrics - we can choose some managable number of metrics that are printed at the end
  ! and read by our training program from the command line standard output
  num_metrics = 8
  allocate(errors(num_metrics), metric_names(num_metrics))
  
  ! metric_names(1) = 'HR (all) '
  ! metric_names(2) = 'HR (PD)'
  ! metric_names(3) = 'HR (future-all)'
  ! metric_names(4) = 'HR (preindustrial)'
  ! metric_names(5) = 'Bias surface downwelling'
  ! metric_names(6) = 'RF-SFC (PI->future)'
  ! metric_names(7) = 'RF-TOA (PI->future)'
  ! metric_names(8) = 'RF-SFC CH4 (PI->PD)'
  metric_names(1) = 'MAE HR (all) '
  metric_names(2) = 'MAE HR (PD)'
  metric_names(3) = 'MAE HR (future-all)'
  metric_names(4) = 'MAE HR (PI)'
  metric_names(5) = 'MAE sfc downwelling'
  metric_names(6) = 'Bias RF-TOA (PI->future)'
  metric_names(7) = 'Bias RF-SFC (PI->future)'
  metric_names(8) = 'Bias RF-SFC CH4 (PI->PD)'
  ! Heating rates
  allocate(hr_nn(nlay,ncol,nexp), hr_lbl(nlay,ncol,nexp))

  do iexp = 1, nexp
    hr_nn(:,:,iexp)   = calc_heating_rate(ncol, nlay, rsu_new(:,:,iexp), rsd_new(:,:,iexp), plev)
    hr_lbl(:,:,iexp)  = calc_heating_rate(ncol, nlay, rsu_lbl(:,:,iexp), rsd_lbl(:,:,iexp), plev)
  end do
  
  val =  mae_presweight(nlay,ncol*nexp, hr_nn, hr_lbl, plev)
  print *, 'Pres-weighted heating rate        ', val

  print *, 'Heating rate MAE, all experiments ', mae_flat(ncol*nlay*nexp, hr_nn, hr_lbl)
  print *, 'Heating rate MAE (<100 hPa)       ', mae_flat(ncol*26*nexp, hr_nn(1:26,:,:), hr_lbl(1:26,:,:))
  print *, 'Heating rate MAE (>100 hPa)       ', mae_flat(ncol*35*nexp, hr_nn(26:,:,:), hr_lbl(26:,:,:))
  print *, 'Heating rate MAE sfc              ', mae_flat(ncol*nexp, hr_nn(nlay,:,:), hr_lbl(nlay,:,:))

  print *, 'Heating rate MAE, all experiments  ', mae_flat(ncol*nlay*nexp, hr_nn, hr_lbl)
  print *, 'Heating rate MAE, present-day      ', mae_flat(ncol*nlay, hr_nn(:,:,1), hr_lbl(:,:,1))
  print *, 'Heating rate MAE, future-all       ', mae_flat(ncol*nlay, hr_nn(:,:,17), hr_lbl(:,:,17))
  print *, 'Heating rate MAE, preindustrial    ', mae_flat(ncol*nlay, hr_nn(:,:,2), hr_lbl(:,:,2))

  ! print *, 'RMSE heating rate error, present-day ', rmse_flat(ncol*nlay, hr_nn(:,:,1), hr_lbl(:,:,1)

  errors(1) = mae_presweight(nlay,ncol*nexp, hr_nn, hr_lbl, plev)
  errors(2) = mae_presweight(nlay,ncol, hr_nn(:,:,1), hr_lbl(:,:,1), plev(:,:,1))
  errors(3) = mae_presweight(nlay,ncol, hr_nn(:,:,17), hr_lbl(:,:,17), plev(:,:,17))
  errors(4) = mae_presweight(nlay,ncol, hr_nn(:,:,2), hr_lbl(:,:,2), plev(:,:,2))


  print *, "bias in downwelling flux (sfc):       ", bias_flat(ncol*nexp, rsd_new(nlay+1,:,:), rsd_lbl(nlay+1,:,:))
  print *, "RMSE in downwelling flux (sfc):       ", rmse_flat(ncol*nexp, rsd_new(nlay+1,:,:), rsd_lbl(nlay+1,:,:))
  print *, "RMSE in downwelling flux, P.D. (sfc): ", rmse_flat(ncol, rsd_new(nlay+1,:,1), rsd_lbl(nlay+1,:,1))
  print *, "mae in downwelling flux (sfc):       ", mae_flat(ncol*nexp, rsd_new(nlay+1,:,:), rsd_lbl(nlay+1,:,:))

  errors(5) = mae_flat(ncol*nexp, rsd_new(nlay+1,:,:), rsd_lbl(nlay+1,:,:))

  ! print *, "------------- FLUX ERRORS ------------ "
  ! print *, "------------- UPWELLING -------------- "
  ! print *, "MAE in upwelling flux, present-day:              ", &
  !  mae(reshape(rsu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsu_new(:,:,1), shape = [1*ncol*(nlay+1)]))
  ! print *, "bias in upwelling flux, ALL EXPS, top-of-atm.:   ", &
  !   bias(reshape(rsu_lbl(1,:,:), shape = [nexp*ncol]),    reshape(rsu_new(1,:,:), shape = [nexp*ncol]))

  ! print *, "-------------- DOWNWELLING --------------"
  ! print *, "MAE in downwelling flux, present-day:            ", &
  !  mae(reshape(rsd_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsd_new(:,:,1), shape = [1*ncol*(nlay+1)]))


  ! print *, "-------------- NET FLUX --------------"
  ! val  =  mae(reshape(rsdu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rsdu_new(:,:,1), shape = [1*ncol*(nlay+1)]))
  ! print *, "MAE in net flux, present-day:               ", val
  ! errors(2) = val

  ! val = mae(reshape(rsdu_lbl(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_new(:,:,:), shape = [nexp*ncol*(nlay+1)]))
  ! print *, "MAE in net flux, ALL EXPS:                  ", val
  ! errors(3) = val

  ! val = rmse(reshape(rsdu_lbl(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_new(nlay+1,:,1), shape = [1*ncol]))
  ! print *, "RMSE in net flux, present-day, SURFACE:      ", val

  
  iref = 1
  iexp = 2
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp))  -  mean(diff(rsd_new,nlay+1,iref,iexp))
  ! val = mean(rsd_lbl(nlay+1,:,iref) - rsd_lbl(nlay+1,:,iexp)) -   mean(rsd_new(nlay+1,:,iref) - rsd_new(nlay+1,:,iexp))
  print *, "radiative forcing error at surface, present-day - preindustrial:      ", val

  ! val = -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  ! print *, "radiative forcing error at TOA, present-day - preindustrial:          ", val
  
  iref = 4
  iexp = 2
  val = -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  print *, "radiative forcing error at TOA, future - preindustrial:               ", val
  errors(6) = val

  iref = 17
  iexp = 1
  val = -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  print *, "radiative forcing error at TOA, future-ALL - present-day:             ", val

  iref = 4
  iexp = 2
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, future - preindustrial:           ", val
  errors(7) = val

  iref = 4
  iexp = 1
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, future - present-day:             ", val

  iref = 17
  iexp = 1
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, future-ALL - present-day:         ", val



  iexp = 9
  iref = 8
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, 8x CO2 - preindustrial CO2:       ", val

  val =   -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  print *, "radiative forcing error at TOA, 8x CO2 - preindustrial CO2            ", val

  iref = 1
  iexp = 11
  val =   mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, present-day - preindustrial N2O:  ", val
  
  val =   -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  print *, "radiative forcing error at TOA, present-day - preindustrial N2O:      ", val

  iref = 1
  iexp = 10
  val = mean(diff(rsd_lbl,nlay+1,iref,iexp)) -   mean(diff(rsd_new,nlay+1,iref,iexp))
  print *, "radiative forcing error at surface, present-day - preindustrial CH4:  ", val 
  errors(8) = val

  val = -mean(diff(rsu_lbl,1,iref,iexp)) -   -mean(diff(rsu_new,1,iref,iexp))
  print *, "radiative forcing error at TOA, present-day - preindustrial CH4:      ", val


  write(stdout, '(a)')   '--------'

  do i = 1,num_metrics
    if (i==num_metrics) then
      write(stdout, fmt="(a)", advance="no") trim(metric_names(i))
    else 
      write(stdout, fmt="(1x, a, a)", advance="no") trim(metric_names(i)), ","
    end if
  end do

  write(stdout, '(a)')   ' '
  write(stdout, '(a)')   '--------'

  do i = 1,num_metrics
    if (i==num_metrics) then
      write(stdout, fmt="(1x, F8.4)", advance="no") errors(i)
    else 
      write(stdout, fmt="(1x, F8.4, a)", advance="no") errors(i), ","
    end if
  end do
  print *, ' '
  ! print *, "vals ", (errors(iexp) // " ", iexp = 1, size(errors))

  ! write(stdout, '(a1, F6.4, a2, F6.4, a2, F6.4)') ' ', errors(1), '  ', errors(2), '  ', errors(3)


  deallocate(rsd_new,rsu_new,rsd_lbl,rsu_lbl,rsdu_new,rsdu_lbl)

  deallocate(flux_up, flux_dn)

  contains

  function diff(flux,ilay,i1,i2) result(difference)
    real(wp), dimension(:,:,:), intent(in) :: flux
    integer, intent(in) :: ilay,i1,i2
    real(wp), dimension(size(flux,2)) :: difference

    difference = flux(ilay,:,i1) - flux(ilay,:,i2)

  end function

      ! calculate heating rates
  function calc_heating_rate(ncol, nlay, flux_up, flux_dn, pressure_hl) result(hr_K_day)
    !  calc_heatingrate(Fdown - Fup, pres)
    ! dF = F[:,1:] - F[:,0:-1] 
    ! dp = p[:,1:] - p[:,0:-1] 
    ! dFdp = dF/dp
    ! g = 9.81 # m s-2
    ! cp = 1004 # J K-1  kg-1
    ! dTdt = -(g/cp)*(dFdp) # K / s
    ! dTdt_day = (24*3600)*dTdt

    use mo_rrtmgp_constants, only : grav
    integer, intent(in) :: ncol, nlay
    real(wp), dimension(nlay+1, ncol), intent(in) :: flux_up, flux_dn, pressure_hl
    real(wp), dimension(nlay,   ncol) :: hr_K_day
    ! Local variables
     real(wp), dimension(nlay+1,   ncol) :: flux_net
    real(wp), dimension(nlay,   ncol) :: dF, dP
    ! "Cp" (J kg-1 K-1)
    real(wp), parameter :: SpecificHeatDryAir = 1004.0
    real(wp) :: scaling
    integer :: jlay
    scaling = -(24.0_wp * 3600.0_wp * grav / SpecificHeatDryAir)

    flux_net = flux_dn - flux_up
    dF = flux_net(2:nlay+1,:) - flux_net(1:nlay,:)
    dP = pressure_hl(2:nlay+1,:) - pressure_hl(1:nlay,:)
    hr_K_day = scaling * dF / dP
    ! hr_K_day = scaling * (flux_net(2:nlay+1,:) - flux_net(1:nlay,:)) / (pressure_hl(2:nlay+1,:) - pressure_hl(1:nlay,:))

  end function calc_heating_rate

  function rmse(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = x1 - x2
    res = sqrt( sum(diff**2)/size(diff) )
  end function rmse

  function rmse_flat(ndim, x1,x2) result(res)
    implicit none 
    integer, intent(in) :: ndim
    real(wp), dimension(ndim), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(ndim) :: diff 
    
    diff = x1 - x2
    res = sqrt( (sum(diff**2)/ndim) )
  end function rmse_flat


  function mae(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1)) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff, dim=1)/size(diff, dim=1)
  end function mae

  function mae_flat(ndim, x1,x2) result(res)
    implicit none 
    integer, intent(in) :: ndim
    real(wp), dimension(ndim), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(ndim) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff, dim=1)/size(diff, dim=1)
  end function mae_flat

  function mae_presweight(nlay,ndim, x1,x2, plev) result(res)
    implicit none 
    integer, intent(in) :: nlay,ndim
    real(wp), dimension(nlay,  ndim), intent(in) :: x1,x2
    real(wp), dimension(nlay+1,ndim), intent(in) :: plev

    real(wp) :: res
    real(wp), dimension(nlay,ndim) :: diff 
    
    diff = abs(x1 - x2)
    diff = diff*(sqrt(plev(2:nlay+1,:))-sqrt(plev(1:nlay,:))) ! times delta(sqrt(p))
    res = sum(diff)/ (ndim*mean(sqrt(plev(nlay+1,:)) - sqrt(plev(1,:))))

  end function mae_presweight

  function bias_flat(ndim, x1,x2) result(res)
    implicit none 
    integer, intent(in) :: ndim
    real(wp), dimension(ndim), intent(in) :: x1,x2
    real(wp) :: mean1,mean2, res
    
    mean1 = sum(x1)/ndim
    mean2 = sum(x2)/ndim
    res = mean1 - mean2

  end function bias_flat 
  function bias(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:), intent(in) :: x1,x2
    real(wp) :: mean1,mean2, res
    
    mean1 = sum(x1, dim=1)/size(x1, dim=1)
    mean2 = sum(x2, dim=1)/size(x2, dim=1)
    res = mean1 - mean2

  end function bias

  function mean(x) result(mean1)
    implicit none 
    real(wp), dimension(:), intent(in) :: x
    real(wp) :: mean1
    mean1 = sum(x) / size(x)
  end function mean

  function mean_2d(x) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x
    real(wp) :: mean2
    mean2 = sum(x) / size(x)
  end function mean_2d

  function mean_3d(x) result(mean3)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x
    real(wp) :: mean3
    mean3 = sum(x) / size(x)
  end function mean_3d

end program rrtmgp_rfmip_sw
