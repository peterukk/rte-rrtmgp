! This program is for generating training data for neural network emulators of RRTMGP and RTE,
! as well as demonstrating their use.
! Three general machine learning approaches are compared:
!    1) emulation of gas optics=RRTMGP only (as done in Ukkonen et al. 2020, Menno et al. 2021),
!       mapping atmospheric conditions to gas optical properties
!    2) emulation of radiative solver=RTE, mapping optical properties to fluxes
!    3) emulation of RTE+RRTMGP, mapping atmospheric conditions to fluxes (as done in some other papers)
! 
! Since we are interested in the trade-off of accuracy and speedup of these methods for realistic use cases, 
! clouds will be included when generating training data for 2-3, and in the evaluation of 1-3.
! What this means is that the NN in method 3 has a fully implicit treatment of clouds, but in 1) and 2) cloud optical 
! properties are computed as a separate step using the relatively cheap cloud optics extension 
! However, the NNs in both 2) and 3) account for the radiative effect of clouds
! 
! The idea is that this program can by called by a Python program to
!  -- generate training data for 1), 2), or 3)
!  -- evaluate 1), 2), or 3) by loading NNs and replacing the appropriate computations by NN predictions,
!     timing the computations and saving fluxes for offline validation.
! 
! The data comes from CAMS which has been extended into RFMIP-style experiments where gas
! concentrations are varied. The large problem is divided into blocks
!
! following machine learning methods for shortwave radiation computations could be tested (preliminary ideas):
! 
!                   "emulate"   cloud   NN input (shape)	NN output (shape)     	#NN	#NN iterations needed
!                               optics                                                  models  (= #training samples) 
!   RTE + RRTMGP      none      orig.                  
!   (RTE+RRTMGP)-NN   both      NN      concs+BC (8*nlay +2)  bb flux (2*nlay)            1	ncol                    		   	        
!  ^RTE + RRTMGP-NN   rrtmgp    orig.   layer concs (8)       layer gas OP (ng)           2	nlay*ncol 
!   RTE-NN1 + RRTMGP  rte       orig.   OPs+BC (3*nlay +2)    gp flux (2*nlay)            1	ng*ncol
!  *RTE-NN2 + RRTMGP  reftrans  orig.   layer OPs (3 + 1)     layer rdif,tdif,rdir,tdir   1	ng*nlay*ncol
!  ?RTE-NN3 + RRTMGP  rte-bb    orig.   OPs+BC (3*nlay*ng +2) bb flux (2*nlay)            1	ncol
!  ?RTE-NN4 + RRTMGP  rte-gpvec orig.   OPs+BC (3*nlay*ng +2) gp flux vectors (2*nlay*ng) 1	ncol

! ^method used in Ukkonen 2020; separate NNs are used to predict absorption and Rayleigh absorption cross-sections
! *this model would only emulate one component of RTE, which is relatively expensive and has no layer dependency
! ?these methods are questionable, because radiative transfer computations are independent for g-points.
! however, they could be tested anyway
!
! concs = gas concentrations + temperature + pressure; needed for optical property computations
! OPs = shortwave optical properties (tau, ssa, g), for gases only first two are needed 
! BC = boundary conditions for radiative transfer (sfc albedo and solar angle; incoming flux at TOA considered constant)
! bb = broadband. bb fluxes are obtained by summing fluxes for different g-points together
! gp = g-point (pseudo-independent spectral dimension) 
! typical dimension sizes : ng=100-250, nlay=60-120, ncol=1000+; here ng=112 or 224 and nlay=&0
! the number 2 in outputs comes from upwelling+downwelling flux; direct downwelling may be needed as well
!
! Fortran program arguments (k_distribution file and cloud_optics file are fixed):
! ml_allsky_sw [block_size] [input file] [component to emulate] [NN model file(s)]"
!
! Developed by Peter Ukkonen (built on allsky example, and existing RTE+RRTMGP code by Robert Pincus)
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
#ifdef USE_OPENMP
  use omp_lib
#endif
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp
  !
  ! Array utilities
  !
  use mo_rte_util_array,     only: zero_array
  !
  ! Optical properties of the atmosphere as array of values
  !   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  !
  use mo_optical_props,      only: ty_optical_props_2str 
  !
  ! Gas optics: maps physical state of the atmosphere to optical properties
  !
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp, compute_nn_inputs, get_col_dry
  !
  ! Gas optics uses a derived type to represent gas concentrations compactly
  !
  use mo_gas_concentrations, only: ty_gas_concs
  ! !
  ! ! Coefficients used to scale NN inputs
  ! !
  ! use mo_rrtmgp_nn_constants,only: nn_input_names, nn_input_maxvals, nn_input_minvals
  !
  !
  ! Cloud optics extension
  use mo_cloud_optics,       only: ty_cloud_optics
  use mo_load_cloud_coefficients, only: load_cld_lutcoeff, load_cld_padecoeff
  !
  !
  ! RTE shortwave driver
  !
  use mo_rte_sw,             only: rte_sw
  !
  ! RTE driver uses a derived type to reduce spectral fluxes to whatever the user wants
  !   Here we're using a flexible type which outputs broadband fluxes and optionally g-point fluxes
  !
  use mo_fluxes,             only: ty_fluxes_flexible
  !
  ! Neural network library
  !
  use mod_network  
  ! --------------------------------------------------
  !
  ! modules for reading and writing files
  !
  ! RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  !
  use mo_load_coefficients,  only: load_and_init
  use mo_io_rfmipstyle_generic, only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   read_and_block_sw_bc, read_and_block_clouds_cams, determine_gas_names, unblock                          
  use mo_simple_netcdf,      only: read_field, write_field, get_dim_size
  use netcdf
  use easy_netcdf
#ifdef USE_OPENACC  
  !
  ! GPU library
  !
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
  character(len=132)  ::  input_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                          kdist_file = '../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc', &
                          cloud_optics_file='../../extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc'
  character(len=132)  ::  flx_file, flx_file_ref, timing_file, nndev_file='', nn_input_str, cmt
  integer             ::  nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size
  logical             ::  top_at_1, do_scattering
  integer             ::  b, icol, ilay, igpt, igas, ngas, ninputs_rrtmgp, num_gases, ret, i, istat, ncid
  character(len=5)    ::  block_size_char
  character(len=12)    ::  emulated_component
  character(len=32 ), dimension(:),     allocatable :: kdist_gas_names, input_file_gas_names, input_names
  ! Output fluxes
  real(wp), dimension(:,:,:),         allocatable :: rsu_ref, rsd_ref, rsu_nn, rsd_nn, rsdu_ref, rsdu_nn

  ! Neural network objects/variables  
  ! 1-2 neural network models depending on emulated component (2 for RRTMGP)
  type(network_type), dimension(:),     allocatable :: nn_models   
  type(network_type)                                :: nn_model    
  character (len = 80)                              :: nn_modelfile_1, nn_modelfile_2 
    ! RRTMGP inputs for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: input_rrtmgp ! (nfeatures,nlay,block_size,nblocks)
                                        ! RRMTGP-NN-SW: first model is for tau, second for tau_rayleigh
  real(sp), dimension(:,:,:,:),         allocatable :: nn_input ! (nfeatures,nlay,block_size,nblocks)
  ! Output fluxes
  real(wp), dimension(:,:,:),   target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:,:), target, allocatable :: gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir
  ! Thermodynamic and other variables
  real(wp), dimension(:,:,:),           allocatable :: p_lay, p_lev, t_lay, t_lev ! nlay, block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
                                                     ! block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  ! Optical properties for NN development - eiher combined from gas and clouds, or just gas (described by tau and ssa)
  real(sp), dimension(:,:,:,:),         allocatable :: tau_sw, ssa_sw, g_sw 
  real(wp), dimension(:,:,:),           allocatable :: col_dry, vmr_h2o
  !
  ! Cloud variables
  !
  real(wp), allocatable, dimension(:,:,:) :: clwc, ciwc, cloud_fraction
  real(wp), allocatable, dimension(:,:,:) :: lwp, iwp, rel, rei
  logical,  allocatable, dimension(:,:,:) :: cloud_mask
  !
  ! various logical to control program
  logical :: use_rrtmgp_nn=.false., use_rte_nn=.false., use_rtegpt_nn=.false., use_reftrans_nn = .false., use_rte_rrtmgp_nn =.false.
  logical :: include_clouds=.true., compare_flux=.false., save_inputs_outputs = .false., do_gpt_flux, save_flux
  !
  ! Derived types from the RTE and RRTMGP libraries
  !
  type(ty_gas_optics_rrtmgp)                    :: k_dist
  type(ty_cloud_optics)                         :: cloud_optics
  type(ty_optical_props_2str)                   :: atmos, clouds
  type(ty_fluxes_flexible)                      :: fluxes

  real(wp), dimension(:,:), allocatable         :: toa_flux       ! block_size, ngpt
  real(wp), dimension(:  ), allocatable         :: def_tsi, mu0   ! block_size
  logical , dimension(:,:), allocatable         :: usecol         ! block_size, nblocks
  !
  ! ty_gas_concentration holds multiple columns; we make an array of these objects to
  !   leverage what we know about the input file
  !
  type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array
  real(wp), parameter :: deg_to_rad = acos(-1._wp)/180._wp
  real(wp) :: def_tsi_s, factor, rel_val, rei_val
  ! ecRAD type netCDF IO
  type(netcdf_file)  :: flux_file_netcdf

  ! Initialize GPU kernel - important not to include this substantial overhead in timing
#ifdef USE_OPENACC  
  type(cublasHandle) :: h
  istat = cublasCreate(h) 
#endif
  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
  !
  !  ------------ I/O and settings -----------------
  ! Compute fluxes per g-point?
  do_gpt_flux = .false.
  ! Save fluxes to netCDF file?
  save_flux   = .true.
  ! Compare fluxes?
  compare_flux = .true.

  print *, "Usage: ml_allsky_sw [block_size] [input file] [k-distribution file] [cloud coeff. file] [flux file] (5 args: use reference code) "
  print *, "OR   : ml_allsky_sw [block_size] [input file] [k-distribution file] [cloud coeff. file] [flux file] [emulated component] ", &
      "[NN model file(s)] (7-8 args, replace a component with NN)"
  print *, "Provide 'none' for cloud coeff. file to skip clouds (clear-sky computation)"

  nargs = command_argument_count()
  if (nargs <  4) call stop_on_err("Need to provide at least block_size input_file k-distribution file [cloud coeff. file]")

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i5)') block_size

  call get_command_argument(2, input_file)
  call get_command_argument(3, kdist_file)
  call get_command_argument(4, cloud_optics_file)
  if (trim(cloud_optics_file)=='none' ) include_clouds =.false.

  call get_command_argument(5, flx_file) 
  
  if(nargs == 6) then
     print *, "provide 5 (reference code) or 7-8 arguments (emulation mode), not 5"
     stop
  end if
  if(nargs > 6) then
    call get_command_argument(6, emulated_component) 
    call get_command_argument(7, nn_modelfile_1)

    print *, "program called in EMULATION MODE, using neural nets to emulate ", emulated_component

    if (emulated_component == 'rrtmgp') then
      use_rrtmgp_nn         = .true.
      print *, "EMULATING RRTMGP COMPUTATIONS WITH NN"
      if (nargs < 8) then
        call stop_on_err("Need to supply two model files to emulate RRTMGP (absorption and Rayleigh cross-section models, respectively")
      else
        call get_command_argument(8, nn_modelfile_2)
      end if
    else if (emulated_component == 'both') then
      use_rte_rrtmgp_nn     = .true.
      print *, "EMULATING ENTIRE RADIATION SCHEME WITH NN"
    else if (emulated_component == 'rte') then
      use_rte_nn            = .true.
    else if (emulated_component == 'rte-reftrans') then
      use_reftrans_nn       = .true.
      print *, "EMULATING REFTRANS COMPUTATIONS WITH NN"
    else
      call stop_on_err("If nargs>6, third argument specifies which code to replace with NNs and must be one of the following: & 
      &'both', 'rrtmgp', 'rte',' 'rte-reftrans'")
    end if
  end if


  ! How big is the problem? Does it fit into blocks of the size we've specified?
  !
  call read_size(input_file, ncol, nlay, nexp)
  print *, "input file:", input_file
  print *, "ncol:", ncol
  print *, "nexp:", nexp
  print *, "nlay:", nlay

  if(mod(ncol*nexp, block_size) /= 0 ) call stop_on_err("rrtmgp_rfmip_lw: number of columns doesn't fit evenly into blocks.")
  nblocks = (ncol*nexp)/block_size
  print *, "Doing ",  nblocks, "blocks of size ", block_size

  !
  ! Identify the set of gases used in the calculation 
  ! A gas might have a different name in the k-distribution than in the files
  ! provided (e.g. 'co2' and 'carbon_dioxide'), user needs to provide the correct ones
  !
  ! kdist_gas_names = ["h2o  ","co2  ","ch4  ","o2   ","o3   ", "n2o  ", "n2   "] !,"no2  "]
  ! input_file_gas_names =  ['water_vapor   ', &
  !                   'carbon_dioxide', &
  !                   'methane       ', &
  !                   'oxygen        ', &
  !                   'ozone         ', &
  !                   'nitrous_oxide ', &
  !                   'nitrogen      ']!,'no2           ']  
  kdist_gas_names = ["h2o  ","o3   ","co2  ","n2o  ", "ch4  ","o2   ", "n2   "] !,"no2  "]
  input_file_gas_names =  ['water_vapor   ', &
                    'ozone         ', &
                    'carbon_dioxide', &
                    'nitrous_oxide ', &
                    'methane       ', &
                    'oxygen        ', &
                    'nitrogen      ']!,'no2           ']  
  num_gases = size(kdist_gas_names)
  print *, "Calculation uses gases: ", (trim(kdist_gas_names(b)) // " ", b = 1, size(kdist_gas_names))

  ! How many gas optics inputs does this correspond to?
  ninputs_rrtmgp = 2 ! NN inputs consist of temperature, pressure and..
  do b = 1, num_gases
    if (trim(kdist_gas_names(b))=='o2' .or. trim(kdist_gas_names(b))=='n2') cycle
    ninputs_rrtmgp = ninputs_rrtmgp + 1 ! ..mixing ratios of all selected gases except N2 and O2 (constants)
  end do

  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
  ! Load neural network models
  ! nn_modelfile_1           = "../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.txt" 
  ! nn_modelfile_2           = "../../neural/data/BEST_tau-sw-ray-7-16-16_2.txt" 
  if (use_rrtmgp_nn) then
    allocate(nn_models(2))
    print *, "-------- Loading gas optics neural networks ---------"
	  print *, 'loading shortwave absorption model from ', nn_modelfile_1
    call nn_models(1) % load(nn_modelfile_1)
    print *, 'loading rayleigh model from ', nn_modelfile_2
    call nn_models(2) % load(nn_modelfile_2)
    ! ninputs_rrtmgp = size(nn_models(1) % layers(1) % w_transposed, 2)
    if (ninputs_rrtmgp /= size(nn_models(1) % layers(1) % w_transposed, 2)) then
      call stop_on_err("Number of RRTMGP-NN inputs as determined from the model differs from the provided gases")
    end if
    print *, "Number of NN inputs: ", ninputs_rrtmgp
  end if  
  if (use_reftrans_nn) then
    print *, 'loading reflectance-transmittance model from ', nn_modelfile_1
    call nn_model % load(nn_modelfile_1)
  end if
  if (use_rte_rrtmgp_nn) then
    print *, 'loading radscheme model from ', nn_modelfile_1
    allocate(input_names(   ninputs_rrtmgp)) ! temperature + pressure + gases
    allocate(input_rrtmgp(  ninputs_rrtmgp, nlay, block_size, nblocks))

    print *, "shape nn gasopt inp", shape(input_rrtmgp)
    call nn_model % load(nn_modelfile_1)
  end if
  ! Note: The coefficients for scaling RRTMGP-NN inputs and outputs are currently hard-coded in mo_rrtmgp_nn_constants.F90

  !
  ! Allocation on assignment within reading routines
  !
  call read_and_block_pt(input_file, block_size, p_lay, p_lev, t_lay, t_lev)
  !
  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(nlay, 1, 1)
  !
  ! Read the gas concentrations and surface properties
  !
  call read_and_block_gases_ty(input_file, block_size, kdist_gas_names, input_file_gas_names, gas_conc_array)
  ! do b = 1, size(gas_conc_array(1)%concs)
  !   print *, "max of gas ", gas_conc_array(1)%gas_name(b), ":", maxval(gas_conc_array(1)%concs(b)%conc)
  ! end do

  call read_and_block_sw_bc(input_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)

  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))
  if(.not. k_dist%source_is_external()) stop "rrtmgp_rfmip_sw: k-distribution file isn't SW"
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()

  allocate(toa_flux(k_dist%get_ngpt(), block_size), &
           def_tsi(block_size), usecol(block_size,nblocks))
  !$acc enter data create (toa_flux, def_tsi)

  !
  ! RRTMGP won't run with pressure less than its minimum. The top level in the RFMIP file
  !   is set to 10^-3 Pa. Here we pretend the layer is just a bit less deep.
  !   This introduces an error but shows input sanitizing.
  !
  if(top_at_1) then
    p_lev(1,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lev(nlay+1,:,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  end if

  !
  ! Load cloud optics coefficients
  !
  if (include_clouds) then
    ! Initialize and allocate derived type
    call stop_on_err(clouds%init(k_dist%get_band_lims_wavenumber()))
    call stop_on_err(clouds%alloc_2str(block_size, nlay))

    ! if(use_luts) then
      call load_cld_lutcoeff (cloud_optics, cloud_optics_file)
    ! else
    !   call load_cld_padecoeff(cloud_optics, cloud_optics_file)
    ! end if

    allocate(lwp(nlay,block_size,nblocks), iwp(nlay,block_size,nblocks))
    allocate(rel(nlay,block_size,nblocks), rei(nlay,block_size,nblocks))
    allocate(cloud_mask(nlay,block_size,nblocks))
    cloud_mask = .false.
    ! Load CAMS cloud data (cloud liquid water and ice contents, cloud fraction)
    call read_and_block_clouds_cams(input_file, block_size, clwc, ciwc, cloud_fraction)

    ! Particle effective size/radius
    rel_val = 0.5_wp * (cloud_optics%get_min_radius_liq() + cloud_optics%get_max_radius_liq())
    rei_val = 0.5_wp * (cloud_optics%get_min_radius_ice() + cloud_optics%get_max_radius_ice())
    rel = 0.0_wp
    rei = 0.0_wp
    ! Compute ice and liquid water paths from mixing ratios (from ecRAD)
    do b = 1, nblocks
      ! print *, "CLWC", maxval(clwc(:,:,b))
      do icol = 1, block_size
        do ilay = 1, nlay
          if (cloud_fraction(ilay,icol,b) > 0.0_wp) then
            cloud_mask(ilay,icol,b) = .true.
          end if
          ! Compute in-cloud liquid and ice water path
          ! if (config%is_homogeneous) then
            ! Homogeneous solvers assume cloud fills the box
            ! horizontally, so we don't divide by cloud fraction
            factor = ( p_lev(ilay+1,icol,b) -p_lev(ilay,icol,b)  ) / 9.80665_wp
          ! else
          !   factor = ( p_lev(ilay+1,icol,b) -p_lev(ilay,icol,b)  ) / (9.80665_wp * cloud_fraction(ilay,icol,b))
          ! end if
          lwp(ilay,icol,b) = factor * clwc(ilay,icol,b)
          iwp(ilay,icol,b) = factor * ciwc(ilay,icol,b)

          rel(ilay,icol,b) = rel_val
          rei(ilay,icol,b) = rei_val
        end do
      end do
    end do
    ! looks like the cloud optics extension takes lwp and iwp in g/kg 
    lwp = 1000 * lwp
    iwp = 1000 * iwp
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

  !
  ! Allocate g-point fluxes if desired
  !
  if (do_gpt_flux) then
    allocate(gpt_flux_up(ngpt, nlay+1, block_size, nblocks), &
             gpt_flux_dn(ngpt, nlay+1, block_size, nblocks))
    allocate(gpt_flux_dn_dir(ngpt, nlay+1, block_size, nblocks))
  end if

  ! allocate(mu0(block_size), sfc_alb_spec(nbnd,block_size))
  allocate(mu0(block_size), sfc_alb_spec(ngpt,block_size))

  !$acc enter data create (sfc_alb_spec, mu0) copyin(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle) 

  ! Allocate derived types - optical properties of gaseous atmosphere
  ! Device allocation happens inside procedures
  call stop_on_err(atmos%alloc_2str(block_size, nlay, k_dist))

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
  ret =  gptlinitialize()
#endif

  print *, "-------------------------------------------------------------------------"
  if (include_clouds) then
    print *, "starting all-sky shortwave computations which includes gases and clouds"
  else
    print *, "starting clear-sky shortwave computations which just includes gases"
  end if
  if (use_rrtmgp_nn) print *, "Using neural networks as RRTMGP kernel"

  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
  ret =  gptlstart('radiation_total_shortwave')
#endif
#ifdef USE_OPENMP
  !$OMP PARALLEL shared(nn_models, k_dist) firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0,fluxes,atmos)
  !$OMP DO 
#endif
  if (use_rte_rrtmgp_nn) def_tsi_s = sum(k_dist%solar_source)
  do b = 1, nblocks

    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)
    fluxes%flux_dn_dir => flux_dn_dir(:,:,b)
    if (do_gpt_flux) then
      ! If g-point fluxes are allocated, the RTE kernels write to 3D arrays, otherwise broadband
      ! computation is inlined
      fluxes%gpt_flux_up => gpt_flux_up(:,:,:,b)
      fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:,b)
      fluxes%gpt_flux_dn_dir => gpt_flux_dn_dir(:,:,:,b)
    end if

    if (use_rte_rrtmgp_nn) then 
#ifdef USE_TIMING
        ret =  gptlstart('predict_nn_radscheme_sw')
#endif
      call stop_on_err(get_gasopt_nn_inputs(block_size, nlay, ninputs_rrtmgp, &
                                    p_lay(:,:,b), t_lay(:,:,b), gas_conc_array(b),           &
                                    input_rrtmgp(:,:,:,b), input_names))
      ! Use emulator for whole radiation scheme
#ifdef USE_TIMING
        ret =  gptlstart('preproc1')
#endif
      ! Cosine of the solar zenith angle
      !$acc parallel loop
      do icol = 1, block_size
        mu0(icol) = merge(cos(solar_zenith_angle(icol,b)*deg_to_rad), 1._wp, usecol(icol,b))

        do igpt  = 1, ngpt
          ! Normalize incoming solar flux to match RFMIP specification
          toa_flux(igpt,icol) = k_dist%solar_source(igpt) * total_solar_irradiance(icol,b)/def_tsi_s
          ! Apply boundary condition
          toa_flux(igpt,icol) = toa_flux(igpt,icol) * mu0(icol)
        end do
      end do
#ifdef USE_TIMING
        ret =  gptlstop('preproc1')
#endif

      call predict_nn_radscheme_sw(block_size, nlay, ninputs_rrtmgp, & 
                        input_rrtmgp(:,:,:,b), & ! RRTMGP inputs (gas concentrations + T + p)
                        lwp(:,:,b), iwp(:,:,b), & ! cloud liquid and ice water paths
                        mu0, surface_albedo(:,b),  & ! RTE inputs; no inc. flux because its assumed constant across columns
                        nn_model,    & ! pre-trained neural network 
                        fluxes, sum(toa_flux,dim=1))
#ifdef USE_TIMING
        ret =  gptlstop('predict_nn_radscheme_sw')
#endif
    else
    
      !
      ! Compute the optical properties of clouds
      !
      if (include_clouds) then
#ifdef USE_TIMING
        ret =  gptlstart('cloud_optics')
#endif
        call stop_on_err( cloud_optics%cloud_optics(lwp(:,:,b), iwp(:,:,b), &
                          rel(:,:,b), rei(:,:,b), clouds))
           
#ifdef USE_TIMING
        ret =  gptlstop('cloud_optics')
#endif
      end if
      !
      ! Compute the optical properties of the atmosphere and the Planck source functions
      !    from pressures, temperatures, and gas concentrations...
      !
#ifdef USE_TIMING
      ret =  gptlstart('gas_optics_sw')
#endif
      if (use_rrtmgp_nn) then
        call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                          p_lev(:,:,b),       &
                                          t_lay(:,:,b),       &
                                          gas_conc_array(b),  &
                                          atmos,      &
                                          toa_flux, neural_nets=nn_models))
      else
        call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                          p_lev(:,:,b),       &
                                          t_lay(:,:,b),       &
                                          gas_conc_array(b),  &
                                          atmos,      &
                                          toa_flux))
      end if
      ! print *, "mean tau", mean_3d(atmos%tau)
      ! print *, "mean ssa", mean_3d(atmos%ssa)
      ! print *," max, min (tau)",   maxval(atmos%tau), minval(atmos%tau)
      ! print *," max, min (ssa)",   maxval(atmos%ssa), minval(atmos%ssa)
      ! print *," max, min (g)",   maxval(atmos%g), minval(atmos%g)

#ifdef USE_TIMING
      ret =  gptlstop('gas_optics_sw')
      ret =  gptlstart('clouds_deltascale_increment')
#endif       
      if (include_clouds) then

        call stop_on_err(clouds%delta_scale())
        call stop_on_err(clouds%increment(atmos))
        ! print *, "mean tau after adding cloud optics", mean_3d(atmos%tau)
      end if
#ifdef USE_TIMING
      ret =  gptlstop('clouds_deltascale_increment')
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
        ! print *, "icol", icol, "sza", solar_zenith_angle(icol,b), "usecol", usecol(icol,b)
      end do
      !$acc end parallel

      !
      ! ... and compute the spectrally-resolved fluxes, providing reduced values
      !    via ty_fluxes_broadband
      !
#ifdef USE_TIMING
      ret =  gptlstart('rte_sw')
#endif
      ! Emulate entire solver?
      if (use_rte_nn) then

      else 
        ! Emulate reflectance-transmittance computations?
        if (use_reftrans_nn) then
            call stop_on_err(rte_sw(atmos,   &
                                    top_at_1,        &
                                    mu0,             &
                                    toa_flux,        &
                                    sfc_alb_spec,  sfc_alb_spec,  &
                                    fluxes, &
                                    neural_net=nn_model))
        else ! reference
            call stop_on_err(rte_sw(atmos,   &
                                    top_at_1,        &
                                    mu0,             &
                                    toa_flux,        &
                                    sfc_alb_spec, sfc_alb_spec,  &
                                    fluxes))
        end if
      end if
#ifdef USE_TIMING
      ret =  gptlstop('rte_sw')
#endif
  
  end if ! Use emulator for whole radiation scheme?

  end do !blocks
#ifdef USE_OPENMP
  !$OMP END DO
  !$OMP END PARALLEL
  !$OMP barrier
#endif
  !
  ! End timers
  !
#ifdef USE_TIMING
  ret =  gptlstop('radiation_total_shortwave')
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  !$acc exit data delete(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)
  !$acc exit data delete(sfc_alb_spec, mu0, toa_flux, def_tsi)
  call atmos%finalize() ! Also deallocates arrays on device

#ifdef USE_OPENACC  
  istat = cublasDestroy(h) 
#endif
  print *, "Finished with computations!"
  print *, "------------------------------------------------------------------------------"


  !
  ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
  
  ! do b = 1, nblocks
  !   do icol = 1, block_size
  !     if(.not. usecol(icol,b)) then
  !       flux_up(:,icol,b)  = 0._wp
  !       flux_dn(:,icol,b)  = 0._wp
  !     end if
  !   end do
  ! end do

  ! print *, "mean of flux_down is:", mean_3d(flux_dn)  ! 
  ! print *, "mean of flux_up is:", mean_3d(flux_up)    !
  ! print *, "mean of flux_net is:", mean_3d(flux_dn - flux_up)    ! 
  !  if(do_gpt_flux) print *, "mean of gpt_flux_up for gpt=1 is:", mean_3d(gpt_flux_up(1,:,:,:))

  if (compare_flux) then
    print *, "-- FLUX AND HEATING RATE ERRORS OF RESULTS COMPARED TO REFERENCE RTE+RRTMGP --"
    allocate(rsd_ref( nlay+1, ncol, nexp))
    allocate(rsu_ref( nlay+1, ncol, nexp))  
    allocate(rsdu_ref( nlay+1, ncol, nexp))  
    allocate(rsd_nn( nlay+1, ncol, nexp))
    allocate(rsu_nn( nlay+1, ncol, nexp))
    allocate(rsdu_nn( nlay+1, ncol, nexp))

    flx_file_ref = 'fluxes/CAMS_2015_rsud_REFERENCE.nc'
    print *, "reference file:", flx_file_ref
    print *, "------------------------------------------------------------------------------"

    call unblock(flux_up, rsu_nn)
    call unblock(flux_dn, rsd_nn)

    rsdu_nn = rsd_nn - rsu_nn

    if(nf90_open(trim(flx_file_ref), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_ref))

    rsu_ref = read_field(ncid, "rsu", nlay+1, ncol, nexp)
    rsd_ref = read_field(ncid, "rsd", nlay+1, ncol, nexp)
    rsdu_ref = rsd_ref - rsu_ref

    print *, "mean net flux NEW", mean_3d(rsdu_nn), "REFERENCE", mean_3d(rsdu_ref)


    print *, "-------------- UPWELLING ----------------"

    print *, "bias in upwelling flux, top-of-atmosphere:", &
      bias(reshape(rsu_ref(1,:,1), shape = [1*ncol]),    reshape(rsu_nn(1,:,1), shape = [1*ncol]))

    print *, "MAE in upwelling flux, top-of-atmosphere: ", &
      mae(reshape(rsu_ref(1,:,1), shape = [1*ncol]),              reshape(rsu_nn(1,:,1), shape = [1*ncol]))   

    print *, "MAE in upwelling flux:                    ", &
      mae(reshape(rsu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)]))

    print *, "-------------- DOWNWELLING --------------"

    print *, "MAE in downwelling flux:                  ", &
    mae(reshape(rsd_ref(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsd_nn(:,:,:), shape = [nexp*ncol*(nlay+1)]))

    print *, "-------------- NET FLUX -----------------"

    print *, "MAE in net flux:                          ", &
     mae(reshape(rsdu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)]))

    print *, "RMSE in net flux:                         ", &
     rmse(reshape(rsdu_ref(:,:,:), shape = [nexp*ncol*(nlay+1)]),    reshape(rsdu_nn(:,:,:), shape = [nexp*ncol*(nlay+1)]))

    print *, "RMSE in net flux, SURFACE:                ", &
     rmse(reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol]))

    print *, "bias in net flux, SURFACE:                ", &
     bias(reshape(rsdu_ref(nlay+1,:,1), shape = [1*ncol]),    reshape(rsdu_nn(nlay+1,:,1), shape = [1*ncol]))

    print *, "---------"

    print *, "Max-diff in downwelling flux:     ", &
     maxval(abs(rsd_ref(:,:,:)-rsd_nn(:,:,:)))
 
    print *, "Max-diff in upwelling flux:       ", &
     maxval(abs(rsu_ref(:,:,:)-rsu_nn(:,:,:)))

    print *, "Max-diff in net flux:             ", &
     maxval(abs(rsdu_ref(:,:,:)-rsdu_nn(:,:,:))) 

    deallocate(rsd_ref,rsu_ref,rsd_nn,rsu_nn,rsdu_ref,rsdu_nn)

  end if

  ! Save fluxes?  we might want to evaluate the fluxes predicted with neural networks
  if ((save_flux) .and. (len(trim(flx_file))>0 )) then
    ! flx_file = 'fluxes/CAMS_2018_rsud_REF.nc'
    print *, "Attempting to save broadband fluxes to ", flx_file

      ! Create file
    ! call flux_file_netcdf%create(trim(flx_file),override_file=.true.)

    call flux_file_netcdf%open(trim(flx_file), redefine_existing=.true.,is_hdf5_file=.true.)

    ! Define dimensions
    ! call flux_file_netcdf%define_dimension("expt", nexp)
    ! call flux_file_netcdf%define_dimension("site", ncol)
    ! call flux_file_netcdf%define_dimension("level", nlay+1)

    ! call flux_file_netcdf%define_variable("plev", &
    !   dim3_name="expt", dim2_name="site", dim1_name="level", &
    !   units_str="Pa", standard_name="air_pressure", long_name="Pressure at layer edge")

    ! call flux_file_netcdf%define_variable("rsu", &
    !   dim3_name="time", dim2_name="site", dim1_name="level", long_name="upwelling shortwave flux")
      
    ! call flux_file_netcdf%define_variable("rsd", &
    !   dim3_name="time", dim2_name="site", dim1_name="level", long_name="downwelling shortwave flux")

    call flux_file_netcdf%end_define_mode()
    ! call unblock_and_write(trim(flx_file), 'plev', p_lev)
    call unblock_and_write(trim(flx_file), 'rsu', flux_up)
    call unblock_and_write(trim(flx_file), 'rsd', flux_dn)

    call unblock_and_write(trim(flx_file), 'solar_zenith_angle', solar_zenith_angle)


    call flux_file_netcdf%close()
    print *, "Broadband fluxes saved to ", flx_file
    
    deallocate(flux_up, flux_dn)

  end if 

  contains

  ! -------------------------------------------------------------------------------------------------
  ! Routine for preparing neural network inputs from the gas concentrations, temperature and pressure
  ! This routine, used for generating training data, differs from the compute_nn_inputs in gas_optics_rrtmgp
  ! because "operationally" the loaded NN model specifies which gases are used, and if a gas is missing
  ! from available gases (gas_desc) it needs to be set to zero or a reference concentration is used.
  ! Here we just use the available gases
  function get_gasopt_nn_inputs(ncol, nlay, ninputs_rrtmgp, &
                              play, tlay, gas_desc,           &
                              nn_inputs, input_names) result(error_msg)

    integer,                                  intent(in   ) ::  ncol, nlay, ninputs_rrtmgp
    real(wp), dimension(nlay,ncol),           intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                                tlay
    type(ty_gas_concs),                       intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  
    real(sp), dimension(ninputs_rrtmgp, nlay, ncol),  intent(inout) ::  nn_inputs !
    character(len=32 ), dimension(ninputs_rrtmgp),    intent(inout) ::  input_names 
    character(len=128)                                  :: error_msg
    ! ----------------------------------------------------------
    ! Local variables
    integer :: igas, ilay, icol, ndims, idx_h2o, idx_o3, idx_gas, i
    character(len=32)                           :: gas_name    
    real(wp),       dimension(nlay,ncol)        :: vmr

    !  Neural network inputs are a vector consisting of temperature and pressure followed by gas concentrations
    ! These inputs are scaled to a range of (0-1), additionally some are power or log scaled: 
    ! The inputs are:   tlay,    log(play),   h2o**(1/4), o3**(1/4), co2, ..

    ! First lets write temperature, pressure, water vapor and ozone into the inputs
    ! These are assumed to always be present!
    error_msg = gas_desc%get_conc_dims_and_igas('h2o', ndims, idx_h2o)
    error_msg = gas_desc%get_conc_dims_and_igas('o3',  ndims, idx_o3)
    if(error_msg  /= '') return

    input_names(1) = 'tlay'
    input_names(2) = 'play'
    input_names(3) = 'h2o'
    input_names(4) = 'o3'
    do icol = 1, ncol
      do ilay = 1, nlay
        nn_inputs(1,ilay,icol)    =  tlay(ilay,icol)  
        nn_inputs(2,ilay,icol)    = log(play(ilay,icol))
        nn_inputs(3,ilay,icol)    = sqrt(sqrt(gas_desc%concs(idx_h2o)%conc(ilay,icol)))
        nn_inputs(4,ilay,icol)    = sqrt(sqrt(gas_desc%concs(idx_o3) %conc(ilay,icol)))
      end do
    end do

    ! Write the remaining gases
    ! The scaling coefficients are tied to a string specifying the gas names, these are all loaded from rrtmgp_constants.F90
    ! Lets find the indices which map the available gases to the scaling coefficients of each gas, 
    ! and also the dimensions of the concentration array
    i = 5
    do igas = 1, size(gas_desc%gas_name)
      gas_name = gas_desc%gas_name(igas)
      if(gas_name=='h2o' .or. gas_name=='o3' .or. gas_name=='o2' .or. gas_name=='n2') cycle

      ! Save gas name
      input_names(i) = gas_name

      ! Fill 2D (lay,col) array with gas concentration
      error_msg = gas_desc%get_vmr(gas_name, vmr(:,:))

      ! Write to nn_input non-contiguously
      nn_inputs(i,:,:) = vmr(:,:)

      ! print *, "i", i, "GAS NAME", gas_name
    
      i = i + 1
    end do
    
    ! do igas = 1, ninputs_rrtmgp
    !   print '(A25,I2,A2,A8,F6.3,F6.3)', "Min,max of NN-input ", igas, " =", input_names(igas), &
    !               minval(nn_inputs(igas,:,:)), maxval(nn_inputs(igas,:,:))
    ! end do


  end function get_gasopt_nn_inputs


  ! Interface for NN emulation of the entire radiation scheme (shortwave)
  ! Inputs are vertical COLUMNS of atmospheric conditions (T,p, gas concentrations) + boundary conditions,
  ! and outputs are COLUMNS of broadband fluxes (upwelling and downwelling)
  subroutine predict_nn_radscheme_sw(nbatch, nlay, nx_gasopt, & 
                                    rrtmgp_inputs, & ! RRTMGP inputs (gas concentrations + T + p)
                                    cloud_lwp, cloud_iwp, & ! cloud liquid and ice water paths
                                    mu0, sfc_alb,  & ! RTE inputs; no inc. flux because its assumed constant across columns
                                    neural_net,    & ! pre-trained neural network 
                                    fluxes, incflux)
    integer,                                    intent(in)  ::  nbatch, nlay, nx_gasopt
    real(sp), dimension(nx_gasopt, nlay, nbatch), intent(in)  ::  rrtmgp_inputs 
    real(wp), dimension(nlay, nbatch),            intent(in)  ::  cloud_lwp, cloud_iwp
    real(wp), dimension(nbatch),                  intent(in)  ::  mu0, sfc_alb 
    real(wp), dimension(nbatch)             :: incflux

    type(network_type),                         intent(in)  ::  neural_net 
    class(ty_fluxes_flexible),                  intent(inout) :: fluxes  

    ! Local variables
    integer :: nlev, nx, ny, i, j
    integer :: i1e, i2s, i2e, i3s, i3e, i4, i5
    real(wp), dimension(:,:), allocatable   :: nn_inputs, nn_outputs
    real(sp), dimension(542)                :: xmin, xmax
    ! load input scaling coefficients
    open(20,file="../../neural/data/nn_radscheme_xmin_xmax.txt",status="old",action="read")
    do i = 1, 542
      read(20,*) xmax(i)
    end do
    close(20)
    xmin = 0.0_wp
    
    ! number of samples (profiles) nbatch = ncol

    ! number of features (inputs)
    ! the inputs are stacked columns of atmospheric conditions (gas concentrations + T + p), cloud conditions and
    ! scalars mu+ and sfc_alb 
    ! Incoming flux at top of the atmosphere is assumed constant! (not spectrally constant, but the solar gpt flux is still an array of constants)
    !           gases,         clouds,  mu0, sfc_alb
#ifdef USE_TIMING
        ret =  gptlstart('preproc2')
#endif
    nx    =    nlay*nx_gasopt + 2*nlay  + 1 + 1     +1
    ! outputs
    nlev = nlay + 1
    ny   = 2*nlev ! 3 if dir downward flux needed
    allocate(nn_inputs (nx, nbatch))
    allocate(nn_outputs(ny, nbatch))

    i1e = nlay*nx_gasopt  ! 1-420
    i2s = i1e + 1         ! 421-480
    i2e = i1e + nlay  
    i3s = i2e + 1         ! 481-540
    i3e = i2e + nlay 
    i4  = i3e + 1         ! 541
    i5  = i4 + 1          ! 542

    
    do i = 1, nx
      if (xmax(i) < 1e-9) then 
         xmax(i) = 1.0_sp
      end if
    end do

    ! print *, "xmax", xmax
    ! print *, "CLOUD LWP MAX", maxval(cloud_lwp), "IWP", maxval(cloud_iwp)

    ! print *, "shape nn input", shape(nn_inputs), "shape cloud lwp", shape(cloud_lwp), "nbatch", nbatch
    ! print *, "shape rrtmgp inputs", shape(rrtmgp_inputs)

    do j = 1, nbatch
      nn_inputs(1:i1e,j)     = reshape(rrtmgp_inputs(:,:,j),(/nx_gasopt*nlay/))
      nn_inputs(i2s:i2e,j)   = cloud_lwp(:,j)
      nn_inputs(i3s:i3e,j)   = cloud_iwp(:,j)
      nn_inputs(i4,j)        = mu0(j)
      nn_inputs(i5,j)        = sfc_alb(j)

      do i = 1, nx
        ! if (xmax(i) /= 0.0_sp) then 
          nn_inputs(i,j) = nn_inputs(i,j) / xmax(i)
        ! end if
      end do
      ! print *, "i 177 2",nn_inputs(177,j)

      ! print *, "1", nn_inputs(1:i1e,j)
      ! print *, "2", nn_inputs(i2s:i2e,j)
      ! print *, "mean nn inp",j,":", mean(nn_inputs(1:i1e,j))
    end do 
#ifdef USE_TIMING
        ret =  gptlstop('preproc2')
#endif
    ! do i = 1, nx
    !     print *, i, ":", maxval(nn_inputs(i,:))
    ! end do

#ifndef DOUBLE_PRECISION
    call neural_net % output_sgemm_flat(nx, ny, nbatch, nn_inputs, nn_outputs)
#endif
#ifdef USE_TIMING
        ret =  gptlstart('postproc')
#endif
    ! print *, "mean nn outp 1", mean_2d(nn_outputs)
    do j = 1, nbatch
      ! Postprocess: reverse standard scaling
      do i = 1, ny
        nn_outputs(i, j) =  nn_outputs(i, j) * incflux(j)
        nn_outputs(i, j) = max(0.0, nn_outputs(i,j))
      end do
      ! Save to flux
      fluxes%flux_up(1:nlev,j) = nn_outputs(1:nlev,j)
      fluxes%flux_dn(1:nlev,j) = nn_outputs(nlev+1:2*nlev,j)
    end do
#ifdef USE_TIMING
        ret =  gptlstop('postproc')
#endif
    ! print *, "mean nn outp 2", mean_2d(nn_outputs)

  end subroutine predict_nn_radscheme_sw


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

  function mean_2d(x2) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x2
    real(wp) :: mean2
    
    mean2 = sum(sum(x2, dim=1),dim=1) / (size(x2))
  end function mean_2d

  function mean_3d(x3) result(mean3)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x3
    real(wp) :: mean3
    
    mean3 = sum(sum(sum(x3, dim=1),dim=1),dim=1) / (size(x3))
  end function mean_3d

end program rrtmgp_rfmip_sw
