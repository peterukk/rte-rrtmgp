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
! Fortran program arguments:
! ml_allsky_sw [block_size] [input file] [k-distribution file] [cloud coeff. file] [optional file to save NN inputs/outputs]"
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
  !
  ! Coefficients used to scale NN inputs
  !
  use mo_rrtmgp_nn_constants,only: nn_gasopt_input_names, nn_input_maxvals, nn_input_minvals
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
                                   read_and_block_sw_bc, read_and_block_clouds_cams, determine_gas_names                             
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
  character(len=132)  ::  flx_file, timing_file, nndev_file='', nn_input_str, cmt
  integer             ::  nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size
  logical             ::  top_at_1, do_scattering
  integer             ::  b, icol, ilay, igpt, igas, ngas, ninputs, num_gases, ret, i, istat
  character(len=4)    ::  block_size_char
  character(len=6)    ::  emulated_component
  character(len=32 ), dimension(:),     allocatable :: kdist_gas_names, input_file_gas_names, gasopt_input_names
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
  ! RRTMGP inputs for NN development
  real(sp), dimension(:,:,:,:),         allocatable :: nn_gasopt_input ! (nfeatures,nlay,block_size,nblocks)
  ! RTE inputs for NN development
  real(sp),                             allocatable :: toa_flux_save(:,:,:), sfc_alb_spec_save(:,:,:), mu0_save(:,:)
  ! RTE outputs for NN development
  real(wp), dimension(:,:,:,:),         allocatable :: reftrans_variables
  real(sp), dimension(:,:,:,:),         allocatable :: Rdif_save, Tdif_save, Rdir_save, Tdir_save
  !
  ! Cloud variables
  !
  real(wp), allocatable, dimension(:,:,:) :: clwc, ciwc, cloud_fraction
  real(wp), allocatable, dimension(:,:,:) :: lwp, iwp, rel, rei
  logical,  allocatable, dimension(:,:,:) :: cloud_mask
  !
  ! various logical to control program
  logical ::  include_clouds=.true., compare_flux=.false., save_inputs_outputs = .false., &
          &   do_gpt_flux, save_reftrans, save_rrtmgp, preprocess_rrtmgp_inputs, clouds_provided = .true.
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
  type(netcdf_file)                      :: nndev_file_netcdf, flux_file_netcdf

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
  do_gpt_flux   = .false.
  ! When writing inputs and outputs for ML training, save also gas optics output variables?
  save_rrtmgp   = .false.
  ! When writing inputs and outputs for ML training, save also reflectance-transmittance variables?
  save_reftrans = .false.
  ! When writing RRTMGP inputs for ML training, preprocess inputs like in Ukkonen 2020?
  preprocess_rrtmgp_inputs = .false.

  print *, "Usage: ./allsky_sw_gendata [block_size] [input file] [k-distribution file] [cloud coeff. file] (4 args: use reference code) "
  print *, "OR   : ./allsky_sw_gendata [block_size] [input file] [k-distribution file] [cloud coeff. file] [input-output file] ", &
      " (5 args: ref. code, save training data)"
  ! print *, "OR   : ml_allsky_sw [block_size] [input file] [k-distribution file] [cloud coeff. file] [emulated component] ", &
  !     "[NN model file(s)] (6-7 args, replace a component with NN)"
  print *, "Provide 'none' for cloud coeff. file to skip clouds (clear-sky computation)"

  nargs = command_argument_count()
  if (nargs <  4) call stop_on_err("Need to provide at least block_size input_file k-distribution file [cloud coeff. file]")

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size

  call get_command_argument(2, input_file)
  call get_command_argument(3, kdist_file)
  call get_command_argument(4, cloud_optics_file)
  if (trim(cloud_optics_file)=='none' ) include_clouds =.false.

  if (nargs == 5) then
    call get_command_argument(5, nndev_file)
    save_inputs_outputs = .true.
  end if

  if (save_reftrans) then
    save_rrtmgp = .false.
    if (.not. include_clouds) call stop_on_err("Need to provide cloud coeff. file for saving reftrans data")
  end if

  if (save_rrtmgp) then
    save_reftrans = .false.
    if (include_clouds) call stop_on_err("clouds should not be included when saving rrtmgp inout data")
  end if

  if (.not. save_inputs_outputs) then
    save_rrtmgp = .false.
    save_reftrans = .false.
  end if
  
  ! How big is the problem? Does it fit into blocks of the size we've specifed?
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
  print *, "Calculation uses gases: ", (trim(input_file_gas_names(b)) // " ", b = 1, size(input_file_gas_names))

  ! How many neural network input features does this correspond to?
  ninputs = 2 ! RRTMGP-NN inputs consist of temperature, pressure and..
  do b = 1, num_gases
    if (trim(kdist_gas_names(b))=='o2' .or. trim(kdist_gas_names(b))=='n2') cycle
    ninputs = ninputs + 1 ! ..mixing ratios of all selected gases except N2 and O2 (constants)
  end do

  ! --------------------------------------------------
  !
  ! Prepare data for use in rte+rrtmgp
  !
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

  call read_and_block_sw_bc(input_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle,  sza_fill_randoms_in=.true.)
  ! call read_and_block_sw_bc(input_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)

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
    allocate(clwc(nlay, block_size,   nblocks), ciwc(nlay, block_size,   nblocks))
    allocate(cloud_fraction(nlay, block_size,   nblocks))
    allocate(lwp(nlay,block_size,nblocks), iwp(nlay,block_size,nblocks))
    allocate(rel(nlay,block_size,nblocks), rei(nlay,block_size,nblocks))
    allocate(cloud_mask(nlay,block_size,nblocks))
    cloud_mask = .false.

    ! clouds_provided = .true.
    ! Load CAMS cloud data (cloud liquid water and ice contents, cloud fraction)
    ! if (clouds_provided) then
      call read_and_block_clouds_cams(input_file, block_size, clwc, ciwc, cloud_fraction)
    ! end if

    ! Particle effective size/radius
    rel_val = 0.5_wp * (cloud_optics%get_min_radius_liq() + cloud_optics%get_max_radius_liq())
    rei_val = 0.5_wp * (cloud_optics%get_min_radius_ice() + cloud_optics%get_max_radius_ice())
    rel = 0.0_wp
    rei = 0.0_wp
    ! Compute ice and liquid water paths from mixing ratios (from ecRAD)
    do b = 1, nblocks
      do icol = 1, block_size
        do ilay = 1, nlay
          ! if (clouds_provided) then
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
            ! looks like the cloud optics extension takes lwp and iwp in g/kg 
            lwp(ilay,icol,b) = 1000 * lwp(ilay,icol,b)
            iwp(ilay,icol,b) = 1000 * iwp(ilay,icol,b)
            
          ! else

          !   cloud_mask(ilay,icol,b) = p_lev(ilay,icol,b) > 100._wp * 100._wp .and. &
          !                       p_lev(ilay,icol,b) < 900._wp * 100._wp .and. &
          !                       mod(icol, 3) /= 0
          !   !
          !   ! Ice and liquid will overlap in a few layers
          !   !
          !   lwp(ilay,icol,b) = merge(10._wp,  0._wp, cloud_mask(ilay,icol,b) .and. t_lay(ilay,icol,b) > 263._wp)
          !   iwp(ilay,icol,b) = merge(10._wp,  0._wp, cloud_mask(ilay,icol,b) .and. t_lay(ilay,icol,b) < 273._wp)
          !   rel(ilay,icol,b) = merge(rel_val, 0._wp, lwp(ilay,icol,b) > 0._wp)
          !   rei(ilay,icol,b) = merge(rei_val, 0._wp, iwp(ilay,icol,b) > 0._wp)
          ! end if
        end do
      end do
    end do
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

  if (save_inputs_outputs) then
    ! RRTMGP inputs
    allocate(gasopt_input_names(ninputs)) ! temperature + pressure + gases
    allocate(nn_gasopt_input(   ninputs, nlay, block_size, nblocks))
    ! RTE inputs
    allocate(toa_flux_save(k_dist%get_ngpt(), block_size, nblocks))
    allocate(mu0_save(block_size,nblocks), sfc_alb_spec_save(ngpt,block_size,nblocks))
    if (save_reftrans) then
      allocate(Rdif_save(ngpt,nlay,block_size,nblocks), Tdif_save(ngpt,nlay,block_size,nblocks))
      allocate(Rdir_save(ngpt,nlay,block_size,nblocks), Tdir_save(ngpt,nlay,block_size,nblocks))
      allocate(reftrans_variables(ngpt,nlay,block_size,4))
    end if
    if (save_rrtmgp) then
      ! number of dry air molecules
      allocate(col_dry(nlay, block_size, nblocks), vmr_h2o(nlay, block_size, nblocks)) 
    end if
    if (save_rrtmgp .or. save_reftrans) then
      allocate(tau_sw(ngpt, nlay, block_size, nblocks), ssa_sw(ngpt, nlay, block_size, nblocks))
      ! cloud optical properties
      if (save_reftrans) then
        allocate(g_sw(ngpt, nlay, block_size, nblocks))
      end if
    end if
  end if

#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate 
  ret =  gptlinitialize()
#endif

  print *, "-------------------------------------------------------------------------"
  if (include_clouds) then
    print *, "starting all-sky shortwave computations which includes gases and clouds"
  else
    print *, "starting clear-sky shortwave computations which just includes gases"
  end if

  !
  ! Loop over blocks
  !

#ifdef USE_OPENMP
  !$OMP PARALLEL shared(neural_nets, k_dist) firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0,fluxes,atmos)
  !$OMP DO 
#endif
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

      !
      ! Compute the optical properties of clouds
      !
      if (include_clouds) then
        call stop_on_err( cloud_optics%cloud_optics(lwp(:,:,b), iwp(:,:,b), &
                          rel(:,:,b), rei(:,:,b), clouds))
      end if
      !
      ! Compute the optical properties of the atmosphere and the Planck source functions
      !    from pressures, temperatures, and gas concentrations...
      !

    call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        atmos,      &
                                        toa_flux))

      ! Save RRTMGP inputs and outputs for NN training? 
      if (save_inputs_outputs)  then
        ! Compute NN inputs: this is just a 3D array (ninputs,nlay,ncol),
        ! where the inner dimension vector consists of (tlay, play, vmr_h2o, vmr_o3, vmr_co2...)
        ! if last argument (preprocess) is true, inputs are scaled to 0...1, and 
        ! play, H2O and O3 are additionally power-scaled, like in RRTMGP-NN
        ! For convenience the NN inputs are computed here using a procedure (ensures that the outputs
        ! correspond to inputs, and allows the user to specify which gases are used by changing _gas_names)
        call stop_on_err(get_gasopt_nn_inputs(                  &
                      block_size, nlay, ninputs,                        &
                      p_lay(:,:,b), t_lay(:,:,b), gas_conc_array(b),      &
                      nn_gasopt_input(:,:,:,b), gasopt_input_names, preprocess_rrtmgp_inputs))
        if (save_rrtmgp) then
          ! column dry amount, needed to normalize outputs (could also be computed within Python)
          call stop_on_err(gas_conc_array(b)%get_vmr('h2o', vmr_h2o(:,:,b)))
          call get_col_dry(vmr_h2o(:,:,b), p_lev(:,:,b), col_dry(:,:,b))
          tau_sw(:,:,:,b)     = atmos%tau
          ssa_sw(:,:,:,b)     = atmos%ssa 
        end if
      end if

      ! print *, "mean tau after gas optics", mean_3d(atmos%tau)
    
      if (include_clouds) then
        call stop_on_err(clouds%delta_scale())
        call stop_on_err(clouds%increment(atmos))
      
        ! print *, "mean tau after adding clouds", mean_3d(atmos%tau)
        
        if (save_reftrans)  then
            g_sw(:,:,:,b)       = atmos%g 
            tau_sw(:,:,:,b)     = atmos%tau
            ssa_sw(:,:,:,b)     = atmos%ssa 
        end if
      end if

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


      if (save_reftrans) then
        call stop_on_err(rte_sw(atmos,   &
                                top_at_1,        &
                                mu0,             &
                                toa_flux,        &
                                sfc_alb_spec,  sfc_alb_spec,  &
                                fluxes, &
                                reftrans_vars=reftrans_variables))
      else
        call stop_on_err(rte_sw(atmos,   &
                                top_at_1,        &
                                mu0,             &
                                toa_flux,        &
                                sfc_alb_spec, sfc_alb_spec,  &
                                fluxes))
      end if
      if (save_inputs_outputs) then
        ! Save TOA flux, mu0 and sfc_alb
        mu0_save(:,b) = mu0
        toa_flux_save(:,:,b) = toa_flux
        sfc_alb_spec_save(:,:,b) = sfc_alb_spec 
        if (save_reftrans) then
          Rdif_save(:,:,:,b) = reftrans_variables(:,:,:,1)
          Tdif_save(:,:,:,b) = reftrans_variables(:,:,:,2)
          Rdir_save(:,:,:,b) = reftrans_variables(:,:,:,3)
          Tdir_save(:,:,:,b) = reftrans_variables(:,:,:,4)
        end if
      end if
            
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
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  !$acc exit data delete(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)
  !$acc exit data delete(sfc_alb_spec, mu0, toa_flux, def_tsi)
  call atmos%finalize() ! Also deallocates arrays on device
  if (allocated(reftrans_variables))  deallocate(reftrans_variables)

#ifdef USE_OPENACC  
  istat = cublasDestroy(h) 
#endif
  print *, "mean cloud optical depth for last column block", mean_3d(clouds%tau)

  print *, "Finished with computations!"
  print *, "-------------------------------------------------------------------------"

  ! Save inputs and outputs for neural network gas optics development?
  if(save_inputs_outputs) then 
    print *, "Attempting to save full RTE and RRTMGP input/output to ", nndev_file
    ! Create file
    call nndev_file_netcdf%create(trim(nndev_file),is_hdf5_file=.true.)

    ! Put global attributes
    if (include_clouds) then 
      cmt = "All-sky computation which includes gases and clouds"
    else
      cmt = "Clear-sky computation which only includes gases"
    end if
    call nndev_file_netcdf%put_global_attributes( &
         &   title_str="Input - output from computations with RTE+RRTMGP, can be used as training data for machine learning", &
         &   input_str=input_file, comment_str = trim(cmt))

    ! Define dimensions
    call nndev_file_netcdf%define_dimension("expt", nexp)
    call nndev_file_netcdf%define_dimension("site", ncol)
    call nndev_file_netcdf%define_dimension("layer", nlay)
    call nndev_file_netcdf%define_dimension("level", nlay+1)
    call nndev_file_netcdf%define_dimension("feature", ninputs)
    call nndev_file_netcdf%define_dimension("gpt", ngpt)
    call nndev_file_netcdf%define_dimension("bnd", nbnd)

    ! RTE inputs and outputs (broadband fluxes), always saved
    call nndev_file_netcdf%define_variable("rsu", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="upwelling shortwave flux")

    call nndev_file_netcdf%define_variable("rsd", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="downwelling shortwave flux")

    call nndev_file_netcdf%define_variable("rsd_dir", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="direct downwelling shortwave flux")

    call nndev_file_netcdf%define_variable("toa_flux", &
    &   dim3_name="expt", dim2_name="site", dim1_name="gpt", &
    &   long_name="top-of-atmosphere incoming flux")

    call nndev_file_netcdf%define_variable("sfc_alb", &
    &   dim3_name="expt", dim2_name="site", dim1_name="gpt", &
    &   long_name="surface albedo")

    call nndev_file_netcdf%define_variable("mu0", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="cosine of solar zenith angle")

    call nndev_file_netcdf%define_variable("pres_level", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="pressure at half-level")

    if (preprocess_rrtmgp_inputs) then
      cmt = "preprocessed inputs for RRTMGP shortwave gas optics"
    else 
      cmt = "inputs for RRTMGP shortwave gas optics"
    end if

    ! RRTMGP inputs
    nn_input_str = 'Features:'
    do b  = 1, size(gasopt_input_names)
      nn_input_str = trim(nn_input_str) // " " // trim(gasopt_input_names(b)) 
    end do
    call nndev_file_netcdf%define_variable("rrtmgp_sw_input", &
    &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="feature", &
    &   long_name =cmt, comment_str=nn_input_str, &
    &   data_type_name="float")

    if (save_rrtmgp) then
      call nndev_file_netcdf%define_variable("tau_sw_gas", &
      &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
      &   long_name="gas optical depth", data_type_name="float")
      call nndev_file_netcdf%define_variable("ssa_sw_gas", &
      &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
      &   long_name="gas single scattering albedo", data_type_name="float")
      call nndev_file_netcdf%define_variable("col_dry", &
      &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
      &   long_name="layer number of dry air molecules")
    end if

    if(include_clouds) then
      call nndev_file_netcdf%define_variable("cloud_lwp", &
      &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
      &   long_name="cloud liquid water path", units_str="g/kg")
      call nndev_file_netcdf%define_variable("cloud_iwp", &
      &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
      &   long_name="cloud ice water path", units_str="g/kg")
      call nndev_file_netcdf%define_variable("cloud_fraction", &
      &   dim3_name="expt", dim2_name="site", dim1_name="layer", &
      &   long_name="cloud fraction")
      if (save_reftrans) then
        call nndev_file_netcdf%define_variable("tau_sw", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="optical depth", data_type_name="float")
        call nndev_file_netcdf%define_variable("ssa_sw", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="single scattering albedo", data_type_name="float")
        call nndev_file_netcdf%define_variable("g_sw", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="asymmetry parameter", data_type_name="float")

        call nndev_file_netcdf%define_variable("rdif", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="diffuse reflectance", data_type_name="float")
        call nndev_file_netcdf%define_variable("tdif", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="diffuse transmittance", data_type_name="float")
        call nndev_file_netcdf%define_variable("rdir", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="direct reflectance", data_type_name="float")
        call nndev_file_netcdf%define_variable("tdir", &
        &   dim4_name="expt", dim3_name="site", dim2_name="layer", dim1_name="gpt", &
        &   long_name="direct transmittance", data_type_name="float")
      end if

    end if

    call nndev_file_netcdf%end_define_mode()

    call unblock_and_write(trim(nndev_file), 'rrtmgp_sw_input',nn_gasopt_input)
    deallocate(nn_gasopt_input)
    print *, "RRTMGP inputs (gas concs + T + p) were successfully saved"

    if (save_rrtmgp) then
      ! print *," min max col dry", minval(col_dry), maxval(col_dry)
      call unblock_and_write(trim(nndev_file), 'col_dry', col_dry)
      deallocate(col_dry)

      call unblock_and_write(trim(nndev_file), 'tau_sw_gas', tau_sw)
      call unblock_and_write(trim(nndev_file), 'ssa_sw_gas', ssa_sw)
      deallocate(tau_sw,ssa_sw)
      print *, "Optical properties (RRTMGP output) were successfully saved"
    end if
    if(include_clouds) then
      call unblock_and_write(trim(nndev_file), 'cloud_lwp', lwp)
      call unblock_and_write(trim(nndev_file), 'cloud_iwp', iwp)
      call unblock_and_write(trim(nndev_file), 'cloud_fraction', cloud_fraction)
      print *, "Cloud variables were successfully saved"
      deallocate(lwp, iwp, cloud_fraction)
    end if
    if (save_reftrans) then
      call unblock_and_write(trim(nndev_file), 'tau_sw', tau_sw)
      deallocate(tau_sw)
      call unblock_and_write(trim(nndev_file), 'ssa_sw', ssa_sw)
      deallocate(ssa_sw)
      call unblock_and_write(trim(nndev_file), 'g_sw',   g_sw)
      deallocate(g_sw)
      print *, "Optical properties (RRTMGP+cloud optics) were successfully saved"

      print *,  "minmax Rdif", minval(Rdif_save), maxval(Rdif_save), &
                "minmax Tdif", minval(Tdif_save), maxval(Tdif_save)
      print *,  "minmax Rdir", minval(Rdir_save), maxval(Rdir_save), &
                "minmax Tdir", minval(Tdir_save), maxval(Tdir_save)
      call unblock_and_write(trim(nndev_file), 'rdif', Rdif_save)
      deallocate(Rdif_save)
      call unblock_and_write(trim(nndev_file), 'tdif', Tdif_save)
      deallocate(Tdif_save)
      call unblock_and_write(trim(nndev_file), 'rdir', Rdir_save)
      deallocate(Rdir_save)
      call unblock_and_write(trim(nndev_file), 'tdir', Tdir_save)
      deallocate(Tdir_save)
      print *, "Reflectance-transmittance data succesfully saved"
    end if

    ! call nndev_file_netcdf%close()
    ! call nndev_file_netcdf%open(trim(nndev_file), redefine_existing=.true.,is_hdf5_file=.true.)
    ! call nndev_file_netcdf%end_define_mode()

    call unblock_and_write(trim(nndev_file), 'pres_level', p_lev)

    call unblock_and_write(trim(nndev_file), 'rsu', flux_up)
    call unblock_and_write(trim(nndev_file), 'rsd', flux_dn)
    call unblock_and_write(trim(nndev_file), 'rsd_dir', flux_dn_dir)

    call unblock_and_write(trim(nndev_file), 'toa_flux', toa_flux_save)
    call unblock_and_write(trim(nndev_file), 'sfc_alb', sfc_alb_spec_save)
    call unblock_and_write(trim(nndev_file), 'mu0', mu0_save)
    deallocate(toa_flux_save, sfc_alb_spec_save, mu0_save)

    if (do_gpt_flux) then

      call nndev_file_netcdf%close()
      call nndev_file_netcdf%open(trim(nndev_file), redefine_existing=.true.,is_hdf5_file=.true.)

      call nndev_file_netcdf%define_variable("rsu_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="gpt", &
      &   long_name="upwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_file_netcdf%define_variable("rsd_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="gpt", &
      &   long_name="downwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_file_netcdf%define_variable("rsd_dir_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="gpt", &
      &   long_name="direct downwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_file_netcdf%end_define_mode()

      call unblock_and_write(trim(nndev_file), 'rsu_gpt', gpt_flux_up)
      call unblock_and_write(trim(nndev_file), 'rsd_gpt', gpt_flux_dn)
      call unblock_and_write(trim(nndev_file), 'rsd_dir_gpt', gpt_flux_dn_dir)
      deallocate(gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir)
    end if 

    print *, "RTE outputs were successfully saved. All done!"

    call nndev_file_netcdf%close()

    print *, "-------------------------------------------------------------------------"

  end if

  !
  ! Zero out fluxes for which the original solar zenith angle is > 90 degrees.
  !
  do b = 1, nblocks
    do icol = 1, block_size
      if(.not. usecol(icol,b)) then
        flux_up(:,icol,b)  = 0._wp
        flux_dn(:,icol,b)  = 0._wp
      end if
    end do
  end do

  print *, "mean of flux_down is:", mean_3d(flux_dn)  ! mean of flux_down is:   292.71945410963957     
  print *, "mean of flux_up is:", mean_3d(flux_up)    ! mean of flux_up is:   41.835381782065106 
  !  if(do_gpt_flux) print *, "mean of gpt_flux_up for gpt=1 is:", mean_3d(gpt_flux_up(1,:,:,:))

  contains

  ! -------------------------------------------------------------------------------------------------
  ! Routine for preparing neural network inputs from the gas concentrations, temperature and pressure
  ! This routine, used for generating training data, differs from the compute_nn_inputs in gas_optics_rrtmgp
  ! because "operationally" the loaded NN model specifies which gases are used, and if a gas is missing
  ! from available gases (gas_desc) it needs to be set to zero or a reference concentration is used.
  ! Here we just use the available gases
  function get_gasopt_nn_inputs(ncol, nlay, ninputs, &
                              play, tlay, gas_desc,           &
                              nn_inputs, gasopt_input_names, preprocess) result(error_msg)

    integer,                                  intent(in   ) ::  ncol, nlay, ninputs
    real(wp), dimension(nlay,ncol),           intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                                tlay
    type(ty_gas_concs),                       intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  
    real(sp), dimension(ninputs, nlay, ncol),  intent(inout) ::  nn_inputs !
    character(len=32 ), dimension(ninputs),    intent(inout) ::  gasopt_input_names 
    logical,                                  intent(in)    :: preprocess
    character(len=128)                                  :: error_msg
    ! ----------------------------------------------------------
    ! Local variables
    integer :: igas, ilay, icol, ndims, idx_h2o, idx_o3, idx_gas, i
    character(len=32)                           :: gas_name    
    real(wp),       dimension(nlay,ncol)        :: vmr
    real(sp),       dimension(:), allocatable   :: xmin, xmax

    !  Neural network inputs are a vector consisting of temperature and pressure followed by gas concentrations
    ! These inputs are scaled to a range of (0-1), additionally some are power or log scaled: 
    ! The inputs are:   tlay,    log(play),   h2o**(1/4), o3**(1/4), co2, ..
    xmin = nn_input_minvals
    xmax = nn_input_maxvals

    ! First lets write temperature, pressure, water vapor and ozone into the inputs
    ! These are assumed to always be present!
    error_msg = gas_desc%get_conc_dims_and_igas('h2o', ndims, idx_h2o)
    error_msg = gas_desc%get_conc_dims_and_igas('o3',  ndims, idx_o3)
    if(error_msg  /= '') return

    gasopt_input_names(1) = 'tlay'

    if (.not. preprocess) then
      gasopt_input_names(2) = 'play'
      gasopt_input_names(3) = 'h2o'
      gasopt_input_names(4) = 'o3'
      do icol = 1, ncol
        do ilay = 1, nlay
            nn_inputs(1,ilay,icol)    =  tlay(ilay,icol)
            nn_inputs(2,ilay,icol)    =  play(ilay,icol)
            nn_inputs(3,ilay,icol)    =  gas_desc%concs(idx_h2o)%conc(ilay,icol)
            nn_inputs(4,ilay,icol)    =  gas_desc%concs(idx_o3) %conc(ilay,icol)
        end do
      end do
    else
      gasopt_input_names(2) = 'log(play)'
      gasopt_input_names(3) = 'h2o**(1/4)'
      gasopt_input_names(4) = 'o3**(1/4)'
      do icol = 1, ncol
        do ilay = 1, nlay
            nn_inputs(1,ilay,icol)    =  (tlay(ilay,icol)     - xmin(1)) / (xmax(1) - xmin(1))
            nn_inputs(2,ilay,icol)    = (log(play(ilay,icol)) - xmin(2)) / (xmax(2) - xmin(2))
            nn_inputs(3,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_h2o)%conc(ilay,icol))) - xmin(3)) / (xmax(3) - xmin(3))
            nn_inputs(4,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_o3) %conc(ilay,icol))) - xmin(4)) / (xmax(4) - xmin(4))
        end do
      end do
    end if

    ! Write the remaining gases
    ! The scaling coefficients are tied to a string specifying the gas names, these are all loaded from rrtmgp_constants.F90
    ! Lets find the indices which map the available gases to the scaling coefficients of each gas, 
    ! and also the dimensions of the concentration array
    i = 5
    do igas = 1, size(gas_desc%gas_name)
    
      gas_name = gas_desc%gas_name(igas)
      if(gas_name=='h2o' .or. gas_name=='o3' .or. gas_name=='o2' .or. gas_name=='n2') cycle

      ! Save gas name
      gasopt_input_names(i) = gas_name

      ! Fill 2D (lay,col) array with gas concentration
      error_msg = gas_desc%get_vmr(gas_name, vmr(:,:))

      ! Write to nn_input non-contiguously
      if (.not. preprocess) then
        nn_inputs(i,:,:) = vmr(:,:)
      else 
        ! which index in nn_input_name, which corresponds to the scaling coefficient arrays
        idx_gas = findloc(nn_gasopt_input_names,gas_name,dim=1)
        if (idx_gas == 0) then
          error_msg = 'get_gasopt_nn_inputs: trying to write ' // trim(gas_name) // ' but name not found in nn_gasopt_input_names'
          return
        end if
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(i,ilay,icol)    =  (vmr(ilay,icol)  - xmin(idx_gas)) / (xmax(idx_gas) - xmin(idx_gas))
            end do
          end do
      end if
      i = i + 1
    end do
    
    ! do igas = 1, ninputs
    !   print '(A25,I2,A2,A8,F6.3,F6.3)', "Min,max of NN-input ", igas, " =", gasopt_input_names(igas), &
    !               minval(nn_inputs(igas,:,:)), maxval(nn_inputs(igas,:,:))
    ! end do

  end function get_gasopt_nn_inputs


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
