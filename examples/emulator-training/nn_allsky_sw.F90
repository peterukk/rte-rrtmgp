! This program is for generating training data for neural network emulators of RRTMGP and RTE,
! as well as demonstrating their use.
! Three general machine learning approaches are possible:
!    1) emulation of gas optics=RRTMGP only (as in the paper by Ukkonen et al. 2020);
!       mapping atmospheric conditions to gas optical properties
!    2) emulation of radiative solver=RTE; mapping optical properties to fluxes
!    3) emulation of RTE+RRTMGP;. mapping atmospheric conditions to fluxes
! 
! Since we are interested in the trade-off of accuracy and speedup of these methods for realistic use cases, 
! clouds will be included when generating training data for 2-3, and in the evaluation of 1-3.
! What this means is that the NN in method 3) includes the effect of clouds, but in 1) and 2) cloud optical 
! properties are added as a separate step and computed from a description of clouds by the (relatively cheap) 
! cloud optics extension, i.e. not emulated by NNs. 
! 
! The idea is that this program can by called by a Python program to
!  -- generate training data for 1), 2), or 3)
!  -- evaluate 1), 2), or 3) by loading NNs and replacing the appropriate computations by NN predictions.
! 
! The evaluation data comes from CAMS which has been extended into RFMIP-style "experiments" where gas
! concentrations are varied. The large problem is divided into blocks
!
! the software will support the following methods for radiation computations:
!
!                     emulate       cloud   NN output (shape)		        NN iterations   NN models
!                                   optics                              
! RTE + RRTMGP        none          orig.		-						                                0
! (RTE+RRTMGP)-NN			both			    NN		  bb fluxes (3*1)			        nlay*ncol       1
! RTE + RRTMGP-NN			rrtmgp	      orig.		opt. prop. gpt vector (ng) 	~2*nlay*ncol    2 (abs,scat)
! RTE-NN1 + RRTMGP		rte			      orig.		bb fluxes (3*1)	            nlay*ncol       1
!(RTE-NN2 + RRTMGP		rte-gptvec		orig.		gpt flux vectors (3*ng)	    nlay*ncol       1)
! RTE-NN3 + RRTMGP		rte-gptscal		orig.	  gpt flux scalars (3*1)			nlay*ncol*ng	  1
! RTE-NN4 + RRTMGP		rte-reftrans  orig.	  gpt REFTRANS scalars (4*1)	nlay*ncol*ng    1
! 
! bb = broadband, gpt = g-point, ng = number of g-points (e.g. 224), 
!
! Fortran program arguments (k_distribution file and cloud_optics file are fixed):
! nn_allsky_sw [block_size] [input file] [emulate] [NN model files] [optional file to save NN inputs/outputs]"
!
! Developed by Peter Ukkonen
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
  ! Cloud optics extension
  use mo_cloud_optics,       only: ty_cloud_optics
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
  use mo_rfmip_io,           only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, &
                                   read_and_block_sw_bc, determine_gas_names                             
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
  character(len=132) :: input_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc', &
                        kdist_file = '../../rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc', &
                        cloud_optics_file='../..//extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc'
  character(len=132) :: flx_file, flx_file_ref, timing_file, nndev_inout_file=''
  integer            :: nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size, forcing_index
  logical 		       :: top_at_1, do_scattering
  integer            :: b, icol, ilay,ibnd, igpt, igas, ncid, ngas, ninputs, num_gases, ret, i, istat
  character(len=4)   :: block_size_char, forcing_index_char = '1'
  character(len=6)   :: nn_emulated_code
  character(len=32 ), &
            dimension(:),             allocatable :: kdist_gas_names, rfmip_gas_names
  ! Neural network variables  
  type(network_type), dimension(2)    :: neural_nets ! First model for predicting tau, second for tau_rayleigh
  character (len = 80)                :: nn_modelfile_1, nn_modelfile_2
  real(sp), dimension(:,:,:,:),         allocatable :: nn_input ! (nfeatures,nlay,block_size,nblocks)
  ! Output fluxes
  real(wp), dimension(:,:,:),   target, allocatable :: flux_up, flux_dn, flux_dn_dir
  real(wp), dimension(:,:,:,:), target, allocatable :: gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir
  ! Thermodynamic and other variables
  real(wp), dimension(:,:,:),           allocatable :: p_lay, p_lev, t_lay, t_lev ! nlay, block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
                                                     ! block_size, nblocks
  real(wp), dimension(:,:  ),           allocatable :: sfc_alb_spec ! nbnd, block_size; spectrally-resolved surface albedo
  ! RRTMGP outputs (absorption and Rayleigh optical depths) for NN development
  real(wp), dimension(:,:,:,:),         allocatable :: tau_sw, tau_sw_ray
  real(wp), dimension(:,:,:),           allocatable :: col_dry, vmr_h2o
  ! RTE inputs for NN development
  real(wp),                             allocatable :: toa_flux_save(:,:,:), sfc_alb_spec_save(:,:,:), mu0_save(:,:)
  !
  ! Cloud variables
  !
  real(wp), allocatable, dimension(:,:) :: lwp, iwp, rel, rei
  logical,  allocatable, dimension(:,:) :: cloud_mask
  !
  ! various logical to control program
  logical 		                        :: use_rrtmgp_nn=.false., use_rte_nn=.false., use_rtegpt_nn=.false., use_reftrans_nn = .false., use_rte_rrtmgp_nn =.false.
  logical 		                        :: do_gpt_flux=.false., compare_flux=.false., save_flux=.false., save_all_input_output=.false.
  !
  ! Classes used by rte+rrtmgp
  !
  type(ty_gas_optics_rrtmgp)                    :: k_dist
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
  real(wp) :: def_tsi_s
  ! ecRAD type netCDF IO
  type(netcdf_file)                      :: nndev_inout_netcdf

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

  ! Save fluxes
  save_flux    = .false.
  ! Compute fluxes per g-point?
  do_gpt_flux = .false.

  print *, "Usage: nn_allsky_sw [block_size] [input file]                                         (2 args: reference code) "
  print *, "OR   : nn_allsky_sw [block_size] [input file] [input/output file for NN development]  (3 args: ref, save input and output)"
  print *, "OR   : nn_allsky_sw [block_size] [input file] [emulated component] [NN model file(s)] (4-5 args, replace a component with NN)"

  nargs = command_argument_count()
  if (nargs <  2) call stop_on_err("Need to supply at least block_size input_file")
  if (nargs == 3) then
    call get_command_argument(3, nndev_inout_file)
    save_all_input_output   = .true.
    ninputs = 7
  end if

  call get_command_argument(1, block_size_char)
  read(block_size_char, '(i4)') block_size

  call get_command_argument(2, input_file)

  if(nargs > 3) then
    call get_command_argument(3, nn_emulated_code) 
    if (nn_emulated_code == 'rrtmgp') then
      use_rrtmgp_nn      = .true.
      if (nargs < 5) call stop_on_err("Need to supply two model files to emulate RRTMGP (absorption and Rayleigh cross-section models, respectively")
    else if (nn_emulated_code == 'both') then
      use_rte_rrtmgp_nn  = .true.
    else if (nn_emulated_code == 'rte') then
      use_rte_nn         = .true.
    ! else if (nn_emulated_code == 'rte-gptvec') then
    !   use_rte_nn         = .true.
    else if (nn_emulated_code == 'rte-gptscal') then
      use_rtegpt_nn         = .true.
    else if (nn_emulated_code == 'rte-reftrans') then
      use_reftrans_nn         = .true.
    else
      call stop_on_err("If nargs>3, third argument specifies which code to replace with NNs and must be one of the following: & 
      &'both', 'rrtmgp', 'rte',' 'rte-gptvec', 'rte-gptscal', 'rte-reftrans'")
    end if
  end if

  if(nargs >= 4) call get_command_argument(4, nn_modelfile_1)
  if(nargs >= 5) call get_command_argument(5, nn_modelfile_2)

  ! if (len_trim(nndev_inout_file) > 0) then
  !   save_all_input_output   = .true.
  ! end if

  ! Neural network models
  ! nn_modelfile_1           = "../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.txt" 
  ! nn_modelfile_2           = "../../neural/data/BEST_tau-sw-ray-7-16-16_2.txt" 
  if (use_rrtmgp_nn) then
    print *, "-------- Loading gas optics neural networks ---------"
	  print *, 'loading shortwave absorption model from ', nn_modelfile_1
    call neural_nets(1) % load(nn_modelfile_1)
    print *, 'loading rayleigh model from ', nn_modelfile_2
    call neural_nets(2) % load(nn_modelfile_2)
    ninputs = size(neural_nets(1) % layers(1) % w_transposed, 2)
    print *, "Number of NN inputs: ", ninputs
  end if  
  ! Note: The coefficients for scaling the inputs and outputs are currently hard-coded in mo_gas_optics_rrtmgp.F90

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

  if (use_rrtmgp_nn) then
    flx_file = 'output_fluxes/rsud_RTE-RRTMGP-NN.nc'
  else
    flx_file = 'output_fluxes/rsud_RTE-RRTMGP.nc'
  end if
  !
  ! Identify the set of gases used in the calculation based on the forcing index
  !   A gas might have a different name in the k-distribution than in the files
  !   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  !
  ! ALL SHORTWAVE GASES
  num_gases = 8
  allocate(kdist_gas_names(num_gases), rfmip_gas_names(num_gases))            
  kdist_gas_names = ["h2o  ","co2  ","ch4  ","o2   ","o3   ", "n2o  ", "n2   ","no2  "]
  rfmip_gas_names =  ['water_vapor   ', &
                    'carbon_dioxide', &
                    'methane       ', &
                    'oxygen        ', &
                    'ozone         ', &
                    'nitrous_oxide ', &
                    'nitrogen      ', &
                    'no2           ']   
  ! call determine_gas_names(input_file, kdist_file, 5, kdist_gas_names, rfmip_gas_names)
  print *, "Calculation uses RFMIP gases: ", (trim(rfmip_gas_names(b)) // " ", b = 1, size(rfmip_gas_names))
  ! print *, "Calculation uses K-DIST gases: ", (trim(kdist_gas_names(b)) // " ", b = 1, size(kdist_gas_names))
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
  call read_and_block_gases_ty(input_file, block_size, kdist_gas_names, rfmip_gas_names, gas_conc_array)
  ! do b = 1, size(gas_conc_array(1)%concs)
  !   print *, "max of gas ", gas_conc_array(1)%gas_name(b), ":", maxval(gas_conc_array(1)%concs(b)%conc)
  ! end do

  call read_and_block_sw_bc(input_file, block_size, surface_albedo, total_solar_irradiance, solar_zenith_angle)
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
  !$acc enter data create (toa_flux, def_tsi)

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
  !$acc enter data create (sfc_alb_spec, mu0) 

  !$acc enter data copyin(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)

  ! Allocate derived types - optical properties of both gaseous atmosphere and clouds
  call stop_on_err(atmos%alloc_2str(block_size, nlay, k_dist))
  call stop_on_err(clouds%alloc_2str(block_size, nlay, k_dist))
  ! Device allocation happens inside procedures

  if (save_all_input_output) then
    allocate(nn_input( 	ninputs, nlay, block_size, nblocks)) ! temperature + pressure + gases
    ! number of dry air molecules
    allocate(col_dry(nlay, block_size, nblocks), vmr_h2o(nlay, block_size, nblocks)) 
    allocate(tau_sw(    	ngpt, nlay, block_size, nblocks))
    allocate(tau_sw_ray(  ngpt, nlay, block_size, nblocks))
  end if

  if (save_all_input_output) then
    allocate(toa_flux_save(k_dist%get_ngpt(), block_size, nblocks))
    allocate(mu0_save(block_size,nblocks), sfc_alb_spec_save(ngpt,block_size,nblocks))
  end if

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

  ! --------------------------------------------------

  if (use_rrtmgp_nn) then
    print *, "starting clear-sky shortwave computations, using neural networks as RRTMGP kernel"
  else
    print *, "starting clear-sky shortwave computations, using lookup-table as RRTMGP kernel"
  end if

  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
  ret =  gptlstart('clear_sky_total (SW)')
#endif
#ifdef USE_OPENMP
  !$OMP PARALLEL shared(neural_nets, k_dist) firstprivate(def_tsi,toa_flux,sfc_alb_spec,mu0,fluxes,atmos)
  !$OMP DO 
#endif
  do b = 1, nblocks

    fluxes%flux_up => flux_up(:,:,b)
    fluxes%flux_dn => flux_dn(:,:,b)
    fluxes%flux_dn_dir => flux_dn_dir(:,:,b)
    if (do_gpt_flux) then
      fluxes%gpt_flux_up => gpt_flux_up(:,:,:,b)
      fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:,b)
      fluxes%gpt_flux_dn_dir => gpt_flux_dn_dir(:,:,:,b)
    end if
    
    !
    ! Compute the optical properties of clouds
    !
! #ifdef USE_TIMING
!     ret =  gptlstart('cloud_optics')
! #endif
!     call stop_on_err(                                      &
!       cloud_optics%cloud_optics(lwp, iwp, rel, rei, clouds))
! #ifdef USE_TIMING
!     ret =  gptlstop('cloud_optics')
! #endif

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
                                        toa_flux, neural_nets=neural_nets))
    else
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b), &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        gas_conc_array(b),  &
                                        atmos,      &
                                        toa_flux))
    end if
    if (save_all_input_output) then
        call stop_on_err(compute_nn_inputs(                  &
                      block_size, nlay, ninputs,  &
                      p_lay(:,:,b), t_lay(:,:,b), gas_conc_array(b),     &
                      nn_input(:,:,:,b)))
        ! column dry amount
        call stop_on_err(gas_conc_array(b)%get_vmr('h2o', vmr_h2o(:,:,b)))
        call get_col_dry(vmr_h2o(:,:,b), p_lev(:,:,b), col_dry(:,:,b))
        tau_sw(:,:,:,b)      = atmos%tau
        tau_sw_ray(:,:,:,b)  = (atmos%ssa * atmos%tau)
    end if

    ! !$acc update host(atmos%tau, atmos%ssa, atmos%g)
    ! print *, "mean tau after gas optics", mean_3d(atmos%tau)
#ifdef USE_TIMING
    ret =  gptlstop('gas_optics_sw')
    ret =  gptlstart('clouds_deltascale_increment')
#endif       
      ! call stop_on_err(clouds%delta_scale())
      ! call stop_on_err(clouds%increment(atmos))
    ! !$acc update host(atmos%tau, atmos%ssa, atmos%g)
    ! print *, "mean tau after adding cloud optics", mean_3d(atmos%tau)
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
    end do
    !$acc end parallel

    !
    ! ... and compute the spectrally-resolved fluxes, providing reduced values
    !    via ty_fluxes_broadband
    !
#ifdef USE_TIMING
    ret =  gptlstart('rte_sw')
#endif
    
    call stop_on_err(rte_sw(atmos,   &
                            top_at_1,        &
                            mu0,             &
                            toa_flux,        &
                            sfc_alb_spec,    &
                            sfc_alb_spec,    &
                            fluxes))
    if (save_all_input_output) then
      ! Save TOA flux, mu0 and sfc_alb
      mu0_save(:,b) = mu0
      toa_flux_save(:,:,b) = toa_flux
      sfc_alb_spec_save(:,:,b) = sfc_alb_spec 
    end if
            
#ifdef USE_TIMING
    ret =  gptlstop('rte_sw')
#endif

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
  ret =  gptlstop('clear_sky_total (SW)')
  timing_file = "timing.sw-" // adjustl(trim(block_size_char))
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif

  !$acc exit data delete(total_solar_irradiance, surface_albedo, usecol, solar_zenith_angle)
  !$acc exit data delete(sfc_alb_spec, mu0)
  !$acc exit data delete(toa_flux, def_tsi)
  call atmos%finalize() ! Also deallocates arrays on device

#ifdef USE_OPENACC  
  istat = cublasDestroy(h) 
#endif

  ! Save inputs and outputs for neural network gas optics development?
  if(save_all_input_output) then 
    print *, "Attempting to save full RTE and RRTMGP input/output to ", nndev_inout_file
    ! Create file
    call nndev_inout_netcdf%create(trim(nndev_inout_file))

    ! Define dimensions
    call nndev_inout_netcdf%define_dimension("expt", nexp)
    call nndev_inout_netcdf%define_dimension("site", ncol)
    call nndev_inout_netcdf%define_dimension("layer", nlay)
    call nndev_inout_netcdf%define_dimension("level", nlay+1)
    call nndev_inout_netcdf%define_dimension("feature", ninputs)
    call nndev_inout_netcdf%define_dimension("ngpt", ngpt)

    call nndev_inout_netcdf%define_variable("nn_input", &
    &   dim4_name="expt", dim3_name="site", &
    &   dim2_name="layer", dim1_name="feature", &
    &   long_name="RRTMGP-NN input", &
    &   data_type_name="float")

    call nndev_inout_netcdf%define_variable("tau_sw", &
    &   dim4_name="expt", dim3_name="site", &
    &   dim2_name="layer", dim1_name="ngpt", &
    &   long_name="shortwave absorption optical depth", &
    &   data_type_name="float")

    call nndev_inout_netcdf%define_variable("tau_sw_ray", &
    &   dim4_name="expt", dim3_name="site", &
    &   dim2_name="layer", dim1_name="ngpt", &
    &   long_name="shortwave Rayleigh optical depth", &
    &   data_type_name="float")

    call nndev_inout_netcdf%define_variable("col_dry", &
    &   dim3_name="expt", dim2_name="site", &
    &   dim1_name="layer", &
    &   long_name="layer number of dry air molecules")

    call nndev_inout_netcdf%end_define_mode()

    ! This function also deallocates its input 
    call unblock_and_write(trim(nndev_inout_file), 'nn_input',nn_input)
    ! print *," min max col dry", minval(col_dry), maxval(col_dry)
    call unblock_and_write(trim(nndev_inout_file), 'col_dry', col_dry)
    print *, "RRTMGP inputs were successfully saved"
    print *, "shape tau_sw, ray", shape(tau_sw_ray)
    call unblock_and_write(trim(nndev_inout_file), 'tau_sw', tau_sw)
    call unblock_and_write(trim(nndev_inout_file), 'tau_sw_ray', tau_sw_ray)
    print *, "RRTMGP outputs were successfully saved"

    call nndev_inout_netcdf%close()

    call nndev_inout_netcdf%open(trim(nndev_inout_file), redefine_existing=.true.)

    call nndev_inout_netcdf%define_variable("rsu", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="upwelling shortwave flux")

    call nndev_inout_netcdf%define_variable("rsd", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="downwelling shortwave flux")

    call nndev_inout_netcdf%define_variable("rsd_dir", &
    &   dim3_name="expt", dim2_name="site", dim1_name="level", &
    &   long_name="direct downwelling shortwave flux")

    call nndev_inout_netcdf%define_variable("toa_flux", &
    &   dim3_name="expt", dim2_name="site", dim1_name="ngpt", &
    &   long_name="top-of-atmosphere incoming flux")

    call nndev_inout_netcdf%define_variable("sfc_alb", &
    &   dim3_name="expt", dim2_name="site", dim1_name="ngpt", &
    &   long_name="surface albedo")

    call nndev_inout_netcdf%define_variable("mu0", &
    &   dim2_name="expt", dim1_name="site", &
    &   long_name="cosine of solar zenith angle")

    call nndev_inout_netcdf%end_define_mode()

    call unblock_and_write(trim(nndev_inout_file), 'rsu', flux_up)
    call unblock_and_write(trim(nndev_inout_file), 'rsd', flux_dn)
    call unblock_and_write(trim(nndev_inout_file), 'rsd_dir', flux_dn_dir)

    call unblock_and_write(trim(nndev_inout_file), 'toa_flux', toa_flux_save)
    call unblock_and_write(trim(nndev_inout_file), 'sfc_alb', sfc_alb_spec_save)
    call unblock_and_write(trim(nndev_inout_file), 'mu0', mu0_save)

    if (do_gpt_flux) then

      call nndev_inout_netcdf%close()
      call nndev_inout_netcdf%open(trim(nndev_inout_file), redefine_existing=.true.)

      call nndev_inout_netcdf%define_variable("rsu_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="ngpt", &
      &   long_name="upwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_inout_netcdf%define_variable("rsd_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="ngpt", &
      &   long_name="downwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_inout_netcdf%define_variable("rsd_dir_gpt", &
      &   dim4_name="expt", dim3_name="site", &
      &   dim2_name="level", dim1_name="ngpt", &
      &   long_name="direct downwelling shortwave flux by g-point", &
      &   data_type_name="float")

      call nndev_inout_netcdf%end_define_mode()

      call unblock_and_write(trim(nndev_inout_file), 'rsu_gpt', gpt_flux_up)
      call unblock_and_write(trim(nndev_inout_file), 'rsd_gpt', gpt_flux_dn)
      call unblock_and_write(trim(nndev_inout_file), 'rsd_dir_gpt', gpt_flux_dn_dir)

    end if 

    print *, "RTE outputs were successfully saved"
    call nndev_inout_netcdf%close()

    print *, "-----------------------------------------------------------------------------------------"

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

  ! Save fluxes ?
  if (save_flux) then
    print *, "Attempting to save fluxes to ", flx_file
    call unblock_and_write(trim(flx_file), 'rsu', flux_up)
    call unblock_and_write(trim(flx_file), 'rsd', flux_dn)
    print *, "Fluxes saved to ", flx_file
  end if 

  deallocate(flux_up, flux_dn)
  print *, "SUCCESS!"

  contains

  ! -------------------------------------------------------------------------------------------------
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

  function mean_3d(x3) result(mean3)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x3
    real(wp) :: mean3
    
    mean3 = sum(sum(sum(x3, dim=1),dim=1),dim=1) / (size(x3))

  end function mean_3d

  function mean_2d(x2) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x2
    real(wp) :: mean2
    
    mean2 = sum(sum(x2, dim=1),dim=1) / (size(x2))

  end function mean_2d


end program rrtmgp_rfmip_sw
