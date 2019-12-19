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
! Program is invoked as rrtmgp_rfmip_lw [block_size input_file  coefficient_file upflux_file downflux_file]
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
#ifdef USE_TIMING
  use omp_lib
#endif
  ! Working precision for real variables
  !
  use mo_rte_kind,           only: wp, sp
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
                                   unblock_and_write_3D, unblock_and_write_3D_notrans, unblock_and_write_3D_sp, &
                                   unblock_and_write_3D_notrans_sp, &
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
  ! --------------------------------------------------
  !
  ! Local variables
  !
  !character(len=132) :: rfmip_file = 'multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-1_none.nc', &
  !                      kdist_file = 'coefficients_lw.nc'
  character(len=132) :: rfmip_file,kdist_file
  character(len=132) :: flxdn_file, flxup_file, output_file, input_file, flx_file, flx_file_ref, flx_file_lbl
  integer            :: nargs, ncol, nlay, nbnd, ngas, ngpt, nexp, nblocks, block_size, forcing_index, physics_index, n_quad_angles = 1
  logical            :: top_at_1
  integer            :: b, icol, ibnd, igpt, count_rate, iTime1, iTime2, ncid
  character(len=4)   :: block_size_char, forcing_index_char = '1', physics_index_char = '1'
  
  character(len=32 ), &
            dimension(:),             allocatable :: kdist_gas_names, rfmip_gas_games
  real(wp), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev ! block_size, nlay, nblocks
  real(wp), dimension(:,:,:), target, allocatable :: flux_up, flux_dn
  real(wp), dimension(:,:,:),         allocatable :: rlu_ref, rld_ref, rlu_nn, rld_nn, rlu_lbl, rld_lbl
  real(sp), dimension(:,:,:,:),       allocatable :: tau_lw, planck_frac
  real(wp), dimension(:,:,:,:),       allocatable :: nn_inputs
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis, sfc_t  ! block_size, nblocks (emissivity is spectrally constant)
  real(wp), dimension(:,:  ),         allocatable :: sfc_emis_spec    ! nbands, block_size (spectrally-resolved emissivity)
  real(wp), dimension(:),             allocatable :: means,stdevs ,temparray

  character (len = 80)                :: modelfile_tau_tropo, modelfile_tau_strato, modelfile_source
  type(network_type)                  :: net_tau_tropo, net_tau_strato, net_pfrac
  type(network_type), dimension(2)    :: neural_nets

  logical 		:: use_nn, save_output, save_input, save_flux, compare_flux

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
#endif
  ! -------------------------------------------------------------------------------------------------
  !
  ! Code starts
  !   all arguments are optional
! call mkl_set_num_threads( 4 )

  !  ------------ I/O and settings -----------------
  ! Use neural networks for gas optics? 
  use_nn      = .true.
  ! Save outputs (tau, planck fracs) and inputs (scaled gases)
  save_input  = .false.
  save_output = .false.
  ! Save fluxes
  save_flux   = .true.
  ! Compare fluxes to original (RFMIP only)
  compare_flux = .true.

  if (compare_flux) save_flux = .true.
  

  ! Where neural network model weights are located (required!)

  modelfile_tau_tropo     = "../../neural/data/tau-lw-tropstrat-19-46-46-46-ynorm-pow8-3.txt"
  modelfile_tau_strato    = modelfile_tau_tropo
  modelfile_source        = "../../neural/data/pfrac-tropstrat-36-36-pow2.txt"

  ! Save upwelling and downwelling fluxes in the same file
  flx_file = 'rlud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_NN.nc'

  ! FOR NN MODEL DEVELOPMENT
  ! Where to save g-point optical depths and planck fractions
  output_file = '/data/puk/rrtmgp/outp_lw_RFMIP-BIGALL_1f1_REF.nc'
  ! output_file = '/data/puk/rrtmgp/outp_lw_NWPSAF_1f1_REF.nc'
  
  ! Where to save neural network inputs (scaled gases)
  !input_file =  '/data/puk/rrtmgp/inp2_lw_NWPSAF_1f1_NN.nc'
  input_file =  '/data/puk/rrtmgp/inp2_lw_RFMIP-BIGALL_1f1_NN.nc'

  ! The coefficients for scaling the INPUTS are currently still hard-coded in mo_gas_optics_rrtmgp.F90

  if (use_nn) then
	  print *, 'loading tau model from ', modelfile_tau_tropo
	  call net_tau_tropo % load(modelfile_tau_tropo)

	  ! print *, 'loading tau-strato model from ', modelfile_tau_strato
	  ! call net_tau_strato % load(modelfile_tau_strato)

	  print *, 'loading planck fraction model from ', modelfile_source
	  call net_pfrac % load(modelfile_source)

    neural_nets(1) = net_pfrac
    neural_nets(2) = net_tau_tropo
	  ! neural_nets(2) = net_tau_strato
	  ! neural_nets(3) = net_tau_tropo
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

  ! block_size = 450
  ! rfmip_file = "inputs_RFMIP.nc"
  ! kdist_file = "../../rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc"
  ! forcing_index_char = "1"
  ! physics_index_char = "1"
  !
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
  if(forcing_index < 1 .or. forcing_index > 3) &
    stop "Forcing index is invalid (must be 1,2 or 3)"

  read(physics_index_char, '(i4)') physics_index
  if(physics_index < 1 .or. physics_index > 2) &
    stop "Physics index is invalid (must be 1 or 2)"
  if(physics_index == 2) n_quad_angles = 3
                
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

  ! Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  !
  top_at_1 = p_lay(1, 1, 1) < p_lay(1, nlay, 1)

  !
  ! Read the gas concentrations and surface properties
  !

  call read_and_block_gases_ty(rfmip_file, block_size, kdist_gas_names, rfmip_gas_games, gas_conc_array)

  call read_and_block_lw_bc(rfmip_file, block_size, sfc_emis, sfc_t)

  !
  ! Read k-distribution information. load_and_init() reads data from netCDF and calls
  !   k_dist%init(); users might want to use their own reading methods
  !
  
  call load_and_init(k_dist, trim(kdist_file), gas_conc_array(1))
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
    p_lev(:,1,:) = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  else
    p_lev(:,nlay+1,:) &
                 = k_dist%get_press_min() + epsilon(k_dist%get_press_min())
  end if

  !
  ! Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  !   gas optical properties, and source functions. The %alloc() routines carry along
  !   the spectral discretization from the k-distribution.
  !

  allocate(flux_up(    	block_size, nlay+1, nblocks), &
           flux_dn(    	block_size, nlay+1, nblocks))
  allocate(sfc_emis_spec(nbnd, block_size))

  if (save_output) then 
    allocate(tau_lw(    	block_size, nlay, ngpt, nblocks))
    tau_lw = 0.0_sp
    allocate(planck_frac(	block_size, nlay, ngpt, nblocks))
    planck_frac = 0.0_sp
  end if
  allocate(nn_inputs( 	ngas, nlay, block_size, nblocks)) ! dry air + gases + temperature + pressure
  
  call stop_on_err(source%alloc            (block_size, nlay, k_dist))
  call stop_on_err(optical_props%alloc_1scl(block_size, nlay, k_dist))
  !
  ! OpenACC directives put data on the GPU where it can be reused with communication
  ! NOTE: these are causing problems right now, most likely due to a compiler
  ! bug related to the use of Fortran classes on the GPU.
  !
  !$acc enter data create(sfc_emis_spec)
  !$acc enter data create(optical_props, optical_props%tau)
  !$acc enter data create(source, source%lay_source, source%lev_source_inc, source%lev_source_dec, source%sfc_source)
  ! --------------------------------------------------
#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
  ret = gptlinitialize()
#endif
#ifdef OMP
print *, "OpenMP processes available:", omp_get_num_procs()
#endif
    call system_clock(count_rate=count_rate)
    call system_clock(iTime1)
  !
  ! Loop over blocks
  !
#ifdef USE_TIMING
  do i = 1, 4
#endif

!bo !$OMP PARALLEL DO
  do b = 1, nblocks
#ifdef OMP
    PRINT *, "Hello from process: ", OMP_GET_THREAD_NUM()
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
    ! print *, "starting computations"

    if (use_nn) then
      print *, "Using neural networks for predicting optical depths"

      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b),          &
                                          p_lev(:,:,b),         &
                                          t_lay(:,:,b),         &
                                          sfc_t(:  ,b),         &
                                          gas_conc_array(b),    &
                                          optical_props,        &
                                          source,               &
                                          nn_inputs(:,:,:,b),   &
                                          neural_nets,          & !net_pfrac, net_tau
                                          tlev = t_lev(:,:,b)))
    else 
      print *, "Using original code (interpolation routine) for predicting optical depths"
      call stop_on_err(k_dist%gas_optics(p_lay(:,:,b),      &
                                        p_lev(:,:,b),       &
                                        t_lay(:,:,b),       &
                                        sfc_t(:  ,b),       &
                                        gas_conc_array(b),  &
                                        optical_props,      &
                                        source,             &
                                        tlev = t_lev(:,:,b)))
    end if

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
    call stop_on_err(rte_lw(optical_props,   &
                            top_at_1,        &
                            source,          &
                            sfc_emis_spec,   &
                            fluxes, n_gauss_angles = n_quad_angles))
#ifdef USE_TIMING
    ret =  gptlstop('rte_lw')
#endif
    ! Save optical depths
    if (save_output) then
      do igpt = 1, ngpt
        tau_lw(:,:,igpt,b)      = optical_props%tau(:,:,igpt)
        planck_frac(:,:,igpt,b) = source%planck_frac(:,:,igpt)
      end do
    end if
  end do ! blocks

  deallocate(sfc_emis_spec, gas_conc_array, optical_props%tau, & 
  source%planck_frac, source%lay_source, source%lev_source_inc, source%lev_source_dec, source%sfc_source)

!bo !$OMP END PARALLEL DO
#ifdef USE_TIMING
  end do
  !
  ! End timers
  !
  ret = gptlpr(block_size)
  ret = gptlfinalize()
#endif
  !$acc exit data delete(sfc_emis_spec)
  !$acc exit data delete(optical_props%tau, optical_props)
  !$acc exit data delete(source%lay_source, source%lev_source_inc, source%lev_source_dec, source%sfc_source)
  !$acc exit data delete(source)
  ! --------------------------------------------------m
  call system_clock(iTime2)
  print *,'Elapsed time on everything ',real(iTime2-iTime1)/real(count_rate)
 !  mean of flux_down is:   103.2458

  allocate(temparray(   block_size*nlay*nblocks)) 
  temparray = pack(flux_dn(:,:,:),.true.)
  print *, "mean of flux_down is:", sum(temparray, dim=1)/size(temparray, dim=1)
  deallocate(temparray)

  if (save_output) then 
    allocate(temparray(   block_size*nlay*ngpt*nblocks)) 
    temparray = pack(tau_lw(:,:,:,:),.true.)
    print *, "mean of tau is", sum(temparray, dim=1)/size(temparray, dim=1)
    print *, "max of tau is", maxval(tau_lw)
    print *, "min of tau is", minval(tau_lw)
    deallocate(temparray)
      ! Save optical depths and planck fractions
    print *, "Attempting to save outputs..."
    call unblock_and_write_3D_sp(trim(output_file), 'tau_lw',tau_lw)
    call unblock_and_write_3D_sp(trim(output_file), 'planck_frac',planck_frac)
    print *, "outputs saved to", output_file
  end if

    ! Save neural network inputs
  if (save_input) then
    print *, "Attempting to save neural network inputs..."
    call unblock_and_write_3D_notrans(trim(input_file), 'col_gas',nn_inputs)
    deallocate(nn_inputs)
    print *, "inputs saved to", input_file
  end if

  ! Save fluxes
  if (save_flux) then
    call unblock_and_write(trim(flx_file), 'rlu', flux_up)
    call unblock_and_write(trim(flx_file), 'rld', flux_dn)
  end if 

  deallocate(flux_up, flux_dn)

  print *, "success"

  if (compare_flux) then

    print *, "comparing fluxes to original scheme:"

    allocate(rld_ref( nlay+1, ncol, nexp))
    allocate(rlu_ref( nlay+1, ncol, nexp))
    allocate(rld_nn( nlay+1, ncol, nexp))
    allocate(rlu_nn( nlay+1, ncol, nexp))
    allocate(rld_lbl( nlay+1, ncol, nexp))
    allocate(rlu_lbl( nlay+1, ncol, nexp))

    flx_file_ref = 'rlud_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p1f1_gn.nc'
    flx_file_lbl = 'rlud_Efx_LBLRTM-12-8_rad-irf_r1i1p1f1_gn.nc'

    if(nf90_open(trim(flx_file), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file))

    rlu_nn = read_field(ncid, "rlu", nlay+1, ncol, nexp)
    rld_nn = read_field(ncid, "rld", nlay+1, ncol, nexp)

    if(nf90_open(trim(flx_file_ref), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_ref))

    rlu_ref = read_field(ncid, "rlu", nlay+1, ncol, nexp)
    rld_ref = read_field(ncid, "rld", nlay+1, ncol, nexp)

    if(nf90_open(trim(flx_file_lbl), NF90_NOWRITE, ncid) /= NF90_NOERR) &
    call stop_on_err("read_and_block_gases_ty: can't find file " // trim(flx_file_lbl))

    rlu_lbl = read_field(ncid, "rlu", nlay+1, ncol, nexp)
    rld_lbl = read_field(ncid, "rld", nlay+1, ncol, nexp)

    print *, "---- UPWELLING ----"

    print *, "MAE in upwelling fluxes of NN w.r.t LBL, present-day:", &
     mae(reshape(rlu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in upwelling fluxes of NN w.r.t LBL, future:     ", &
     mae(reshape(rlu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in upwelling fluxes of RRTMGP w.r.t LBL, present-day:", &
     mae(reshape(rlu_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in upwelling fluxes of RRTMGP w.r.t LBL, future:     ", &
     mae(reshape(rlu_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rlu_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "Max-vertical-error in upwelling fluxes of NN w.r.t LBL, present-day:", &
      maxval(rlu_lbl(:,:,1)-rlu_nn(:,:,1))

    print *, "Max-vertical-error in upwelling fluxes of NN w.r.t LBL, future:     ", &
      maxval(rlu_lbl(:,:,4)-rlu_nn(:,:,4))

    print *, "Max-vertical-error in upwelling fluxes of RRTMGP w.r.t LBL, present-day:", &
      maxval(rlu_lbl(:,:,1)-rlu_ref(:,:,1))

    print *, "Max-vertical-error in upwelling fluxes of RRTMGP w.r.t LBL, future:     ", &
      maxval(rlu_lbl(:,:,4)-rlu_ref(:,:,4))

    print *, "---- DOWNWELLING ----"

    print *, "MAE in downwelling fluxes of NN w.r.t LBL, present-day:", &
     mae(reshape(rld_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN w.r.t LBL, future:     ", &
     mae(reshape(rld_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of RRTMGP w.r.t LBL, present-day:", &
     mae(reshape(rld_lbl(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_ref(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of RRTMGP w.r.t LBL, future:     ", &
     mae(reshape(rld_lbl(:,:,4), shape = [1*ncol*(nlay+1)]), reshape(rld_ref(:,:,4), shape = [1*ncol*(nlay+1)]))

    print *, "Max-vertical-error in downwelling fluxes of NN w.r.t LBL, present-day:", &
      maxval(rld_lbl(:,:,1)-rld_nn(:,:,1))

    print *, "Max-vertical-error in downwelling fluxes of NN w.r.t LBL, future:     ", &
      maxval(rld_lbl(:,:,4)-rld_nn(:,:,4))

    print *, "Max-vertical-error in downwelling fluxes of RRTMGP w.r.t LBL, present-day:", &
      maxval(rld_lbl(:,:,1)-rld_ref(:,:,1))

    print *, "Max-vertical-error in downwelling fluxes of RRTMGP w.r.t LBL, future:     ", &
      maxval(rld_lbl(:,:,4)-rld_ref(:,:,4))

    print *, "---------"


    print *, "MAE in upwelling fluxes of NN w.r.t RRTMGP, present-day:", &
     mae(reshape(rlu_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rlu_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    print *, "MAE in downwelling fluxes of NN w.r.t RRTMGP, present-day:", &
     mae(reshape(rld_ref(:,:,1), shape = [1*ncol*(nlay+1)]), reshape(rld_nn(:,:,1), shape = [1*ncol*(nlay+1)]))

    deallocate(rld_ref,rlu_ref,rld_nn,rlu_nn,rld_lbl,rlu_lbl)

  end if



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

end program rrtmgp_rfmip_lw

