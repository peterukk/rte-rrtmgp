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
! Class for computing spectrally-resolved gas optical properties and source functions
!   given atmopsheric physical properties (profiles of temperature, pressure, and gas concentrations)
!   The class must be initialized with data (provided as a netCDF file) before being used.
!
! Two variants apply to internal Planck sources (longwave radiation in the Earth's atmosphere) and to
!   external stellar radiation (shortwave radiation in the Earth's atmosphere).
!   The variant is chosen based on what information is supplied during initialization.
!   (It might make more sense to define two sub-classes)
!
! -------------------------------------------------------------------------------------------------
module mo_gas_optics_rrtmgp
  use mo_rte_kind,           only: wp, wl, dp, sp
  use mo_rte_rrtmgp_config,  only: check_extents, check_values, nn_scenario_index
  use mo_rte_util_array,     only: zero_array, any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props,      only: ty_optical_props
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_optics_kernels, only: interpolation,                                         &
                                   compute_tau_absorption, compute_tau_rayleigh,          &
                                   combine_2str, combine_nstr,                            &
                                   compute_Planck_source, compute_Planck_source_nn, &
                                   predict_nn_lw_blas, predict_nn_sw_blas
  use mo_rrtmgp_constants,   only: avogad, m_dry, m_h2o, grav
  use mo_rrtmgp_util_string, only: lower_case, string_in_array, string_loc_in_array
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_gas_ref_concentrations, only: get_ref_vmr
  use mo_optical_props,      only: ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_gas_optics,         only: ty_gas_optics
  use mo_rrtmgp_util_reorder
  use mod_network_rrtmgp
  use omp_lib

  use,intrinsic :: ISO_Fortran_env
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif
  implicit none
  private
  real(wp), parameter :: pi = acos(-1._wp)

#ifdef USE_TIMING
  integer :: ret, i
#endif

  ! -------------------------------------------------------------------------------------------------
  type, extends(ty_gas_optics), public :: ty_gas_optics_rrtmgp
    private
    !
    ! RRTMGP computes absorption in each band arising from
    !   two major species in each band, which are combined to make
    !     a relative mixing ratio eta and a total column amount (col_mix)
    !   contributions from zero or more minor species whose concentrations
    !     may be scaled by other components of the atmosphere
    !
    ! Absorption coefficients are interpolated from tables on a pressure/temperature/(eta) grid
    !
    ! ------------------------------------
    ! Interpolation variables: Temperature and pressure grids
    !
    real(wp),      dimension(:),     allocatable :: press_ref,  press_ref_log, temp_ref
    !
    ! Derived and stored for convenience:
    !   Min and max for temperature and pressure intepolation grids
    !   difference in ln pressure between consecutive reference levels
    !   log of reference pressure separating the lower and upper atmosphere
    !
    real(wp) :: press_ref_min, press_ref_max, &
                temp_ref_min,  temp_ref_max
    real(wp) :: press_ref_log_delta, temp_ref_delta, press_ref_trop_log
    ! ------------------------------------
    ! Major absorbers ("key species")
    !   Each unique set of major species is called a flavor.
    !
    ! Names  and reference volume mixing ratios of major gases
    !
    character(32), dimension(:),  allocatable :: gas_names     ! gas names
    real(wp), dimension(:,:,:),   allocatable :: vmr_ref       ! vmr_ref(lower or upper atmosphere, gas, temp)
    !
    ! Which two gases are in each flavor? By index
    !
    integer,  dimension(:,:),     allocatable :: flavor        ! major species pair; (2,nflav)
    !
    ! Which flavor for each g-point? One each for lower, upper atmosphere
    !
    integer,  dimension(:,:),     allocatable :: gpoint_flavor ! flavor = gpoint_flavor(2, g-point)
    !
    ! Major gas absorption coefficients
    !
    real(wp), dimension(:,:,:,:), allocatable :: kmajor        !  kmajor(g-point,eta,pressure,temperature)
    !
    ! ------------------------------------
    ! Minor species, independently for upper and lower atmospheres
    !   Array extents in the n_minor dimension will differ between upper and lower atmospheres
    !   Each contribution has starting and ending g-points
    !
    integer, dimension(:,:), allocatable :: minor_limits_gpt_lower, &
                                            minor_limits_gpt_upper
    !
    ! Minor gas contributions might be scaled by other gas amounts; if so we need to know
    !   the total density and whether the contribution is scaled by the partner gas
    !   or its complement (i.e. all other gases)
    ! Water vapor self- and foreign continua work like this, as do
    !   all collision-induced abosption pairs
    !
    logical(wl), dimension(:), allocatable :: minor_scales_with_density_lower, &
                                              minor_scales_with_density_upper
    logical(wl), dimension(:), allocatable :: scale_by_complement_lower, scale_by_complement_upper
    integer,     dimension(:), allocatable :: idx_minor_lower,           idx_minor_upper
    integer,     dimension(:), allocatable :: idx_minor_scaling_lower,   idx_minor_scaling_upper
    !
    ! Index into table of absorption coefficients
    !
    integer, dimension(:), allocatable :: kminor_start_lower,        kminor_start_upper
    !
    ! The absorption coefficients themselves
    !
    real(wp), dimension(:,:,:), allocatable :: kminor_lower, kminor_upper ! kminor_lower(n_minor,eta,temperature)
    !
    ! -----------------------------------------------------------------------------------
    !
    ! Rayleigh scattering coefficients
    !
    real(wp), dimension(:,:,:,:), allocatable :: krayl ! krayl(g-point,eta,temperature,upper/lower atmosphere)
    !
    ! -----------------------------------------------------------------------------------
    ! Planck function spectral mapping
    !   Allocated only when gas optics object is internal-source
    !
    real(wp), dimension(:,:,:,:), allocatable :: planck_frac_stored   ! stored fraction of Planck irradiance in band for given g-point
                                                               ! planck_frac(g-point, eta, pressure, temperature)
    real(wp), dimension(:,:),     allocatable :: totplnk       ! integrated Planck irradiance by band; (Planck temperatures,band)
    real(wp)                                  :: totplnk_delta ! temperature steps in totplnk
    real(wp), dimension(:,:),     allocatable :: optimal_angle_fit ! coefficients of linear function
                                                                   ! of vertical path clear-sky transmittance that is used to
                                                                   ! determine the secant of single angle used for the
                                                                   ! no-scattering calculation,
                                                                   ! optimal_angle_fit(coefficient, band)
    ! -----------------------------------------------------------------------------------
    ! Solar source function spectral mapping with solar variability capability
    !   Allocated  when gas optics object is external-source
    !   n-solar-terms: quiet sun, facular brightening and sunspot dimming components
    !   following the NRLSSI2 model of Coddington et al. 2016, doi:10.1175/BAMS-D-14-00265.1.
    !
    real(wp), dimension(:), allocatable, public :: solar_source         ! incoming solar irradiance, computed from other three terms (g-point)
    real(wp), dimension(:), allocatable :: solar_source_quiet   ! incoming solar irradiance, quiet sun term (g-point)
    real(wp), dimension(:), allocatable :: solar_source_facular ! incoming solar irradiance, facular term (g-point)
    real(wp), dimension(:), allocatable :: solar_source_sunspot ! incoming solar irradiance, sunspot term (g-point)

    !
    ! -----------------------------------------------------------------------------------
    ! Ancillary
    ! -----------------------------------------------------------------------------------
    ! Index into %gas_names -- is this a key species in any band?
    logical, dimension(:), allocatable :: is_key
    ! -----------------------------------------------------------------------------------
    
  contains
    ! Type-bound procedures
    ! Public procedures
    ! public interface
    generic,   public :: load       => load_int,       load_ext
    procedure, public :: source_is_internal
    procedure, public :: source_is_external
    procedure, public :: get_ngas
    procedure, public :: get_gases
    procedure, public :: get_press_min
    procedure, public :: get_press_max
    procedure, public :: get_temp_min
    procedure, public :: get_temp_max
    procedure, public :: compute_optimal_angles
    procedure, public :: set_solar_variability
    procedure, public :: set_tsi
    ! Internal procedures
    procedure, private :: load_int
    procedure, private :: load_ext
    procedure, public  :: gas_optics_int
    procedure, public  :: gas_optics_ext
    procedure, private :: check_key_species_present
    ! Interpolation table dimensions
    procedure, private :: get_nflav
    procedure, private :: get_neta
    procedure, private :: get_npres
    procedure, private :: get_ntemp
    procedure, private :: get_nPlanckTemp
  end type
  ! -------------------------------------------------------------------------------------------------
  !
  ! col_dry is the number of molecules per cm-2 of dry air
  !
  public :: get_col_dry ! Utility function, not type-bound
  public :: compute_nn_inputs
contains
  ! --------------------------------------------------------------------------------------
  !
  ! Public procedures
  !
  ! --------------------------------------------------------------------------------------
  !
  ! Two functions to define array sizes needed by gas_optics()
  !
  pure function get_ngas(this)
    ! return the number of gases registered in the spectral configuration
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                                 :: get_ngas

    get_ngas = size(this%gas_names)
  end function get_ngas
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the number of distinct major gas pairs in the spectral bands (referred to as
  ! "flavors" - all bands have a flavor even if there is one or no major gas)
  !
  pure function get_nflav(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                                 :: get_nflav

    get_nflav = size(this%flavor,dim=2)
  end function get_nflav
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Compute gas optical depth and Planck source functions,
  !  given temperature, pressure, and composition
  !
  function gas_optics_int(this,                             &
                          play, plev, tlay, tsfc, gas_desc, &
                          optical_props, sources,           &
                          col_dry, tlev, neural_nets       & ! optional inputs
                          ) result(error_msg)
    ! inputs
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp), dimension(:,:), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                               plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                               tlay      ! layer temperatures [K]; (nlay,ncol)
    real(wp), dimension(:),   intent(in   ) :: tsfc      ! surface skin temperatures [K]; (ncol)
    type(ty_gas_concs),       intent(in   ) :: gas_desc  ! Gas volume mixing ratios
    ! output
    class(ty_optical_props_arry),  &
                              intent(inout) :: optical_props ! Optical properties
    class(ty_source_func_lw    ),  &
                              intent(inout) :: sources       ! Planck sources
    character(len=128)                      :: error_msg
    ! Optional inputs
    real(wp), dimension(:,:),   intent(in   ), &
                           optional, target :: col_dry, &  ! Column dry amount; dim(nlay,ncol)
                                                tlev        ! level temperatures [K]; (nlay+1,ncol)
    ! Optional input: neural network model (uses NN kernel if present)
    type(rrtmgp_network_type), dimension(:), intent(in), optional      :: neural_nets ! Emission and absorption models 

    ! ----------------------------------------------------------                              
    ! Local variables
    real(sp), dimension(:,:,:), allocatable           :: nn_inputs
    real(wp), dimension(:,:), allocatable,  target    :: col_dry_arr, vmr_h2o
    real(wp), dimension(:,:),   contiguous, pointer   :: col_dry_wk => NULL()
    ! ----------------------------------------------------------
    real(wp), dimension(size(play,dim=1)+1, size(play,dim=2)), &
                               target         :: tlev_arr
    real(wp), dimension(:,:),   pointer       :: tlev_wk

    integer :: ncol, nlay, ngpt, nband, ngas, ninputs, icol, ilay, idx_h2o
    ! ----------------------------------------------------------
    ncol  = size(play,dim=2)
    nlay  = size(play,dim=1)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()

    !$acc enter data copyin(tlay, tsfc, plev, play) 

    !
    ! check arrays sizes and values
    !
    error_msg  = ''
    if(check_extents) then
      if(.not. extents_are(play, nlay, ncol  )) error_msg = "gas_optics(): array play has wrong size"
      if(.not. extents_are(tlay, nlay, ncol  )) error_msg = "gas_optics(): array tlay has wrong size"
      if(.not. extents_are(plev, nlay+1, ncol)) error_msg = "gas_optics(): array plev has wrong size"
      if(.not. extents_are(tsfc, ncol)) error_msg = "gas_optics(): array tsfc has wrong size"
      if(present(tlev)) then
        if(.not. extents_are(tlev, nlay+1, ncol)) error_msg = "gas_optics(): array tlev has wrong size" 
      end if
      if(present(col_dry)) then
        if(.not. extents_are(col_dry, nlay, ncol)) error_msg = "gas_optics(): array col_dry has wrong size"
      end if
      if(any([sources%get_ncol(), sources%get_nlay(), sources%get_ngpt()] /= [ncol, nlay, ngpt])) &
      error_msg = "gas_optics%gas_optics: source function arrays inconsistently sized"
    end if
    if(error_msg  /= '') return

    if(check_values) then
      if(any_vals_outside(play, this%press_ref_min,this%press_ref_max)) error_msg = "gas_optics(): array play has values outside range"
      if(any_vals_less_than(plev, 0._wp)) error_msg = "gas_optics(): array plev has values outside range"
      if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max)) error_msg = "gas_optics(): array tlay has values outside range"
      if(any_vals_outside(tsfc, this%temp_ref_min,  this%temp_ref_max)) error_msg = "gas_optics(): array tsfc has values outside range"
      if(present(tlev)) then
        if(any_vals_outside(tlev, this%temp_ref_min, this%temp_ref_max)) error_msg = "gas_optics(): array tlev has values outside range"
      end if
      if(present(col_dry)) then
        if(any_vals_less_than(col_dry, 0._wp)) error_msg = "gas_optics(): array col_dry has values outside range"
      end if
    end if
    if(error_msg  /= '') return

    if(present(tlev)) then
      tlev_wk => tlev
       !$acc enter data copyin(tlev)
    else
      tlev_wk => tlev_arr
      !
      ! Interpolate temperature to levels if not provided
      !   Interpolation and extrapolation at boundaries is weighted by pressure
      !
      do icol = 1, ncol
        tlev_arr(1,icol) = tlay(1,icol) + (plev(1,icol)-play(1,icol))*(tlay(2,icol)-tlay(1,icol)) / (play(2,icol)-play(1,icol))
        do ilay = 2, nlay
          tlev_arr(ilay,icol) = (play(ilay-1,icol)*tlay(ilay-1,icol)*(plev(ilay,icol  )-play(ilay,icol)) &
                                +  play(ilay,icol  )*tlay(ilay,icol  )*(play(ilay-1,icol)-plev(ilay,icol))) /  &
                                  (plev(ilay,icol)*(play(ilay-1,icol) - play(ilay,icol)))
        end do
        tlev_arr(nlay+1,icol) = tlay(nlay,icol) + (plev(nlay+1,icol)-play(nlay,icol))*(tlay(nlay,icol)-tlay(nlay-1,icol))  &
                                 / (play(nlay,icol)-play(nlay-1,icol))
      end do
      !$acc enter data copyin(tlev_arr)
    end if

      !
      ! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
      !
    if (present(col_dry)) then
      !$acc enter data copyin(col_dry)
      col_dry_wk => col_dry
    else
      ! allocate(col_dry_arr(nlay,ncol), vmr_h2o(nlay,ncol))
      ! !$acc enter data create(col_dry_arr, vmr_h2o)
      ! error_msg = gas_desc%get_vmr('h2o', vmr_h2o)
      ! call get_col_dry(vmr_h2o, plev, col_dry_arr)
      ! col_dry_wk => col_dry_arr
      ! !$acc exit data delete(vmr_h2o)

      allocate(col_dry_arr(nlay,ncol))
      idx_h2o = string_loc_in_array('h2o', gas_desc%gas_name)
      !$acc enter data create(col_dry_arr)
      call get_col_dry(gas_desc%concs(idx_h2o)%conc, plev, col_dry_arr)
      col_dry_wk => col_dry_arr
    end if
    !$acc enter data attach(tlev_wk, col_dry_wk) 

    ! !! FOUND STRANGE OPENMP BUG ON GCC 9.3: with many threads, the content of col_dry_wk 
    ! can be different from col_dry_arr!
    ! PRINT *, "proc ", OMP_GET_THREAD_NUM(), " mean col_dry_wk, arr ", mean_2d(col_dry_wk), mean_2d(col_dry_arr)
    
    !
    ! Gas optics
    !
    if (present(neural_nets)) then
      ! ----------------------------------------------------------------------------------
      ! Use neural network for gas optics
      ninputs =  size(neural_nets(1) % layers(1) % w_transposed, 2)
      allocate(nn_inputs(ninputs,nlay,ncol))
      !$acc enter data create(nn_inputs)

      error_msg = compute_nn_inputs(              &
                          ncol, nlay, ninputs,    &
                          play, tlay, gas_desc,   &
                          neural_nets, &
                          nn_inputs)                     
                          
#ifdef USE_TIMING
    ret =  gptlstart('predict_nn_lw_blas')
#endif
      ! trick to reduce memory use: use already allocated variable
      associate (planck_fraction =>sources%lay_source)

        call predict_nn_lw_blas(              &
                ncol, nlay, ngpt, ninputs,    &  ! dimensions
                nn_inputs,  col_dry_arr, & ! col_dry_wk, !! OPENMP BUG, SEE ABOVE
                neural_nets,                  &  ! NN models (input)
                optical_props%tau, planck_fraction) 
        !$acc exit data delete(nn_inputs) 
        deallocate(nn_inputs)
#ifdef USE_TIMING
    ret =  gptlstop('predict_nn_lw_blas')
    ret =  gptlstart('compute_source')
#endif
        ! Compute Planck source function
        ! In this case the Planck fraction has already been computed, so we call a custom function
        call compute_Planck_source_nn(ncol, nlay, nband, ngpt,      &
          this%get_nPlanckTemp(),                                   &
          tlay, tlev_wk, tsfc, merge(1,nlay,play(1,1) > play(nlay,1)), &
          this%get_gpoint_bands(), this%get_band_lims_gpoint(),     &
          this%temp_ref_min, this%totplnk_delta, this%totplnk,      &
          sources%sfc_source, sources%sfc_source_Jac,               &
          sources%lay_source, sources%lev_source) ! argument lay_source (inout) is pfrac on input,
                                                  ! lay_source on output
        end associate
#ifdef USE_TIMING
    ret =  gptlstop('compute_source')
#endif
              
    else

      ! ----------------------------------------------------------------------------------
      ! Use interpolation routine for gas optics, NOT neural network
      error_msg = compute_gas_optics(this,                         &
                                   ncol, nlay, ngpt, nband,                 &
                                   play, plev, tlay, gas_desc, col_dry_arr, & ! col_dry_wk, !! OPENMP BUG, SEE ABOVE
                                   optical_props,                           &
                                   sources, tlev_wk, tsfc)             
      if(error_msg  /= '') return
      ! ----------------------------------------------------------------------------------
    end if

    !$acc exit data delete(col_dry, col_dry_arr, tlay, tlev, tlev_arr, tsfc, plev, play) detach(tlev_wk, col_dry_wk)

  end function gas_optics_int
  !------------------------------------------------------------------------------------------
  !
  ! Compute gas optical depth given temperature, pressure, and composition
  !
  function gas_optics_ext(this,                         &
                          play, plev, tlay, gas_desc,   & ! mandatory inputs
                          optical_props, toa_src,       & ! mandatory outputs
                          col_dry, neural_nets         &         ! optional input
                          ) result(error_msg)      

    class(ty_gas_optics_rrtmgp),  intent(in) :: this
    real(wp), dimension(:,:),     intent(in) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                               plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                               tlay      ! layer temperatures [K]; (nlay,ncol)
    type(ty_gas_concs),           intent(in) :: gas_desc  ! Gas volume mixing ratios
    ! output
    class(ty_optical_props_arry),  &
                                  intent(inout) :: optical_props
    real(wp), dimension(:,:),     intent(  out) :: toa_src     ! Incoming solar irradiance(ngpt,ncol)
    character(len=128)                      :: error_msg

    ! Optional inputs 
    real(wp), dimension(:,:),     intent(in   ), &
                            optional, target :: col_dry ! Column dry amount; dim(nlay,ncol)
    !  neural network model (uses NN kernel if present)
    type(rrtmgp_network_type), dimension(2), intent(in), optional      :: neural_nets ! Absorption model, Rayleigh model 
    ! ----------------------------------------------------------
    ! Local variables
    real(sp), dimension(:,:,:), allocatable           :: nn_inputs
    real(wp), dimension(:,:), allocatable,  target  :: col_dry_arr, vmr_h2o
    real(wp), dimension(:,:),   contiguous, pointer   :: col_dry_wk => NULL()

    integer :: ncol, nlay, ngpt, nband, ngas, idx_h2o, ninputs
    integer :: igpt, icol, ilay
    ! ----------------------------------------------------------
    nlay  = size(play,dim=1)
    ncol  = size(play,dim=2)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()
    ngas  = this%get_ngas()

    !$acc enter data copyin(tlay, play, plev) 

    !
    ! check arrays sizes and values
    !
    error_msg  = ''
    if(check_extents) then
      if(.not. extents_are(play, nlay, ncol  )) error_msg = "gas_optics(): array play has wrong size"
      if(.not. extents_are(tlay, nlay, ncol  )) error_msg = "gas_optics(): array tlay has wrong size"
      if(.not. extents_are(plev, nlay+1, ncol)) error_msg = "gas_optics(): array plev has wrong size"
      if(present(col_dry)) then
        if(.not. extents_are(col_dry, nlay, ncol)) error_msg = "gas_optics(): array col_dry has wrong size"
      end if
    end if
    if(error_msg  /= '') return

    if(check_values) then
      if(any_vals_outside(play, this%press_ref_min,this%press_ref_max))  error_msg = "gas_optics(): array play has values outside range"
      if(any_vals_less_than(plev, 0._wp)) error_msg = "gas_optics(): array plev has values outside range"
      if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max))  error_msg = "gas_optics(): array tlay has values outside range"
      if(present(col_dry)) then
        if(any_vals_less_than(col_dry, 0._wp)) error_msg = "gas_optics(): array col_dry has values outside range"
      end if
    end if
    if(error_msg  /= '') return

    !
    ! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
    !
    if (present(col_dry)) then
      !$acc enter data copyin(col_dry)
      col_dry_wk => col_dry
    else
      ! allocate(col_dry_arr(nlay,ncol), vmr_h2o(nlay,ncol))
      ! !$acc enter data create(col_dry_arr, vmr_h2o)
      ! error_msg = gas_desc%get_vmr('h2o', vmr_h2o)
      ! call get_col_dry(vmr_h2o, plev, col_dry_arr)
      
      ! col_dry_wk => col_dry_arr
      ! !$acc exit data delete(vmr_h2o)

      !
      ! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
      !
      allocate(col_dry_arr(nlay,ncol))
      idx_h2o = string_loc_in_array('h2o', gas_desc%gas_name)
      !$acc enter data create(col_dry_arr)
      call get_col_dry(gas_desc%concs(idx_h2o)%conc, plev, col_dry_arr)
      col_dry_wk => col_dry_arr
    end if
    !$acc enter data attach(col_dry_wk)
    ! !! FOUND STRANGE OPENMP BUG ON GCC 9.3: with many threads, the content of col_dry_wk 
    ! can be different from col_dry_arr!
    ! PRINT *, "proc ", OMP_GET_THREAD_NUM(), " mean col_dry_wk, arr ", mean_2d(col_dry_wk), mean_2d(col_dry_arr)


    !
    ! Gas optics
    !
    if (present(neural_nets)) then
      ! ----------------------------------------------------------------------------------
      ! Use neural network for gas optics
      ninputs =  size(neural_nets(1) % layers(1) % w_transposed, 2)
      allocate(nn_inputs(ninputs,nlay,ncol))
      !$acc enter data create(nn_inputs)

      error_msg = compute_nn_inputs(              &
                          ncol, nlay, ninputs,    &
                          play, tlay, gas_desc,   &
                          neural_nets, &
                          nn_inputs)  

      select type(optical_props)
        type is (ty_optical_props_1scl)
          ! User is asking for absorption optical depth only
          call predict_nn_sw_blas(              &
                    ncol, nlay, ngpt, ninputs,    &  ! dimensions
                    nn_inputs, col_dry_arr, & ! col_dry_wk, !! OPENMP BUG, SEE ABOVE
                    neural_nets,                  &  ! NN models (input)
                    optical_props%tau)             ! outputs    

        type is (ty_optical_props_2str)
          ! User is also asking for single scattering albedo

          call predict_nn_sw_blas(              &
                  ncol, nlay, ngpt, ninputs,    &  ! dimensions
                  nn_inputs,  col_dry_arr, & ! col_dry_wk, !! OPENMP BUG, SEE ABOVE
                  neural_nets,                  &  ! NN models (input)
                  optical_props%tau, optical_props%ssa)    ! outputs    

          !$acc parallel loop collapse(3) present(optical_props%g)
          do icol = 1, ncol
            do ilay = 1, nlay
              do igpt = 1, ngpt
                optical_props%g(igpt,ilay,icol) = 0.0_wp
              end do
            end do
          end do
        
      end select
      
      !$acc exit data delete(nn_inputs) 
      deallocate(nn_inputs)
     
    else
    ! ----------------------------------------------------------------------------------
    ! Use interpolation routine for gas optics, NOT neural networks

      error_msg = compute_gas_optics(this,                     &
                                  ncol, nlay, ngpt, nband,     &
                                  play, plev, tlay, gas_desc,  col_dry_arr, & ! col_dry_wk, 
                                  optical_props) 
    end if

    if(error_msg  /= '') return

    ! ----------------------------------------------------------
    !
    ! External source function is constant
    !
    if(.not. extents_are(toa_src, ngpt, ncol)) &
      error_msg = "gas_optics(): array toa_src has wrong size"
    if(error_msg  /= '') return

    !$acc parallel loop collapse(2) present(toa_src,this%solar_source)
    do icol = 1,ncol
       do igpt = 1,ngpt
          toa_src(igpt,icol) = this%solar_source(igpt)
       end do
    end do
    !$acc exit data delete(col_dry, col_dry_arr, tlay, play, plev) detach(col_dry_wk)

  end function gas_optics_ext
  function mean_2d(x) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x
    real(wp) :: mean2
    mean2 = sum(x) / size(x)
  end function mean_2d
  !------------------------------------------------------------------------------------------
  ! Routine for preparing RRTMGP-NN inputs from the gas concentrations, temperature and pressure
  ! The input names are loaded from the model, along with the input scaling coefficients:
  ! currently, it's assumed that min-max scaling of inputs to 0-1 range was done prior to training,
  ! and furthermore that prior to this min-max scaling, the following power scalings were performed
  ! p_input = log(p), h2o_input = h2o**(1/4), o3_input = o3**(1/4). 
  ! These are hardcoded assumptions, but not the min-max coefficients themselves!
  ! If a gas used by the NN is missing from the provided concentrations, a concentration of zero
  ! or alternatively a global-mean reference concentration can be used (pre-industrial, present, or future?)
  function compute_nn_inputs(ncol, nlay, ninputs,       &
                              play, tlay, gas_desc,     &
                              neural_nets,              &
                              nn_inputs) result(error_msg)

    integer,                                  intent(in   ) ::  ncol, nlay, ninputs
    real(wp), dimension(nlay,ncol),           intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                                tlay
    type(ty_gas_concs),                       intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  
    type(rrtmgp_network_type), dimension(:),  intent(in)    :: neural_nets ! needed to match inputs, load coefficients
    real(sp), dimension(ninputs, nlay, ncol), intent(inout) ::  nn_inputs 
    character(len=128)                                  :: error_msg
    ! ----------------------------------------------------------
    ! Local variables
    integer :: igas, ilay, icol, ndims, idx_h2o, idx_o3, idx_gas
    ! Reference gas concentrations are stored in rrtmgp_ref_concentrations for each gas 
    ! (except H2O O3 O2 N2) for three different scenarios (present-day, pre-industrial or future)
    ! Set below value to either 0 (zero concentration), 1 (Present-day), 2 (Pre-industrial) or 3 (Future)
    integer                                   :: nn_scenario_index  = 0 
    real(sp)                                  :: ref_vmr
    character(18), dimension(3)               :: scenario_names = &
                                                [character(len=18) :: 'present-day', 'pre-industrial', 'future']   
    character(len=32)                         :: gas_name 
    character(32),  dimension(:), allocatable :: input_names
    real(sp),       dimension(:), allocatable :: input_maxvals, input_minvals
    logical                                   :: print_warnings     = .false.
    ! ----------------------------------------------------------
    error_msg  = ''

    ! Old code with hardcoded assumptions about inputs!
    ! if (ninputs == 18) then        
    !   if (print_warnings) print *, "using longwave neural network models which take all 16 non-constant RRTMGP LW gases as input"
    !   input_names  = nn_gasopt_input_names(1:18)
    !   input_minvals = nn_input_minvals(1:18)
    !   input_maxvals = nn_input_maxvals(1:18)
    ! ! else if (ninputs == 9) then
    ! !   if (print_warnings) print *, "using less complex LW neural network which only uses h2o, o3, co2, n2o, ch4, cfc11-EQ and cfc12"
    ! !   input_names    = nn_gasopt_input_names(1:9)
    ! !   input_minvals   = nn_input_minvals(1:9)               
    ! !   input_maxvals   = nn_input_maxvals(1:9)
    ! else if (ninputs == 7) then
    !   if (print_warnings) print *, "using shortwave neural network model which take as input the concentrations of h2o, o3, co2, n2o and ch4"
    !   input_names    = nn_gasopt_input_names(1:7)
    !   input_minvals   = nn_input_minvals(1:7)               
    !   input_maxvals   = nn_input_maxvals(1:7)
    ! else 
    !  ! error_msg = "ninputs should be either 18 (full longwave model), 9 (reduced longwave model ala CKDMIP using CFC11-eq) or 7 (shortwave)"
    !   error_msg = "ninputs should be either 18 (longwave) or 7 (shortwave)"
    ! end if
    ! if(error_msg  /= '') return

    input_minvals = neural_nets(1)%coeffs_input_min
    input_maxvals = neural_nets(1)%coeffs_input_max
    input_names   = neural_nets(1)%input_names

    ! Check that the two neural nets use the same inputs and scaling coefficients, otherwise stop
    ! (alternatively, we could construct different nn_inputs, calling the routine separately for each network)
    !  if(check_values) then
    !   if (.not. all(neural_nets(1)%input_names == neural_nets(2)%input_names)) &
    !     error_msg = "compute_nn_inputs: The two neural networks seem to use different inputs"
    !   if(error_msg  /= '') return
    !   if (.not. all(abs(neural_nets(1)%coeffs_input_max - neural_nets(2)%coeffs_input_max) < 1e-12)) &
    !     error_msg =  "compute_nn_inputs: The two neural networks seem to use different input scaling coeffs"
    !   if(error_msg  /= '') return
    !   if (.not. (string_in_array('o3', input_names) .and. string_in_array('h2o', input_names))) &
    !   error_msg = "compute_nn_inputs: code assumes that H2O and O3 are both inputs, but not found in input_names"
    !   if(error_msg  /= '') return
    !  end if

    ! if check_concs_within_training_range: 
    ! do igas = 1, n inputs
    !   error_msg = gas_desc%get_conc_dims_and_igas(input_names(igas), ndims, idx_gas)  
    !   minval = input_minvals(igas)
    !   maxval = input_maxvals(igas)
    !   if(any_vals_outside(gas_desc%concs(idx_gas), minval, maxval)) then
    !     error_msg = "rrtmgp: gas",input_names(igas), "was outside the range of the NN training data: <=", minval, " or >" maxval
    !     exit
    ! end do
    ! if(error_msg  /= '') return

#ifdef USE_TIMING
    ret =  gptlstart('compute_nn_inputs')
#endif
    ! Start writing NN inputs from the gas concentrations
    ! Here the gas concentrations are accessed directly from gas_conc, instead of the original method 
    ! of filling a 2D vmr array for each gas like gas_array(idx_gas,:,:) = get_vmr
    ! This is considerably faster (previously preprocessing became somewhat costly with small block sizes)

    !$acc enter data copyin(input_names, input_maxvals, input_minvals)

    ! First lets write temperature, pressure, water vapor and ozone - these are assumed to be present and 2D!
    error_msg = gas_desc%get_conc_dims_and_igas(input_names(3), ndims, idx_h2o)
    error_msg = gas_desc%get_conc_dims_and_igas(input_names(4), ndims, idx_o3)
    if(error_msg  /= '') return

    !$acc parallel loop collapse(2) default(present)
    do icol = 1, ncol
      do ilay = 1, nlay
          nn_inputs(1,ilay,icol)    =  (tlay(ilay,icol)     - input_minvals(1)) / (input_maxvals(1) - input_minvals(1))
          nn_inputs(2,ilay,icol)    = (log(play(ilay,icol)) - input_minvals(2)) / (input_maxvals(2) - input_minvals(2))
          nn_inputs(3,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_h2o)%conc(ilay,icol))) - input_minvals(3)) / (input_maxvals(3) - input_minvals(3))
          nn_inputs(4,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_o3) %conc(ilay,icol))) - input_minvals(4)) / (input_maxvals(4) - input_minvals(4))
      end do
    end do
    
    ! Now lets write the remaining gases
    do igas = 5, ninputs

      ! Get the index and number of dimensions (concentration could be scalar, 1D or 2D)
      error_msg = gas_desc%get_conc_dims_and_igas(input_names(igas), ndims, idx_gas)  
      ! idx_gas is the location of the gas in gas_desc

      ! If successful, the gas was found and we can continue filling array
      if (error_msg == '') then
        if (ndims == 0) then
          !$acc parallel loop collapse(2) default(present)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(1,1)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        else if (ndims == 1) then
          !$acc parallel loop collapse(2) default(present)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(ilay,1)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        else 
          !$acc parallel loop collapse(2) default(present)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(ilay,icol)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        end if 
      
      ! If not successful, the gas was not found in gas_desc, and we need to use a reference concentration
      else
        if (nn_scenario_index==0) then
          ref_vmr = 0.0_wp
          error_msg = ''
        else
          error_msg = get_ref_vmr(nn_scenario_index, input_names(igas), ref_vmr) 
        end if 
        if (print_warnings) then
          print *, 'WARNING: Neural network uses the gas '// trim(input_names(igas)) //  ' as input but it was not provided'
          if (nn_scenario_index==0) then
            print *, 'Scenario_index in rte_rrtmgp_config was set to 0 -> using a concentration of zero'
          else
            print '("Scenario index was set to ", I0, " (", A, ") -> using a reference concentration of ", E10.3)', &
                  nn_scenario_index, trim(scenario_names(nn_scenario_index)), ref_vmr
          end if
        end if 

        !$acc parallel loop collapse(2) default(present)
        do icol = 1, ncol
          do ilay = 1, nlay
              nn_inputs(igas,ilay,icol)    =  (ref_vmr  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
          end do
        end do

      end if

    end do
    if(error_msg  /= '') return

    !$acc exit data delete(input_names, input_maxvals, input_minvals)

    ! do igas = 1, ninputs
    !   print '(A25,I2,A2,A8,F6.3,F6.3)', "Min,max of NN-input ", igas, " =", input_names(igas), &
    !               minval(nn_inputs(igas,:,:)), maxval(nn_inputs(igas,:,:))
    ! end do

    ! nn_inputs = max(0.0_wp, nn_inputs)

#ifdef USE_TIMING
    ret =  gptlstop('compute_nn_inputs')
#endif

  end function compute_nn_inputs

  !
  ! Returns optical properties and interpolation coefficients
  !
  function compute_gas_optics(this,                        &
                            ncol, nlay, ngpt, nband,                &
                            play, plev, tlay, gas_desc, col_dry,    &
                            optical_props,                          &
                            sources, tlev, tsfc) result(error_msg)

    class(ty_gas_optics_rrtmgp), &
                                      intent(in   ) :: this
    integer,                          intent(in   ) :: ncol, nlay, ngpt, nband
    real(wp),   dimension(nlay,ncol), intent(in   ) :: play, tlay ! layer pressures [Pa, mb]; temperatures[K] (nlay,ncol)
    real(wp), dimension(nlay+1,ncol), intent(in   ) :: plev   ! level pressures
    type(ty_gas_concs),               intent(in   ) :: gas_desc  ! Gas volume mixing ratios
    real(wp), dimension(nlay,ncol),   intent(in   ) :: col_dry ! Column dry amount; dim(nlay,ncol)
    class(ty_optical_props_arry),     intent(inout) :: optical_props !inout because components are allocated
    ! Optional inputs used for long-wave
    class(ty_source_func_lw    ),     intent(inout), &
                                                  optional :: sources ! Planck sources
    real(wp), dimension(nlay+1,ncol), intent(in), optional :: tlev
    real(wp), dimension(ncol       ), intent(in), optional :: tsfc  

    ! ----------------------------------------------------------
    ! Local variables
    !
    ! Interpolation coefficients for use in internal source function
    integer,     dimension(                      nlay, ncol)  :: jtemp, jpress
    integer,     dimension(2,    get_nflav(this),nlay, ncol)  :: jeta
    logical(wl), dimension(                      nlay, ncol)  :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),nlay, ncol)  :: fmajor
    character(len=128)                                        :: error_msg

    real(wp), dimension(:,:,:), allocatable               :: tau_rayleigh  ! absorption, Rayleigh scattering optical depths
    real(wp), dimension(nlay, ncol, this%get_ngas())      :: vmr     ! volume mixing ratios
    !
    ! Interpolation variables used in major gas but not elsewhere, so don't need exporting
    !
    real(wp), dimension(nlay, ncol,  0:this%get_ngas())   :: col_gas ! column amounts for each gas (Num. of molecules per cm^2), plus col_dry
    real(wp), dimension(2,    get_nflav(this),nlay,ncol)  :: col_mix ! combination of major species's column amounts
                                                         ! index(1) : reference temperature level
                                                         ! index(2) : flavor
                                                         ! index(3) : layer
    real(wp), dimension(2,2,  get_nflav(this),nlay,ncol) :: fminor ! interpolation fractions for minor species
                                                          ! index(1) : reference eta level (temperature dependent)
                                                          ! index(2) : reference temperature level
                                                          ! index(3) : flavor
                                                          ! index(4) : layer
    integer :: ngas, nflav, neta, npres, ntemp
    integer :: icol, ilay, igas
    integer :: idx_h2o ! index of water vapor
    integer :: nminorlower, nminorklower,nminorupper, nminorkupper
    logical :: use_rayl
    ! ----------------------------------------------------------
    !
    ! Error checking
    !
    use_rayl = allocated(this%krayl)
    error_msg = ''
    ! Check for initialization
    if (.not. this%is_initialized()) then
      error_msg = 'ERROR: spectral configuration not loaded'
      return
    end if
    !
    ! Check for presence of key species in ty_gas_concs; return error if any key species are not present
    !
    error_msg = this%check_key_species_present(gas_desc)
    if (error_msg /= '') return

    ! ----------------------------------------------------------
    ngas  = this%get_ngas()
    nflav = get_nflav(this)
    neta  = this%get_neta()
    npres = this%get_npres()
    ntemp = this%get_ntemp()
    ! number of minor contributors, total num absorption coeffs
    nminorlower  = size(this%minor_scales_with_density_lower)
    nminorklower = size(this%kminor_lower, 1)
    nminorupper  = size(this%minor_scales_with_density_upper)
    nminorkupper = size(this%kminor_upper, 1)

    !
    ! compute column gas amounts [molec/cm^2]
    !
    !$acc enter data create(col_gas, vmr)
    do igas = 1, ngas
      !
      ! Get vmr if  gas is provided in ty_gas_concs
      !
      if (any (lower_case(this%gas_names(igas)) == gas_desc%gas_name(:))) then
        error_msg = gas_desc%get_vmr(this%gas_names(igas), vmr(:,:,igas))
        if (error_msg /= '') return
      endif
    end do
  
    !$acc parallel default(present)
    !$acc loop gang vector collapse(2) 
    do icol = 1, ncol
      do ilay = 1, nlay
        col_gas(ilay,icol,0) = col_dry(ilay,icol)
      end do
    end do
    
    !$acc loop gang vector collapse(3)
    do igas = 1, ngas
      do icol = 1, ncol
        do ilay = 1, nlay
          col_gas(ilay,icol,igas) = vmr(ilay,icol,igas) * col_dry(ilay,icol)
        end do
      end do
    end do
    !$acc end parallel

    !$acc exit data delete(vmr)

    !
    ! ---- calculate gas optical depths ----
    !
    !$acc enter data create(jtemp, jpress, jeta, tropo, fmajor, fminor, col_mix)
    !$acc enter data copyin(this)
    !$acc enter data copyin(this%gpoint_flavor)
    !$acc enter data copyin(this%flavor,this%press_ref_log,this%temp_ref,this%vmr_ref) 
    call interpolation(               &
            ncol,nlay,                &        ! problem dimensions
            ngas, nflav, neta, npres, ntemp, & ! interpolation dimensions
            this%flavor,              &
            this%press_ref_log,       &
            this%temp_ref,            &
            this%press_ref_log_delta, &
            this%temp_ref_min,        &
            this%temp_ref_delta,      &
            this%press_ref_trop_log,  &
            this%vmr_ref, &
            play,         &
            tlay,         &
            col_gas,      &
            jtemp,        & ! outputs
            fmajor,fminor,&
            col_mix,      &
            tropo,        &
            jeta,jpress)
    !$acc exit data delete(this%flavor,this%press_ref_log,this%temp_ref,this%vmr_ref) 

    !idx_h2o = string_loc_in_array('h2o', this%gas_names)
    idx_h2o = string_loc_in_array('h2o', gas_desc%gas_name)

#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_kernel')
#endif
    call compute_tau_absorption(                     &
            ncol,nlay,nband,ngpt,                    &  ! dimensions
            ngas,nflav,neta,npres,ntemp,             &
            nminorlower, nminorklower,               & ! number of minor contributors, total num absorption coeffs
            nminorupper, nminorkupper,               &
            idx_h2o,                                 &
            this%gpoint_flavor,                      &
            this%get_band_lims_gpoint(),             &
            this%kmajor,                             &
            this%kminor_lower,                       &
            this%kminor_upper,                       &
            this%minor_limits_gpt_lower,             &
            this%minor_limits_gpt_upper,             &
            this%minor_scales_with_density_lower,    &
            this%minor_scales_with_density_upper,    &
            this%scale_by_complement_lower,          &
            this%scale_by_complement_upper,          &
            this%idx_minor_lower,                    &
            this%idx_minor_upper,                    &
            this%idx_minor_scaling_lower,            &
            this%idx_minor_scaling_upper,            &
            this%kminor_start_lower,                 &
            this%kminor_start_upper,                 &
            tropo,                                   &
            col_mix,fmajor,fminor,                   &
            play,tlay,col_gas,                       &
            jeta,jtemp,jpress,                       &
            optical_props%tau)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_kernel')
#endif

    if (allocated(this%krayl)) then
      allocate(tau_rayleigh(ngpt,nlay,ncol))
#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_ray_kernel')
#endif
      !$acc enter data copyin(this%krayl) create(tau_rayleigh)
      call compute_tau_rayleigh(         & !Rayleigh scattering optical depths
            ncol,nlay,nband,ngpt,        &
            ngas,nflav,neta,npres,ntemp, & ! dimensions
            this%gpoint_flavor,          &
            this%get_band_lims_gpoint(), &
            this%krayl,                  & ! inputs from object
            idx_h2o, col_dry, col_gas, &
            fminor,jeta,tropo,jtemp,     & ! local input
            tau_rayleigh)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_ray_kernel')
    ret =  gptlstart('combine_taus_compute_ssa')
#endif
      ! Combine taus
      call combine(tau_rayleigh, allocated(this%krayl), optical_props)
      !$acc exit data delete(this%krayl, tau_rayleigh)

#ifdef USE_TIMING
    ret =  gptlstop('combine_taus_compute_ssa')
#endif
    end if

    !$acc exit data delete(col_gas, col_mix, fminor)

    if(present(sources)) then
#ifdef USE_TIMING
    ret =  gptlstart('compute_source')
#endif
      if (allocated(sources%planck_frac)) then
        call compute_Planck_source (ncol, nlay, nband, ngpt,                &
        nflav, neta, npres, ntemp, this%get_nPlanckTemp(),&
        tlay, tlev, tsfc, merge(1,nlay,play(1,1) > play(nlay,1)),             &
        fmajor, jeta, tropo, jtemp, jpress,    &
        this%get_gpoint_bands(), this%get_band_lims_gpoint(),           &
        this%temp_ref_min, this%totplnk_delta, this%planck_frac_stored, this%totplnk, this%gpoint_flavor,  &
        sources%sfc_source, sources%sfc_source_Jac, sources%lay_source, sources%lev_source, sources%planck_frac)
      else
        call compute_Planck_source (ncol, nlay, nband, ngpt,                &
        nflav, neta, npres, ntemp, this%get_nPlanckTemp(),&
        tlay, tlev, tsfc, merge(1,nlay,play(1,1) > play(nlay,1)),             &
        fmajor, jeta, tropo, jtemp, jpress,    &
        this%get_gpoint_bands(), this%get_band_lims_gpoint(),           &
        this%temp_ref_min, this%totplnk_delta, this%planck_frac_stored, this%totplnk, this%gpoint_flavor,  &
        sources%sfc_source, sources%sfc_source_Jac, sources%lay_source, sources%lev_source)
      end if
#ifdef USE_TIMING
    ret =  gptlstop('compute_source')
#endif
    end if
    if (error_msg /= '') return

    !$acc exit data delete(this%gpoint_flavor)
    !$acc exit data delete(jtemp, jpress, jeta, tropo, fmajor)
    
  end function compute_gas_optics

  !------------------------------------------------------------------------------------------
  !
  ! Compute the spectral solar source function adjusted to account for solar variability
  !   following the NRLSSI2 model of Coddington et al. 2016, doi:10.1175/BAMS-D-14-00265.1.
  ! as specified by the facular brightening (mg_index) and sunspot dimming (sb_index)
  ! indices provided as input.
  !
  ! Users provide the NRLSSI2 facular ("Bremen") index and sunspot ("SPOT67") index.
  !   Changing either of these indicies will change the total solar irradiance (TSI)
  !   Code in extensions/mo_solar_variability may be used to compute the value of these
  !   indices through an average solar cycle
  ! Users may also specify the TSI, either alone or in conjunction with the facular and sunspot indices
  !
  !------------------------------------------------------------------------------------------
  function set_solar_variability(this,                      &
                                 mg_index, sb_index, tsi)   &
                                 result(error_msg)
    !
    ! Updates the spectral distribution and, optionally,
    !   the integrated value of the solar source function
    !   Modifying either index will change the total solar irradiance
    !
    class(ty_gas_optics_rrtmgp), intent(inout) :: this
    !
    real(wp),           intent(in) :: mg_index, & ! facular brightening index (NRLSSI2 facular "Bremen" index)
                                      sb_index    ! sunspot dimming index     (NRLSSI2 sunspot "SPOT67" index)
    real(wp), optional, intent(in) :: tsi         ! total solar irradiance
    character(len=128)             :: error_msg
    ! ----------------------------------------------------------
    integer :: igpt
    real(wp), parameter :: a_offset = 0.1495954_wp
    real(wp), parameter :: b_offset = 0.00066696_wp
    ! ----------------------------------------------------------
    error_msg = ""
    if(mg_index < 0._wp) error_msg = 'mg_index out of range'
    if(sb_index < 0._wp) error_msg = 'sb_index out of range'
    if(error_msg /= "") return
    !
    ! Calculate solar source function for provided facular and sunspot indices
    !
    !$acc parallel loop
    do igpt = 1, size(this%solar_source_quiet)
      this%solar_source(igpt) = this%solar_source_quiet(igpt) + &
                                (mg_index - a_offset) * this%solar_source_facular(igpt) + &
                                (sb_index - b_offset) * this%solar_source_sunspot(igpt)
    end do
    !
    ! Scale solar source to input TSI value
    !
    if (present(tsi)) error_msg = this%set_tsi(tsi)

  end function set_solar_variability
  !------------------------------------------------------------------------------------------
  function set_tsi(this, tsi) result(error_msg)
    !
    ! Scale the solar source function without changing the spectral distribution
    !
    class(ty_gas_optics_rrtmgp), intent(inout) :: this
    real(wp),                    intent(in   ) :: tsi ! user-specified total solar irradiance;
    character(len=128)                         :: error_msg

    real(wp) :: norm
    ! ----------------------------------------------------------
    error_msg = ""
    if(tsi < 0._wp) then
      error_msg = 'tsi out of range'
    else
      !
      ! Scale the solar source function to the input tsi
      !
      !$acc kernels
      norm = 1._wp/sum(this%solar_source(:))
      this%solar_source(:) = this%solar_source(:) * tsi * norm
      !$acc end kernels
    end if

  end function set_tsi
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Initialization
  !
  !--------------------------------------------------------------------------------------------------------------------
  ! Initialize object based on data read from netCDF file however the user desires.
  !  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  ! This interface is for the internal-sources object -- includes Plank functions and fractions
  !
  function load_int(this, available_gases, gas_names, key_species,  &
                    band2gpt, band_lims_wavenum,                    &
                    press_ref, press_ref_trop, temp_ref,            &
                    temp_ref_p, temp_ref_t, vmr_ref,                &
                    kmajor, kminor_lower, kminor_upper,             &
                    gas_minor,identifier_minor,                     &
                    minor_gases_lower, minor_gases_upper,           &
                    minor_limits_gpt_lower, minor_limits_gpt_upper, &
                    minor_scales_with_density_lower,                &
                    minor_scales_with_density_upper,                &
                    scaling_gas_lower, scaling_gas_upper,           &
                    scale_by_complement_lower,                      &
                    scale_by_complement_upper,                      &
                    kminor_start_lower,                             &
                    kminor_start_upper,                             &
                    totplnk, planck_frac,                           &
                    rayl_lower, rayl_upper,                         &
                    optimal_angle_fit) result(err_message)
    class(ty_gas_optics_rrtmgp),     intent(inout) :: this
    class(ty_gas_concs),                    intent(in   ) :: available_gases ! Which gases does the host model have available?
    character(len=*),   dimension(:),       intent(in   ) :: gas_names
    integer,            dimension(:,:,:),   intent(in   ) :: key_species
    integer,            dimension(:,:),     intent(in   ) :: band2gpt
    real(wp),           dimension(:,:),     intent(in   ) :: band_lims_wavenum
    real(wp),           dimension(:),       intent(in   ) :: press_ref, temp_ref
    real(wp),                               intent(in   ) :: press_ref_trop, temp_ref_p, temp_ref_t
    real(wp),           dimension(:,:,:),   intent(in   ) :: vmr_ref
    real(wp),           dimension(:,:,:,:), intent(in   ) :: kmajor
    real(wp),           dimension(:,:,:),   intent(in   ) :: kminor_lower, kminor_upper
    real(wp),           dimension(:,:),     intent(in   ) :: totplnk
    real(wp),           dimension(:,:,:,:), intent(in   ) :: planck_frac
    real(wp),           dimension(:,:,:),   intent(in   ), &
                                              allocatable :: rayl_lower, rayl_upper
    real(wp),           dimension(:,:),     intent(in   ) :: optimal_angle_fit
    character(len=*),   dimension(:),       intent(in   ) :: gas_minor,identifier_minor
    character(len=*),   dimension(:),       intent(in   ) :: minor_gases_lower, &
                                                             minor_gases_upper
    integer,            dimension(:,:),     intent(in   ) :: minor_limits_gpt_lower, &
                                                             minor_limits_gpt_upper
    logical(wl),        dimension(:),       intent(in   ) :: minor_scales_with_density_lower, &
                                                             minor_scales_with_density_upper
    character(len=*),   dimension(:),       intent(in   ) :: scaling_gas_lower, &
                                                             scaling_gas_upper
    logical(wl),        dimension(:),       intent(in   ) :: scale_by_complement_lower,&
                                                             scale_by_complement_upper
    integer,            dimension(:),       intent(in   ) :: kminor_start_lower,&
                                                             kminor_start_upper
    character(len = 128) :: err_message
    ! ----
    !$acc enter data create(this)
    err_message = init_abs_coeffs(this, &
                                  available_gases, &
                                  gas_names, key_species,    &
                                  band2gpt, band_lims_wavenum, &
                                  press_ref, temp_ref,       &
                                  press_ref_trop, temp_ref_p, temp_ref_t, &
                                  vmr_ref,                   &
                                  kmajor, kminor_lower, kminor_upper, &
                                  gas_minor,identifier_minor,&
                                  minor_gases_lower, minor_gases_upper, &
                                  minor_limits_gpt_lower, &
                                  minor_limits_gpt_upper, &
                                  minor_scales_with_density_lower, &
                                  minor_scales_with_density_upper, &
                                  scaling_gas_lower, scaling_gas_upper, &
                                  scale_by_complement_lower, &
                                  scale_by_complement_upper, &
                                  kminor_start_lower, &
                                  kminor_start_upper, &
                                  rayl_lower, rayl_upper)
    ! Planck function tables
    !
    ! ?????????? These allocations are unneeded ?????????????
    !  allocate(this%totplnk    (size(totplnk,    1), size(totplnk,   2)), &
    !           this%planck_frac_stored(size(planck_frac,1), size(planck_frac,2), size(planck_frac,3), size(planck_frac,4)), &
    !           this%optimal_angle_fit(size(optimal_angle_fit,    1), size(optimal_angle_fit,   2)))
    ! !$acc enter data create(this%totplnk, this%planck_frac_stored )
    ! !$acc kernels
    this%totplnk = totplnk
    this%planck_frac_stored = planck_frac
    this%optimal_angle_fit = optimal_angle_fit
    ! !$acc end kernels

    !$acc enter data copyin(this%totplnk,this%planck_frac_stored)

    ! Temperature steps for Planck function interpolation
    !   Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    !   Planck grid and the Planck grid is equally spaced
    this%totplnk_delta =  (this%temp_ref_max-this%temp_ref_min) / (size(this%totplnk,dim=1)-1)
    !$acc update device(this%totplnk_delta, this%temp_ref_min)
    
  end function load_int

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Initialize object based on data read from netCDF file however the user desires.
  !  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  ! This interface is for the external-sources object -- includes TOA source function table
  !
  function load_ext(this, available_gases, gas_names, key_species,        &
                    band2gpt, band_lims_wavenum,           &
                    press_ref, press_ref_trop, temp_ref, &
                    temp_ref_p, temp_ref_t, vmr_ref,     &
                    kmajor, kminor_lower, kminor_upper, &
                    gas_minor,identifier_minor, &
                    minor_gases_lower, minor_gases_upper, &
                    minor_limits_gpt_lower, minor_limits_gpt_upper, &
                    minor_scales_with_density_lower, &
                    minor_scales_with_density_upper, &
                    scaling_gas_lower, scaling_gas_upper, &
                    scale_by_complement_lower, &
                    scale_by_complement_upper, &
                    kminor_start_lower, &
                    kminor_start_upper, &
                    solar_quiet, solar_facular, solar_sunspot, &
                    tsi_default, mg_default, sb_default, &
                    rayl_lower, rayl_upper)  result(err_message)
    class(ty_gas_optics_rrtmgp), intent(inout) :: this
    class(ty_gas_concs),         intent(in   ) :: available_gases ! Which gases does the host model have available?
    character(len=*), &
              dimension(:),       intent(in) :: gas_names
    integer,  dimension(:,:,:),   intent(in) :: key_species
    integer,  dimension(:,:),     intent(in) :: band2gpt
    real(wp), dimension(:,:),     intent(in) :: band_lims_wavenum
    real(wp), dimension(:),       intent(in) :: press_ref, temp_ref
    real(wp),                     intent(in) :: press_ref_trop, temp_ref_p, temp_ref_t
    real(wp), dimension(:,:,:),   intent(in) :: vmr_ref
    real(wp), dimension(:,:,:,:), intent(in) :: kmajor
    real(wp), dimension(:,:,:),   intent(in) :: kminor_lower, kminor_upper
    character(len=*),   dimension(:), &
                                  intent(in) :: gas_minor, &
                                                identifier_minor
    character(len=*),   dimension(:), &
                                  intent(in) :: minor_gases_lower, &
                                                minor_gases_upper
    integer,  dimension(:,:),     intent(in) :: &
                                                minor_limits_gpt_lower, &
                                                minor_limits_gpt_upper
    logical(wl),    dimension(:), intent(in) :: &
                                                minor_scales_with_density_lower, &
                                                minor_scales_with_density_upper
    character(len=*),dimension(:),intent(in) :: &
                                                scaling_gas_lower, &
                                                scaling_gas_upper
    logical(wl),    dimension(:), intent(in) :: &
                                                scale_by_complement_lower, &
                                                scale_by_complement_upper
    integer,        dimension(:), intent(in) :: &
                                                kminor_start_lower, &
                                                kminor_start_upper
    real(wp),       dimension(:), intent(in) :: solar_quiet, &
                                                solar_facular, &
                                                solar_sunspot
    real(wp),                     intent(in) :: tsi_default, &
                                                mg_default, sb_default
    real(wp), dimension(:,:,:),   intent(in), &
                                 allocatable :: rayl_lower, rayl_upper
    character(len = 128) err_message

    integer :: ngpt
    ! ----
    !$acc enter data create(this)
    err_message = init_abs_coeffs(this, &
                                  available_gases, &
                                  gas_names, key_species,    &
                                  band2gpt, band_lims_wavenum, &
                                  press_ref, temp_ref,       &
                                  press_ref_trop, temp_ref_p, temp_ref_t, &
                                  vmr_ref,                   &
                                  kmajor, kminor_lower, kminor_upper, &
                                  gas_minor,identifier_minor, &
                                  minor_gases_lower, minor_gases_upper, &
                                  minor_limits_gpt_lower, &
                                  minor_limits_gpt_upper, &
                                  minor_scales_with_density_lower, &
                                  minor_scales_with_density_upper, &
                                  scaling_gas_lower, scaling_gas_upper, &
                                  scale_by_complement_lower, &
                                  scale_by_complement_upper, &
                                  kminor_start_lower, &
                                  kminor_start_upper, &
                                  rayl_lower, rayl_upper)
    if(err_message /= "") return
    !
    ! Spectral solar irradiance terms init
    !
    ngpt = size(solar_quiet)
    allocate(this%solar_source_quiet(ngpt), this%solar_source_facular(ngpt), &
             this%solar_source_sunspot(ngpt), this%solar_source(ngpt))
    !$acc enter data create(this%solar_source_quiet, this%solar_source_facular, this%solar_source_sunspot, this%solar_source)
    !$acc kernels
    this%solar_source_quiet   = solar_quiet
    this%solar_source_facular = solar_facular
    this%solar_source_sunspot = solar_sunspot
    !$acc end kernels
    err_message = this%set_solar_variability(mg_default, sb_default)
  end function load_ext
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Initialize absorption coefficient arrays,
  !   including Rayleigh scattering tables if provided (allocated)
  !
  function init_abs_coeffs(this, &
                           available_gases, &
                           gas_names, key_species,    &
                           band2gpt, band_lims_wavenum, &
                           press_ref, temp_ref,       &
                           press_ref_trop, temp_ref_p, temp_ref_t, &
                           vmr_ref,                   &
                           kmajor, kminor_lower, kminor_upper, &
                           gas_minor,identifier_minor,&
                           minor_gases_lower, minor_gases_upper, &
                           minor_limits_gpt_lower, &
                           minor_limits_gpt_upper, &
                           minor_scales_with_density_lower, &
                           minor_scales_with_density_upper, &
                           scaling_gas_lower, scaling_gas_upper, &
                           scale_by_complement_lower, &
                           scale_by_complement_upper, &
                           kminor_start_lower, &
                           kminor_start_upper, &
                           rayl_lower, rayl_upper) result(err_message)
    class(ty_gas_optics_rrtmgp), intent(inout) :: this
    class(ty_gas_concs),                intent(in   ) :: available_gases
    character(len=*), &
              dimension(:),       intent(in) :: gas_names
    integer,  dimension(:,:,:),   intent(in) :: key_species
    integer,  dimension(:,:),     intent(in) :: band2gpt
    real(wp), dimension(:,:),     intent(in) :: band_lims_wavenum
    real(wp), dimension(:),       intent(in) :: press_ref, temp_ref
    real(wp),                     intent(in) :: press_ref_trop, temp_ref_p, temp_ref_t
    real(wp), dimension(:,:,:),   intent(in) :: vmr_ref
    real(wp), dimension(:,:,:,:), intent(in) :: kmajor
    real(wp), dimension(:,:,:),   intent(in) :: kminor_lower, kminor_upper
    character(len=*),   dimension(:), &
                                  intent(in) :: gas_minor, &
                                                identifier_minor
    character(len=*),   dimension(:), &
                                  intent(in) :: minor_gases_lower, &
                                                minor_gases_upper
    integer,  dimension(:,:),     intent(in) :: minor_limits_gpt_lower, &
                                                minor_limits_gpt_upper
    logical(wl), dimension(:),    intent(in) :: minor_scales_with_density_lower, &
                                                minor_scales_with_density_upper
    character(len=*),   dimension(:),&
                                  intent(in) :: scaling_gas_lower, &
                                                scaling_gas_upper
    logical(wl), dimension(:),    intent(in) :: scale_by_complement_lower, &
                                                scale_by_complement_upper
    integer,  dimension(:),       intent(in) :: kminor_start_lower, &
                                                kminor_start_upper
    real(wp), dimension(:,:,:),   intent(in), &
                                 allocatable :: rayl_lower, rayl_upper
    character(len=128)                       :: err_message
    ! --------------------------------------------------------------------------
    logical,  dimension(:),     allocatable :: gas_is_present
    logical,  dimension(:),     allocatable :: key_species_present_init
    integer,  dimension(:,:,:), allocatable :: key_species_red
    real(wp), dimension(:,:,:), allocatable :: vmr_ref_red
    character(len=256), &
              dimension(:),     allocatable :: minor_gases_lower_red, &
                                               minor_gases_upper_red
    character(len=256), &
              dimension(:),     allocatable :: scaling_gas_lower_red, &
                                               scaling_gas_upper_red
    integer :: i, j, idx
    integer :: ngas
    ! --------------------------------------
    err_message = this%ty_optical_props%init(band_lims_wavenum, band2gpt)
    if(len_trim(err_message) /= 0) return
    !
    ! Which gases known to the gas optics are present in the host model (available_gases)?
    !
    ngas = size(gas_names)
    allocate(gas_is_present(ngas))
    do i = 1, ngas
      gas_is_present(i) = string_in_array(gas_names(i), available_gases%gas_name)
    end do
    !
    ! Now the number of gases is the union of those known to the k-distribution and provided
    !   by the host model
    !
    ngas = count(gas_is_present)
    !
    ! Initialize the gas optics object, keeping only those gases known to the
    !   gas optics and also present in the host model
    !
    this%gas_names = pack(gas_names,mask=gas_is_present)

    allocate(vmr_ref_red(size(vmr_ref,dim=1),0:ngas, &
                         size(vmr_ref,dim=3)))
    ! Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    vmr_ref_red(:,0,:) = vmr_ref(:,1,:)
    do i = 1, ngas
      idx = string_loc_in_array(this%gas_names(i), gas_names)
      vmr_ref_red(:,i,:) = vmr_ref(:,idx+1,:)
    enddo
    call move_alloc(vmr_ref_red, this%vmr_ref)
    !
    ! Reduce minor arrays so variables only contain minor gases that are available
    ! Reduce size of minor Arrays
    !
    call reduce_minor_arrays(available_gases, &
                             gas_names, &
                             gas_minor,identifier_minor, &
                             kminor_lower, &
                             minor_gases_lower, &
                             minor_limits_gpt_lower, &
                             minor_scales_with_density_lower, &
                             scaling_gas_lower, &
                             scale_by_complement_lower, &
                             kminor_start_lower, &
                             this%kminor_lower, &
                             minor_gases_lower_red, &
                             this%minor_limits_gpt_lower, &
                             this%minor_scales_with_density_lower, &
                             scaling_gas_lower_red, &
                             this%scale_by_complement_lower, &
                             this%kminor_start_lower)
    call reduce_minor_arrays(available_gases, &
                             gas_names, &
                             gas_minor,identifier_minor,&
                             kminor_upper, &
                             minor_gases_upper, &
                             minor_limits_gpt_upper, &
                             minor_scales_with_density_upper, &
                             scaling_gas_upper, &
                             scale_by_complement_upper, &
                             kminor_start_upper, &
                             this%kminor_upper, &
                             minor_gases_upper_red, &
                             this%minor_limits_gpt_upper, &
                             this%minor_scales_with_density_upper, &
                             scaling_gas_upper_red, &
                             this%scale_by_complement_upper, &
                             this%kminor_start_upper)

    ! Arrays not reduced by the presence, or lack thereof, of a gas
    allocate(this%press_ref(size(press_ref)), this%temp_ref(size(temp_ref)), &
             this%kmajor(size(kmajor,1),size(kmajor,2),size(kmajor,3),size(kmajor,4)))
    this%press_ref = press_ref
    this%temp_ref  = temp_ref
    this%kmajor    = kmajor
    !$acc enter data copyin(this%kmajor)


    if(allocated(rayl_lower) .neqv. allocated(rayl_upper)) then
      err_message = "rayl_lower and rayl_upper must have the same allocation status"
      return
    end if
    if (allocated(rayl_lower)) then
      allocate(this%krayl(size(rayl_lower,dim=1),size(rayl_lower,dim=2),size(rayl_lower,dim=3),2))
      this%krayl(:,:,:,1) = rayl_lower
      this%krayl(:,:,:,2) = rayl_upper
    end if

    ! ---- post processing ----
    ! creates log reference pressure
    allocate(this%press_ref_log(size(this%press_ref)))
    this%press_ref_log(:) = log(this%press_ref(:))
    !$acc enter data copyin(this%press_ref_log)


    ! log scale of reference pressure
    this%press_ref_trop_log = log(press_ref_trop)

    ! Get index of gas (if present) for determining col_gas
    call create_idx_minor(this%gas_names, gas_minor, identifier_minor, minor_gases_lower_red, &
      this%idx_minor_lower)
    call create_idx_minor(this%gas_names, gas_minor, identifier_minor, minor_gases_upper_red, &
      this%idx_minor_upper)
    ! Get index of gas (if present) that has special treatment in density scaling
    call create_idx_minor_scaling(this%gas_names, scaling_gas_lower_red, &
      this%idx_minor_scaling_lower)
    call create_idx_minor_scaling(this%gas_names, scaling_gas_upper_red, &
      this%idx_minor_scaling_upper)

    ! create flavor list
    ! Reduce (remap) key_species list; checks that all key gases are present in incoming
    call create_key_species_reduce(gas_names,this%gas_names, &
      key_species,key_species_red,key_species_present_init)
    err_message = check_key_species_present_init(gas_names,key_species_present_init)
    if(len_trim(err_message) /= 0) return
    ! create flavor list
    call create_flavor(key_species_red, this%flavor)
    ! create gpoint_flavor list
    call create_gpoint_flavor(key_species_red, this%get_gpoint_bands(), this%flavor, this%gpoint_flavor)

    ! minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    !   for T, high-to-low ordering for p
    this%temp_ref_min  = this%temp_ref (1)
    this%temp_ref_max  = this%temp_ref (size(this%temp_ref))
    this%press_ref_min = this%press_ref(size(this%press_ref))
    this%press_ref_max = this%press_ref(1)

    ! creates press_ref_log, temp_ref_delta
    this%press_ref_log_delta = (log(this%press_ref_min)-log(this%press_ref_max))/(size(this%press_ref)-1)
    this%temp_ref_delta      = (this%temp_ref_max-this%temp_ref_min)/(size(this%temp_ref)-1)

    ! Which species are key in one or more bands?
    !   this%flavor is an index into this%gas_names
    !
    if (allocated(this%is_key)) deallocate(this%is_key) ! Shouldn't ever happen...
    allocate(this%is_key(this%get_ngas()))
    this%is_key(:) = .False.
    do j = 1, size(this%flavor, 2)
      do i = 1, size(this%flavor, 1) ! extents should be 2
        if (this%flavor(i,j) /= 0) this%is_key(this%flavor(i,j)) = .true.
      end do
    end do

  end function init_abs_coeffs
  ! ----------------------------------------------------------------------------------------------------
  function check_key_species_present_init(gas_names, key_species_present_init) result(err_message)
    logical,          dimension(:), intent(in) :: key_species_present_init
    character(len=*), dimension(:), intent(in) :: gas_names
    character(len=128)                             :: err_message

    integer :: i

    err_message=''
    do i = 1, size(key_species_present_init)
      if(.not. key_species_present_init(i)) &
        err_message = ' ' // trim(gas_names(i)) // trim(err_message)
    end do
    if(len_trim(err_message) > 0) err_message = "gas_optics: required gases" // trim(err_message) // " are not provided"

  end function check_key_species_present_init
  !------------------------------------------------------------------------------------------
  !
  ! Ensure that every key gas required by the k-distribution is
  !    present in the gas concentration object
  !
  function check_key_species_present(this, gas_desc) result(error_msg)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    class(ty_gas_concs),                intent(in) :: gas_desc
    character(len=128)                             :: error_msg

    ! Local variables
    character(len=32), dimension(count(this%is_key(:)  )) :: key_gas_names
    integer                                               :: igas
    ! --------------------------------------
    error_msg = ""
    key_gas_names = pack(this%gas_names, mask=this%is_key)
    do igas = 1, size(key_gas_names)
      if(.not. string_in_array(key_gas_names(igas), gas_desc%gas_name)) &
        error_msg = ' ' // trim(lower_case(key_gas_names(igas))) // trim(error_msg)
    end do
    if(len_trim(error_msg) > 0) error_msg = "gas_optics: required gases" // trim(error_msg) // " are not provided"

  end function check_key_species_present
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Inquiry functions
  !
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return true if initialized for internal sources, false otherwise
  !
  pure function source_is_internal(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    logical                          :: source_is_internal
    source_is_internal = allocated(this%totplnk) .and. allocated(this%planck_frac_stored)
  end function source_is_internal
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return true if initialized for external sources, false otherwise
  !
  pure function source_is_external(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    logical                          :: source_is_external
    source_is_external = allocated(this%solar_source)
  end function source_is_external

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the gas names
  !
  pure function get_gases(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    character(32), dimension(get_ngas(this))     :: get_gases

    get_gases = this%gas_names
  end function get_gases
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the minimum pressure on the interpolation grids
  !
  pure function get_press_min(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp)                                       :: get_press_min

    get_press_min = this%press_ref_min
  end function get_press_min

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the maximum pressure on the interpolation grids
  !
  pure function get_press_max(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp)                                       :: get_press_max

    get_press_max = this%press_ref_max
  end function get_press_max

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the minimum temparature on the interpolation grids
  !
  pure function get_temp_min(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp)                                       :: get_temp_min

    get_temp_min = this%temp_ref_min
  end function get_temp_min

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return the maximum temparature on the interpolation grids
  !
  pure function get_temp_max(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp)                                       :: get_temp_max

    get_temp_max = this%temp_ref_max
  end function get_temp_max
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Utility function, provided for user convenience
  ! computes column amounts of dry air using hydrostatic equation
  !
  pure subroutine get_col_dry(vmr_h2o, plev, col_dry, latitude) !result(col_dry)
    ! input
    real(wp), dimension(:,:), intent(in) :: vmr_h2o  ! volume mixing ratio of water vapor to dry air; (nlay,ncol)
    real(wp), dimension(:,:), intent(in) :: plev     ! Layer boundary pressures [Pa] (nlay+1,ncol)
    real(wp), dimension(:),   optional, &
                              intent(in) :: latitude ! Latitude [degrees] (ncol)
    ! output
    real(wp), dimension(size(plev,dim=1)-1,size(plev,dim=2)), intent(out) :: col_dry ! Column dry amount (nlay,ncol)
    ! ------------------------------------------------
    ! first and second term of Helmert formula
    real(wp), parameter :: helmert1 = 9.80665_wp
    real(wp), parameter :: helmert2 = 0.02586_wp
    ! local variables
    real(wp), dimension(size(plev,dim=2)) :: g0 ! (ncol)
    real(wp):: delta_plev, m_air, fact
    integer :: ncol, nlev
    integer :: icol, ilev ! nlay = nlev-1
    ! ------------------------------------------------
    ncol = size(plev, dim=2)
    nlev = size(plev, dim=1)
    !$acc enter data create(g0)
    if(present(latitude)) then
      ! A purely OpenACC implementation would probably compute g0 within the kernel below
      !$acc parallel loop
      do icol = 1, ncol
        g0(icol) = helmert1 - helmert2 * cos(2.0_wp * pi * latitude(icol) / 180.0_wp) ! acceleration due to gravity [m/s^2]
      end do
    else
      !$acc parallel loop
      do icol = 1, ncol
        g0(icol) = grav
      end do
    end if

    !$acc parallel loop gang vector collapse(2) present(plev,vmr_h2o, col_dry)
    do icol = 1, ncol
      do ilev = 1, nlev-1
        delta_plev = abs(plev(ilev,icol) - plev(ilev+1,icol))
        ! Get average mass of moist air per mole of moist air
        fact = 1._wp / (1.+vmr_h2o(ilev,icol))
        m_air = (m_dry + m_h2o * vmr_h2o(ilev,icol)) * fact
        col_dry(ilev,icol) = 10._wp * delta_plev * avogad * fact/(1000._wp*m_air*100._wp*g0(icol))
      end do
    end do
    !$acc exit data delete (g0)
  end subroutine get_col_dry
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Compute a transport angle that minimizes flux errors at surface and TOA based on empirical fits
  !
  function compute_optimal_angles(this, optical_props, optimal_angles) result(err_msg)
    ! input
    class(ty_gas_optics_rrtmgp),  intent(in   ) :: this
    class(ty_optical_props_arry), intent(in   ) :: optical_props
    real(wp), dimension(:,:),     intent(  out)  :: optimal_angles
    character(len=128)                           :: err_msg
    !----------------------------
    integer  :: ncol, nlay, ngpt
    integer  :: icol, ilay, igpt, bnd
    real(wp) :: t, trans_total
    !----------------------------
    ncol = optical_props%get_ncol()
    nlay = optical_props%get_nlay()
    ngpt = optical_props%get_ngpt()

    err_msg=""
    if(.not. this%gpoints_are_equal(optical_props)) &
      err_msg = "gas_optics%compute_optimal_angles: optical_props has different spectral discretization than gas_optics"
    if(.not. extents_are(optimal_angles, ncol, ngpt)) &
      err_msg = "gas_optics%compute_optimal_angles: optimal_angles different dimension (ncol)"
    if (err_msg /=  "") return

    !
    ! column transmissivity
    !
    !$acc parallel loop gang vector collapse(2) copyin(optical_props, optical_props%tau, optical_props%gpt2band) copyout(optimal_angles)
    do icol = 1, ncol
      do igpt = 1, ngpt
        !
        ! Column transmissivity
        !
        t = 0._wp
        trans_total = 0._wp
        do ilay = 1, nlay
          t = t + optical_props%tau(igpt,ilay,icol)
        end do
        trans_total = exp(-t)
        !
        ! Optimal transport angle is a linear fit to column transmissivity
        !
        bnd = optical_props%gpt2band(igpt)
        optimal_angles(icol,igpt) = this%optimal_angle_fit(1,bnd)*trans_total + &
                                    this%optimal_angle_fit(2,bnd)
      end do
    end do

  end function compute_optimal_angles
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Internal procedures
  !
  !--------------------------------------------------------------------------------------------------------------------
  pure function rewrite_key_species_pair(key_species_pair)
    ! (0,0) becomes (2,2) -- because absorption coefficients for these g-points will be 0.
    integer, dimension(2) :: rewrite_key_species_pair
    integer, dimension(2), intent(in) :: key_species_pair
    rewrite_key_species_pair = key_species_pair
    if (all(key_species_pair(:).eq.(/0,0/))) then
      rewrite_key_species_pair(:) = (/2,2/)
    end if
  end function
  
  subroutine rewrite_key_species_pair_sr(key_species_pair)
      ! (0,0) becomes (2,2) -- because absorption coefficients for these g-points will be 0.
    integer, dimension(2), intent(inout) :: key_species_pair
    if (all(key_species_pair(:).eq.(/0,0/))) then
      key_species_pair(:) = (/2,2/)
    end if
  end subroutine rewrite_key_species_pair_sr

  ! ---------------------------------------------------------------------------------------
  ! true is key_species_pair exists in key_species_list
  pure function key_species_pair_exists(key_species_list, key_species_pair)
    logical                             :: key_species_pair_exists
    integer, dimension(:,:), intent(in) :: key_species_list
    integer, dimension(2),   intent(in) :: key_species_pair
    integer :: i
    do i=1,size(key_species_list,dim=2)
      if (all(key_species_list(:,i).eq.key_species_pair(:))) then
        key_species_pair_exists = .true.
        return
      end if
    end do
    key_species_pair_exists = .false.
  end function key_species_pair_exists
  ! ---------------------------------------------------------------------------------------
  ! create flavor list --
  !   an unordered array of extent (2,:) containing all possible pairs of key species
  !   used in either upper or lower atmos
  !
  subroutine create_flavor(key_species, flavor)
    integer, dimension(:,:,:), intent(in) :: key_species
    integer, dimension(:,:), allocatable, intent(out) :: flavor
    integer, dimension(2,size(key_species,3)*2) :: key_species_list

    integer :: ibnd, iatm, i, iflavor
    ! prepare list of key_species
    i = 1
    do ibnd=1,size(key_species,3)
      do iatm=1,size(key_species,1)
        key_species_list(:,i) = key_species(:,iatm,ibnd)
        i = i + 1
      end do
    end do
    ! rewrite single key_species pairs
    do i=1,size(key_species_list,2)
        call rewrite_key_species_pair_sr(key_species_list(:,i))
    end do
    ! count unique key species pairs
    iflavor = 0
    do i=1,size(key_species_list,2)
      if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then
        iflavor = iflavor + 1
      end if
    end do
    ! fill flavors
    allocate(flavor(2,iflavor))
    iflavor = 0
    do i=1,size(key_species_list,2)
      if (.not.key_species_pair_exists(key_species_list(:,1:i-1),key_species_list(:,i))) then
        iflavor = iflavor + 1
        flavor(:,iflavor) = key_species_list(:,i)
      end if
    end do
  end subroutine create_flavor
  ! ---------------------------------------------------------------------------------------
  !
  ! create index list for extracting col_gas needed for minor gas optical depth calculations
  !
  subroutine create_idx_minor(gas_names, &
    gas_minor, identifier_minor, minor_gases_atm, idx_minor_atm)
    character(len=*), dimension(:), intent(in) :: gas_names
    character(len=*), dimension(:), intent(in) :: &
                                                  gas_minor, &
                                                  identifier_minor
    character(len=*), dimension(:), intent(in) :: minor_gases_atm
    integer, dimension(:), allocatable, &
                                   intent(out) :: idx_minor_atm

    ! local
    integer :: imnr
    integer :: idx_mnr
    allocate(idx_minor_atm(size(minor_gases_atm,dim=1)))
    do imnr = 1, size(minor_gases_atm,dim=1) ! loop over minor absorbers in each band
          ! Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
          idx_mnr     = string_loc_in_array(minor_gases_atm(imnr), identifier_minor)
          ! Find name of gas associated with minor species identifier (e.g. h2o)
          idx_minor_atm(imnr) = string_loc_in_array(gas_minor(idx_mnr),    gas_names)
    enddo

  end subroutine create_idx_minor

  ! ---------------------------------------------------------------------------------------
  !
  ! create index for special treatment in density scaling of minor gases
  !
  subroutine create_idx_minor_scaling(gas_names, &
    scaling_gas_atm, idx_minor_scaling_atm)
    character(len=*), dimension(:), intent(in) :: gas_names
    character(len=*), dimension(:), intent(in) :: scaling_gas_atm
    integer, dimension(:), allocatable, &
                                   intent(out) :: idx_minor_scaling_atm

    ! local
    integer :: imnr
    allocate(idx_minor_scaling_atm(size(scaling_gas_atm,dim=1)))
    do imnr = 1, size(scaling_gas_atm,dim=1) ! loop over minor absorbers in each band
          ! This will be -1 if there's no interacting gas
          idx_minor_scaling_atm(imnr) = string_loc_in_array(scaling_gas_atm(imnr), gas_names)
    enddo

  end subroutine create_idx_minor_scaling
  ! ---------------------------------------------------------------------------------------
  subroutine create_key_species_reduce(gas_names,gas_names_red, &
    key_species,key_species_red,key_species_present_init)
    character(len=*), &
              dimension(:),       intent(in) :: gas_names
    character(len=*), &
              dimension(:),       intent(in) :: gas_names_red
    integer,  dimension(:,:,:),   intent(in) :: key_species
    integer,  dimension(:,:,:), allocatable, intent(out) :: key_species_red

    logical, dimension(:), allocatable, intent(out) :: key_species_present_init
    integer :: ip, ia, it, np, na, nt

    np = size(key_species,dim=1)
    na = size(key_species,dim=2)
    nt = size(key_species,dim=3)
    allocate(key_species_red(size(key_species,dim=1), &
                             size(key_species,dim=2), &
                             size(key_species,dim=3)))
    allocate(key_species_present_init(size(gas_names)))
    key_species_present_init = .true.

    do ip = 1, np
      do ia = 1, na
        do it = 1, nt
          if (key_species(ip,ia,it) .ne. 0) then
            key_species_red(ip,ia,it) = string_loc_in_array(gas_names(key_species(ip,ia,it)),gas_names_red)
            if (key_species_red(ip,ia,it) .eq. -1) key_species_present_init(key_species(ip,ia,it)) = .false.
          else
            key_species_red(ip,ia,it) = key_species(ip,ia,it)
          endif
        enddo
      end do
    enddo

  end subroutine create_key_species_reduce

! ---------------------------------------------------------------------------------------
  subroutine reduce_minor_arrays(available_gases, &
                           gas_names, &
                           gas_minor,identifier_minor,&
                           kminor_atm, &
                           minor_gases_atm, &
                           minor_limits_gpt_atm, &
                           minor_scales_with_density_atm, &
                           scaling_gas_atm, &
                           scale_by_complement_atm, &
                           kminor_start_atm, &
                           kminor_atm_red, &
                           minor_gases_atm_red, &
                           minor_limits_gpt_atm_red, &
                           minor_scales_with_density_atm_red, &
                           scaling_gas_atm_red, &
                           scale_by_complement_atm_red, &
                           kminor_start_atm_red)

    class(ty_gas_concs),                intent(in) :: available_gases
    character(len=*), dimension(:),     intent(in) :: gas_names
    real(wp),         dimension(:,:,:), intent(in) :: kminor_atm
    character(len=*), dimension(:),     intent(in) :: gas_minor, &
                                                      identifier_minor
    character(len=*), dimension(:),     intent(in) :: minor_gases_atm
    integer,          dimension(:,:),   intent(in) :: minor_limits_gpt_atm
    logical(wl),      dimension(:),     intent(in) :: minor_scales_with_density_atm
    character(len=*), dimension(:),     intent(in) :: scaling_gas_atm
    logical(wl),      dimension(:),     intent(in) :: scale_by_complement_atm
    integer,          dimension(:),     intent(in) :: kminor_start_atm
    real(wp),         dimension(:,:,:), allocatable, &
                                        intent(out) :: kminor_atm_red
    character(len=*), dimension(:), allocatable, &
                                        intent(out) :: minor_gases_atm_red
    integer,          dimension(:,:), allocatable, &
                                        intent(out) :: minor_limits_gpt_atm_red
    logical(wl),      dimension(:),    allocatable, &
                                        intent(out) ::minor_scales_with_density_atm_red
    character(len=*), dimension(:), allocatable, &
                                        intent(out) ::scaling_gas_atm_red
    logical(wl),      dimension(:), allocatable, intent(out) :: &
                                                scale_by_complement_atm_red
    integer,          dimension(:), allocatable, intent(out) :: &
                                                kminor_start_atm_red

    ! Local variables
    integer :: i, j, ks
    integer :: idx_mnr, nm, tot_g, red_nm
    integer :: icnt, n_elim, ng
    logical, dimension(:), allocatable :: gas_is_present
    integer, dimension(:), allocatable :: indexes

    nm = size(minor_gases_atm)
    tot_g=0
    allocate(gas_is_present(nm))
    do i = 1, size(minor_gases_atm)
      idx_mnr = string_loc_in_array(minor_gases_atm(i), identifier_minor)
      gas_is_present(i) = string_in_array(gas_minor(idx_mnr),available_gases%gas_name)
      if(gas_is_present(i)) then
        tot_g = tot_g + (minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1)
      endif
    enddo
    red_nm = count(gas_is_present)

    allocate(minor_gases_atm_red              (red_nm),&
             minor_scales_with_density_atm_red(red_nm), &
             scaling_gas_atm_red              (red_nm), &
             scale_by_complement_atm_red      (red_nm), &
             kminor_start_atm_red             (red_nm))
    allocate(minor_limits_gpt_atm_red(2, red_nm))
    allocate(kminor_atm_red(tot_g, size(kminor_atm,2), size(kminor_atm,3)))

    if ((red_nm .eq. nm)) then
      ! Character data not allowed in OpenACC regions?
      minor_gases_atm_red         = minor_gases_atm
      scaling_gas_atm_red         = scaling_gas_atm
      kminor_atm_red              = kminor_atm
      minor_limits_gpt_atm_red    = minor_limits_gpt_atm
      minor_scales_with_density_atm_red = minor_scales_with_density_atm
      scale_by_complement_atm_red = scale_by_complement_atm
      kminor_start_atm_red        = kminor_start_atm
    else
      allocate(indexes(red_nm))
      ! Find the integer indexes for the gases that are present
      indexes = pack([(i, i = 1, size(minor_gases_atm))], mask=gas_is_present)

      minor_gases_atm_red  = minor_gases_atm        (indexes)
      scaling_gas_atm_red  = scaling_gas_atm        (indexes)
      minor_scales_with_density_atm_red = &
                             minor_scales_with_density_atm(indexes)
      scale_by_complement_atm_red = &
                             scale_by_complement_atm(indexes)
      kminor_start_atm_red = kminor_start_atm       (indexes)

      icnt = 0
      n_elim = 0
      do i = 1, nm
        ng = minor_limits_gpt_atm(2,i)-minor_limits_gpt_atm(1,i)+1
        if(gas_is_present(i)) then
          icnt = icnt + 1
          minor_limits_gpt_atm_red(1:2,icnt) = minor_limits_gpt_atm(1:2,i)
          kminor_start_atm_red(icnt) = kminor_start_atm(i)-n_elim
          ks = kminor_start_atm_red(icnt)
          do j = 1, ng
            kminor_atm_red(kminor_start_atm_red(icnt)+j-1,:,:) = &
              kminor_atm(kminor_start_atm(i)+j-1,:,:)
          enddo
        else
          n_elim = n_elim + ng
        endif
      enddo
    endif
    !$acc enter data copyin(kminor_atm_red)

  end subroutine reduce_minor_arrays

! ---------------------------------------------------------------------------------------
  ! returns flavor index; -1 if not found
  pure function key_species_pair2flavor(flavor, key_species_pair)
    integer :: key_species_pair2flavor
    integer, dimension(:,:), intent(in) :: flavor
    integer, dimension(2), intent(in) :: key_species_pair
    integer :: iflav
    do iflav=1,size(flavor,2)
      if (all(key_species_pair(:).eq.flavor(:,iflav))) then
        key_species_pair2flavor = iflav
        return
      end if
    end do
    key_species_pair2flavor = -1
  end function key_species_pair2flavor

  ! ---------------------------------------------------------------------------------------
  !
  ! create gpoint_flavor list
  !   a map pointing from each g-point to the corresponding entry in the "flavor list"
  !
  subroutine create_gpoint_flavor(key_species, gpt2band, flavor, gpoint_flavor)
    integer, dimension(:,:,:), intent(inout) :: key_species
    integer, dimension(:), intent(in) :: gpt2band
    integer, dimension(:,:), intent(in) :: flavor
    integer, dimension(:,:), intent(out), allocatable :: gpoint_flavor
    integer :: ngpt, igpt, iatm
    ngpt = size(gpt2band)
    allocate(gpoint_flavor(2,ngpt))
    do igpt=1,ngpt
      do iatm=1,2
        call rewrite_key_species_pair_sr(key_species(:,iatm,gpt2band(igpt)))
        gpoint_flavor(iatm,igpt) = key_species_pair2flavor( flavor, key_species(:,iatm,gpt2band(igpt)) )
        ! gpoint_flavor(iatm,igpt) = key_species_pair2flavor( flavor, &
        !    rewrite_key_species_pair(key_species(:,iatm,gpt2band(igpt))) )
      end do
    end do
  end subroutine create_gpoint_flavor

  !--------------------------------------------------------------------------------------------------------------------
 !
 ! Utility function to combine optical depths from gas absorption and Rayleigh scattering
 !
 subroutine combine(tau_rayleigh, has_rayleigh, optical_props)
  real(wp), dimension(:,:,:),   intent(in) :: tau_rayleigh
  logical,                      intent(in) :: has_rayleigh
  class(ty_optical_props_arry), intent(inout) :: optical_props

  integer :: ncol, nlay, ngpt, nmom, igpt

  ncol = size(optical_props%tau, 3)
  nlay = size(optical_props%tau, 2)
  ngpt = size(optical_props%tau, 1)

  if (.not. has_rayleigh) then
    select type(optical_props)
      type is (ty_optical_props_1scl)
        ! do nothing
      type is (ty_optical_props_2str)
        !$acc enter data create(optical_props%ssa, optical_props%g)
        call zero_array(     ngpt,nlay,ncol,optical_props%ssa)
        call zero_array(     ngpt,nlay,ncol,optical_props%g  )
        !$acc exit data copyout(optical_props%ssa, optical_props%g)
      type is (ty_optical_props_nstr) ! We ought to be able to combine this with above
        nmom = size(optical_props%p, 1)
        !$acc enter data create(optical_props%ssa, optical_props%p)
        call zero_array(     ngpt,nlay,ncol,optical_props%ssa)
        call zero_array(nmom,ngpt,nlay,ncol,optical_props%p  )
        !$acc exit data copyout(optical_props%ssa, optical_props%p)
      end select
  else
    ! combine optical depth and rayleigh scattering
    select type(optical_props)
      type is (ty_optical_props_1scl)
        ! User is asking for absorption optical depth
        ! do nothing
      type is (ty_optical_props_2str)
        call combine_2str(ncol, nlay, ngpt,        tau_rayleigh, &
                                      optical_props%tau, optical_props%ssa, optical_props%g)
                        
      type is (ty_optical_props_nstr) ! We ought to be able to combine this with above
        nmom = size(optical_props%p, 1)
        call combine_nstr(ncol, nlay, ngpt, nmom, tau_rayleigh, &
                                      optical_props%tau, optical_props%ssa, optical_props%p)
    end select
  end if
end subroutine combine


  !--------------------------------------------------------------------------------------------------------------------
  ! Sizes of tables: pressure, temperate, eta (mixing fraction)
  !   Equivalent routines for the number of gases and flavors (get_ngas(), get_nflav()) are defined above because they're
  !   used in function defintions
  ! Table kmajor has dimensions (ngpt, neta, npres, ntemp)
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return extent of eta dimension
  !
  pure function get_neta(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                          :: get_neta

    get_neta = size(this%kmajor,dim=2)
  end function
  ! --------------------------------------------------------------------------------------
  !
  ! return the number of pressures in reference profile
  !   absorption coefficient table is one bigger since a pressure is repeated in upper/lower atmos
  !
  pure function get_npres(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                          :: get_npres

    get_npres = size(this%kmajor,dim=3)-1
  end function get_npres
  ! --------------------------------------------------------------------------------------
  !
  ! return the number of temperatures
  !
  pure function get_ntemp(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                          :: get_ntemp

    get_ntemp = size(this%kmajor,dim=4)
  end function get_ntemp
  ! --------------------------------------------------------------------------------------
  !
  ! return the number of temperatures for Planck function
  !
  pure function get_nPlanckTemp(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    integer                          :: get_nPlanckTemp

    get_nPlanckTemp = size(this%totplnk,dim=1) ! dimensions are Planck-temperature, band
  end function get_nPlanckTemp

  function mean2(x) result(mean)
    real(sp), dimension(:,:), intent(in) :: x
    real(sp) :: mean
    
    mean = sum(sum(x, dim=1),dim=1) / size(x)

  end function mean2

  function mean_3d(x) result(mean3)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x
    real(wp) :: mean3
    mean3 = sum(x) / size(x)
  end function mean_3d
end module mo_gas_optics_rrtmgp

