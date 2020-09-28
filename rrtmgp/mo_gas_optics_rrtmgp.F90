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
  use mo_rte_config,         only: check_extents, check_values
  use mo_rte_util_array,     only: zero_array, any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props,      only: ty_optical_props
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_optics_kernels, only: interpolation,                                         &
                                   compute_tau_absorption, compute_tau_rayleigh,          &
                                   combine_2str, combine_nstr,                            &
                                   compute_source_bybnd_pfrac_bygpt, compute_source_bybnd, &
                                   predict_nn_lw_blas, predict_nn_sw_blas
  use mo_rrtmgp_constants,   only: avogad, m_dry, m_h2o, grav
  use mo_rrtmgp_util_string, only: lower_case, string_in_array, string_loc_in_array
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_gas_ref_concentrations, only: get_ref_vmr
  use mo_optical_props,      only: ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_gas_optics,         only: ty_gas_optics
  use mo_rrtmgp_util_reorder
  use mod_network
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
    real(wp), dimension(:), allocatable :: solar_source         ! incoming solar irradiance, computed from other three terms (g-point)
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
                          optical_props, sources,          &
                          col_dry, tlev, neural_nets       &
#ifdef DEV_MODE
                          ,nn_inputs, col_dry_arr          &
#endif
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
    type(network_type), dimension(2), intent(in), optional      :: neural_nets ! Planck fraction model, optical depth model                                
    ! Outputs for neural network model development
#ifdef DEV_MODE
    real(sp), dimension(:,:,:), intent(inout)         :: nn_inputs
    real(sp), dimension(:,:),   intent(inout), target :: col_dry_arr
#else
    real(sp), dimension(:,:,:), allocatable           :: nn_inputs
    real(wp), dimension(size(play,dim=1), size(play,dim=2)), &
                                              target  :: col_dry_arr
#endif
    real(wp), dimension(:,:),   contiguous, pointer   :: col_dry_wk => NULL()
    ! ----------------------------------------------------------
    ! Local variables
    ! real(wp), dimension(:,:),   allocatable, target   :: col_dry_arr
    ! real(wp), dimension(:,:),   pointer               :: col_dry_wk => NULL()
    real(wp), dimension(size(play,dim=1)+1, size(play,dim=2)), &
                               target         :: tlev_arr
    real(wp), dimension(:,:),   pointer       :: tlev_wk

    integer :: ncol, nlay, ngpt, nband, ngas, ninputs, nflav, icol, ilay, idx_h2o
    ! ----------------------------------------------------------
    ncol  = size(play,dim=2)
    nlay  = size(play,dim=1)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()

    ! OpenACC: copy everything in here, remove in the end?
    !$acc enter data copyin(tlay, tlev, tsfc, plev, play) 

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
      if(any_vals_outside(plev, this%press_ref_min,this%press_ref_max)) error_msg = "gas_optics(): array plev has values outside range"
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
    !$acc enter data attach(tlev_wk)

    !
    ! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
    !
    if (present(col_dry)) then
      !$acc enter data copyin(col_dry)
      col_dry_wk => col_dry
    else
      !$acc enter data create(col_dry_arr)

      ! idx_h2o needs to be obtained from gas_desc%gas_names and NOT this%gas_names, since these 
      ! can be in different order.
      ! It's confusing that the gas optics class and gas concentration class both have their own
      ! gas name array. Probably a good reason for it? If not, the gas optics name array should be removed.

      !idx_h2o = string_loc_in_array('h2o', this%gas_names)
      idx_h2o = string_loc_in_array('h2o', gas_desc%gas_name)
      ! print *, "GAS NAMES, this-gas-names:" ,this%gas_names
      ! print *, "GAS NAMES, gas_desc%gas_name ", gas_desc%gas_name
      ! print *, "idx_h2o", idx_h2o

      ! NOTE: The above change was necessary since in the NN code gas concentrations are accessed directly from gas_conc 
      ! like below, instead of the original method filling a 2D vmr array for each gas like gas_array(idx_gas,:,:) = get_vmr
      ! This is much faster (input preprocessing previously had a significant cost at small block sizes)
            
      ! Original code:
      ! if (any (lower_case(this%gas_names(igas)) == gas_desc%gas_name(:))) then
      !   error_msg = gas_desc%get_vmr(this%gas_names(igas), vmr(:,:,igas))
      !   if (error_msg /= '') return
      ! endif
      call get_col_dry(gas_desc%concs(idx_h2o)%conc, plev, col_dry_arr)
      col_dry_wk => col_dry_arr

    end if
    !$acc enter data attach(col_dry_wk)

    !
    ! Gas optics
    !
#ifdef USE_TIMING
      ret =  gptlstart('compute_gas_opticss')
#endif

    if (present(neural_nets)) then
      ! ----------------------------------------------------------------------------------
      ! Use neural network for gas optics

      call  compute_source_bybnd(                                     &
            ncol, nlay, nband,                                        &
            this%get_ntemp(),this%get_nPlanckTemp(),                  &
            tlay, tlev_wk , tsfc,                                     &
            this%temp_ref_min, this%totplnk_delta, this%totplnk,      &
            sources%sfc_source_bnd, sources%sfc_source_bnd_Jac,       &
            sources%lay_source_bnd, sources%lev_source_bnd)  

      ninputs =  size(neural_nets(1) % layers(1) % w_transposed, 2)
#ifndef DEV_MODE
      allocate(nn_inputs(ninputs,nlay,ncol))
#endif
      !$acc enter data create(nn_inputs)

      error_msg = compute_nn_inputs(this,             &
                          ncol, nlay, ngas, ninputs,  &
                          play, tlay, gas_desc,       &
                          nn_inputs)
                          
#ifdef USE_TIMING
    ret =  gptlstart('predict_nn_lw_blas')
#endif
      call predict_nn_lw_blas(              &
              ncol, nlay, ngpt, ninputs,    &  ! dimensions
              nn_inputs, col_dry_wk,        &  ! data inputs
              neural_nets,                  &  ! NN models (input)
              optical_props%tau, sources%planck_frac)    ! outputs    
      !$acc exit data delete(nn_inputs) 
#ifdef USE_TIMING
    ret =  gptlstop('predict_nn_lw_blas')
#endif
              
    else

      ! ----------------------------------------------------------------------------------
      ! Use interpolation routine for gas optics, NOT neural network
      error_msg = compute_gas_optics(this,                         &
                                   ncol, nlay, ngpt, nband,                 &
                                   play, plev, tlay, gas_desc, col_dry_wk,  &
                                   optical_props,                           &
                                   sources, tlev_wk, tsfc)
      if(error_msg  /= '') return
      ! ----------------------------------------------------------
      ! Use this code block to compute nn inputs for model development, commented out as default
#ifdef DEV_MODE
      error_msg = compute_nn_inputs(this,                         &
                      ncol, nlay, ngas, size(nn_inputs,dim=1),    &
                      play, tlay, gas_desc,  nn_inputs)   
      if(error_msg  /= '') return  
#endif
      ! ----------------------------------------------------------

      ! ----------------------------------------------------------------------------------
    end if

#ifdef USE_TIMING
    ret =  gptlstop('compute_gas_opticss')
#endif
    !$acc exit data delete(col_dry_arr, tlay, tlev, tlev_arr, tsfc, plev, play) detach(tlev_wk, col_dry_wk)

  end function gas_optics_int
  !------------------------------------------------------------------------------------------
  !
  ! Compute gas optical depth given temperature, pressure, and composition
  !
  function gas_optics_ext(this,                         &
                          play, plev, tlay, gas_desc,   & ! mandatory inputs
                          optical_props, toa_src,       & ! mandatory outputs
                          col_dry, neural_nets) result(error_msg)      ! optional input

    class(ty_gas_optics_rrtmgp),  intent(in) :: this
    real(wp), dimension(:,:),     intent(in) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                               plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                               tlay      ! layer temperatures [K]; (nlay,ncol)
    type(ty_gas_concs),           intent(in) :: gas_desc  ! Gas volume mixing ratios
    ! output
    class(ty_optical_props_arry),  &
                                  intent(inout) :: optical_props
    real(wp), dimension(:,:),     intent(  out) :: toa_src     ! Incoming solar irradiance(ncol,ngpt)
    character(len=128)                      :: error_msg

    ! Optional inputs
    real(wp), dimension(:,:),     intent(in   ), &
                           optional, target :: col_dry ! Column dry amount; dim(nlay,ncol)
    ! Optional input: neural network model (uses NN kernel if present)
    type(network_type), dimension(2), intent(in), optional      :: neural_nets ! Planck fraction model, optical depth model      

    ! Optional outputs (for neural network model development)
    ! real(sp), dimension(:,:,:), intent(inout), optional     :: nn_inputs
    ! real(sp), dimension(:,:),   intent(inout), optional     :: col_dry_arr
    ! ----------------------------------------------------------
    ! Local variables
    ! real(wp), dimension(:,:),   allocatable, target   :: col_dry_arr
    real(wp), dimension(size(play,dim=1), size(play,dim=2)), &
                                                target  :: col_dry_arr
    real(wp), dimension(:,:),   pointer, contiguous     :: col_dry_wk => NULL()
    real(sp), dimension(:,:,:), allocatable   :: nn_inputs

    integer :: ncol, nlay, ngpt, nband, ngas, idx_h2o, ninputs
    integer :: igpt, icol
    ! ----------------------------------------------------------
    nlay  = size(play,dim=1)
    ncol  = size(play,dim=2)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()
    ngas  = this%get_ngas()

      ! OpenACC: copy everything in here, remove in the end?
    !$acc enter data copyin(tlay, plev, play) 

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
      if(any_vals_outside(plev, this%press_ref_min,this%press_ref_max))  error_msg = "gas_optics(): array plev has values outside range"
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
      !$acc enter data create(col_dry_arr)
      !idx_h2o = string_loc_in_array('h2o', this%gas_names)
      idx_h2o = string_loc_in_array('h2o', gas_desc%gas_name)
      call get_col_dry(gas_desc%concs(idx_h2o)%conc, plev, col_dry_arr)
      col_dry_wk => col_dry_arr
    end if
    !$acc enter data attach(col_dry_wk)

    !
    ! Gas optics
    !
#ifdef USE_TIMING
    ret =  gptlstart('compute_gas_taus')
#endif

    if (present(neural_nets)) then
    ! ----------------------------------------------------------------------------------
    ! Use neural network for gas optics

      ninputs =  size(neural_nets(1) % layers(1) % w_transposed, 2)
      allocate(nn_inputs(ninputs,nlay,ncol))
      !$acc enter data create(nn_inputs)

      error_msg = compute_nn_inputs(this,             &
                          ncol, nlay, ngas, ninputs,  &
                          play, tlay, gas_desc,       &
                          nn_inputs)   

      select type(optical_props)
        type is (ty_optical_props_1scl)
          ! User is asking for absorption optical depth
          ! do nothing

        type is (ty_optical_props_2str)

          call predict_nn_sw_blas(              &
                  ncol, nlay, ngpt, ninputs,    &  ! dimensions
                  nn_inputs, col_dry_wk,        &  ! data inputs
                  neural_nets,                  &  ! NN models (input)
                  optical_props%tau, optical_props%ssa)    ! outputs    
#ifdef USE_TIMING
    ret =  gptlstart('set_g_to_zero')
#endif
          optical_props%g = 0.0_wp
#ifdef USE_TIMING
    ret =  gptlstop('set_g_to_zero')
#endif
      end select
      !$acc exit data delete(nn_inputs) 
     
    else
    ! ----------------------------------------------------------------------------------
    ! Use interpolation routine for gas optics, NOT neural network

      error_msg = compute_gas_optics(this,                     &
                                  ncol, nlay, ngpt, nband,             &
                                  play, plev, tlay, gas_desc, col_dry_wk, &
                                  optical_props)
    end if

#ifdef USE_TIMING
    ret =  gptlstop('compute_gas_taus')
#endif
    if(error_msg  /= '') return

    ! if(save_inputs)) then
    !   error_msg = compute_nn_inputs(this,                         &
    !                   ncol, nlay, ngas, size(nn_inputs,dim=1),    &
    !                   play, tlay, gas_desc,  nn_inputs)   

    !   if(error_msg  /= '') return  
    ! end if

    ! ----------------------------------------------------------
    !
    ! External source function is constant
    !
    !$acc enter data create(toa_src)
    if(.not. extents_are(toa_src, ngpt, ncol)) &
      error_msg = "gas_optics(): array toa_src has wrong size"
    if(error_msg  /= '') return

    !$acc parallel loop collapse(2)
    do icol = 1,ncol
       do igpt = 1,ngpt
          toa_src(igpt,icol) = this%solar_source(igpt)
       end do
    end do
    !$acc exit data copyout(toa_src) delete(col_dry_arr, tlay, play, plev, plev) detach(col_dry_wk)
  end function gas_optics_ext

!------------------------------------------------------------------------------------------
! Routine for preparing neural network inputs from the gas concentrations, temperature and pressure
! The model needs all 16 RRTMGP long-wave gases as input. If a gas is missing, a global-mean reference concentration
! is used which can be either pre-industrial, present, or future
  function compute_nn_inputs(this,        &
    ncol, nlay, ngas, ninputs,            &
    play, tlay, gas_desc,           &
    nn_inputs) result(error_msg)

    class(ty_gas_optics_rrtmgp),          intent(in   ) ::  this
    integer,                              intent(in   ) ::  ncol, nlay, ngas, ninputs
    real(wp), dimension(nlay,ncol),       intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                            tlay
    type(ty_gas_concs),                   intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  

    real(sp), dimension(ninputs, nlay, ncol),  intent(inout) :: nn_inputs !

    character(len=128)                                  :: error_msg
 
    ! ----------------------------------------------------------
    ! Local variables
    integer :: igas, ilay, idx_gas, icol, ndims,idx_h2o,idx_o3 
    real(wp), dimension(nlay, ncol)           :: gas_array
    ! Handle missing gases: 
    ! Reference gas concentrations are stored in rrtmgp_ref_concentrations for each greenhouse gas except H2O and O3,
    ! for three different scenarios (present-day, pre-industrial or future). If a given gas in nn_gas_names was not provided by user, 
    ! one of the three reference concentrations is used (default: present-day)
    character(len=32)                         :: gas_name 
    integer                                   :: scenario_index     = 1 ! =  1 (zero concentration), 2 (Present-day), 3 (Pre-industrial) or 4 (Future)
    logical                                   :: print_warnings     = .false.
    logical                                   :: all_gases_exist    = .false.
    character(1) :: a_string
    character(18), dimension(4)  :: scenario_names = &
        [character(len=18) :: 'zero concentration', 'present-day', 'pre-industrial', 'future']   
    character(32), dimension(16)        :: nn_gas_names_all = [character(len=32)  :: 'h2o',   'o3',      'co2',    'n2o',   'ch4',   &
    'cfc11', 'cfc12', 'co',  'ccl4',  'cfc22',  'hfc143a', 'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4'] 
    ! ----------------------------------------------------------
    ! Neural network inputs must be preprocessed using min-max (0,1) normalization
    ! The inputs are:   tlay,    log(play),   h2o**(1/4), o3**(1/4), co2, ... 
    real(sp), dimension(18)   :: input_minvals_all =  (/ 1.60E2, 5.15E-3, 1.01E-2, 4.36E-3, 1.41E-4, 0.00E0, 2.55E-8, 0.00E0, 0.00E0, &
    0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0 /)
    real(sp), dimension(18)   :: input_maxvals_all =  (/ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3, 5.8135214E-7, & 
    3.6000001E-6, 2.0000002E-9, 5.3385213E-10, 1.3127458E-6, 1.0316801E-10, 2.3845328E-10, &
     7.7914392E-10, 9.8880004E-10, 3.1067642E-11, 1.3642075E-11, 4.2330001E-10, 1.6702625E-10 /)
    character(32),  dimension(ninputs-2) :: nn_gas_names
    real(sp),       dimension(ninputs)   :: input_maxvals, input_minvals

    all_gases_exist = .false.

    error_msg = ''
    ! Check for initialization
    if (.not. this%is_initialized()) then
      error_msg = 'ERROR: spectral configuration not loaded'
      return
    end if
    !
    ! Check input data sizes and values
    if(.not. extents_are(play, nlay, ncol  )) &
    error_msg = "gas_optics(): array play has wrong size"
    if(.not. extents_are(tlay, nlay, ncol  )) &
    error_msg = "gas_optics(): array tlay has wrong size"
    if(error_msg  /= '') return

    if(any_vals_outside(play, this%press_ref_min,this%press_ref_max)) &
    error_msg = "gas_optics(): array play has values outside range"
    if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max)) &
    error_msg = "gas_optics(): array tlay has values outside range"
    if(error_msg  /= '') return

    if (ninputs == 18) then         !  tlay,    log(play),   h2o**(1/4), o3**(1/4), co2, ... 
      if (print_warnings) print *, "using more complex neural network which takes all 16 non-constant RRTMGP longwave gases as input"
      nn_gas_names  = nn_gas_names_all
      input_minvals = input_minvals_all
      input_maxvals = input_maxvals_all
    else if (ninputs == 9) then
      if (print_warnings) print *, "using less complex neural network which only uses h2o, o3, co2, n2o, ch4, cfc11-EQ and cfc12"
      nn_gas_names    = nn_gas_names_all(1:7)
      input_minvals   = input_minvals_all(1:9)               
      input_maxvals   = input_maxvals_all(1:9)
    else if (ninputs == 7) then
      if (print_warnings) print *, "using short-wave neural network which only uses h2o, o3, co2, n2o, ch4"
      nn_gas_names    = nn_gas_names_all(1:5)
      input_minvals   = input_minvals_all(1:7)               
      input_maxvals   = input_maxvals_all(1:7)
    else 
      error_msg = "ninputs should be either 18 (full longwave model), 9 (reduced longwave model ala CKDMIP using CFC11-eq) or 7 (shortwave)"
    end if

    if(error_msg  /= '') return

#ifdef USE_TIMING
    ret =  gptlstart('compute_nn_inputs')
#endif

    if (all_gases_exist) then
      !$acc enter data copyin(nn_gas_names, input_maxvals, input_minvals)
      !$acc data present(tlay,play,gas_desc,nn_inputs,input_maxvals,input_minvals)
      error_msg = gas_desc%get_conc_dims_and_igas(nn_gas_names(1), ndims, idx_h2o)
      error_msg = gas_desc%get_conc_dims_and_igas(nn_gas_names(2), ndims, idx_o3)
      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do ilay = 1, nlay
            nn_inputs(1,ilay,icol)    =  (tlay(ilay,icol)     - input_minvals(1)) / (input_maxvals(1) - input_minvals(1))
            nn_inputs(2,ilay,icol)    = (log(play(ilay,icol)) - input_minvals(2)) / (input_maxvals(2) - input_minvals(2))
            nn_inputs(3,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_h2o)%conc(ilay,icol))) - input_minvals(3)) / (input_maxvals(3) - input_minvals(3))
            nn_inputs(4,ilay,icol)    = ( sqrt(sqrt(gas_desc%concs(idx_o3) %conc(ilay,icol))) - input_minvals(4)) / (input_maxvals(4) - input_minvals(4))
        end do
      end do
  
      do igas = 5, ninputs
        error_msg = gas_desc%get_conc_dims_and_igas(nn_gas_names(igas-2), ndims, idx_gas)
        
        if (ndims == 0) then
          !$acc parallel loop collapse(2)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(1,1)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        else if (ndims == 1) then
          !$acc parallel loop collapse(2)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(ilay,1)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        else 
          !$acc parallel loop collapse(2)
          do icol = 1, ncol
            do ilay = 1, nlay
                nn_inputs(igas,ilay,icol)    =  (gas_desc%concs(idx_gas)%conc(ilay,icol)  - input_minvals(igas)) / (input_maxvals(igas) - input_minvals(igas))
            end do
          end do
        end if 
      end do
      !$acc end data
      !$acc exit data delete(nn_gas_names, input_maxvals, input_minvals)
      if(error_msg  /= '') return

    else 
      !$acc enter data create(gas_array)
      !$acc enter data copyin(nn_gas_names, input_maxvals, input_minvals)

      !$acc kernels present(nn_inputs, tlay,play, input_maxvals, input_minvals)
      nn_inputs(1,:,:) =  (tlay(:,:)        -  input_minvals(1) ) / (input_maxvals(1) - input_minvals(1))
      nn_inputs(2,:,:) =  (log(play(:,:))    -  input_minvals(2) ) / (input_maxvals(2) - input_minvals(2))
      !$acc end kernels

      do igas = 1, ninputs-2
        ! Get the 2D array with the gas concentration for this gas
        ! print *, "igas", igas, "nn_gas_names", nn_gas_names(igas)
        error_msg = gas_desc%get_vmr(nn_gas_names(igas), gas_array)

        ! If not successful, the gas was not provided, and we need to use a reference concentration
        if (error_msg /= '') then 
          if (scenario_index==1) then
            gas_array = 0.0_wp
            error_msg = ''
          else
            error_msg = get_ref_vmr(scenario_index, nn_gas_names(igas), gas_array) 
          end if 
          if (print_warnings) then
            print *, 'WARNING: Neural network uses the gas '// trim(nn_gas_names(igas)) //  ' as input but it was not provided'
            write(a_string,'(i1)') scenario_index
            print *, 'Scenario_index in gas_optics_rrtmgp was set to ' // a_string // ' (' // scenario_names(scenario_index) // &
            '), using a constant reference concentration of:', gas_array(1,1) 
          end if 
        end if

#ifdef USE_TIMING
    ret =  gptlstart('nn_inputs_write')
#endif
        !$acc kernels present(nn_inputs, gas_array, input_maxvals, input_minvals)
        
        ! nn_inputs(igas+3,:,:) = gas_array(:,:)

        if ((nn_gas_names(igas) == 'h2o') .or. (nn_gas_names(igas) == 'o3')) then
          gas_array(:,:) = sqrt(sqrt(gas_array(:,:)))
        end if
        nn_inputs(igas+2,:,:) =  (gas_array(:,:)        -  input_minvals(igas+2) ) / (input_maxvals(igas+2) - input_minvals(igas+2))

#ifdef USE_TIMING
    ret =  gptlstop('nn_inputs_write')
#endif
        !$acc end kernels
      end do
      !$acc exit data delete(nn_gas_names, input_maxvals, input_minvals, gas_array)
      if(error_msg  /= '') return
    end if

    ! do igas = 1, ninputs
    !   print *, 'Neural network inputs: max of', igas, ":", maxval(nn_inputs(igas,:,:))
    ! end do

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
    real(wp),   dimension(nlay,ncol), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                       plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                                       tlay      ! layer temperatures [K]; (nlay,ncol)
    type(ty_gas_concs),               intent(in   ) :: gas_desc  ! Gas volume mixing ratios
    real(wp), dimension(nlay,ncol),   intent(in   ) :: col_dry ! Column dry amount; dim(nlay,ncol)
    class(ty_optical_props_arry),     intent(inout) :: optical_props !inout because components are allocated
    ! Optional inputs used for long-wave
    class(ty_source_func_lw    ),     intent(inout), &
                                                  optional :: sources       ! Planck sources
    real(wp), dimension(nlay+1,ncol), intent(in), optional :: tlev
    real(wp), dimension(ncol       ), intent(in), optional :: tsfc  

    ! Interpolation coefficients for use in internal source function
    integer,     dimension(                      nlay, ncol)  :: jtemp, jpress
    integer,     dimension(2,    get_nflav(this),nlay, ncol)  :: jeta
    logical(wl), dimension(                      nlay, ncol)  :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),nlay, ncol)  :: fmajor
    character(len=128)                                         :: error_msg
    ! ----------------------------------------------------------
    ! Local variables
    real(wp), dimension(:,:,:), allocatable           :: tau_rayleigh  ! absorption, Rayleigh scattering optical depths
    real(wp), dimension(nlay, ncol, this%get_ngas())             :: vmr     ! volume mixing ratios
    !
    ! Interpolation variables used in major gas but not elsewhere, so don't need exporting
    !
    real(wp), dimension(nlay,ncol,0:this%get_ngas())  :: col_gas ! column amounts for each gas (Num. of molecules per cm^2), plus col_dry
    real(wp), dimension(2,    get_nflav(this),nlay,ncol) :: col_mix ! combination of major species's column amounts
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

    !$acc enter data create(jtemp, jpress, tropo, fmajor, jeta, col_gas)

    !
    ! compute column gas amounts [molec/cm^2]
    !
  
    !$acc enter data create(vmr)
    do igas = 1, ngas
      !
      ! Get vmr if  gas is provided in ty_gas_concs
      !
      if (any (lower_case(this%gas_names(igas)) == gas_desc%gas_name(:))) then
        error_msg = gas_desc%get_vmr(this%gas_names(igas), vmr(:,:,igas))
        if (error_msg /= '') return
      endif
    end do
  
    !$acc parallel loop gang vector collapse(2) present(col_dry)
    do icol = 1, ncol
      do ilay = 1, nlay
        col_gas(ilay,icol,0) = col_dry(ilay,icol)
      end do
    end do
    
    !$acc parallel loop gang vector collapse(3) present(col_gas, vmr, col_dry)
    do igas = 1, ngas
      do icol = 1, ncol
        do ilay = 1, nlay
          col_gas(ilay,icol,igas) = vmr(ilay,icol,igas) * col_dry(ilay,icol)
        end do
      end do
    end do
    !$acc exit data delete(vmr)

    !
    ! ---- calculate gas optical depths ----
    !
    !$acc enter data create(col_mix, fminor)
    !$acc enter data copyin(this)
    !$acc enter data copyin(this%gpoint_flavor)
    !$acc enter data copyin(this%kmajor)

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
    !$acc exit data delete(this%kmajor)
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
            idx_h2o, col_dry ,col_gas, &
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

    !$acc exit data delete(col_mix, fminor)

    if(present(sources)) then
#ifdef USE_TIMING
    ret =  gptlstart('compute_source')
#endif
    ! Interpolate per-band source function at levels and layers, and per-g-point planck fraction at layers
    ! This reduces the size of the arrays going into the solver.
    ! The g-point source functions at layers and levels are instead calculated in-place inside the solver,
    ! saving memory (furthermore another solver might not need upward and downward sources at both layers and levels)
      call compute_source_bybnd_pfrac_bygpt(ncol, nlay, nband, ngpt,  &
      nflav, neta, npres, ntemp, this%get_nPlanckTemp(),              &
      tlay, tlev, tsfc,                                            &
      fmajor, jeta, tropo, jtemp, jpress,                             &
      this%get_gpoint_bands(), this%get_band_lims_gpoint(), this%temp_ref_min,        &
      this%totplnk_delta, this%planck_frac_stored, this%totplnk, this%gpoint_flavor,  &
      sources%sfc_source_bnd, sources%sfc_source_bnd_Jac, sources%lay_source_bnd, sources%lev_source_bnd, sources%planck_frac)
#ifdef USE_TIMING
    ret =  gptlstop('compute_source')
#endif
    end if
    if (error_msg /= '') return

    !$acc exit data delete(this%gpoint_flavor)
    !$acc exit data delete(jtemp, jpress, jeta, tropo, fmajor, col_gas)
    
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
end module mo_gas_optics_rrtmgp

