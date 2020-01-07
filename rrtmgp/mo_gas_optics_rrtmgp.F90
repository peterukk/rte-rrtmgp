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
  use mo_rte_kind,           only: wp, wl, dp
  use mo_rrtmgp_constants,   only: avogad, m_dry, m_h2o, grav
  use mo_rte_util_array,     only: zero_array, any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props,      only: ty_optical_props
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_optics_kernels, only: interpolation,                                                       &
                                   compute_tau_absorption, compute_tau_rayleigh, compute_Planck_source, &
                                   combine_and_reorder_2str, combine_and_reorder_nstr,  &
                                   compute_Planck_source_pfracin, predict_nn_lw, predict_nn_lw_blas, &
                                   gas_optical_depths_major

  use mo_rrtmgp_util_string, only: lower_case, string_in_array, string_loc_in_array
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_optical_props,      only: ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_gas_optics,         only: ty_gas_optics
  use mo_rrtmgp_util_reorder
  use mod_network
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
    real(wp), dimension(:,:,:,:), allocatable :: planck_frac   ! stored fraction of Planck irradiance in band for given g-point
                                                               ! planck_frac(g-point, eta, pressure, temperature)
    real(wp), dimension(:,:),     allocatable :: totplnk       ! integrated Planck irradiance by band; (Planck temperatures,band)
    real(wp)                                  :: totplnk_delta ! temperature steps in totplnk
    ! -----------------------------------------------------------------------------------
    ! Solar source function spectral mapping
    !   Allocated only when gas optics object is external-source
    !
    real(wp), dimension(:), allocatable :: solar_src ! incoming solar irradiance(g-point)
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
    ! Internal procedures
    procedure, private :: load_int
    procedure, private :: load_ext
    procedure, public  :: gas_optics_int
    procedure, public  :: gas_optics_int_nn
    procedure, public  :: gas_optics_ext
    procedure, private :: check_key_species_present
    procedure, private :: get_minor_list
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
    integer                                        :: get_ngas

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
                          col_dry, tlev) result(error_msg)
    ! inputs
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp), dimension(:,:), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (ncol,nlay)
                                               plev, &   ! level pressures [Pa, mb]; (ncol,nlay+1)
                                               tlay      ! layer temperatures [K]; (ncol,nlay)
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
                           optional, target :: col_dry, &  ! Column dry amount; dim(ncol,nlay)
                                               tlev        ! level temperatures [K]; (ncol,nlay+1)
    ! ----------------------------------------------------------
    ! Local variables
    ! Interpolation coefficients for use in source function
    integer,     dimension(size(play,dim=1), size(play,dim=2)) :: jtemp, jpress
    logical(wl), dimension(size(play,dim=1), size(play,dim=2)) :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),size(play,dim=1), size(play,dim=2)) :: fmajor
    integer,     dimension(2,    get_nflav(this),size(play,dim=1), size(play,dim=2)) :: jeta

    integer :: ncol, nlay, ngpt, nband, ngas, nflav, count_rate, iTime1, iTime2
    ! ----------------------------------------------------------
    ncol  = size(play,dim=1)
    nlay  = size(play,dim=2)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()
    !
    ! Gas optics
    !
    ! call system_clock(count_rate=count_rate)
    ! call system_clock(iTime1)
#ifdef USE_TIMING
    ret =  gptlstart('compute_gas_taus')
#endif

    !$acc enter data create(jtemp, jpress, tropo, fmajor, jeta)
    error_msg = compute_gas_taus(this,                       &
                                 ncol, nlay, ngpt, nband,    &
                                 play, plev, tlay, gas_desc, &
                                 optical_props,              &
                                 jtemp, jpress, jeta, tropo, fmajor, &
                                 col_dry)
    if(error_msg  /= '') return

#ifdef USE_TIMING
    ret =  gptlstop('compute_gas_taus')
#endif
    ! call system_clock(iTime2)
    ! print *,'Elapsed time on optical depths: ',real(iTime2-iTime1)/real(count_rate)

    ! ----------------------------------------------------------
    !
    ! External source -- check arrays sizes and values
    ! input data sizes and values
    !
    !$acc enter data copyin(tsfc,tlev)
    if(.not. extents_are(tsfc, ncol)) &
      error_msg = "gas_optics(): array tsfc has wrong size"
    if(any_vals_outside(tsfc, this%temp_ref_min,  this%temp_ref_max)) &
      error_msg = "gas_optics(): array tsfc has values outside range"
    if(error_msg  /= '') return

    if(present(tlev)) then
      if(.not. extents_are(tlev, ncol, nlay+1)) &
        error_msg = "gas_optics(): array tlev has wrong size"
      if(any_vals_outside(tlev, this%temp_ref_min, this%temp_ref_max)) &
        error_msg = "gas_optics(): array tlev has values outside range"
      if(error_msg  /= '') return
    end if

    !
    !   output extents
    !
    if(any([sources%get_ncol(), sources%get_nlay(), sources%get_ngpt()] /= [ncol, nlay, ngpt])) &
      error_msg = "gas_optics%gas_optics: source function arrays inconsistently sized"
    if(error_msg  /= '') return

    !
    ! Interpolate source function
    !
    error_msg = source(this,                               &
                       ncol, nlay, nband, ngpt,            &
                       play, plev, tlay, tsfc,             &
                       jtemp, jpress, jeta, tropo, fmajor, &
                       sources,                            &
                       tlev)
    !$acc exit data delete(tsfc,tlev)
    !$acc exit data delete(jtemp, jpress, tropo, fmajor, jeta)
  end function gas_optics_int

  function gas_optics_int_nn(this,                              &
                          play, plev, tlay, tsfc, gas_desc,     &
                          optical_props, sources, nn_inputs,    &
                          neural_nets,                          &
                          col_dry, tlev) result(error_msg)

  
    ! inputs
    class(ty_gas_optics_rrtmgp),    intent(in)    :: this
    real(wp), dimension(:,:),       intent(in)    :: play, &   ! layer pressures [Pa, mb]; (ncol,nlay)
                                                  plev, &   ! level pressures [Pa, mb]; (ncol,nlay+1)
                                                  tlay      ! layer temperatures [K]; (ncol,nlay)
    real(wp),    dimension(:),      intent(in)    :: tsfc      ! surface skin temperatures [K]; (ncol)
    type(ty_gas_concs),             intent(in)    :: gas_desc  ! Gas volume mixing ratios
    real(wp),    dimension(:,:,:),  intent(inout) :: nn_inputs

    type(network_type), dimension(2), intent(inout) :: neural_nets ! Planck fraction model, optical depth model

    class(ty_optical_props_arry),   intent(inout)   :: optical_props ! Optical properties
    class(ty_source_func_lw    ),   intent(inout)   :: sources       ! Planck sources
    character(len=128)                              :: error_msg
    ! Optional inputs
    real(wp), dimension(:,:),   intent(in   ), &
                           optional, target :: col_dry, &  ! Column dry amount; dim(ncol,nlay)
                                               tlev        ! level temperatures [K]; (ncol,nlay+1)
    ! ----------------------------------------------------------
    ! Local variables
    ! Interpolation coefficients for use in source function
                   
    !integer,     dimension(ncol,nlay)                           :: jtemp, jpress
    !logical(wl), dimension(ncol,nlay)                           :: tropo
    !real(wp),    dimension(2,2,2,get_nflav(this),ncol,nlay)     :: fmajor
    !integer,     dimension(2,    get_nflav(this),ncol,nlay)     :: jeta
    !integer,     dimension(ncol,2)                              :: itropo, istrato
    !real(wp),    dimension(ncol,nlay)                           :: play_log
    real(wp),     dimension(:,:,:),     allocatable              :: pfrac
    !real(wp),    dimension(this%get_ngpt(),ncol,nlay)           :: tau_maj
    !integer :: ngpt, nband, count_rate, iTime1, iTime2
    integer :: ncol, nlay, ngpt, nband, ngas, nflav, count_rate, iTime1, iTime2
    logical :: original_source, top_at_1

    !original_source = .false.
    ! ----------------------------------------------------------

    ncol  = size(play,dim=1)
    nlay  = size(play,dim=2)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()
    ngas  = this%get_ngas()
    
    allocate(pfrac(ngpt,ncol,nlay))

#ifdef USE_TIMING
    ret =  gptlstart('compute_taus_pfracs_nnlw')
#endif

    ! Predict g-point taus and planck fractions using neural networks
    error_msg = compute_taus_pfracs_nnlw(this,                  &
                                 ncol, nlay, ngas, ngpt, nband, &
                                 play, plev, tlay, gas_desc,    &
                                 optical_props, pfrac,          &
                                 neural_nets,                   &
                                 nn_inputs, col_dry) 
    
    ! call system_clock(iTime2)
    !print *,'Elapsed time on optical depths: ',real(iTime2-iTime1)/real(count_rate)


#ifdef USE_TIMING
    ret =  gptlstop('compute_taus_pfracs_nnlw')
#endif

    ! ----------------------------------------------------------
    !
    ! External source -- check arrays sizes and values
    ! input data sizes and values
    !
    !$acc enter data copyin(tsfc,tlev)
    if(.not. extents_are(tsfc, ncol)) &
      error_msg = "gas_optics(): array tsfc has wrong size"
    if(any_vals_outside(tsfc, this%temp_ref_min,  this%temp_ref_max)) &
      error_msg = "gas_optics(): array tsfc has values outside range"
    if(error_msg  /= '') return

    if(present(tlev)) then
      if(.not. extents_are(tlev, ncol, nlay+1)) &
        error_msg = "gas_optics(): array tlev has wrong size"
      if(any_vals_outside(tlev, this%temp_ref_min, this%temp_ref_max)) &
        error_msg = "gas_optics(): array tlev has values outside range"
      if(error_msg  /= '') return
    end if

    !
    !   output extents
    !
    if(any([sources%get_ncol(), sources%get_nlay(), sources%get_ngpt()] /= [ncol, nlay, ngpt])) &
      error_msg = "gas_optics%gas_optics: source function arrays inconsistently sized"
    if(error_msg  /= '') return

#ifdef USE_TIMING
    ret =  gptlstart('Planck-source')
#endif

    !if (original_source) then
    ! Use original source computations, including planck fractions by interpolating
    !error_msg = source(this,                               &
    !                   ncol, nlay, nband, ngpt,            &
    !                   play, plev, tlay, tsfc,             &
    !                   jtemp, jpress, jeta, tropo, fmajor, &
    !                   sources,                            &
    !                   tlev)
    !else

    error_msg = source_nn(this,                     & !
                        ncol, nlay, nband, ngpt,    &
                        play, plev, tlay, tsfc,     &
                        sources,                    & ! inout
                        pfrac,                      & ! in
                        tlev)                         ! optional input

    !end if

#ifdef USE_TIMING
    ret =  gptlstop('Planck-source')
#endif

  end function gas_optics_int_nn
  !------------------------------------------------------------------------------------------
  !
  ! Compute gas optical depth given temperature, pressure, and composition
  !
  function gas_optics_ext(this,                         &
                          play, plev, tlay, gas_desc,   & ! mandatory inputs
                          optical_props, toa_src,       & ! mandatory outputs
                          col_dry) result(error_msg)      ! optional input

    class(ty_gas_optics_rrtmgp), intent(in) :: this
    real(wp), dimension(:,:), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (ncol,nlay)
                                               plev, &   ! level pressures [Pa, mb]; (ncol,nlay+1)
                                               tlay      ! layer temperatures [K]; (ncol,nlay)
    type(ty_gas_concs),       intent(in   ) :: gas_desc  ! Gas volume mixing ratios
    ! output
    class(ty_optical_props_arry),  &
                              intent(inout) :: optical_props
    real(wp), dimension(:,:), intent(  out) :: toa_src     ! Incoming solar irradiance(ncol,ngpt)
    character(len=128)                      :: error_msg

    ! Optional inputs
    real(wp), dimension(:,:), intent(in   ), &
                           optional, target :: col_dry ! Column dry amount; dim(ncol,nlay)
    ! ----------------------------------------------------------
    ! Local variables
    ! Interpolation coefficients for use in source function
    integer,     dimension(size(play,dim=1), size(play,dim=2)) :: jtemp, jpress
    logical(wl), dimension(size(play,dim=1), size(play,dim=2)) :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),size(play,dim=1), size(play,dim=2)) :: fmajor
    integer,     dimension(2,    get_nflav(this),size(play,dim=1), size(play,dim=2)) :: jeta

    integer :: ncol, nlay, ngpt, nband, ngas, nflav, count_rate, iTime1, iTime2
    integer :: igpt, icol
    ! ----------------------------------------------------------
    ncol  = size(play,dim=1)
    nlay  = size(play,dim=2)
    ngpt  = this%get_ngpt()
    nband = this%get_nband()
    ngas  = this%get_ngas()
    nflav = get_nflav(this)
    !
    ! Gas optics

    !
    !$acc enter data create(jtemp, jpress, tropo, fmajor, jeta)
    error_msg = compute_gas_taus(this,                       &
                                 ncol, nlay, ngpt, nband,    &
                                 play, plev, tlay, gas_desc, &
                                 optical_props,              &
                                 jtemp, jpress, jeta, tropo, fmajor, &
                                 col_dry)
    !$acc exit data delete(jtemp, jpress, tropo, fmajor, jeta)
    if(error_msg  /= '') return


    ! ----------------------------------------------------------
    !
    ! External source function is constant
    !
    !$acc enter data create(toa_src)
    if(.not. extents_are(toa_src, ncol, ngpt)) &
      error_msg = "gas_optics(): array toa_src has wrong size"
    if(error_msg  /= '') return

    !$acc parallel loop collapse(2)
    do igpt = 1,ngpt
       do icol = 1,ncol
          toa_src(icol,igpt) = this%solar_src(igpt)
       end do
    end do
    !$acc exit data copyout(toa_src)
  end function gas_optics_ext
  !------------------------------------------------------------------------------------------
  !
  ! Returns optical properties and interpolation coefficients
  !
  function compute_gas_taus(this,                       &
                            ncol, nlay, ngpt, nband,    &
                            play, plev, tlay, gas_desc, &
                            optical_props,              &
                            jtemp, jpress, jeta, tropo, fmajor, &
                            col_dry) result(error_msg)

    class(ty_gas_optics_rrtmgp), &
                                      intent(in   ) :: this
    integer,                          intent(in   ) :: ncol, nlay, ngpt, nband
    real(wp), dimension(:,:),         intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (ncol,nlay)
                                                       plev, &   ! level pressures [Pa, mb]; (ncol,nlay+1)
                                                       tlay      ! layer temperatures [K]; (ncol,nlay)
    type(ty_gas_concs),               intent(in   ) :: gas_desc  ! Gas volume mixing ratios
    class(ty_optical_props_arry),     intent(inout) :: optical_props !inout because components are allocated
    ! Interpolation coefficients for use in internal source function
    integer,     dimension(                      ncol, nlay), intent(  out) :: jtemp, jpress
    integer,     dimension(2,    get_nflav(this),ncol, nlay), intent(  out) :: jeta
    logical(wl), dimension(                      ncol, nlay), intent(  out) :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),ncol, nlay), intent(  out) :: fmajor
    character(len=128)                                         :: error_msg

    ! Optional inputs
    real(wp), dimension(:,:), intent(in   ), &
                           optional, target :: col_dry ! Column dry amount; dim(ncol,nlay)
    ! ----------------------------------------------------------
    ! Local variables
    real(wp), dimension(ngpt,nlay,ncol) :: tau, tau_rayleigh  ! absorption, Rayleigh scattering optical depths
    ! Number of molecules per cm^2
    real(wp), dimension(ncol,nlay), target  :: col_dry_arr
    real(wp), dimension(:,:),       pointer :: col_dry_wk
    real(wp), dimension(ncol,nlay)          :: play_log
    !
    ! Interpolation variables used in major gas but not elsewhere, so don't need exporting
    !
    real(wp), dimension(ncol,nlay,  this%get_ngas()) :: vmr     ! volume mixing ratios
    real(wp), dimension(ncol,nlay,0:this%get_ngas()) :: col_gas ! column amounts for each gas, plus col_dry
    real(wp), dimension(2,    get_nflav(this),ncol,nlay) :: col_mix ! combination of major species's column amounts
                                                         ! index(1) : reference temperature level
                                                         ! index(2) : flavor
                                                         ! index(3) : layer
    real(wp), dimension(2,2,  get_nflav(this),ncol,nlay) :: fminor ! interpolation fractions for minor species
                                                          ! index(1) : reference eta level (temperature dependent)
                                                          ! index(2) : reference temperature level
                                                          ! index(3) : flavor
                                                          ! index(4) : layer
    integer :: ngas, nflav, neta, npres, ntemp, count_rate, iTime1, iTime2
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

    !
    ! Check input data sizes and values
    !
    !$acc enter data copyin(play,plev,tlay)
    if(.not. extents_are(play, ncol, nlay  )) &
      error_msg = "gas_optics(): array play has wrong size"
    if(.not. extents_are(tlay, ncol, nlay  )) &
      error_msg = "gas_optics(): array tlay has wrong size"
    if(.not. extents_are(plev, ncol, nlay+1)) &
      error_msg = "gas_optics(): array plev has wrong size"
    if(error_msg  /= '') return

    if(any_vals_outside(play, this%press_ref_min,this%press_ref_max)) &
      error_msg = "gas_optics(): array play has values outside range"
    if(any_vals_outside(plev, this%press_ref_min,this%press_ref_max)) &
      error_msg = "gas_optics(): array plev has values outside range"
    if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max)) &
      error_msg = "gas_optics(): array tlay has values outside range"
    if(error_msg  /= '') return

    if(present(col_dry)) then
      if(.not. extents_are(col_dry, ncol, nlay)) &
        error_msg = "gas_optics(): array col_dry has wrong size"
      if(any_vals_less_than(col_dry, 0._wp)) &
        error_msg = "gas_optics(): array col_dry has values outside range"
      if(error_msg  /= '') return
    end if

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
    ! Fill out the array of volume mixing ratios
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

    !
    ! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
    !
    idx_h2o = string_loc_in_array('h2o', this%gas_names)
    !$acc enter data create(col_dry_wk, col_dry_arr, col_gas)
    if (present(col_dry)) then
      col_dry_wk => col_dry
    else
      col_dry_arr = get_col_dry(vmr(:,:,idx_h2o), plev) ! dry air column amounts computation
      col_dry_wk => col_dry_arr
    end if
    !
    ! compute column gas amounts [molec/cm^2]
    !
    !$acc parallel loop gang vector collapse(2)
    do ilay = 1, nlay
      do icol = 1, ncol
        col_gas(icol,ilay,0) = col_dry_wk(icol,ilay)
      end do
    end do
    !$acc parallel loop gang vector collapse(3)
    do igas = 1, ngas
      do ilay = 1, nlay
        do icol = 1, ncol
          col_gas(icol,ilay,igas) = vmr(icol,ilay,igas) * col_dry_wk(icol,ilay)
        end do
      end do
    end do
    !$acc exit data delete(vmr)

    !
    ! ---- calculate gas optical depths ----
    !
    !$acc enter data create(jtemp, jpress, jeta, tropo, fmajor)
    !$acc enter data create(tau, tau_rayleigh)
    !$acc enter data create(col_mix, fminor)
    !$acc enter data copyin(this)
    !$acc enter data copyin(this%gpoint_flavor)
    call zero_array(ngpt, nlay, ncol, tau)
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
            jeta,jpress,play_log)

    !call system_clock(count_rate=count_rate)
    !call system_clock(iTime1)

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
            tau)
    !call system_clock(iTime2)
    !print *,'Elapsed time on compute_tau_absorption ',real(iTime2-iTime1)/real(count_rate)

    if (allocated(this%krayl)) then
      !$acc enter data attach(col_dry_wk) copyin(this%krayl)
      call compute_tau_rayleigh(         & !Rayleigh scattering optical depths
            ncol,nlay,nband,ngpt,        &
            ngas,nflav,neta,npres,ntemp, & ! dimensions
            this%gpoint_flavor,          &
            this%get_band_lims_gpoint(), &
            this%krayl,                  & ! inputs from object
            idx_h2o, col_dry_wk,col_gas, &
            fminor,jeta,tropo,jtemp,     & ! local input
            tau_rayleigh)
      !$acc exit data detach(col_dry_wk) delete(this%krayl)
    end if
    if (error_msg /= '') return

    call system_clock(count_rate=count_rate)
    call system_clock(iTime1)

    ! Combine optical depths and reorder for radiative transfer solver.
    call combine_and_reorder(tau, tau_rayleigh, allocated(this%krayl), optical_props)

    call system_clock(iTime2)
    print *,'Elapsed time on combine and reorder ',real(iTime2-iTime1)/real(count_rate)
    !$acc exit data delete(play, tlay, plev)
    !$acc exit data delete(tau, tau_rayleigh)
    !$acc exit data delete(col_dry_wk, col_dry_arr, col_gas, col_mix, fminor)
    !$acc exit data delete(this%gpoint_flavor)
    !$acc exit data copyout(jtemp, jpress, jeta, tropo, fmajor)
  end function compute_gas_taus
  !------------------------------------------------------------------------------------------
  !
  ! Compute Planck source functions at layer centers and levels
  !
  function source(this,                               &
                  ncol, nlay, nbnd, ngpt,             &
                  play, plev, tlay, tsfc,             &
                  jtemp, jpress, jeta, tropo, fmajor, &
                  sources,                            & ! Planck sources
                  tlev)                               & ! optional input
                  result(error_msg)
    ! inputs
    class(ty_gas_optics_rrtmgp),    intent(in ) :: this
    integer,                               intent(in   ) :: ncol, nlay, nbnd, ngpt
    real(wp), dimension(ncol,nlay),        intent(in   ) :: play   ! layer pressures [Pa, mb]
    real(wp), dimension(ncol,nlay+1),      intent(in   ) :: plev   ! level pressures [Pa, mb]
    real(wp), dimension(ncol,nlay),        intent(in   ) :: tlay   ! layer temperatures [K]
    real(wp), dimension(ncol),             intent(in   ) :: tsfc   ! surface skin temperatures [K]
    ! Interplation coefficients
    integer,     dimension(ncol,nlay),     intent(in   ) :: jtemp, jpress
    logical(wl), dimension(ncol,nlay),     intent(in   ) :: tropo
    real(wp),    dimension(2,2,2,get_nflav(this),ncol,nlay),  &
                                           intent(in   ) :: fmajor
    integer,     dimension(2,    get_nflav(this),ncol,nlay),  &
                                           intent(in   ) :: jeta
    class(ty_source_func_lw    ),          intent(inout) :: sources
    real(wp), dimension(ncol,nlay+1),      intent(in   ), &
                                      optional, target :: tlev          ! level temperatures [K]
    character(len=128)                                 :: error_msg
    ! ----------------------------------------------------------
    integer                                      :: icol, ilay, igpt
    integer                                      :: count_rate, iTime1, iTime2
    real(wp), dimension(ngpt,nlay,ncol)          :: lay_source_t, lev_source_inc_t, lev_source_dec_t, planck_frac_out
    ! lay_source_t  - (ngot)
    real(wp), dimension(ngpt,     ncol)          :: sfc_source_t
    ! Variables for temperature at layer edges [K] (ncol, nlay+1)
    real(wp), dimension(   ncol,nlay+1), target  :: tlev_arr
    real(wp), dimension(:,:),            pointer :: tlev_wk
    ! ----------------------------------------------------------
    error_msg = ""
    !
    ! Source function needs temperature at interfaces/levels and at layer centers
    !
    if (present(tlev)) then
      !   Users might have provided these
      tlev_wk => tlev
    else
      tlev_wk => tlev_arr
      !
      ! Interpolate temperature to levels if not provided
      !   Interpolation and extrapolation at boundaries is weighted by pressure
      !
      do icol = 1, ncol
         tlev_arr(icol,1) = tlay(icol,1) &
                           + (plev(icol,1)-play(icol,1))*(tlay(icol,2)-tlay(icol,1))  &
              &                                           / (play(icol,2)-play(icol,1))
      end do
      do ilay = 2, nlay
        do icol = 1, ncol
           tlev_arr(icol,ilay) = (play(icol,ilay-1)*tlay(icol,ilay-1)*(plev(icol,ilay  )-play(icol,ilay)) &
                                +  play(icol,ilay  )*tlay(icol,ilay  )*(play(icol,ilay-1)-plev(icol,ilay))) /  &
                                  (plev(icol,ilay)*(play(icol,ilay-1) - play(icol,ilay)))
        end do
      end do
      do icol = 1, ncol
         tlev_arr(icol,nlay+1) = tlay(icol,nlay)                                                             &
                                + (plev(icol,nlay+1)-play(icol,nlay))*(tlay(icol,nlay)-tlay(icol,nlay-1))  &
                                                                      / (play(icol,nlay)-play(icol,nlay-1))
      end do
    end if

    !-------------------------------------------------------------------
    ! Compute internal (Planck) source functions at layers and levels,
    !  which depend on mapping from spectral space that creates k-distribution.
    !$acc enter data copyin(sources)
    !$acc enter data create(sources%lay_source, sources%lev_source_inc, sources%lev_source_dec, sources%sfc_source)
    !$acc enter data create(sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t) attach(tlev_wk)
    call compute_Planck_source(ncol, nlay, nbnd, ngpt, &
                get_nflav(this), this%get_neta(), this%get_npres(), this%get_ntemp(), this%get_nPlanckTemp(), &
                tlay, tlev_wk, tsfc, merge(1,nlay,play(1,1) > play(1,nlay)), &
                fmajor, jeta, tropo, jtemp, jpress,                    &
                this%get_gpoint_bands(), this%get_band_lims_gpoint(), this%planck_frac, this%temp_ref_min,&
                this%totplnk_delta, this%totplnk, this%gpoint_flavor,  &
                sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t, planck_frac_out)
    !$acc parallel loop collapse(2)
    do igpt = 1, ngpt
      do icol = 1, ncol
        sources%sfc_source(icol,igpt) = sfc_source_t(igpt,icol)
      end do
    end do

#ifdef USE_TIMING
    ret =  gptlstart('reorder-source')
#endif

    call reorder123x321(lay_source_t, sources%lay_source)
    call reorder123x321(lev_source_inc_t, sources%lev_source_inc)
    call reorder123x321(lev_source_dec_t, sources%lev_source_dec)
    call reorder123x321(planck_frac_out,  sources%planck_frac)

#ifdef USE_TIMING
    ret =  gptlstop('reorder-source')
#endif
    !$acc exit data delete(sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t) detach(tlev_wk)
    !$acc exit data copyout(sources%lay_source, sources%lev_source_inc, sources%lev_source_dec, sources%sfc_source)
    !$acc exit data copyout(sources)
  end function source
  !------------------------------------------------------------------------------------------
  !
  ! Compute Planck source functions at layer centers and levels
  ! The difference between this and source():
  ! 1) the planck fractions have already been calculated and are inputs
  ! 2) the reordering procedure is applied on the planck fractions, and another function (compute_Plank_source_nn)
  !    is then called to calculate lay_source and lev_sources directly in the right format (ncol,nlay,ngpt)
  function source_nn(this,                              &
                  ncol, nlay, nbnd, ngpt,             &
                  play, plev, tlay, tsfc,             &
                  sources,                            & ! Planck sources
                  pfrac,                              & ! planck fractions (input) 
                  tlev)                               & ! optional input
                  result(error_msg)
    ! inputs
    class(ty_gas_optics_rrtmgp),    intent(in ) :: this
    integer,                               intent(in ) :: ncol, nlay, nbnd, ngpt
    real(wp), dimension(ncol,nlay),        intent(in ) :: play   ! layer pressures [Pa, mb]
    real(wp), dimension(ncol,nlay+1),      intent(in ) :: plev   ! level pressures [Pa, mb]
    real(wp), dimension(ncol,nlay),        intent(in ) :: tlay   ! layer temperatures [K]
    real(wp), dimension(ncol),             intent(in ) :: tsfc   ! surface skin temperatures [K]
    real(wp), dimension(ngpt,nlay,ncol),   intent(in)  :: pfrac  ! planck fractions

    ! Interpolation and post processing coefficients
    class(ty_source_func_lw    ),          intent(inout) :: sources
    real(wp), dimension(ncol,nlay+1),      intent(in ), &
                                      optional, target :: tlev          ! level temperatures [K]

    character(len=128)                                 :: error_msg
    ! ----------------------------------------------------------
    integer                                      :: icol, ilay
    integer, dimension(2, nbnd)                  :: band_lims_gpt
    real(wp), dimension(ncol,nlay,ngpt)          :: pfrac_reverse
    ! real(wp), dimension(ngpt,nlay,ncol)          :: lay_source_t, lev_source_inc_t, lev_source_dec_t
    ! real(wp), dimension(ngpt,     ncol)          :: sfc_source_t
    ! Variables for temperature at layer edges [K] (ncol, nlay+1)
    real(wp), dimension(   ncol,nlay+1), target  :: tlev_arr
    real(wp), dimension(:,:),            pointer :: tlev_wk => NULL()
    ! ----------------------------------------------------------
    error_msg = ""
    !
    ! Source function needs temperature at interfaces/levels and at layer centers
    !
    if (present(tlev)) then
      !   Users might have provided these
      tlev_wk => tlev
    else
      tlev_wk => tlev_arr
      !
      ! Interpolate temperature to levels if not provided
      !   Interpolation and extrapolation at boundaries is weighted by pressure
      !
      do icol = 1, ncol
         tlev_arr(icol,1) = tlay(icol,1) &
                           + (plev(icol,1)-play(icol,1))*(tlay(icol,2)-tlay(icol,1))  &
              &                                           / (play(icol,2)-play(icol,1))
      end do
      do ilay = 2, nlay
        do icol = 1, ncol
           tlev_arr(icol,ilay) = (play(icol,ilay-1)*tlay(icol,ilay-1)*(plev(icol,ilay  )-play(icol,ilay)) &
                                +  play(icol,ilay  )*tlay(icol,ilay  )*(play(icol,ilay-1)-plev(icol,ilay))) /  &
                                  (plev(icol,ilay)*(play(icol,ilay-1) - play(icol,ilay)))
        end do
      end do
      do icol = 1, ncol
         tlev_arr(icol,nlay+1) = tlay(icol,nlay)                                                             &
                                + (plev(icol,nlay+1)-play(icol,nlay))*(tlay(icol,nlay)-tlay(icol,nlay-1))  &
                                                                      / (play(icol,nlay)-play(icol,nlay-1))
      end do
    end if

    !-------------------------------------------------------------------
    ! Compute internal (Planck) source functions at layers and levels,
    !  which depend on mapping from spectral space that creates k-distribution.

    ! call system_clock(count_rate=count_rate)
    ! call system_clock(iTime1)

#ifdef USE_TIMING
    ret =  gptlstart('reorder-source-nn')
#endif

    call reorder123x321(pfrac, sources%planck_frac)

    ! call system_clock(iTime2)
    ! print *,'Elapsed time on reorder: ',real(iTime2-iTime1)/real(count_rate)
    band_lims_gpt = this%get_band_lims_gpoint()
    call compute_Planck_source_pfracin(ncol, nlay, nbnd, ngpt, &
            this%get_ntemp(),this%get_nPlanckTemp(), &
            tlay, tlev_wk, tsfc, merge(1,nlay,play(1,1) > play(1,nlay)), &
            band_lims_gpt, &
            this%temp_ref_min, this%totplnk_delta, this%totplnk, &
            sources%planck_frac, &
            sources%sfc_source, sources%lay_source, sources%lev_source_inc, &
            sources%lev_source_dec)

#ifdef USE_TIMING
    ret =  gptlstop('reorder-source-nn')
#endif

    !call reorder123x321(planck_frac_t, sources%planck_frac)

  end function source_nn
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
                    totplnk, planck_frac, rayl_lower, rayl_upper) result(err_message)
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
    allocate(this%totplnk    (size(totplnk,    1), size(totplnk,   2)), &
             this%planck_frac(size(planck_frac,1), size(planck_frac,2), size(planck_frac,3), size(planck_frac,4)) )
    !$acc enter data create(this%totplnk, this%planck_frac)
    !$acc kernels
    this%totplnk = totplnk
    this%planck_frac = planck_frac
    !$acc end kernels
    ! Temperature steps for Planck function interpolation
    !   Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    !   Planck grid and the Planck grid is equally spaced
    this%totplnk_delta =  (this%temp_ref_max-this%temp_ref_min) / (size(this%totplnk,dim=1)-1)
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
                    solar_src, rayl_lower, rayl_upper)  result(err_message)
    class(ty_gas_optics_rrtmgp), intent(inout) :: this
    class(ty_gas_concs),                intent(in   ) :: available_gases ! Which gases does the host model have available?
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
    logical(wl), dimension(:),    intent(in) :: &
                                                minor_scales_with_density_lower, &
                                                minor_scales_with_density_upper
    character(len=*),   dimension(:),intent(in) :: &
                                                scaling_gas_lower, &
                                                scaling_gas_upper
    logical(wl), dimension(:),    intent(in) :: &
                                                scale_by_complement_lower, &
                                                scale_by_complement_upper
    integer,  dimension(:),       intent(in) :: &
                                                kminor_start_lower, &
                                                kminor_start_upper
    real(wp), dimension(:),       intent(in), allocatable :: solar_src
                                                            ! allocatable status to change when solar source is present in file
    real(wp), dimension(:,:,:), intent(in), allocatable :: rayl_lower, rayl_upper
    character(len = 128) err_message
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
    !
    ! Solar source table init
    !
    allocate(this%solar_src(size(solar_src)))
    !$acc enter data create(this%solar_src)
    !$acc kernels
    this%solar_src = solar_src
    !$acc end kernels
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

!------------------------------------------------------------------------------------------
  !
  ! Returns interpolation coefficients only
  !
function compute_interp_coeffs(this,                       &
  ncol, nlay, ngpt, nband,    &
  play, plev, tlay, gas_desc, &
  jtemp, jpress, jeta, tropo, fmajor, play_log, &
  col_dry) result(error_msg)

class(ty_gas_optics_rrtmgp), &
            intent(in   ) :: this
integer,                          intent(in   ) :: ncol, nlay, ngpt, nband
real(wp), dimension(ncol,nlay),   intent(in   ) :: play, tlay   ! layer pressures [Pa, mb] and temp (K); (ncol,nlay)
real(wp), dimension(ncol,nlay+1), intent(in   ) :: plev   ! level pressures [Pa, mb]; (ncol,nlay+1)
type(ty_gas_concs),               intent(in   ) :: gas_desc  ! Gas volume mixing ratios
! Interpolation coefficients for use in internal source function
integer,     dimension(                       ncol, nlay), intent(  out)  :: jtemp, jpress
integer,     dimension(2,    get_nflav(this),ncol, nlay), intent(  out)   :: jeta
logical(wl), dimension(                       ncol, nlay), intent(  out)  :: tropo
real(wp),    dimension(2,2,2,get_nflav(this),ncol, nlay), intent(  out)   :: fmajor
real(wp),    dimension(                       ncol, nlay), intent(  out)  :: play_log
character(len=128)                                         :: error_msg

! Optional inputs
real(wp), dimension(ncol,nlay), intent(in   ), &
 optional, target :: col_dry ! Column dry amount; dim(ncol,nlay)
! ----------------------------------------------------------
integer :: igas, idx_h2o ! index of some gases
! Number of molecules per cm^2
real(wp), dimension(ncol,nlay), target  :: col_dry_arr
real(wp), dimension(:,:),       pointer :: col_dry_wk => NULL()
!
! Interpolation variables used in major gas but not elsewhere, so don't need exporting
!
real(wp), dimension(ncol,nlay,  this%get_ngas()) :: vmr     ! volume mixing ratios
real(wp), dimension(ncol,nlay,0:this%get_ngas()) :: col_gas ! column amounts for each gas, plus col_dry
real(wp), dimension(2,    get_nflav(this),ncol,nlay) :: col_mix ! combination of major species's column amounts
                               ! index(1) : reference temperature level
                               ! index(2) : flavor
                               ! index(3) : layer
real(wp), dimension(2,2,  get_nflav(this),ncol,nlay) :: fminor ! interpolation fractions for minor species
                                ! index(1) : reference eta level (temperature dependent)
                                ! index(2) : reference temperature level
                                ! index(3) : flavor
                                ! index(4) : layer
integer :: ngas, nflav, neta, npres, ntemp
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

! Check input data sizes and values
!
!$acc enter data copyin(play,plev,tlay)
if(.not. extents_are(play, ncol, nlay  )) &
error_msg = "gas_optics(): array play has wrong size"
if(.not. extents_are(tlay, ncol, nlay  )) &
error_msg = "gas_optics(): array tlay has wrong size"
if(.not. extents_are(plev, ncol, nlay+1)) &
error_msg = "gas_optics(): array plev has wrong size"
if(error_msg  /= '') return

if(any_vals_outside(play, this%press_ref_min,this%press_ref_max)) &
error_msg = "gas_optics(): array play has values outside range"
if(any_vals_outside(plev, this%press_ref_min,this%press_ref_max)) &
error_msg = "gas_optics(): array plev has values outside range"
if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max)) &
error_msg = "gas_optics(): array tlay has values outside range"
if(error_msg  /= '') return

if(present(col_dry)) then
if(.not. extents_are(col_dry, ncol, nlay)) &
error_msg = "gas_optics(): array col_dry has wrong size"
if(any_vals_less_than(col_dry, 0._wp)) &
error_msg = "gas_optics(): array col_dry has values outside range"
if(error_msg  /= '') return
end if


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
! Fill out the array of volume mixing ratios
!
do igas = 1, ngas
!
! Get vmr if  gas is provided in ty_gas_concs
!
if (any (lower_case(this%gas_names(igas)) == gas_desc%gas_name(:))) then
error_msg = gas_desc%get_vmr(this%gas_names(igas), vmr(:,:,igas))
if (error_msg /= '') return
endif
end do

!
! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
!
idx_h2o = string_loc_in_array('h2o', this%gas_names)
if (present(col_dry)) then
	col_dry_wk => col_dry
else
  col_dry_arr = get_col_dry(vmr(:,:,idx_h2o), plev) ! dry air column amounts computation
  col_dry_wk => col_dry_arr
end if
!
! compute column gas amounts [molec/cm^2]
!
col_gas(1:ncol,1:nlay,0) = col_dry_wk(1:ncol,1:nlay)
do igas = 1, ngas
	col_gas(1:ncol,1:nlay,igas) = vmr(1:ncol,1:nlay,igas) * col_dry_wk(1:ncol,1:nlay)
end do

!
! ---- calculate gas optical depths ----
!
!$acc enter data create(jtemp, jpress, jeta, tropo, fmajor)
!$acc enter data copyin(play, tlay, col_gas)
!$acc enter data create(col_mix, fminor)
!$acc enter data copyin(this)
!$acc enter data copyin(this%flavor, this%press_ref_log, this%vmr_ref, this%gpoint_flavor)
!$acc enter data copyin(this%temp_ref)  ! this one causes problems
!$acc enter data copyin(this%kminor_lower, this%kminor_upper)
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
  jeta,jpress,play_log)


!$acc exit data copyout(jtemp, jpress, jeta, tropo, fmajor)
!$acc exit data delete(play, tlay, col_gas, col_mix, fminor)
!$acc exit data delete(this%flavor, this%press_ref_log, this%vmr_ref, this%gpoint_flavor)
!!!$acc exit data delete(this%temp_ref)  ! this one causes problems
!!!$acc exit data delete(this%kminor_lower, this%kminor_upper)
end function compute_interp_coeffs


! Function for neural-network accelerated long-wave gas optics (optical depths and planck fractions).
! This function prepares the outputs for the neural network and then calls the kernel
function compute_taus_pfracs_nnlw(this,                         &
    ncol, nlay, ngas, ngpt, nband,                                    &
    play, plev, tlay, gas_desc,                                 &
    optical_props, pfrac,                                       &
    neural_nets,                                                &
    nn_inputs, col_dry) result(error_msg)

class(ty_gas_optics_rrtmgp),          intent(in   ) ::  this
integer,                              intent(in   ) ::  ncol, nlay, ngas, ngpt, nband
real(wp), dimension(ncol,nlay),       intent(in   ) ::  play, &   ! layer pressures [Pa, mb]; (ncol,nlay)
                                                        tlay
real(wp), dimension(ncol,nlay+1),     intent(in   ) ::  plev                                                    

type(ty_gas_concs),                   intent(in   ) ::  gas_desc  ! Gas volume mixing ratios  
class(ty_optical_props_arry),         intent(inout) ::  optical_props !inout because components are allocated

type(network_type), dimension(2),     intent(inout) :: neural_nets

real(wp), dimension(ngas,nlay,ncol),    intent(out) :: nn_inputs 
real(wp), dimension(ngpt,nlay,ncol),    intent(out) :: pfrac ! Planck fractions predicted by NN

character(len=128)                                  :: error_msg

! Optional inputs
real(wp), dimension(ncol,nlay), intent(in   ), &
   optional, target :: col_dry ! Column dry amount; dim(ncol,nlay)
! ----------------------------------------------------------
! Local variables
real(wp), dimension(ngpt,nlay,ncol) :: tau  ! absorption, Rayleigh scattering optical depths
integer :: igas, ilay, idx_h2o, icol, inet ! index of some gases
! Number of molecules per cm^2
real(wp), dimension(ncol,nlay), target  :: col_dry_arr
real(wp), dimension(:,:),       pointer :: col_dry_wk => NULL()
!
! Interpolation variables used in major gas but not elsewhere, so don't need exporting
!

real(wp), dimension(ncol,nlay,  ngas)  :: vmr     ! volume mixing ratios
real(wp), dimension(ncol,nlay,0:ngas)  :: col_gas ! column amounts for each gas, plus col_dry
integer                                 :: npres, ntemp,  count_rate, iTime1, iTime2, neurons_first, neurons_last
! ----------------------------------------------------------
! Neural network input scaling coefficients .. should probably be loaded from a file

real(wp), dimension(19) :: input_scaler_means = (/4.32155586E5_wp, 1.50567158E3_wp, 3.03431184E2_wp, 1.67521191E-1_wp, &
       1.35015806E-1_wp, 4.37451436E-2_wp, 6.80378326E-1_wp, 1.59224330E-5_wp, &
       1.05737124E-4_wp, 1.14812313E-4_wp, 4.27522004E-5_wp, 9.92715052E-5_wp, &
       1.33700130E-4_wp, 6.91886629E-6_wp, 1.55411482E-6_wp, 6.77498534E-5_wp, &
       3.03697001E-5_wp, 2.39662574E2_wp, 8.76987892E0_wp /)

real(wp), dimension(19) :: input_scaler_std = (/3.93707999E5_wp, 3.87605107E3_wp, 4.02556768E2_wp, 2.24530064E-1_wp, &
1.27834878E-1_wp, 5.03624184E-2_wp, 7.47975734E-1_wp, 2.58462536E-5_wp, &
2.17398717E-4_wp, 1.60784004E-4_wp, 7.20443774E-5_wp, 1.95084899E-4_wp, &
2.64337382E-4_wp, 9.26325333E-6_wp, 2.61683251E-6_wp, 1.13002146E-4_wp, &
3.63426590E-5_wp, 2.88759196E1_wp, 2.71611697E0_wp /)


real(wp) :: testparam   = 0.5001566410_wp
real(dp) :: doubleparam = 0.5001566410_dp
! ----------------------------------------------------------
! Process all the layers and columns simultaenously, using BLAS for matrix-matrix computations? (fastest on intel compilers)
! logical :: flatten_dims
integer :: flatten_dims
! If levs and cols are not flattened, use BLAS anyway for matrix-vector computations? (usually slower)
! If true, the hidden layers must have equal sizes (flat model)
logical :: use_blas

! Predict one layer at a time, where the neural network computations are matrix-vector dot products, 
! or predict all columns and layers simultaenously, using BLAS for matrix-matrix computations? (usually faster)
use_blas = .true.

!
! Error checking
!

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

!
! Check input data sizes and values
if(.not. extents_are(play, ncol, nlay  )) &
error_msg = "gas_optics(): array play has wrong size"
if(.not. extents_are(tlay, ncol, nlay  )) &
error_msg = "gas_optics(): array tlay has wrong size"
if(.not. extents_are(plev, ncol, nlay+1)) &
error_msg = "gas_optics(): array plev has wrong size"
if(error_msg  /= '') return

if(any_vals_outside(play, this%press_ref_min,this%press_ref_max)) &
error_msg = "gas_optics(): array play has values outside range"
if(any_vals_outside(plev, this%press_ref_min,this%press_ref_max)) &
error_msg = "gas_optics(): array plev has values outside range"
if(any_vals_outside(tlay, this%temp_ref_min,  this%temp_ref_max)) &
error_msg = "gas_optics(): array tlay has values outside range"
if(error_msg  /= '') return

if(present(col_dry)) then
if(.not. extents_are(col_dry, ncol, nlay)) &
error_msg = "gas_optics(): array col_dry has wrong size"
if(any_vals_less_than(col_dry, 0._wp)) &
error_msg = "gas_optics(): array col_dry has values outside range"
if(error_msg  /= '') return
end if

! ----------------------------------------------------------
npres = this%get_npres()
ntemp = this%get_ntemp()


! Fill out the array of volume mixing ratios
!
do igas = 1, ngas
!
! Get vmr if  gas is provided in ty_gas_concs
!
  if (any (lower_case(this%gas_names(igas)) == gas_desc%gas_name(:))) then
    error_msg = gas_desc%get_vmr(this%gas_names(igas), vmr(:,:,igas))
    if (error_msg /= '') return
  endif
end do

!
! Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
!
idx_h2o = string_loc_in_array('h2o', this%gas_names)
if (present(col_dry)) then
  col_dry_wk => col_dry
else
  col_dry_arr = get_col_dry(vmr(:,:,idx_h2o), plev) ! dry air column amounts computation
  col_dry_wk => col_dry_arr
end if
!
! compute column gas amounts [molec/cm^2]
!
col_gas(1:ncol,1:nlay,0) = col_dry_wk(1:ncol,1:nlay)
do igas = 1, ngas
  col_gas(1:ncol,1:nlay,igas) = vmr(1:ncol,1:nlay,igas) * col_dry_wk(1:ncol,1:nlay)
end do

!
! Prepare neural network INPUTS (standard-scaled col_gas + pay + tlay)
! These need to be normalized using standard scaling
!
do ilay = 1, nlay
    ! 8th and 9th gases are constants (oxygen and nitrogen), exclude these. Note col_gas index starts from 0 (dry air)
    
    do igas = 1, 7
      ! print *, this%gas_names(igas-1)
      nn_inputs(igas,ilay,:) = (1.0E-18*col_gas(:,ilay,igas-1) - input_scaler_means(igas)) / input_scaler_std(igas)
    end do
    do igas = 8, ngas-2 ! previously ngas-1, temporary fix to remove NO2
      nn_inputs(igas,ilay,:) = (1.0E-18*col_gas(:,ilay,igas+1) - input_scaler_means(igas)) / input_scaler_std(igas)
      ! print *, "nn_input:", igas, "is gas:", this%gas_names(igas+1)
    end do
    ! Last two inputs are temperature and pressure
    nn_inputs(igas,ilay,:) = (tlay(:,ilay) - input_scaler_means(igas)) / input_scaler_std(igas)
    igas = igas + 1
    nn_inputs(igas,ilay,:) = (log(play(:,ilay)) - input_scaler_means(igas)) / input_scaler_std(igas)

end do

! do igas = 0,ngas
!   print *, "max for gas", igas,":", maxval(col_gas(:,:,igas))
! end do

! do igas = 1,size(nn_inputs,1)
!   print *, "max for nn_input", igas,":", maxval(nn_inputs(igas,:,:))
! end do

! print *, "siz:", size(nn_inputs,1)

! do igas = 1, ngas 
!   print *, "ngas",igas,":",this%gas_names(igas)
! end do
! do igas = 1,7 
!   print *, "nn_input:", igas, "is gas:", this%gas_names(igas-1)
! end do
! do igas = 8, ngas-1
!   print *, "nn_input:", igas, "is gas:", this%gas_names(igas+1)
! end do
! print *, "nn_input:", igas, "is temperature"
! igas = igas + 1
! print *, "nn_input:", igas, "is pressure"

! call zero_array(ngpt, nlay, ncol, tau)
tau = 0.0_wp

! ---- calculate gas optical depths ---- ------------------------------------------------------------------------

call system_clock(count_rate=count_rate)
call system_clock(iTime1)   

#ifdef USE_TIMING
    ret =  gptlstart('predict_nn_lw')
#endif
  
  !   if (abs(testparam-doubleparam)>1.0d-15 ) then
  !     print *, "using BLAS, single precision (SGEMV)"
  !     call change_kernel(net_pfrac,     output_sgemv_flatmodel, output_sgemm_flatmodel)
  !     call change_kernel(net_tau_tropo, output_sgemv_flatmodel, output_sgemm_flatmodel)
  !     call change_kernel(net_tau_strato,output_sgemv_flatmodel, output_sgemm_flatmodel)
  !   else
  !     print *, "using BLAS, double precision (DGEMV)"
  !     ! call change_kernel(net_pfrac,     output_dgemv_flatmodel)
  !     ! call change_kernel(net_tau_tropo, output_dgemv_flatmodel)
  !     ! call change_kernel(net_tau_strato,output_dgemv_flatmodel)
  !   end if

do inet = 1, size(neural_nets)
  ! If NN models have the same amount of neurons in each layer, use an optimized kernel..
  neurons_first = size(neural_nets(inet) % layers(1) % w_transposed, 1)
  neurons_last = size(neural_nets(inet) % layers(size(neural_nets(inet)%layers)-1) % w_transposed, 2)
  if (neurons_first == neurons_last) then
    ! print *, "Flat model"
    call change_kernel(neural_nets(inet), output_opt_flatmodel, output_sgemm_flatmodel)
  end if
end do

if (use_blas) then
  ! print *, "Using BLAS for matrix-matrix computations, predicting all layers and colums simultaenously"
  call predict_nn_lw_blas(                  &
                        ncol,nlay,ngpt, ngas,     &  ! dimensions
                        nn_inputs,                &  ! data inputs
                        neural_nets,              &  ! NN models (input)
                        tau, pfrac)    ! outputs

else
  ! print *, "Not using BLAS, predicting one layer at a time"

  call predict_nn_lw(                           &
                      ncol,nlay,ngpt, ngas,     &  ! dimensions
                      nn_inputs,                &  ! data inputs
                      neural_nets,              &  ! NN models (input)
                      tau, pfrac)    ! outputs
end if


#ifdef USE_TIMING
    ret =  gptlstop('predict_nn_lw')
#endif

call system_clock(iTime2)
print *,'Elapsed time on optical depth kernel: ',real(iTime2-iTime1)/real(count_rate)

if (error_msg /= '') return

#ifdef USE_TIMING
    ret =  gptlstart('combine_and_reorder')
#endif

call system_clock(count_rate=count_rate)
call system_clock(iTime1)   

! Combine optical depths and reorder for radiative transfer solver.
call reorder123x321(tau, optical_props%tau)
!call combine_and_reorder(tau, tau_rayleigh, allocated(this%krayl), optical_props)

call system_clock(iTime2)
print *,'Elapsed time on combine and reorder: ',real(iTime2-iTime1)/real(count_rate)

#ifdef USE_TIMING
    ret =  gptlstop('combine_and_reorder')
#endif

end function compute_taus_pfracs_nnlw

  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Function to define names of key and minor gases to be used by gas_optics().
  ! The final list gases includes those that are defined in gas_optics_specification
  ! and are provided in ty_gas_concs.
  !
  function get_minor_list(this, gas_desc, ngas, names_spec)
    class(ty_gas_optics_rrtmgp), intent(in)       :: this
    class(ty_gas_concs), intent(in)                      :: gas_desc
    integer, intent(in)                                  :: ngas
    character(32), dimension(ngas), intent(in)           :: names_spec

    ! List of minor gases to be used in gas_optics()
    character(len=32), dimension(:), allocatable         :: get_minor_list
    ! Logical flag for minor species in specification (T = minor; F = not minor)
    logical, dimension(size(names_spec))                 :: gas_is_present
    integer                                              :: igas, icnt

    if (allocated(get_minor_list)) deallocate(get_minor_list)
    do igas = 1, this%get_ngas()
      gas_is_present(igas) = string_in_array(names_spec(igas), gas_desc%gas_name)
    end do
    icnt = count(gas_is_present)
    allocate(get_minor_list(icnt))
    get_minor_list(:) = pack(this%gas_names, mask=gas_is_present)
  end function get_minor_list
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
    source_is_internal = allocated(this%totplnk) .and. allocated(this%planck_frac)
  end function source_is_internal
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! return true if initialized for external sources, false otherwise
  !
  pure function source_is_external(this)
    class(ty_gas_optics_rrtmgp), intent(in) :: this
    logical                          :: source_is_external
    source_is_external = allocated(this%solar_src)
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
  function get_col_dry(vmr_h2o, plev, latitude) result(col_dry)
    ! input
    real(wp), dimension(:,:), intent(in) :: vmr_h2o  ! volume mixing ratio of water vapor to dry air; (ncol,nlay)
    real(wp), dimension(:,:), intent(in) :: plev     ! Layer boundary pressures [Pa] (ncol,nlay+1)
    real(wp), dimension(:),   optional, &
                              intent(in) :: latitude ! Latitude [degrees] (ncol)
    ! output
    real(wp), dimension(size(plev,dim=1),size(plev,dim=2)-1) :: col_dry ! Column dry amount (ncol,nlay)
    ! ------------------------------------------------
    ! first and second term of Helmert formula
    real(wp), parameter :: helmert1 = 9.80665_wp
    real(wp), parameter :: helmert2 = 0.02586_wp
    ! local variables
    real(wp), dimension(size(plev,dim=1)) :: g0 ! (ncol)
    real(wp):: delta_plev, m_air, fact
    integer :: ncol, nlev
    integer :: icol, ilev ! nlay = nlev-1
    ! ------------------------------------------------
    ncol = size(plev, dim=1)
    nlev = size(plev, dim=2)
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

    !$acc parallel loop gang vector collapse(2) copyin(plev,vmr_h2o) copyout(col_dry)
    do ilev = 1, nlev-1
      do icol = 1, ncol
        delta_plev = abs(plev(icol,ilev) - plev(icol,ilev+1))
        ! Get average mass of moist air per mole of moist air
        fact = 1._wp / (1.+vmr_h2o(icol,ilev))
        m_air = (m_dry + m_h2o * vmr_h2o(icol,ilev)) * fact
        col_dry(icol,ilev) = 10._wp * delta_plev * avogad * fact/(1000._wp*m_air*100._wp*g0(icol))
      end do
    end do
    !$acc exit data delete (g0)
  end function get_col_dry
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
        key_species_list(:,i) = rewrite_key_species_pair(key_species_list(:,i))
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
      scaling_gas_atm_red = scaling_gas_atm
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
 !   (and reorder them for convenience, while we're at it)
 !
 subroutine combine_and_reorder(tau, tau_rayleigh, has_rayleigh, optical_props)
    real(wp), dimension(:,:,:),   intent(in) :: tau
    real(wp), dimension(:,:,:),   intent(in) :: tau_rayleigh
    logical,                      intent(in) :: has_rayleigh
    class(ty_optical_props_arry), intent(inout) :: optical_props

    integer :: ncol, nlay, ngpt, nmom

    ncol = size(tau, 3)
    nlay = size(tau, 2)
    ngpt = size(tau, 1)
    !$acc enter data copyin(optical_props)
    if (.not. has_rayleigh) then
      ! index reorder (ngpt, nlay, ncol) -> (ncol,nlay,gpt)
      !$acc enter data copyin(tau)
      !$acc enter data create(optical_props%tau)
      call reorder123x321(tau, optical_props%tau)
      select type(optical_props)
        type is (ty_optical_props_2str)
          !$acc enter data create(optical_props%ssa, optical_props%g)
          call zero_array(     ncol,nlay,ngpt,optical_props%ssa)
          call zero_array(     ncol,nlay,ngpt,optical_props%g  )
          !$acc exit data copyout(optical_props%ssa, optical_props%g)
        type is (ty_optical_props_nstr) ! We ought to be able to combine this with above
          nmom = size(optical_props%p, 1)
          !$acc enter data create(optical_props%ssa, optical_props%p)
          call zero_array(     ncol,nlay,ngpt,optical_props%ssa)
          call zero_array(nmom,ncol,nlay,ngpt,optical_props%p  )
          !$acc exit data copyout(optical_props%ssa, optical_props%p)
        end select
      !$acc exit data copyout(optical_props%tau)
      !$acc exit data delete(tau)
    else
      ! combine optical depth and rayleigh scattering
      !$acc enter data copyin(tau, tau_rayleigh)
      select type(optical_props)
        type is (ty_optical_props_1scl)
          ! User is asking for absorption optical depth
          !$acc enter data create(optical_props%tau)
          call reorder123x321(tau, optical_props%tau)
          !$acc exit data copyout(optical_props%tau)
        type is (ty_optical_props_2str)
          !$acc enter data create(optical_props%tau, optical_props%ssa, optical_props%g)
          call combine_and_reorder_2str(ncol, nlay, ngpt,       tau, tau_rayleigh, &
                                        optical_props%tau, optical_props%ssa, optical_props%g)
          !$acc exit data copyout(optical_props%tau, optical_props%ssa, optical_props%g)
        type is (ty_optical_props_nstr) ! We ought to be able to combine this with above
          nmom = size(optical_props%p, 1)
          !$acc enter data create(optical_props%tau, optical_props%ssa, optical_props%p)
          call combine_and_reorder_nstr(ncol, nlay, ngpt, nmom, tau, tau_rayleigh, &
                                        optical_props%tau, optical_props%ssa, optical_props%p)
          !$acc exit data copyout(optical_props%tau, optical_props%ssa, optical_props%p)
      end select
      !$acc exit data delete(tau, tau_rayleigh)
    end if
    !$acc exit data copyout(optical_props)
  end subroutine combine_and_reorder

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
