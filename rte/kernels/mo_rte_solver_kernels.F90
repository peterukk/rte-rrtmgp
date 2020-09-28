! This code is part of Radiative Transfer for Energetics (RTE)
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
! Numeric calculations for radiative transfer solvers.
!   Emission/absorption (no-scattering) calculations
!     solver for multi-angle Gaussian quadrature
!     solver for a single angle, calling
!       source function computation (linear-in-tau)
!       transport
!   Extinction-only calculation (direct solar beam)
!   Two-stream calculations
!     solvers for LW and SW with different boundary conditions and source functions
!       source function calculation for LW, SW
!       two-stream calculations for LW, SW (using different assumtions about phase function)
!       transport (adding)
!   Application of boundary conditions
!
! -------------------------------------------------------------------------------------------------
module mo_rte_solver_kernels
  use,  intrinsic :: iso_c_binding
  use mo_rte_kind, only: wp, sp, dp, wl
  use mo_fluxes_broadband_kernels, only : sum_broadband, sum_broadband_nocol
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif
  implicit none
  private

  ! Temporary solution until a single kernel robust to lower precision is developed. 
  ! sw_two_stream_sp is called when wp=sp and uses dp for some critical variables
  interface sw_two_stream
    module procedure sw_two_stream_sp, sw_two_stream_dp
  end interface sw_two_stream

  interface apply_BC
    module procedure apply_BC, apply_BC_old, apply_BC_nocol, apply_BC_factor, apply_BC_0
  end interface apply_BC

  public :: apply_BC, &
            lw_solver_noscat, lw_solver_noscat_GaussQuad, lw_solver_2stream,  &
            lw_solver_noscat_broadband, lw_solver_noscat_GaussQuad_broadband, &
            sw_solver_noscat,                             sw_solver_2stream, &
            sw_solver_noscat_broadband, sw_solver_2stream_broadband

  public :: lw_solver_1rescl_GaussQuad,  lw_solver_1rescl

  ! These routines don't really need to be visible but making them so is useful for testing.
  public :: lw_source_noscat, lw_combine_sources, &
            lw_source_2str, sw_source_2str, &
            lw_two_stream, sw_two_stream, &
            adding, lw_gpt_source

  real(wp), parameter :: pi = acos(-1._wp)

#ifdef USE_TIMING
  integer :: ret, i
#endif
contains

#ifdef FAST_EXPONENTIAL
  !---------------------------------------------------------------------
  ! Fast exponential for negative arguments: a Pade approximant that
  ! doesn't go negative for negative arguments, applied to arg/8, and
  ! the result is then squared three times
  elemental function exp_fast(arg) result(ex)
    real(wp), intent(in)  :: arg
    real(wp)              :: ex
    ex = 1.0_wp / (1.0_wp + arg*(-0.125_wp &
         + arg*(0.0078125_wp - 0.000325520833333333_wp * arg)))
    ex = ex*ex
    ex = ex*ex
    ex = ex*ex
  end function exp_fast

  elemental function exp_fast_double(arg) result(ex)
  real(dp), intent(in)  :: arg
  real(dp)              :: ex
  ex = 1.0_wp / (1.0_wp + arg*(-0.125_wp &
       + arg*(0.0078125_wp - 0.000325520833333333_wp * arg)))
  ex = ex*ex
  ex = ex*ex
  ex = ex*ex
end function exp_fast_double
#else
#define exp_fast exp
#define exp_fast_double exp
#endif
  ! -------------------------------------------------------------------------------------------------
  !
  ! Top-level longwave kernels
  !
  ! -------------------------------------------------------------------------------------------------
  !
  ! LW fluxes, no scattering, mu (cosine of integration angle) specified by column
  !   Does radiation calculation at user-supplied angles; converts radiances to flux
  !   using user-supplied weights
  !
  ! ---------------------------------------------------------------
subroutine lw_solver_noscat(nbnd, ngpt, nlay, ncol, top_at_1, D, weight, band_limits, &
                              tau, planck_frac, &
                              lay_source_bnd, lev_source_bnd, &
                              sfc_source_bnd, sfc_emis, &
                              radn_up, radn_dn, &
                              sfc_source_bnd_Jac, radn_up_Jac) bind(C, name="lw_solver_noscat")
    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: D            ! secant of propagation angle  []
    real(wp),                              intent(in   ) :: weight       ! quadrature weight
    integer,  dimension(2,nbnd),           intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: planck_frac  ! Planck fractions (fraction of band source function associated with each g-point)
    real(wp), dimension(nbnd,nlay,  ncol), intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in   ) :: lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis            ! Surface emissivity      []
    ! Outputs
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(out) :: radn_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) :: radn_dn      ! Top level must contain incident flux boundary condition
    ! real(wp), dimension(nlay+1,     ncol), optional, &
    !                                       intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(out) ::  radn_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    ! ------------------------------------
    ! Local variables. no col dependency
    real(wp), dimension(:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    real(wp), dimension(ngpt,nlay),   target              :: lev_source_dec, lev_source_inc
    real(wp), dimension(ngpt,nlay)                        :: lay_source
    real(wp), dimension(ngpt)                             :: sfc_src       ! Surface source function by g-point [W/m2]
    real(wp), dimension(ngpt)                             :: sfc_srcJac   ! Jacobian of surface source function by g-point [W/m2]
    real(wp), dimension(ngpt,nlay) :: tau_loc, &  ! path length (tau/mu)
                                        trans       ! transmissivity  = exp_fast(-tau)
    real(wp), dimension(ngpt,nlay) :: source_dn, source_up
    real(wp), dimension(ngpt     ) :: source_sfc, sfc_albedo, source_sfcJac

    real(wp), parameter :: pi = acos(-1._wp)
    integer             :: ilay, icol, igpt, sfc_lay, top_level
    ! ------------------------------------
    ! Which way is up?
    ! Level Planck sources for upward and downward radiation
    ! When top_at_1, lev_source_up => lev_source_dec
    !                lev_source_dn => lev_source_inc, and vice-versa
    if(top_at_1) then
      top_level = 1
      sfc_lay   = nlay  ! the layer (not level) closest to surface
      lev_source_up => lev_source_dec
      lev_source_dn => lev_source_inc
    else
      top_level = nlay+1
      sfc_lay = 1
      lev_source_up => lev_source_inc
      lev_source_dn => lev_source_dec
    end if

    do icol = 1, ncol
    
      !
      ! Transport is for intensity
      !   convert flux at top of domain to intensity assuming azimuthal isotropy
      !
      radn_dn(:,top_level,icol) = radn_dn(:,top_level,icol)/(2._wp * pi * weight)

      !
      ! Optical path and transmission, used in source function and transport calculations
      !
#ifdef USE_TIMING
    ret =  gptlstart('compute_trans_exp()')
#endif     
      do ilay = 1, nlay
        tau_loc(:,ilay)  = tau(:,ilay,icol) * D(:,icol)
        trans(:,ilay)    = exp_fast(-tau_loc(:,ilay)) 
      end do
#ifdef USE_TIMING
    ret =  gptlstop('compute_trans_exp()')
#endif   
      
      !
      ! Compute the source function per g-point from source function per band
      !
#ifdef USE_TIMING
    ret =  gptlstart('gpt_source')
#endif
      call lw_gpt_source_Jac(nbnd, ngpt, nlay, sfc_lay, band_limits, &
                    planck_frac(:,:,icol), lay_source_bnd(:,:,icol), lev_source_bnd(:,:,icol), &
                    sfc_source_bnd(:,icol), sfc_source_bnd_Jac(:,icol), &
                    sfc_src, sfc_srcJac, lay_source, lev_source_dec, lev_source_inc)
#ifdef USE_TIMING
    ret =  gptlstop('gpt_source')
#endif
      !
      ! Source function for diffuse radiation
      !
      call lw_source_noscat(ngpt, nlay, &
                            lay_source, lev_source_up, lev_source_dn, &
                            tau_loc, trans, source_dn, source_up)
      !
      ! Surface albedo, surface source function
      !
      sfc_albedo(:)     = 1._wp - sfc_emis(:,icol)
      source_sfc(:)     = sfc_emis(:,icol) * sfc_src
      source_sfcJac(:)  = sfc_emis(:,icol) * sfc_srcJac
      !
      ! Transport
      !      
      call lw_transport_noscat(ngpt, nlay, top_at_1,  &
                               tau_loc, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                               radn_up(:,:,icol), radn_dn(:,:,icol), &
                               source_sfcJac, radn_up_Jac(:,:,icol))
                               
      !
      ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
      !
      radn_dn(:,:,icol)     = 2._wp * pi * weight * radn_dn(:,:,icol)   
      radn_up(:,:,icol)     = 2._wp * pi * weight * radn_up(:,:,icol)  
      radn_up_Jac(:,:,icol)  = 2._wp * pi * weight * radn_up_Jac(:,:,icol)
    end do  ! column loop

  end subroutine lw_solver_noscat

  subroutine lw_solver_noscat_GaussQuad(nbnd, ngpt, nlay, ncol, top_at_1, nmus, Ds, weights, &
                                   band_limits, tau, planck_frac, &
                                   lay_source_bnd, lev_source_bnd, &
                                   sfc_source_bnd, sfc_emis, &
                                   flux_up, flux_dn, &
                                   sfc_source_bnd_Jac, flux_up_Jac) bind(C, name="lw_solver_noscat_GaussQuad")
    integer,                                intent(in   ) ::  nbnd, ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                            intent(in   ) ::  top_at_1
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(nmus),              intent(in   ) ::  Ds, weights  ! quadrature secants, weights
    ! real(wp), dimension(ngpt,ncol),         intent(in   ) ::  inc_flux    ! incident flux at domain top [W/m2] (ngpts, ncol)
    integer,  dimension(2,nbnd),            intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  planck_frac   ! Planck fractions (fraction of band source associated with each g-point) at layers
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis            ! Surface emissivity      []
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    ! Outputs
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(out) ::  flux_up      ! Radiances [W/m2-str]
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(inout) ::  flux_dn      ! Top level must contain incident flux boundary condition
    ! real(wp), dimension(nlay+1,     ncol),  optional, &
    !                                         intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of radiances [W/m2-str / K]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(out) ::  flux_up_Jac   ! surface temperature Jacobian of radiances [W/m2-str / K]                                        
    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt

    real(wp), dimension(:,:,:),  allocatable      :: radn_up, radn_dn ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:,:),  allocatable      :: radn_up_Jac      ! perturbed Fluxes per quad angle
    integer :: imu, icol, sfc_lay, igpt, ilay
    ! ------------------------------------
    !
    ! For the first angle output arrays store total flux
    !
    Ds_ngpt(:,:) = Ds(1)
  
    call lw_solver_noscat(nbnd, ngpt, nlay, ncol, top_at_1, &
                          Ds_ngpt, weights(1), &
                          band_limits, tau, planck_frac, &
                          lay_source_bnd, lev_source_bnd, &
                          sfc_source_bnd, sfc_emis, &
                          flux_up, flux_dn, sfc_source_bnd_Jac, flux_up_Jac)

    if (nmus > 1) then
      allocate( radn_up(ngpt, nlay+1, ncol) )
      allocate( radn_dn(ngpt, nlay+1, ncol) )
      allocate( radn_up_Jac(ngpt, nlay+1, ncol) )
    end if 

    do imu = 2, nmus

      Ds_ngpt(:,:) = Ds(imu)

      call lw_solver_noscat(nbnd, ngpt, nlay, ncol, top_at_1, &
        Ds_ngpt, weights(imu), &
        band_limits, tau, planck_frac, &
        lay_source_bnd, lev_source_bnd, &
        sfc_source_bnd, sfc_emis, &
        radn_up, radn_dn, sfc_source_bnd_Jac, radn_up_Jac)

      flux_up = flux_up + radn_up
      flux_dn = flux_dn + radn_dn
      flux_up_Jac = flux_up_Jac + radn_up_Jac
    end do                      
  end subroutine lw_solver_noscat_GaussQuad

  subroutine lw_solver_noscat_broadband(nbnd, ngpt, nlay, ncol, top_at_1, D, weight, inc_flux, band_limits, &
                              tau, planck_frac, &
                              lay_source_bnd, lev_source_bnd, &
                              sfc_source_bnd, sfc_emis, &
                              flux_up, flux_dn, &
                              sfc_source_bnd_Jac, flux_up_Jac, compute_Jac) bind(C, name="lw_solver_noscat_broadband")
    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                           intent(in   ) :: top_at_1, compute_Jac
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: D            ! secant of propagation angle  []
    real(wp),                              intent(in   ) :: weight       ! quadrature weight
    real(wp), dimension(ngpt,ncol),        intent(in   ) :: inc_flux    ! incident flux at domain top [W/m2] (ngpts, ncol)
    integer, dimension(2,nbnd),            intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: planck_frac  ! Planck fractions (fraction of band source function associated with each g-point)
    real(wp), dimension(nbnd,nlay,  ncol), intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in   ) :: lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis            ! Surface emissivity      []
    ! Outputs
    real(wp), dimension(nlay+1,     ncol), intent(out)    :: flux_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(nlay+1,     ncol), intent(out)    :: flux_dn      ! Top level must contain incident flux boundary condition
    real(wp), dimension(nlay+1,     ncol), intent(inout)  ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    ! ------------------------------------
    ! Local variables. no col dependency
    ! real(wp), dimension(:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    ! real(wp), dimension(ngpt,nlay),   target              :: lev_source_dec, lev_source_inc
    ! real(wp), dimension(ngpt,nlay)                        :: lay_source
    ! real(wp), dimension(ngpt)                             :: sfc_src       ! Surface source function by g-point [W/m2]
    ! real(wp), dimension(ngpt)                             :: sfc_srcJac   ! Jacobian of surface source function by g-point [W/m2]
    real(wp), dimension(ngpt,nlay+1)                      :: radn_up          ! Radiances per g-point [W/m2-str]
    real(wp), dimension(ngpt,nlay+1)                      :: radn_dn          ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay+1)                      :: radn_up_Jac       ! surface temperature Jacobian of g-point radiances [W/m2-str / K]
    real(wp), dimension(ngpt,nlay) ::   tau_loc     ! path length (tau/mu)
    real(wp), dimension(ngpt,nlay) ::   trans       ! transmissivity  = exp_fast(-tau)
    ! real(wp), dimension(ngpt,nlay) :: source_dn, source_up
    ! real(wp), dimension(ngpt     ) :: source_sfc, sfc_albedo, source_sfcJac

    real(wp), parameter :: pi = acos(-1._wp)
    real(wp)            :: fac
    integer             :: ilev, icol, igpt, ilay, sfc_lay, top_level
    ! ------------------------------------
    ! Which way is up?
    ! Level Planck sources for upward and downward radiation
    ! When top_at_1, lev_source_up => lev_source_dec
    !                lev_source_dn => lev_source_inc, and vice-versa
#ifdef USE_TIMING
    ret =  gptlstart('lw_solver_noscat_broadband')
#endif
    if(top_at_1) then
      top_level = 1
      sfc_lay   = nlay  ! the layer (not level) closest to surface
      ! lev_source_up => lev_source_dec
      ! lev_source_dn => lev_source_inc
    else
      top_level = nlay+1
      sfc_lay = 1
      ! lev_source_up => lev_source_inc
      ! lev_source_dn => lev_source_dec
    end if

    do icol = 1, ncol
    
      ! Apply boundary condition
      radn_dn(:,top_level) = inc_flux(:,icol)
      !
      ! Transport is for intensity
      !   convert flux at top of domain to intensity assuming azimuthal isotropy
      !
      radn_dn(:,top_level) = radn_dn(:,top_level)/(2._wp * pi * weight)

      !
      ! Optical path and transmission, used in source function and transport calculations
      !
#ifdef USE_TIMING
    ret =  gptlstart('compute_trans_exp()')
#endif  
      do ilay = 1, nlay
          tau_loc(:,ilay)  = tau(:,ilay,icol) * D(:,icol)
          trans(:,ilay)    = exp_fast(-tau_loc(:,ilay)) 
      end do
#ifdef USE_TIMING
    ret =  gptlstop('compute_trans_exp()')
#endif  
      !
      ! Source function for diffuse radiation, plus transport without scattering
      !
#ifdef USE_TIMING
    ret =  gptlstart('lw_source_transport_noscat')
#endif  
      call lw_source_transport_noscat(nbnd, ngpt, nlay, sfc_lay, top_at_1, band_limits, &
                          planck_frac(:,:,icol), lay_source_bnd(:,:,icol), lev_source_bnd(:,:,icol), &
                          sfc_source_bnd(:,icol), sfc_source_bnd_Jac(:,icol), & 
                          tau_loc, trans, sfc_emis(:,icol),  &
                          radn_up, radn_dn, radn_up_Jac) ! outputs
#ifdef USE_TIMING
    ret =  gptlstop('lw_source_transport_noscat')
#endif  
      !
      ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
      !
      fac         = 2._wp * pi * weight
      radn_dn     = fac * radn_dn   
      radn_up     = fac * radn_up   
      radn_up_Jac = fac * radn_up_Jac

#ifdef USE_TIMING
    ret =  gptlstart('spectral_reduction')
#endif
      ! Compute broadband fluxes
      call sum_broadband_nocol(ngpt, nlay+1, radn_up, flux_up(:,icol) )
      call sum_broadband_nocol(ngpt, nlay+1, radn_dn, flux_dn(:,icol) )

      if (compute_Jac) then
        call sum_broadband_nocol(ngpt, nlay+1, radn_up_Jac, flux_up_Jac(:,icol) )
      end if
#ifdef USE_TIMING
    ret =  gptlstop('spectral_reduction')
#endif
    end do  ! column loop

#ifdef USE_TIMING
    ret =  gptlstop('lw_solver_noscat_broadband')
#endif

  end subroutine lw_solver_noscat_broadband

  subroutine lw_solver_noscat_GaussQuad_broadband(nbnd, ngpt, nlay, ncol, top_at_1, nmus, Ds, weights, inc_flux, &
                                   band_limits, tau, planck_frac, &
                                   lay_source_bnd, lev_source_bnd, &
                                   sfc_source_bnd, sfc_emis, &
                                   flux_up, flux_dn, &
                                   sfc_source_bnd_Jac, flux_up_Jac, compute_Jac) bind(C, name="lw_solver_noscat_GaussQuad_broadband")
    integer,                                intent(in   ) ::  nbnd, ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                            intent(in   ) ::  top_at_1, compute_Jac
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(nmus),              intent(in   ) ::  Ds, weights  ! quadrature secants, weights
    real(wp), dimension(ngpt,ncol),         intent(in   ) ::  inc_flux    ! incident flux at domain top [W/m2] (ngpts, ncol)
    integer,  dimension(2,nbnd),            intent(in   ) ::  band_limits
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  planck_frac   ! Planck fractions (fraction of band source associated with each g-point) at layers
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis            ! Surface emissivity      []
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]

    ! Outputs
    real(wp), dimension(nlay+1,     ncol),  intent(out) ::  flux_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(nlay+1,     ncol),  intent(out) ::  flux_dn      ! Top level must contain incident flux boundary condition
    ! real(wp), dimension(nlay+1,     ncol),  optional, &
    !                                         intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    real(wp), dimension(nlay+1,     ncol), intent(out) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]                                        
    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt

    real(wp), dimension(:,:),  allocatable      :: radn_up, radn_dn ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:),  allocatable      :: radn_up_Jac      ! perturbed Fluxes per quad angle
    integer :: imu, icol, sfc_lay, igpt, ilay
    ! ------------------------------------
    !
    ! For the first angle output arrays store total flux
    !

    Ds_ngpt(:,:) = Ds(1)
  
    call lw_solver_noscat_broadband(nbnd, ngpt, nlay, ncol, top_at_1, &
                          Ds_ngpt, weights(1), inc_flux, &
                          band_limits, tau, planck_frac, &
                          lay_source_bnd, lev_source_bnd, &
                          sfc_source_bnd, sfc_emis, &
                          flux_up, flux_dn, sfc_source_bnd_Jac, flux_up_Jac, compute_Jac)

    if (nmus > 1) then
      allocate( radn_up(nlay+1, ncol) )
      allocate( radn_dn(nlay+1, ncol) )
      allocate( radn_up_Jac(nlay+1, ncol) )
    end if 

    do imu = 2, nmus

      Ds_ngpt(:,:) = Ds(imu)

      call lw_solver_noscat_broadband(nbnd, ngpt, nlay, ncol, top_at_1, &
        Ds_ngpt, weights(imu), inc_flux, &
        band_limits, tau, planck_frac, &
        lay_source_bnd, lev_source_bnd, &
        sfc_source_bnd, sfc_emis, &
        radn_up, radn_dn, sfc_source_bnd_Jac, radn_up_Jac, compute_Jac)

      flux_up = flux_up + radn_up
      flux_dn = flux_dn + radn_dn
      flux_up_Jac = flux_up_Jac + radn_up_Jac

    end do                      

  end subroutine lw_solver_noscat_GaussQuad_broadband

  ! -------------------------------------------------------------------------------------------------
  !
  ! Longwave two-stream calculation:
  !   combine RRTMGP-specific sources at levels
  !   compute layer reflectance, transmittance
  !   compute total source function at levels using linear-in-tau
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_solver_2stream (nbnd, ngpt, nlay, ncol, top_at_1, &
                              band_limits, tau, ssa, g, planck_frac, &
                              lay_source_bnd, lev_source_bnd, &
                              sfc_source_bnd, sfc_emis, &
                              flux_up, flux_dn) bind(C, name="lw_solver_2stream")
    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g, &        ! asymmetry parameter []
                                                            planck_frac ! planck fraction
    integer, dimension(2,nbnd),            intent(in   ) :: band_limits
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd ! Planck source function at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd ! Surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   )  ::  sfc_emis         ! Surface emissivity      []
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                           intent(  out)  :: flux_up   ! Fluxes [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                           intent(inout) :: flux_dn   ! Top level (= merge(1, nlay+1, top_at_1)
                                                                      ! must contain incident flux boundary condition
    ! ----------------------------------------------------------------------
    integer :: icol, sfc_lay, top_level
    real(wp), dimension(ngpt,nlay) :: lev_source_inc, lev_source_dec
                                        ! Planck source at layer edge for radiation in increasing/decreasing ilay direction [W/m2]
                                        ! Includes spectral weighting that accounts for state-dependent frequency to g-space mapping
    real(wp), dimension(ngpt,nlay) :: lay_source

    real(wp), dimension(ngpt)       :: sfc_src          ! Surface source function [W/m2]

    real(wp), dimension(ngpt,nlay  ) :: Rdif, Tdif, gamma1, gamma2
    real(wp), dimension(ngpt       ) :: sfc_albedo
    real(wp), dimension(ngpt,nlay+1) :: lev_source
    real(wp), dimension(ngpt,nlay  ) :: source_dn, source_up
    real(wp), dimension(ngpt       ) :: source_sfc
    ! ------------------------------------

    if(top_at_1) then
      top_level = 1
      sfc_lay = nlay
    else
      top_level = nlay+1
      sfc_lay = 1
    end if

    do icol = 1, ncol

      !
      ! Source function per g-point from source function per band
      !
#ifdef USE_TIMING
    ret =  gptlstart('gpt_source')
#endif
      call lw_gpt_source(nbnd, ngpt, nlay, sfc_lay, band_limits, &
                      planck_frac(:,:,icol), lay_source_bnd(:,:,icol), lev_source_bnd(:,:,icol), sfc_source_bnd(:,icol), &
                      sfc_src, lay_source, lev_source_dec, lev_source_inc)
#ifdef USE_TIMING
    ret =  gptlstop('gpt_source')
#endif
      !
      ! RRTMGP provides source functions at each level using the spectral mapping
      !   of each adjacent layer. Combine these for two-stream calculations
      !
      call lw_combine_sources(ngpt, nlay, &
                              lev_source_inc, lev_source_dec, &
                              lev_source)
      !
      ! Cell properties: reflection, transmission for diffuse radiation
      !   Coupling coefficients needed for source function
      !
      call lw_two_stream(ngpt, nlay,                                 &
                         tau (:,:,icol), ssa(:,:,icol), g(:,:,icol), &
                         gamma1, gamma2, Rdif, Tdif)
      !
      ! Source function for diffuse radiation
      !
      call lw_source_2str(ngpt, nlay, top_at_1, &
                          sfc_emis(:,icol), sfc_src(:), &
                          lay_source(:,:), lev_source, &
                          gamma1, gamma2, Rdif, Tdif, tau(:,:,icol), &
                          source_dn, source_up, source_sfc)
      !
      ! Transport
      !
      sfc_albedo(:) = 1._wp - sfc_emis(:,icol)
      call adding(ngpt, nlay, top_at_1,              &
                  sfc_albedo,                        &
                  Rdif, Tdif,                        &
                  source_dn, source_up, source_sfc,  &
                  flux_up(:,:,icol), flux_dn(:,:,icol))
    end do
  end subroutine lw_solver_2stream
  ! -------------------------------------------------------------------------------------------------
  !
  !   Top-level shortwave kernels
  !
  ! -------------------------------------------------------------------------------------------------
  !
  !   Extinction-only i.e. solar direct beam
  !
  ! -------------------------------------------------------------------------------------------------
pure subroutine sw_solver_noscat(ngpt, nlay, ncol, &
                              top_at_1, tau, mu0, flux_dir) bind(C, name="sw_solver_noscat")
    integer,                    intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                intent( in) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol), intent( in) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ncol),            intent( in) :: mu0          ! cosine of solar zenith angle
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: flux_dir     ! Direct-beam flux, spectral [W/m2]
                                                                       ! Top level must contain incident flux boundary condition
    integer :: igpt, ilev, icol
    real(wp), dimension(ncol) :: mu0_inv
    ! ------------------------------------
    mu0_inv = 1._wp/mu0
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    ! Downward propagation
    if(top_at_1) then
      ! For the flux at this level, what was the previous level, and which layer has the
      !   radiation just passed through?
      ! layer index = level index - 1
      ! previous level is up (-1)
      do icol = 1, ncol
        do ilev = 2, nlay+1
          flux_dir(:,ilev,icol) = flux_dir(:,ilev-1,icol) * exp_fast(-tau(:,ilev-1,icol)*mu0_inv(icol))
        end do
      end do
    else
      ! layer index = level index
      ! previous level is up (+1)
      do icol = 1, ncol
        do ilev = nlay, 1, -1
          flux_dir(:,ilev,icol) = flux_dir(:,ilev+1,icol) * exp_fast(-tau(:,ilev,icol)*mu0_inv(icol))
        end do
      end do
    end if
  end subroutine sw_solver_noscat
  ! -------------------------------------------------------------------------------------------------
  !
  ! Shortwave two-stream calculation:
  !   compute layer reflectance, transmittance
  !   compute solar source function for diffuse radiation
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  
  subroutine sw_solver_2stream (ngpt, nlay, ncol, top_at_1, &
                                 tau, ssa, g, mu0,           &
                                 sfc_alb_dir, sfc_alb_dif,   &
                                 flux_up, flux_dn, flux_dir) bind(C, name="sw_solver_2stream")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(ncol            ), intent(in   ) :: mu0     ! cosine of solar zenith angle
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_alb_dir, sfc_alb_dif
                                                                    ! Spectral albedo of surface to direct and diffuse radiation
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                           intent(  out) :: flux_up ! Fluxes [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), &                        ! Downward fluxes contain boundary conditions
                                           intent(inout) :: flux_dn, flux_dir
    ! -------------------------------------------
    integer :: icol, igpt
    real(wp), dimension(ngpt,nlay) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    real(wp), dimension(ngpt,nlay) :: source_up, source_dn
    real(wp), dimension(ngpt     ) :: source_srf

    ! ------------------------------------

    do icol = 1, ncol
      !
      ! Cell properties: transmittance and reflectance for direct and diffuse radiation
      !
      call sw_two_stream(ngpt, nlay, mu0(icol),                             &
                         tau (:,:,icol), ssa (:,:,icol), g(:,:,icol),       &
                         Rdif, Tdif, Rdir, Tdir, Tnoscat)   
                         
      !
      ! Direct-beam and source for diffuse radiation
      !
      call sw_source_2str(ngpt, nlay, top_at_1, Rdir, Tdir, Tnoscat, sfc_alb_dir(:,icol),&
                          source_up, source_dn, source_srf, flux_dir(:,:,icol))

      !
      ! Transport
      !
      call adding(ngpt, nlay, top_at_1,            &
                     sfc_alb_dif(:,icol), Rdif, Tdif, &
                     source_dn, source_up, source_srf, flux_up(:,:,icol), flux_dn(:,:,icol))
      !
      ! adding computes only diffuse flux; flux_dn is total
      !
      flux_dn(:,:,icol) = flux_dn(:,:,icol) + flux_dir(:,:,icol)
    end do

  end subroutine sw_solver_2stream
    ! -------------------------------------------------------------------------------------------------
  !
  !   Top-level shortwave kernels, return broadband fluxes
  !
  ! -------------------------------------------------------------------------------------------------
  !
  !   Extinction-only i.e. solar direct beam
  !
  ! -------------------------------------------------------------------------------------------------
pure subroutine sw_solver_noscat_broadband(ngpt, nlay, ncol, &
                              top_at_1, inc_flux, tau, mu0, flux_dir) bind(C, name="sw_solver_noscat_broadband")
    integer,                    intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                intent( in) :: top_at_1
    real(wp), dimension(ngpt,ncol),         intent( in) :: inc_flux     ! incident flux at top of domain [W/m2] (ngpt, ncol)
    real(wp), dimension(ngpt,nlay,  ncol),  intent( in) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ncol),              intent( in) :: mu0          ! cosine of solar zenith angle

    ! real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: flux_dir     ! Direct-beam flux, spectral [W/m2]
                                                                       ! Top level must contain incident flux boundary condition
    real(wp), dimension(nlay+1,ncol), intent(inout) :: flux_dir     ! Direct-beam flux, broadband [W/m2]

    real(wp), dimension(ngpt,nlay+1)                :: radn_dir     ! Direct-beam flux, spectral [W/m2]

    integer :: igpt, ilev, icol
    real(wp), dimension(ncol) :: mu0_inv
    ! ------------------------------------
    mu0_inv = 1._wp/mu0
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    ! Downward propagation
    if(top_at_1) then

      do icol = 1, ncol
    
        ! Apply boundary condition
        radn_dir(:,nlay+1) = inc_flux(:,icol)

        ! For the flux at this level, what was the previous level, and which layer has the
        !   radiation just passed through?
        ! layer index = level index - 1
        ! previous level is up (-1)
        do ilev = 2, nlay+1
          radn_dir(:,ilev) = radn_dir(:,ilev-1) * exp_fast(-tau(:,ilev-1,icol)*mu0_inv(icol))
        end do

        ! Compute broadband fluxes
        call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )

      end do

    else

      ! Apply boundary condition
      radn_dir(:,1) = inc_flux(:,icol)

      do icol = 1, ncol

        ! layer index = level index
        ! previous level is up (+1)
        do ilev = nlay, 1, -1
          radn_dir(:,ilev) = radn_dir(:,ilev+1) * exp_fast(-tau(:,ilev,icol)*mu0_inv(icol))
        end do

        ! Compute broadband fluxes
        call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )

      end do 

    end if
  end subroutine sw_solver_noscat_broadband
  ! -------------------------------------------------------------------------------------------------
  !
  ! Shortwave two-stream calculation:
  !   compute layer reflectance, transmittance
  !   compute solar source function for diffuse radiation
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  
  subroutine sw_solver_2stream_broadband(ngpt, nlay, ncol, top_at_1, &
                                 inc_flux, inc_flux_dif,     &
                                 tau, ssa, g, mu0,           &
                                 sfc_alb_dir, sfc_alb_dif,   &
                                 flux_up, flux_dn, flux_dir) bind(C, name="sw_solver_2stream_broadband")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: inc_flux, inc_flux_dif     ! incident flux at top of domain [W/m2] (ngpt, ncol)
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(ncol            ), intent(in   ) :: mu0     ! cosine of solar zenith angle
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_alb_dir, sfc_alb_dif
                                                                    ! Spectral albedo of surface to direct and diffuse radiation
                                                            ! Broadband fluxes  [W/m2]
    real(wp), dimension(nlay+1,ncol),      intent(inout) :: flux_up, flux_dn, flux_dir
    ! -------------------------------------------
    real(wp), dimension(ngpt,nlay+1) :: radn_up            ! Radiative fluxes [W/m2]
    real(wp), dimension(ngpt,nlay+1) :: radn_dn, radn_dir  ! Downward fluxes get boundary conditions
    integer :: icol, igpt, top_level
    real(wp), dimension(ngpt,nlay) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    real(wp), dimension(ngpt,nlay) :: source_up, source_dn
    real(wp), dimension(ngpt     ) :: source_srf

    ! ------------------------------------

    if(top_at_1) then
      top_level = 1
    else
      top_level = nlay+1
    end if

    do icol = 1, ncol

      ! Apply boundary condition
      radn_dir(:,top_level) = inc_flux(:,icol) * mu0(icol)
      radn_dn(:,top_level)  = inc_flux_dif(:,icol)

      !
      ! Cell properties: transmittance and reflectance for direct and diffuse radiation
      !
#ifdef USE_TIMING
    ret =  gptlstart('sw_two_stream')
#endif
      call sw_two_stream(ngpt, nlay, mu0(icol),                                &
                         tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), &
                         Rdif, Tdif, Rdir, Tdir, Tnoscat)     
      ! call sw_two_stream_sp(ngpt, nlay, mu0(icol),                                &
      !                    tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), &
      !                    Rdif, Tdif, Rdir, Tdir, Tnoscat)                       
#ifdef USE_TIMING
    ret =  gptlstop('sw_two_stream')
#endif    
      !
      ! Direct-beam and source for diffuse radiation
      !
#ifdef USE_TIMING
    ret =  gptlstart('sw_source_2str')
#endif
      call sw_source_2str(ngpt, nlay, top_at_1, Rdir, Tdir, Tnoscat, sfc_alb_dir(:,icol),&
                          source_up, source_dn, source_srf, radn_dir)
#ifdef USE_TIMING
    ret =  gptlstop('sw_source_2str')
#endif
      !
      ! Transport
      !
#ifdef USE_TIMING
    ret =  gptlstart('adding')
#endif
      call adding(ngpt, nlay, top_at_1,            &
                     sfc_alb_dif(:,icol), Rdif, Tdif, &
                     source_dn, source_up, source_srf, radn_up, radn_dn)
#ifdef USE_TIMING
    ret =  gptlstop('adding')
#endif                    
      !
      ! adding computes only diffuse flux; flux_dn is total
      !
      radn_dn = radn_dn + radn_dir
#ifdef USE_TIMING
    ret =  gptlstart('sum_broadband_nocol')
#endif  
      ! Compute broadband fluxes
      call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )
      call sum_broadband_nocol(ngpt, nlay+1, radn_up, flux_up(:,icol) )
      call sum_broadband_nocol(ngpt, nlay+1, radn_dn, flux_dn(:,icol) )
#ifdef USE_TIMING
    ret =  gptlstop('sum_broadband_nocol')
#endif    
    end do

  end subroutine sw_solver_2stream_broadband
  ! -------------------------------------------------------------------------------------------------
  !
  !   Lower-level longwave kernels
  !
  ! -------------------------------------------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption
  ! See Clough et al., 1992, doi: 10.1029/92JD01419, Eq 13
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_noscat(ngpt, nlay, lay_source, lev_source_up, lev_source_dn, tau, trans, &
                              source_dn, source_up) bind(C, name="lw_source_noscat")
    integer,                         intent(in) :: ngpt, nlay
    real(wp), dimension(ngpt, nlay), intent(in) :: lay_source, & ! Planck source at layer center
                                                   lev_source_up, & ! Planck source at levels (layer edges),
                                                   lev_source_dn, & !   increasing/decreasing layer index
                                                   tau,        & ! Optical path (tau/mu)
                                                   trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt, nlay), intent(out):: source_dn, source_up
                                                                   ! Source function at layer edges
                                                                   ! Down at the bottom of the layer, up at the top
    ! --------------------------------
    integer             :: igpt, ilay
    real(wp)            :: fact
    real(wp), parameter :: tau_thresh = sqrt(epsilon(tau))
    ! ---------------------------------------------------------------
    do ilay = 1, nlay
      do igpt = 1, ngpt
      !
      ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
      !   is of order epsilon (smallest difference from 1. in working precision)
      !   Thanks to Peter Blossey
      !
      if(tau(igpt, ilay) > tau_thresh) then
        fact = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
      else
        fact = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
      end if
      !
      ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
      !
      source_dn(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_dn(igpt,ilay) + &
                              2._wp * fact * (lay_source(igpt,ilay) - lev_source_dn(igpt,ilay))
      source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_up(igpt,ilay  ) + &
                              2._wp * fact * (lay_source(igpt,ilay) - lev_source_up(igpt,ilay))
      end do
    end do
  end subroutine lw_source_noscat

  ! -------------------------------------------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption,
  ! ( See Clough et al., 1992, doi: 10.1029/92JD01419, Eq 13 )
  ! COMBINE with longwave no-scattering transport in order to avoid traversing arrays more than necessary
  !
  ! ---------------------------------------------------------------
  pure subroutine lw_source_transport_noscat(nbnd, ngpt, nlay, sfc_lay, top_at_1, band_limits, planck_frac, & ! inputs
                    lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_source_bnd_Jac, &    ! inputs
                    tau, trans, sfc_emis, &                   ! more inputs
                    radn_up, radn_dn, radn_up_Jac)            ! layer outputs
    integer,                          intent(in   ) :: nbnd, ngpt, nlay, sfc_lay
    logical(wl),                      intent(in   ) :: top_at_1
    integer, dimension(2,nbnd),       intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay),   intent(in   ) :: planck_frac 
    real(wp), dimension(nbnd,nlay),   intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1), intent(in )   :: lev_source_bnd
    real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd      ! Surface source by band
    real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd_Jac  ! Surface source by band using perturbed temperature
    real(wp), dimension(ngpt, nlay),  intent(in)    :: tau,        & ! Optical path (tau/mu)
                                                      trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt),        intent(in)    :: sfc_emis
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up_Jac       ! surface temperature Jacobian of Radiances [W/m2-str / K]
    ! --------------------------------
    integer                         :: igpt, ilay, ibnd
    real(wp)                        :: fact
    real(wp), parameter             :: tau_thresh = sqrt(epsilon(tau))
    real(wp)                        :: lay_source, lev_source_dec, lev_source_inc
    real(wp)                        :: source_sfc, source_sfc_Jac, sfc_albedo
    real(wp)                        :: source_dn ! Source function at layer edges; 
    real(wp), dimension(ngpt, nlay) :: source_up !  Down at the bottom of the layer, up at the top
    ! ---------------------------------------------------------------

    if(top_at_1) then
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          !dir$ vector aligned
          do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
            !
            ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
            !   is of order epsilon (smallest difference from 1. in working precision)
            !   Thanks to Peter Blossey
            !
            if(tau(igpt, ilay) > tau_thresh) then
              fact = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              fact = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if

            ! compute level source irradiance for each g-point, one each for upward and downward paths
            lev_source_dec   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay)
            lev_source_inc   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay+1)
            ! compute layer source irradiance for each g-point
            lay_source       = planck_frac(igpt,ilay) * lay_source_bnd(ibnd,ilay)
            !
            ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
            !
            source_dn           = (1._wp - trans(igpt,ilay)) * lev_source_inc + &
                                  2._wp * fact * (lay_source - lev_source_inc)
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_dec + &
                                  2._wp * fact * (lay_source - lev_source_dec)
            ! Downward propagation                    
            radn_dn(igpt,ilay+1) = trans(igpt,ilay)*radn_dn(igpt,ilay) + source_dn
          end do
        end do
      end do

      ! Surface source, reflection and emission
      do ibnd = 1, nbnd
        !dir$ vector aligned
        do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
          source_sfc                = planck_frac(igpt,sfc_lay) * sfc_source_bnd(ibnd)
          source_sfc                = sfc_emis(igpt) * source_sfc
          source_sfc_Jac            = planck_frac(igpt,sfc_lay) * (sfc_source_bnd(ibnd) - sfc_source_bnd_Jac(ibnd))
          source_sfc_Jac            = sfc_emis(igpt) * source_sfc_Jac
          sfc_albedo                = 1._wp - sfc_emis(igpt)
          ! Reflection and emission
          radn_up    (igpt,nlay+1)  = radn_dn(igpt,nlay+1)*sfc_albedo + source_sfc
          radn_up_Jac(igpt,nlay+1)  = source_sfc_Jac
        end do
      end do 

      ! Upward propagation
      do ilay = nlay, 1, -1
        radn_up   (:,ilay)  = trans(:,ilay  )*radn_up   (:,ilay+1) + source_up(:,ilay)
        radn_up_Jac(:,ilay) = trans(:,ilay  )*radn_up_Jac(:,ilay+1)
      end do

    else  ! Top of domain is index nlay+1

      do ilay = nlay, 1, -1
        do ibnd = 1, nbnd
          !dir$ vector aligned
          do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
          ! compute layer source irradiance for each g-point
            lay_source       = planck_frac(igpt,ilay) * lay_source_bnd(ibnd,ilay)
            ! compute level source irradiance for each g-point, one each for upward and downward paths
            lev_source_dec   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay)
            lev_source_inc   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay+1)
            !
            ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
            !   is of order epsilon (smallest difference from 1. in working precision)
            !   Thanks to Peter Blossey
            !
            if(tau(igpt, ilay) > tau_thresh) then
              fact = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              fact = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if
            !
            ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
            !
            source_dn           = (1._wp - trans(igpt,ilay)) * lev_source_dec + &
                                  2._wp * fact * (lay_source - lev_source_dec)
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_inc + &
                                  2._wp * fact * (lay_source - lev_source_inc)
            ! Downward propagation
            radn_dn(igpt,ilay) = trans(igpt,ilay)*radn_dn(igpt,ilay+1) + source_dn
          end do
        end do
      end do
      
      ! Surface source, reflection and emission
      do ibnd = 1, nbnd
        !dir$ vector aligned
        do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
          source_sfc            = planck_frac(igpt,sfc_lay) * sfc_source_bnd(ibnd)
          source_sfc            = sfc_emis(igpt) * source_sfc
          source_sfc_Jac        = planck_frac(igpt,sfc_lay) * (sfc_source_bnd(ibnd) - sfc_source_bnd_Jac(ibnd))
          source_sfc_Jac        = sfc_emis(igpt) * source_sfc_Jac
          sfc_albedo            = 1._wp - sfc_emis(igpt)
          radn_up (igpt, 1)     = radn_dn(igpt,1) * sfc_albedo + source_sfc
          radn_up_Jac(igpt, 1)  = source_sfc_Jac
        end do
      end do 

      ! Upward propagation
      do ilay = 2, nlay+1
        radn_up   (:,ilay) = trans(:,ilay-1) * radn_up   (:,ilay-1) +  source_up(:,ilay-1)
        radn_up_Jac(:,ilay) = trans(:,ilay-1) * radn_up_Jac(:,ilay-1)
      end do

    end if

  end subroutine lw_source_transport_noscat
  ! -------------------------------------------------------------------------------------------------
  !
  ! Longwave no-scattering transport
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_transport_noscat(ngpt, nlay, top_at_1, &
                                 tau, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                                 radn_up, radn_dn, &
                                 source_sfcJac, radn_up_Jac) bind(C, name="lw_transport_noscat")
    integer,                          intent(in   ) :: ngpt, nlay ! Number of columns, layers, g-points
    logical(wl),                      intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: tau, &     ! Absorption optical thickness, pre-divided by mu []
                                                       trans      ! transmissivity = exp_fast(-tau)
    real(wp), dimension(ngpt       ), intent(in   ) :: sfc_albedo ! Surface albedo
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: source_dn, &
                                                       source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt       ), intent(in   ) :: source_sfc ! Surface source function [W/m2]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition


    real(wp), dimension(ngpt       ), intent(in )   :: source_sfcJac    ! surface temperature Jacobian of surface source function [W/m2/K]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up_Jac       ! surface temperature Jacobian of Radiances [W/m2-str / K]
    ! Local variables
    integer :: ilev, igpt
    ! ---------------------------------------------------

    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      ! Downward propagation
      do ilev = 2, nlay+1
        radn_dn(:,ilev) = trans(:,ilev-1)*radn_dn(:,ilev-1) + source_dn(:,ilev-1)
      end do

      ! Surface reflection and emission
      radn_up   (:,nlay+1) = radn_dn(:,nlay+1)*sfc_albedo(:) + source_sfc   (:)
      radn_up_Jac(:,nlay+1) = source_sfcJac(:)

      ! Upward propagation
      do ilev = nlay, 1, -1
        radn_up   (:,ilev) = trans(:,ilev  )*radn_up   (:,ilev+1) + source_up(:,ilev)
        radn_up_Jac(:,ilev) = trans(:,ilev  )*radn_up_Jac(:,ilev+1)
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      ! Downward propagation
      do ilev = nlay, 1, -1
        radn_dn(:,ilev) = trans(:,ilev  )*radn_dn(:,ilev+1) + source_dn(:,ilev)
      end do

      ! Surface reflection and emission
      radn_up   (:, 1) = radn_dn(:,1)*sfc_albedo(:) + source_sfc   (:)
      radn_up_Jac(:, 1) = source_sfcJac(:)

      ! Upward propagation
      do ilev = 2, nlay+1
        radn_up   (:,ilev) = trans(:,ilev-1) * radn_up   (:,ilev-1) +  source_up(:,ilev-1)
        radn_up_Jac(:,ilev) = trans(:,ilev-1) * radn_up_Jac(:,ilev-1)
      end do
    end if
  end subroutine lw_transport_noscat
  ! -------------------------------------------------------------------------------------------------
  !
  ! Longwave two-stream solutions to diffuse reflectance and transmittance for a layer
  !    with optical depth tau, single scattering albedo w0, and asymmetery parameter g.
  !
  ! Equations are developed in Meador and Weaver, 1980,
  !    doi:10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2
  !
  pure subroutine lw_two_stream(ngpt, nlay, tau, w0, g, &
                                gamma1, gamma2, Rdif, Tdif) bind(C, name="lw_two_stream")
    integer,                        intent(in)  :: ngpt, nlay
    real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt,nlay), intent(out) :: gamma1, gamma2, Rdif, Tdif
    ! -----------------------
    integer  :: i, j
    ! Variables used in Meador and Weaver
    real(wp) :: k(ngpt)
    ! Ancillary variables
    real(wp) :: RT_term(ngpt)
    real(wp) :: exp_minusktau(ngpt), exp_minus2ktau(ngpt)
    real(wp), parameter :: LW_diff_sec = 1.66  ! 1./cos(diffusivity angle)
    ! ---------------------------------
    do j = 1, nlay
      do i = 1, ngpt
        !
        ! Coefficients differ from SW implementation because the phase function is more isotropic
        !   Here we follow Fu et al. 1997, doi:10.1175/1520-0469(1997)054<2799:MSPITI>2.0.CO;2
        !   and use a diffusivity sec of 1.66
        !
        gamma1(i,j)= LW_diff_sec * (1._wp - 0.5_wp * w0(i,j) * (1._wp + g(i,j))) ! Fu et al. Eq 2.9
        gamma2(i,j)= LW_diff_sec *          0.5_wp * w0(i,j) * (1._wp - g(i,j))  ! Fu et al. Eq 2.10
      end do
      ! Written to encourage vectorization of exponential, square root
      ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
      !   k = 0 for isotropic, conservative scattering; this lower limit on k
      !   gives relative error with respect to conservative solution
      !   of < 0.1% in Rdif down to tau = 10^-9
      k(:) = sqrt(max((gamma1(:,j) - gamma2(:,j)) * &
                           (gamma1(:,j) + gamma2(:,j)),  &
                           1.e-12_wp))
      exp_minusktau(:) = exp_fast(-tau(:,j)*k(:))
      !
      ! Diffuse reflection and transmission
      !
      do i = 1, ngpt
        exp_minus2ktau(i) = exp_minusktau(i) * exp_minusktau(i)

        ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term(i) = 1._wp / (k     (i  ) * (1._wp + exp_minus2ktau(i))  + &
                              gamma1(i,j) * (1._wp - exp_minus2ktau(i)) )

        ! Equation 25
        Rdif(i,j) = RT_term(i) * gamma2(i,j) * (1._wp - exp_minus2ktau(i))

        ! Equation 26
        Tdif(i,j) = RT_term(i) * 2._wp * k(i) * exp_minusktau(i)
      end do

    end do
  end subroutine lw_two_stream

  ! -------------------------------------------------------------------------------------------------
  !
  ! Source function combination
  ! RRTMGP provides two source functions at each level
  !   using the spectral mapping from each of the adjascent layers.
  !   Need to combine these for use in two-stream calculation.
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_combine_sources(ngpt, nlay, lev_src_inc, lev_src_dec, lev_source)
    integer,                                 intent(in ) :: nlay, ngpt
    real(wp), dimension(ngpt, nlay  ), intent(in ) :: lev_src_inc, lev_src_dec
    real(wp), dimension(ngpt, nlay+1), intent(out) :: lev_source

    integer :: ilay, igpt
    ! ---------------------------------------------------------------

    do igpt = 1,ngpt
	    lev_source(igpt, 1) =      lev_src_dec(igpt, 1)
    end do

    do ilay = 2, nlay
      do igpt = 1,ngpt
        lev_source(igpt, ilay) =     sqrt(lev_src_dec(igpt, ilay) * &
                                lev_src_inc(igpt, ilay-1))
      end do
    end do

    do igpt = 1,ngpt
	    lev_source(igpt, nlay+1) =      lev_src_inc(igpt, nlay+1)
    end do

  end subroutine lw_combine_sources

  ! ---------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption
  !   This version straight from ECRAD
  !   Source is provided as W/m2-str; factor of pi converts to flux units
  !
  ! ---------------------------------------------------------------
  
  subroutine lw_source_2str(ngpt, nlay, top_at_1,   &
                            sfc_emis, sfc_src,      &
                            lay_source, lev_source, &
                            gamma1, gamma2, rdif, tdif, tau, source_dn, source_up, source_sfc) &
                            bind (C, name="lw_source_2str")
    integer,                         intent(in) :: ngpt, nlay
    logical(wl),                     intent(in) :: top_at_1
    real(wp), dimension(ngpt      ), intent(in) :: sfc_emis, sfc_src
    real(wp), dimension(ngpt, nlay), intent(in) :: lay_source,    & ! Planck source at layer center
                                                   tau,           & ! Optical depth (tau)
                                                   gamma1, gamma2,& ! Coupling coefficients
                                                   rdif, tdif       ! Layer reflectance and transmittance
    real(wp), dimension(ngpt, nlay+1), target, &
                                     intent(in)  :: lev_source       ! Planck source at layer edges
    real(wp), dimension(ngpt, nlay), intent(out) :: source_dn, source_up
    real(wp), dimension(ngpt      ), intent(out) :: source_sfc      ! Source function for upward radation at surface
    integer             :: igpt, ilay
    real(wp)            :: Z, Zup_top, Zup_bottom, Zdn_top, Zdn_bottom
    real(wp), dimension(:), pointer :: lev_source_bot, lev_source_top
    ! ---------------------------------------------------------------
    do ilay = 1, nlay
      if(top_at_1) then
        lev_source_top => lev_source(:,ilay)
        lev_source_bot => lev_source(:,ilay+1)
      else
        lev_source_top => lev_source(:,ilay+1)
        lev_source_bot => lev_source(:,ilay)
      end if
      do igpt = 1, ngpt
        if (tau(igpt,ilay) > 1.0e-8_wp) then
          !
          ! Toon et al. (JGR 1989) Eqs 26-27
          !
          Z = (lev_source_bot(igpt)-lev_source_top(igpt)) / (tau(igpt,ilay)*(gamma1(igpt,ilay)+gamma2(igpt,ilay)))
          Zup_top        =  Z + lev_source_top(igpt)
          Zup_bottom     =  Z + lev_source_bot(igpt)
          Zdn_top        = -Z + lev_source_top(igpt)
          Zdn_bottom     = -Z + lev_source_bot(igpt)
          source_up(igpt,ilay) = pi * (Zup_top    - rdif(igpt,ilay) * Zdn_top    - tdif(igpt,ilay) * Zup_bottom)
          source_dn(igpt,ilay) = pi * (Zdn_bottom - rdif(igpt,ilay) * Zup_bottom - tdif(igpt,ilay) * Zdn_top)
        else
          source_up(igpt,ilay) = 0._wp
          source_dn(igpt,ilay) = 0._wp
        end if
      end do
    end do
    do igpt = 1, ngpt
      source_sfc(igpt) = pi * sfc_emis(igpt) * sfc_src(igpt)
    end do
  end subroutine lw_source_2str

  ! -------------------------------------------------------------------------------------------------
  !
  !   Lower-level shortwave kernels
  !
  ! -------------------------------------------------------------------------------------------------
  !
  ! Two-stream solutions to direct and diffuse reflectance and transmittance for a layer
  !    with optical depth tau, single scattering albedo w0, and asymmetery parameter g.
  !
  ! Equations are developed in Meador and Weaver, 1980,
  !    doi:10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2
  !
  pure subroutine sw_two_stream_sp(ngpt, nlay, mu0, tau, w0, g, &
                                Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream_sp")
    integer,                        intent(in)  :: ngpt, nlay
    real(sp),                       intent(in)  :: mu0
    real(sp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(sp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    ! -----------------------
    integer  :: i, j

    ! Variables used in Meador and Weaver
    real(sp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
    ! Ancillary variables
    real(sp), dimension(ngpt) :: exp_minusktau, exp_minus2ktau, RT_term
    real(dp) :: k_gamma3, k_gamma4  ! Need to be in double precision
    real(sp) :: k_mu, k_mu2, mu0_inv
    ! ---------------------------------
    mu0_inv = 1._sp/mu0

    do j = 1, nlay
      do i = 1, ngpt
        ! Zdunkowski Practical Improved Flux Method "PIFM"
        !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
        !
        gamma1(i)= (8._sp - w0(i,j) * (5._sp + 3._sp * g(i,j))) * .25_sp
        gamma2(i)=  3._sp *(w0(i,j) * (1._sp -         g(i,j))) * .25_sp
        gamma3(i)= (2._sp - 3._sp * mu0 *              g(i,j) ) * .25_sp
        gamma4(i)=  1._sp - gamma3(i)

        alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
        alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17

        k(i) = sqrt(max((gamma1(i) - gamma2(i)) * (gamma1(i) + gamma2(i)),  1.e-9_sp))
      end do
      ! After increasing the lower limit to e-10 from e-12,
      !  k and the exp computation below can be in single precision. Also moved the k computation to gpt loop
      exp_minusktau(:) = exp_fast_double(-tau(:,j)*k(:))
      !
      ! Diffuse reflection and transmission
      !
      do i = 1, ngpt
        !exp_minus2ktau(i) = real(exp_minusktau(i),kind(k_mu)) * real(exp_minusktau(i),kind(k_mu))
        exp_minus2ktau(i)  = exp_minusktau(i) * exp_minusktau(i)

        ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term(i) = 1._sp / (k(i) * (1._sp + exp_minus2ktau(i)) + gamma1(i) * (1._sp - exp_minus2ktau(i)) )

        ! Equation 25
        Rdif(i,j) = RT_term(i) * gamma2(i) * (1._sp - exp_minus2ktau(i))

        ! Equation 26
        Tdif(i,j) = RT_term(i) * 2._sp * k(i) * exp_minusktau(i)
      end do

      !
      ! Transmittance of direct, unscattered beam. Also used below
      !
      Tnoscat(:,j) = exp_fast_double(-tau(:,j)*mu0_inv)
      !
      ! Direct reflect and transmission
      !
      do i = 1, ngpt
        k_mu     = k(i) * mu0
        k_mu2    = k_mu*k_mu
        k_gamma3 = k(i) * gamma3(i)
        k_gamma4 = k(i) * gamma4(i)
        !
        ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
        !   and rearranging to avoid div by 0.         
        RT_term(i) =  w0(i,j) *  &
        RT_term(i) / merge(1._sp - k_mu2, epsilon(1._sp), abs(1._sp - k_mu2) >= epsilon(1._sp))
        !  --> divide by (1 - kmu2) when (1-kmu2)> eps, otherwise divide by eps

        Rdir(i,j) = RT_term(i)  *                              &
                (   (1._dp - k_mu) * (alpha2(i) + k_gamma3) -  &
                   (1._dp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i) - &
             2.0_dp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) * Tnoscat(i,j)  )

        ! temp(i) =  (1._sp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i)
        ! this term must be in dp
        ! Rdir still having problems (too large) if k_gammas are sp, even after fixing RT_Term

        !
        ! Equation 15, multiplying top and bottom by exp(-k*tau),
        !   multiplying through by exp(-tau/mu0) to
        !   prefer underflow to overflow
        ! Omitting direct transmittance
        !
        Tdir(i,j) = -RT_term(i) *                                                                 &
                    ((1._dp + k_mu) * (alpha1(i) + k_gamma4)                     * Tnoscat(i,j) - &
                     (1._dp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i) * Tnoscat(i,j) - &
                     2.0_sp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))
      end do
    end do

  end subroutine sw_two_stream_sp

  pure subroutine sw_two_stream_dp(ngpt, nlay, mu0, tau, w0, g, &
                                Rdif, Tdif, Rdir, Tdir, Tnoscat)
    integer,                        intent(in)  :: ngpt, nlay
    real(dp),                       intent(in)  :: mu0
    real(dp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(dp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    ! -----------------------
    integer  :: i, j

    ! Variables used in Meador and Weaver
    real(dp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
    ! Ancillary variables
    real(dp), dimension(ngpt) :: exp_minusktau, exp_minus2ktau, RT_term
    real(dp) :: k_gamma3, k_gamma4  ! Need to be in double precision
    real(dp) :: k_mu, k_mu2, mu0_inv
    ! ---------------------------------
    mu0_inv = 1._dp/mu0

    do j = 1, nlay
      do i = 1, ngpt
        ! Zdunkowski Practical Improved Flux Method "PIFM"
        !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
        !
        gamma1(i)= (8._dp - w0(i,j) * (5._dp + 3._dp * g(i,j))) * .25_dp
        gamma2(i)=  3._dp *(w0(i,j) * (1._dp -         g(i,j))) * .25_dp
        gamma3(i)= (2._dp - 3._dp * mu0 *              g(i,j) ) * .25_dp
        gamma4(i)=  1._dp - gamma3(i)

        alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
        alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17

        k(i) = sqrt(max((gamma1(i) - gamma2(i)) * (gamma1(i) + gamma2(i)),  1.e-12_dp))
      end do
      exp_minusktau(:) = exp_fast_double(-tau(:,j)*k(:))
      !
      ! Diffuse reflection and transmission
      !
      do i = 1, ngpt
        exp_minus2ktau(i)  = exp_minusktau(i) * exp_minusktau(i)

        ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term(i) = 1._dp / (k(i) * (1._dp + exp_minus2ktau(i)) + gamma1(i) * (1._dp - exp_minus2ktau(i)) )
        ! Equation 25
        Rdif(i,j) = RT_term(i) * gamma2(i) * (1._dp - exp_minus2ktau(i))
        ! Equation 26
        Tdif(i,j) = RT_term(i) * 2._dp * k(i) * exp_minusktau(i)
      end do

      !
      ! Transmittance of direct, unscattered beam. Also used below
      !
      Tnoscat(:,j) = exp_fast_double(-tau(:,j)*mu0_inv)
      !
      ! Direct reflect and transmission
      !
      do i = 1, ngpt
        k_mu     = k(i) * mu0
        k_mu2    = k_mu*k_mu
        k_gamma3 = k(i) * gamma3(i)
        k_gamma4 = k(i) * gamma4(i)
        !
        ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
        !   and rearranging to avoid div by 0.         
        RT_term(i) =  w0(i,j) *  &
        RT_term(i) / merge(1._dp - k_mu2, epsilon(1._dp), abs(1._dp - k_mu2) >= epsilon(1._dp))
        !  --> divide by (1 - kmu2) when (1-kmu2)> eps, otherwise divide by eps

        Rdir(i,j) = RT_term(i)  *                              &
                (   (1._dp - k_mu) * (alpha2(i) + k_gamma3) -  &
                   (1._dp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i) - &
             2.0_dp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) * Tnoscat(i,j)  )
        ! Equation 15, multiplying top and bottom by exp(-k*tau),
        !   multiplying through by exp(-tau/mu0) to
        !   prefer underflow to overflow
        ! Omitting direct transmittance
        !
        Tdir(i,j) = -RT_term(i) *                                                                 &
                    ((1._dp + k_mu) * (alpha1(i) + k_gamma4)                     * Tnoscat(i,j) - &
                     (1._dp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i) * Tnoscat(i,j) - &
                     2.0_dp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))
      end do
    end do

  end subroutine sw_two_stream_dp

  pure subroutine sw_two_stream_ecrad(ngpt, nlay, mu0, tau, w0, g, &
                                Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream_ecrad")

    integer, intent(in) :: ngpt, nlay
    ! Cosine of solar zenith angle
    real(wp), intent(in) :: mu0
    ! Optical depth and single scattering albedo
    real(wp), intent(in), dimension(ngpt,nlay) :: tau, w0,g 
    ! The direct reflectance and transmittance, i.e. the fraction of
    ! incoming direct solar radiation incident at the top of a layer
    ! that is either reflected back (Rdir) or scattered but
    ! transmitted through the layer to the base (Tdir)
    real(wp), intent(out), dimension(ngpt,nlay) :: Rdir, Tdir
    ! The diffuse reflectance and transmittance, i.e. the fraction of
    ! diffuse radiation incident on a layer from either top or bottom
    ! that is reflected back or transmitted through
    real(wp), intent(out), dimension(ngpt,nlay) :: Rdif, Tdif
    ! Transmittance of the direct been with no scattering
    real(wp), intent(out), dimension(ngpt,nlay) :: Tnoscat

    real(wp) :: gamma1,gamma2,gamma3,gamma4, alpha1, alpha2, k_exponent, reftrans_factor
    real(wp) :: exponential0 ! = exp(-tau/mu0)
    real(wp) :: exponential  ! = exp(-k_exponent*tau)
    real(wp) :: exponential2 ! = exp(-2*k_exponent*tau)
    real(wp) :: k_mu0, k_gamma3, k_gamma4
    real(wp) :: k_2_exponential, tau_over_w0
    integer    :: ilay, igpt
    
    do ilay = 1, nlay
    
      do igpt = 1, ngpt
          tau_over_w0 = max(tau(igpt,ilay) / mu0, 0.0_wp)
    
          gamma1= (8._wp - w0(igpt,ilay) * (5._wp + 3._wp * g(igpt,ilay))) * .25_wp
          gamma2=  3._wp *(w0(igpt,ilay) * (1._wp -         g(igpt,ilay))) * .25_wp
          gamma3= (2._wp - 3._wp * mu0 *              g(igpt,ilay) ) * .25_wp
          ! factor = 0.75_wp*g(igpt,ilay)
          ! gamma1(i) = 2.0_wp  - w0(igpt,ilay) * (1.25_wp + factor)
          ! gamma2(i) = w0(igpt,ilay) * (0.75_wp - factor)
          ! gamma3(i) = 0.5_wp  - mu0*factor
          gamma4 =  1._wp - gamma3
          alpha1 = gamma1*gamma4     + gamma2*gamma3 ! Eq. 16
          alpha2 = gamma1*gamma3 + gamma2*gamma4    ! Eq. 17
          ! In the IFS this appears to be faster without testing the value
          ! of tau_over_w0:
    
          k_exponent = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), &
              &       1.0e-10_wp)) ! Eq 18
          k_mu0 = k_exponent*mu0
          k_gamma3 = k_exponent*gamma3
          k_gamma4 = k_exponent*gamma4
          ! Check for mu0 <= 0!
          exponential0 = exp_fast(-tau_over_w0)
          Tnoscat(igpt,ilay) = exponential0
          exponential = exp_fast(-k_exponent*tau(igpt,ilay))
          
          exponential2 = exponential*exponential
          k_2_exponential = 2.0_wp * k_exponent * exponential
          
          if (k_mu0 == 1.0_wp) then
            k_mu0 = 1.0_wp - 10.0_wp*epsilon(1.0_wp)
          end if
          
          reftrans_factor = 1.0_wp / (k_exponent + gamma1 + (k_exponent - gamma1)*exponential2)
          
          ! Meador & Weaver (1980) Eq. 25
          Rdif(igpt,ilay) = gamma2 * (1.0_wp - exponential2) * reftrans_factor
          
          ! Meador & Weaver (1980) Eq. 26
          Tdif(igpt,ilay) = k_2_exponential * reftrans_factor
          
          ! Here we need mu0 even though it wasn't in Meador and Weaver
          ! because we are assuming the incoming direct flux is defined
          ! to be the flux into a plane perpendicular to the direction of
          ! the sun, not into a horizontal plane
          reftrans_factor = mu0 * w0(igpt,ilay) * reftrans_factor / (1.0_wp - k_mu0*k_mu0)
          
          ! Meador & Weaver (1980) Eq. 14, multiplying top & bottom by
          ! exp(-k_exponent*tau) in case of very high optical depths
          Rdir(igpt,ilay) = reftrans_factor &
              &  * ( (1.0_wp - k_mu0) * (alpha2 + k_gamma3) &
              &     -(1.0_wp + k_mu0) * (alpha2 - k_gamma3)*exponential2 &
              &     -k_2_exponential*(gamma3 - alpha2*mu0)*exponential0)
          
          ! Meador & Weaver (1980) Eq. 15, multiplying top & bottom by
          ! exp(-k_exponent*tau), minus the 1*exp(-tau/mu0) term representing direct
          ! unscattered transmittance.  
          Tdir(igpt,ilay) = reftrans_factor * ( k_2_exponential*(gamma4 + alpha1*mu0) &
              & - exponential0 &
              & * ( (1.0_wp + k_mu0) * (alpha1 + k_gamma4) &
              &    -(1.0_wp - k_mu0) * (alpha1 - k_gamma4) * exponential2) )
      end do
    end do
  end subroutine sw_two_stream_ecrad

 subroutine sw_two_stream_timing(ngpt, nlay, mu0, tau, w0, g, &
                                Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream_timing")
    integer,                        intent(in)  :: ngpt, nlay
    real(wp),                       intent(in)  :: mu0
    real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    ! -----------------------
    integer  :: i, j

    ! Variables used in Meador and Weaver
    real(dp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
    ! Ancillary variables
    real(dp), dimension(ngpt) :: RT_term, exp_minusktau, exp_minus2ktau
    real(dp) :: k_mu, k_gamma3, k_gamma4, mu0_inv
    ! ---------------------------------
    mu0_inv = 1._dp/mu0
    do j = 1, nlay

#ifdef USE_TIMING
      ret =  gptlstart('gamma')
#endif
      do i = 1, ngpt
        
        ! Zdunkowski Practical Improved Flux Method "PIFM"
        !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
        !
        gamma1(i)= (8._dp - w0(i,j) * (5._dp + 3._dp * g(i,j))) * .25_dp
        gamma2(i)=  3._dp *(w0(i,j) * (1._dp -         g(i,j))) * .25_dp
        gamma3(i)= (2._dp - 3._dp * mu0 *              g(i,j) ) * .25_dp
        gamma4(i)=  1._dp - gamma3(i)

        alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
        alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17
      end do
#ifdef USE_TIMING
    ret =  gptlstop('gamma')
#endif
      ! Written to encourage vectorization of exponential, square root
      ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
      !   k = 0 for isotropic, conservative scattering; this lower limit on k
      !   gives relative error with respect to conservative solution
      !   of < 0.1% in Rdif down to tau = 10^-9
      k(:) = sqrt(max((gamma1(:) - gamma2(:)) * &
                           (gamma1(:) + gamma2(:)),  &
                           1.e-12_dp))
#ifdef USE_TIMING
    ret =  gptlstart('exp1')
#endif
     ! tau_tmp = -tau(:,j)*k(:)
      !exp_minusktau(:) = exp_fast(tau_tmp)
      exp_minusktau(:) = exp_fast_double(-tau(:,j)*k(:))
#ifdef USE_TIMING
    ret =  gptlstop('exp1')
#endif
      !
      ! Diffuse reflection and transmission
      !
#ifdef USE_TIMING
    ret =  gptlstart('diffuse')
#endif
      do i = 1, ngpt
        exp_minus2ktau(i) = exp_minusktau(i) * exp_minusktau(i)

        ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term(i) = 1._dp / (k     (i) * (1._dp + exp_minus2ktau(i))  + &
                              gamma1(i) * (1._dp - exp_minus2ktau(i)) )

        ! Equation 25
        Rdif(i,j) = RT_term(i) * gamma2(i) * (1._dp - exp_minus2ktau(i))

        ! Equation 26
        Tdif(i,j) = RT_term(i) * 2._dp * k(i) * exp_minusktau(i)
      end do
#ifdef USE_TIMING
    ret =  gptlstop('diffuse')
#endif
      !
      ! Transmittance of direct, unscattered beam. Also used below
      !
#ifdef USE_TIMING
    ret =  gptlstart('exp2')
#endif)
      Tnoscat(:,j) = exp_fast_double(-tau(:,j)*mu0_inv)
#ifdef USE_TIMING
    ret =  gptlstop('exp2')
#endif
      !
      ! Direct reflect and transmission
      !
#ifdef USE_TIMING
    ret =  gptlstart('direct')
#endif
      do i = 1, ngpt
        k_mu     = k(i) * mu0
        k_gamma3 = k(i) * gamma3(i)
        k_gamma4 = k(i) * gamma4(i)

        !
        ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
        !   and rearranging to avoid div by 0.
        !
        RT_term(i) =  w0(i,j) * RT_term(i)/merge(1._dp - k_mu*k_mu, &
                                                 epsilon(1._dp),    &
                                                 abs(1._dp - k_mu*k_mu) >= epsilon(1._dp))

        Rdir(i,j) = RT_term(i)  *                                        &
            ((1._dp - k_mu) * (alpha2(i) + k_gamma3)                     - &
             (1._dp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i) - &
             2.0_dp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) * Tnoscat(i,j))

        !
        ! Equation 15, multiplying top and bottom by exp_fast(-k*tau),
        !   multiplying through by exp_fast(-tau/mu0) to
        !   prefer underflow to overflow
        ! Omitting direct transmittance
        !
        Tdir(i,j) = -RT_term(i) *                                                                 &
                    ((1._dp + k_mu) * (alpha1(i) + k_gamma4)                     * Tnoscat(i,j) - &
                     (1._dp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i) * Tnoscat(i,j) - &
                     2.0_dp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))

      end do
#ifdef USE_TIMING
    ret =  gptlstop('direct')
#endif 
    end do

  end subroutine sw_two_stream_timing
  ! ---------------------------------------------------------------
  !
  ! Direct beam source for diffuse radiation in layers and at surface;
  !   report direct beam as a byproduct
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine sw_source_2str(ngpt, nlay, top_at_1, Rdir, Tdir, Tnoscat, sfc_albedo, &
                            source_up, source_dn, source_sfc, flux_dn_dir) bind(C, name="sw_source_2str")
    integer,                           intent(in   ) :: ngpt, nlay
    logical(wl),                       intent(in   ) :: top_at_1
    real(wp), dimension(ngpt, nlay  ), intent(in   ) :: Rdir, Tdir, Tnoscat ! Layer reflectance, transmittance for diffuse radiation
    real(wp), dimension(ngpt        ), intent(in   ) :: sfc_albedo          ! surface albedo for direct radiation
    real(wp), dimension(ngpt, nlay  ), intent(  out) :: source_dn, source_up
    real(wp), dimension(ngpt        ), intent(  out) :: source_sfc          ! Source function for upward radation at surface
    real(wp), dimension(ngpt, nlay+1), intent(inout) :: flux_dn_dir ! Direct beam flux
                                                                    ! intent(inout) because top layer includes incident flux

    integer :: ilev

    if(top_at_1) then
      do ilev = 1, nlay
        source_up(:,ilev) =        Rdir(:,ilev) * flux_dn_dir(:,ilev)
        source_dn(:,ilev) =        Tdir(:,ilev) * flux_dn_dir(:,ilev)
        flux_dn_dir(:,ilev+1) = Tnoscat(:,ilev) * flux_dn_dir(:,ilev)
      end do
      source_sfc(:) = flux_dn_dir(:,nlay+1)*sfc_albedo(:)
    else
      ! layer index = level index
      ! previous level is up (+1)
      do ilev = nlay, 1, -1
        source_up(:,ilev)   =    Rdir(:,ilev) * flux_dn_dir(:,ilev+1)
        source_dn(:,ilev)   =    Tdir(:,ilev) * flux_dn_dir(:,ilev+1)
        flux_dn_dir(:,ilev) = Tnoscat(:,ilev) * flux_dn_dir(:,ilev+1)
      end do
      source_sfc(:) = flux_dn_dir(:,     1)*sfc_albedo(:)
    end if
end subroutine sw_source_2str
! ---------------------------------------------------------------
!
! Transport of diffuse radiation through a vertically layered atmosphere.
!   Equations are after Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08)
!   This routine is shared by longwave and shortwave
!
! -------------------------------------------------------------------------------------------------
  subroutine adding(ngpt, nlay, top_at_1, &
                  albedo_sfc,           &
                  rdif, tdif,           &
                  src_dn, src_up, src_sfc, &
                  flux_up, flux_dn) bind(C, name="adding")
    integer,                          intent(in   ) :: ngpt, nlay
    logical(wl),                      intent(in   ) :: top_at_1
    real(wp), dimension(ngpt       ), intent(in   ) :: albedo_sfc
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: rdif, tdif
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: src_dn, src_up
    real(wp), dimension(ngpt       ), intent(in   ) :: src_sfc
    real(wp), dimension(ngpt,nlay+1), intent(  out) :: flux_up
    ! intent(inout) because top layer includes incident flux
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: flux_dn
    ! ------------------
    integer :: ilev
    real(wp), dimension(ngpt,nlay+1)  :: albedo, &  ! reflectivity to diffuse radiation below this level
                                                    ! alpha in SH08
                                        src        ! source of diffuse upwelling radiation from emission or
                                                    ! scattering of direct beam
                                                    ! G in SH08
    real(wp), dimension(ngpt,nlay  )  :: denom      ! beta in SH08
    ! ------------------
    !
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    !
    if(top_at_1) then
      ilev = nlay + 1
      ! Albedo of lowest level is the surface albedo...
      albedo(:,ilev)  = albedo_sfc(:)
      ! ... and source of diffuse radiation is surface emission
      src(:,ilev) = src_sfc(:)
      !
      ! From bottom to top of atmosphere --
      !   compute albedo and source of upward radiation
      !
      do ilev = nlay, 1, -1
        denom(:, ilev) = 1._wp/(1._wp - rdif(:,ilev)*albedo(:,ilev+1))                 ! Eq 10
        albedo(:,ilev) = rdif(:,ilev) + &
                        tdif(:,ilev)*tdif(:,ilev) * albedo(:,ilev+1) * denom(:,ilev) ! Equation 9
        !
        ! Equation 11 -- source is emitted upward radiation at top of layer plus
        !   radiation emitted at bottom of layer,
        !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
        !
        src(:,ilev) =  src_up(:, ilev) + &
                      tdif(:,ilev) * denom(:,ilev) *       &
                        (src(:,ilev+1) + albedo(:,ilev+1)*src_dn(:,ilev))
      end do
      ! Eq 12, at the top of the domain upwelling diffuse is due to ...
      ilev = 1
      flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! ... reflection of incident diffuse and
                        src(:,ilev)                          ! emission from below
      !
      ! From the top of the atmosphere downward -- compute fluxes
      !
      do ilev = 2, nlay+1
        flux_dn(:,ilev) = (tdif(:,ilev-1)*flux_dn(:,ilev-1) + &  ! Equation 13
                          rdif(:,ilev-1)*src(:,ilev) +       &
                          src_dn(:,ilev-1)) * denom(:,ilev-1)
        flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! Equation 12
                          src(:,ilev)
      end do
    else
      ilev = 1
      ! Albedo of lowest level is the surface albedo...
      albedo(:,ilev)  = albedo_sfc(:)
      ! ... and source of diffuse radiation is surface emission
      src(:,ilev) = src_sfc(:)
      !
      ! From bottom to top of atmosphere --
      !   compute albedo and source of upward radiation
      !
      do ilev = 1, nlay
        denom(:, ilev  ) = 1._wp/(1._wp - rdif(:,ilev)*albedo(:,ilev))                ! Eq 10
        albedo(:,ilev+1) = rdif(:,ilev) + &
                          tdif(:,ilev)*tdif(:,ilev) * albedo(:,ilev) * denom(:,ilev) ! Equation 9
        !
        ! Equation 11 -- source is emitted upward radiation at top of layer plus
        !   radiation emitted at bottom of layer,
        !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
        !
        src(:,ilev+1) =  src_up(:, ilev) +  &
                        tdif(:,ilev) * denom(:,ilev) *       &
                        (src(:,ilev) + albedo(:,ilev)*src_dn(:,ilev))
      end do
      ! Eq 12, at the top of the domain upwelling diffuse is due to ...
      ilev = nlay+1
      flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! ... reflection of incident diffuse and
                        src(:,ilev)                          ! scattering by the direct beam below
      !
      ! From the top of the atmosphere downward -- compute fluxes
      !
      do ilev = nlay, 1, -1
        flux_dn(:,ilev) = (tdif(:,ilev)*flux_dn(:,ilev+1) + &  ! Equation 13
                          rdif(:,ilev)*src(:,ilev) + &
                          src_dn(:, ilev)) * denom(:,ilev)
        flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! Equation 12
                          src(:,ilev)
      end do
    end if
  end subroutine adding
  ! -------------------------------------------------------------------------------------------------
  !
  ! Planck sources by g-point from plank fraction and sources by band
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_gpt_source_Jac(nbnd, ngpt, nlay, sfc_lay, band_limits, planck_frac, &
    lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_source_bnd_Jac, &     ! inputs: band source functions
    sfc_source, sfc_source_Jac,  lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

    integer,                          intent(in   ) :: nbnd, ngpt, nlay, sfc_lay
    integer, dimension(2,nbnd),       intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay),   intent(in   ) :: planck_frac 
    real(wp), dimension(nbnd,nlay),   intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1), intent(in )   :: lev_source_bnd
    real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd      ! Surface source by band
    real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd_Jac  ! Surface source by band using perturbed temperature
    ! outputs
    real(wp), dimension(ngpt     ),   intent(out)   :: sfc_source, sfc_source_Jac ! Surface source by g-point and its Jacobian
    real(wp), dimension(ngpt,nlay),   intent(out)   :: lay_source
    real(wp), dimension(ngpt,nlay),   intent(out)   :: lev_source_dec, lev_source_inc

    integer             ::  ilay, igpt, ibnd, gptS, gptE

    do ibnd = 1, nbnd
      do igpt = band_limits(1, ibnd), band_limits(2, ibnd)
        sfc_source(igpt)     = planck_frac(igpt,sfc_lay) * sfc_source_bnd(ibnd)
        sfc_source_Jac(igpt) = planck_frac(igpt,sfc_lay) * (sfc_source_bnd(ibnd) - sfc_source_bnd_Jac(ibnd))
      end do
    end do 

    do ilay = 1, nlay
      do ibnd = 1, nbnd
        do igpt = band_limits(1, ibnd), band_limits(2, ibnd)
          ! compute layer source irradiance for each g-point
          lay_source(igpt,ilay)       = planck_frac(igpt,ilay) * lay_source_bnd(ibnd,ilay)
          ! compute level source irradiance for each g-point, one each for upward and downward paths
          lev_source_dec(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay)
          lev_source_inc(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay+1)
        end do
      end do 
    end do

  end subroutine lw_gpt_source_Jac
  

  pure subroutine lw_gpt_source(nbnd, ngpt, nlay, sfc_lay, band_limits, planck_frac, &
      lay_source_bnd, lev_source_bnd, sfc_source_bnd, &     ! inputs: band source functions
      sfc_source, lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

      integer,                          intent(in   ) :: nbnd, ngpt, nlay, sfc_lay
      integer, dimension(2,nbnd),       intent(in   ) :: band_limits
      real(wp), dimension(ngpt,nlay),   intent(in   ) :: planck_frac 
      real(wp), dimension(nbnd,nlay),   intent(in   ) :: lay_source_bnd
      real(wp), dimension(nbnd,nlay+1), intent(in )   :: lev_source_bnd
      real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd      ! Surface source by band
      ! outputs
      real(wp), dimension(ngpt     ),   intent(out)   :: sfc_source
      real(wp), dimension(ngpt,nlay),   intent(out)   :: lay_source
      real(wp), dimension(ngpt,nlay),   intent(out)   :: lev_source_dec, lev_source_inc

      integer             ::  ilay, igpt, ibnd, gptS, gptE

    do ibnd = 1, nbnd
      do igpt = band_limits(1, ibnd), band_limits(2, ibnd)
        sfc_source(igpt)     = planck_frac(igpt,sfc_lay) * sfc_source_bnd(ibnd)
      end do
    end do 

    do ilay = 1, nlay
      do ibnd = 1, nbnd
        do igpt = band_limits(1, ibnd), band_limits(2, ibnd)
          ! compute layer source irradiance for each g-point
          lay_source(igpt,ilay)       = planck_frac(igpt,ilay) * lay_source_bnd(ibnd,ilay)
          ! compute level source irradiance for each g-point, one each for upward and downward paths
          lev_source_dec(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay)
          lev_source_inc(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(ibnd,ilay+1)
        end do
      end do 
    end do

  end subroutine lw_gpt_source

  ! ---------------------------------------------------------------
  !
  ! Upper boundary condition
  !
  ! ---------------------------------------------------------------
  pure subroutine apply_BC_nocol(ngpt, nlay, top_level, inc_flux, flux_dn) bind (C, name="apply_BC_nocol")

    integer,                                intent( in) :: ngpt, nlay ! Number of columns, layers, g-points
    integer,                                intent( in) :: top_level
    real(wp), dimension(ngpt      ),        intent( in) :: inc_flux         ! Flux at top of domain
    real(wp), dimension(ngpt,nlay+1),       intent(out) :: flux_dn          ! Flux to be used as input to solvers below
    integer :: igpt

    !   Upper boundary condition

    do igpt = 1, ngpt
      flux_dn(igpt, top_level)  = inc_flux(igpt)
    end do
  end subroutine apply_BC_nocol
  ! ---------------------
  pure subroutine apply_BC(ngpt, nlay, ncol, top_level, inc_flux, flux_dn) bind (C, name="apply_BC")
    integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    integer,                               intent( in) :: top_level
    real(wp), dimension(ngpt, ncol      ), intent( in) :: inc_flux         ! Flux at top of domain
    real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below

    !   Upper boundary condition
    flux_dn(:,  top_level, :)  = inc_flux(:,:)

  end subroutine apply_BC
  ! ---------------------
  pure subroutine apply_BC_old(ngpt, nlay, ncol, top_at_1, inc_flux, flux_dn) bind (C, name="apply_BC_old")
  integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
  logical(wl),                           intent( in) :: top_at_1
  real(wp), dimension(ngpt, ncol      ), intent( in) :: inc_flux         ! Flux at top of domain
  real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below

  !   Upper boundary condition
  if(top_at_1) then
    flux_dn(:,  1, :)  = inc_flux(:,:)
  else
    flux_dn(:,  nlay+1, :)  = inc_flux(:,:) 
  end if 
end subroutine apply_BC_old
  
  ! ---------------------
  pure subroutine apply_BC_factor(ngpt, nlay, ncol, top_at_1, inc_flux, factor, flux_dn) bind (C, name="apply_BC_factor")
    integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent( in) :: top_at_1
    real(wp), dimension(ngpt, ncol      ), intent( in) :: inc_flux         ! Flux at top of domain
    real(wp), dimension(ncol            ), intent( in) :: factor           ! Factor to multiply incoming flux
    real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below
    integer :: igpt
    !   Upper boundary condition

    if(top_at_1) then
      do igpt = 1, ngpt
          flux_dn(igpt, 1, :)  = inc_flux(igpt,:) * factor
      end do
    else
      do igpt = 1, ngpt
          flux_dn(igpt, nlay+1, 1:ncol)  = inc_flux(igpt,:) * factor
      end do
    end if

  end subroutine apply_BC_factor
  ! --------------------- 
  pure subroutine apply_BC_0(ngpt, nlay, ncol, top_at_1, flux_dn) bind (C, name="apply_BC_0")
    integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent( in) :: top_at_1
    real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below

    !   Upper boundary condition

    if(top_at_1) then
      flux_dn(1:ngpt,      1, 1:ncol)  = 0._wp
    else
      flux_dn(1:ngpt, nlay+1, 1:ncol)  = 0._wp
    end if

  end subroutine apply_BC_0


  subroutine lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, D , band_limits, &
                            tau, scaling, planck_frac, &
                            lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
                            radn_up, radn_dn, &
                            sfc_source_bnd_Jac, radn_up_Jac, radn_dn_Jac) bind(C, name="lw_solver_1rescl")
    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: D            ! secant of propagation angle  []
    integer,  dimension(2,nbnd),            intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: scaling
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: planck_frac  ! Planck fractions (fraction of band source function associated with each g-point)
    real(wp), dimension(nbnd,nlay,  ncol), intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in   ) :: lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis            ! Surface emissivity      []
    ! Outputs
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) :: radn_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) :: radn_dn      ! Top level must contain incident flux boundary condition
    ! real(wp), dimension(nlay+1,     ncol), optional, &
    !                                       intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) ::  radn_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) ::  radn_dn_Jac  

    ! ------------------------------------
    ! Local variables. no col dependency
    real(wp), dimension(:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    real(wp), dimension(ngpt,nlay),   target              :: lev_source_dec, lev_source_inc
    real(wp), dimension(ngpt,nlay)                        :: lay_source
    real(wp), dimension(ngpt)                             :: sfc_src       ! Surface source function by g-point [W/m2]
    real(wp), dimension(ngpt)                             :: sfc_srcJac   ! Jacobian of surface source function by g-point [W/m2]
    real(wp), dimension(ngpt,nlay) :: tau_loc, &  ! path length (tau/mu)
                                        trans       ! transmissivity  = exp_fast(-tau)
    real(wp), dimension(ngpt,nlay) :: source_dn, source_up
    real(wp), dimension(ngpt     ) :: source_sfc, sfc_albedo, source_sfcJac
    real(wp), dimension(ngpt,nlay) :: An, Cn

    real(wp), parameter :: pi = acos(-1._wp)
    integer             :: ilev, icol, igpt, ilay, sfc_lay, top_level
    ! ------------------------------------
    real(wp), parameter :: tau_thresh = sqrt(epsilon(tau))
    ! ------------------------------------

    ! Which way is up?
    ! Level Planck sources for upward and downward radiation
    ! When top_at_1, lev_source_up => lev_source_dec
    !                lev_source_dn => lev_source_inc, and vice-versa
    if(top_at_1) then
      top_level = 1
      sfc_lay   = nlay  ! the layer (not level) closest to surface
      lev_source_up => lev_source_dec
      lev_source_dn => lev_source_inc
    else
      top_level = nlay+1
      sfc_lay = 1
      lev_source_up => lev_source_inc
      lev_source_dn => lev_source_dec
    end if

    do icol = 1, ncol
      !
      ! Optical path and transmission, used in source function and transport calculations
      !
      do ilev = 1, nlay
        tau_loc(:,ilev) = tau(:,ilev,icol)*D(:,icol)
        trans  (:,ilev) = exp_fast(-tau_loc(:,ilev))
        !
        ! here scaling is used to store parameter wb/(1-w(1-b)) of Eq.21 of the Tang's paper
        ! explanation of factor 0.4 note A of Table
        !
        Cn(:,ilev) = 0.4_wp*scaling(:,ilev,icol)
        An(:,ilev) = (1._wp-trans(:,ilev)*trans(:,ilev))
      end do

      !
      ! Compute the source function per g-point from source function per band
      !
#ifdef USE_TIMING
    ret =  gptlstart('gpt_source')
#endif
      call lw_gpt_source_Jac(nbnd, ngpt, nlay, sfc_lay, band_limits, &
                      planck_frac(:,:,icol), lay_source_bnd(:,:,icol), lev_source_bnd(:,:,icol), &
                      sfc_source_bnd(:,icol), sfc_source_bnd_Jac(:,icol), &
                      sfc_src, sfc_srcJac, lay_source, lev_source_dec, lev_source_inc)
#ifdef USE_TIMING
    ret =  gptlstop('gpt_source')
#endif
      !
      ! Source function for diffuse radiation
      !
      call lw_source_noscat(ngpt, nlay, &
                          lay_source, lev_source_up, lev_source_dn, &
                          tau_loc, trans, source_dn, source_up)
      !
      ! Surface albedo, surface source function
      !
      sfc_albedo(:)     = 1._wp - sfc_emis(:,icol)
      source_sfc(:)     = sfc_emis(:,icol) * sfc_src
      source_sfcJac(:)  = sfc_emis(:,icol) * sfc_srcJac
      !
      ! Transport
      !
      call lw_transport_noscat(ngpt, nlay, top_at_1,  &
                              tau_loc, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                              radn_up(:,:,icol), radn_dn(:,:,icol), &
                              source_sfcJac, radn_up_Jac(:,:,icol))
                  
      radn_dn_Jac(:,:,icol) = 0._wp
      !  make adjustment
      call lw_transport_1rescl(ngpt, nlay, top_at_1, trans, &
                              source_dn, source_up, &
                              radn_up(:,:,icol), radn_dn(:,:,icol), An, Cn,&
                              radn_up_Jac(:,:,icol), radn_dn_Jac(:,:,icol))

    end do  ! column loop
  end subroutine lw_solver_1rescl

! -------------------------------------------------------------------------------------------------
!
!  Similar to lw_solver_noscat_GaussQuad.
!    It is main solver to use the rescaled-for-scattering approximation for fluxes
!    In addition to the no scattering input parameters the user must provide
!    scattering related properties (ssa and g) that the solver uses to compute scaling
!
! ---------------------------------------------------------------
  subroutine lw_solver_1rescl_GaussQuad(nbnd, ngpt, nlay, ncol, top_at_1, nmus, Ds, weights, &
                                    band_limits, tau, ssa, g, planck_frac, &
                                    lay_source_bnd, lev_source_bnd, &
                                    sfc_source_bnd, sfc_emis, &
                                    flux_up, flux_dn, &
                                    sfc_source_bnd_Jac, flux_up_Jac, flux_dn_Jac) bind(C, name="lw_solver_1rescl_GaussQuad")
    integer,                                intent(in   ) ::  nbnd, ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                            intent(in   ) ::  top_at_1
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(nmus),              intent(in   ) ::  Ds, weights  ! quadrature secants, weights
    integer,  dimension(2,nbnd),            intent(in   ) ::  band_limits
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  ssa          ! single-scattering albedo
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  g            ! asymmetry parameter []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  planck_frac   ! Planck fractions (fraction of band source associated with each g-point) at layers
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis            ! Surface emissivity      []
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    ! Outputs
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(inout) ::  flux_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(inout) ::  flux_dn      ! Top level must contain incident flux boundary condition

    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]                 
    real(wp), dimension(ngpt, nlay+1,     ncol), intent(inout) ::  flux_dn_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]                                        
                        
    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt

    real(wp), dimension(:,:,:),  allocatable      :: radn_up, radn_dn           ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:,:),  allocatable      :: radn_up_Jac, radn_dn_Jac   ! perturbed Fluxes per quad angle

    real(wp), dimension(ncol, nlay,  ngpt) :: tauLoc           ! rescaled Tau
    real(wp), dimension(ncol, nlay,  ngpt) :: scaling          ! scaling
    real(wp), dimension(ncol, ngpt)        :: fluxTOA          ! downward flux at TOA

    integer :: imu, top_level
    real    :: weight
    real(wp), parameter                   :: tresh=1.0_wp - 1e-6_wp

    ! Tang rescaling
    if (any(ssa*g >= tresh)) then
      call scaling_1rescl_safe(ngpt, nlay, ncol, tauLoc, scaling, tau, ssa, g)
    else
      call scaling_1rescl(ngpt, nlay, ncol, tauLoc, scaling, tau, ssa, g)
    endif
    ! ------------------------------------
    !
    ! For the first angle output arrays store total flux
    !
    top_level = MERGE(1, nlay+1, top_at_1)
    fluxTOA = flux_dn(1:ngpt, top_level, 1:ncol)
    Ds_ngpt(:,:) = Ds(1)
    weight = 2._wp*pi*weights(1)
    ! Transport is for intensity
    !   convert flux at top of domain to intensity assuming azimuthal isotropy
    !
    radn_dn(1:ngpt, top_level, 1:ncol)  = fluxTOA(1:ngpt, 1:ncol) / weight
    call lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, Ds_ngpt, band_limits, &
                          tauLoc, scaling, planck_frac, &
                          lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
                          flux_up, flux_dn, &
                          sfc_source_bnd_Jac, flux_up_Jac, flux_dn_Jac)

    flux_up     = flux_up     * weight
    flux_dn     = flux_dn     * weight
    flux_up_Jac = flux_up_Jac * weight
    flux_dn_Jac = flux_dn_Jac * weight

    if (nmus > 1) then
      allocate( radn_up(ngpt, nlay+1, ncol) )
      allocate( radn_dn(ngpt, nlay+1, ncol) )
      allocate( radn_up_Jac(ngpt, nlay+1, ncol) )
      allocate( radn_dn_Jac(ngpt, nlay+1, ncol) )
    end if 

    do imu = 2, nmus
      Ds_ngpt(:,:) = Ds(imu)
      weight = 2._wp*pi*weights(imu)
      ! Transport is for intensity
      !   convert flux at top of domain to intensity assuming azimuthal isotropy
      !
      radn_dn(1:ngpt, top_level, 1:ncol)  = fluxTOA(1:ngpt, 1:ncol) / weight
      call lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, Ds_ngpt, band_limits, &
                          tauLoc, scaling, planck_frac, &
                          lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
                          radn_up, radn_dn, &
                          sfc_source_bnd_Jac, radn_up_Jac, radn_dn_Jac)

      flux_up    (:,:,:) = flux_up    (:,:,:) + weight*radn_up    (:,:,:)
      flux_dn    (:,:,:) = flux_dn    (:,:,:) + weight*radn_dn    (:,:,:)
      flux_up_Jac(:,:,:) = flux_up_Jac(:,:,:) + weight*radn_up_Jac(:,:,:)
      flux_dn_Jac(:,:,:) = flux_dn_Jac(:,:,:) + weight*radn_dn_Jac(:,:,:)
    end do
  end subroutine lw_solver_1rescl_GaussQuad
! -------------------------------------------------------------------------------------------------
!
!  Computes re-scaled layer optical thickness and scaling parameter
!    unsafe if ssa*g =1.
!
! ---------------------------------------------------------------
    pure subroutine scaling_1rescl(ngpt, nlay, ncol, tauLoc, scaling, tau, ssa, g)
    integer ,                              intent(in)    :: ngpt
    integer ,                              intent(in)    :: nlay
    integer ,                              intent(in)    :: ncol
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: tau
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: ssa
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: g

    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: tauLoc
    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: scaling


    integer  :: igpt, ilay, icol
    real(wp) :: wb, ssal, scaleTau
    do icol=1,ncol
      do ilay=1,nlay
        do igpt=1,ngpt
          ssal = ssa(igpt, ilay, icol)
          wb = ssal*(1._wp - g(igpt, ilay, icol)) / 2._wp
          scaleTau = (1._wp - ssal + wb )
          tauLoc(igpt, ilay, icol) = scaleTau * tau(igpt, ilay, icol)   ! Eq.15 of the paper
          !
          ! here scaling is used to store parameter wb/(1-w(1-b)) of Eq.21 of the Tang paper
          ! actually it is in line of parameter rescaling defined in Eq.7
          ! potentialy if g=ssa=1  then  wb/scaleTau = NaN
          ! it should not happen
          scaling(igpt, ilay, icol) = wb / scaleTau
        enddo
      enddo
    enddo
  end subroutine scaling_1rescl
! -------------------------------------------------------------------------------------------------
!
!  Computes re-scaled layer optical thickness and scaling parameter
!    safe implementation
!
! ---------------------------------------------------------------
  pure subroutine scaling_1rescl_safe(ngpt, nlay, ncol, tauLoc, scaling, tau, ssa, g)
    integer ,                              intent(in)    :: ngpt
    integer ,                              intent(in)    :: nlay
    integer ,                              intent(in)    :: ncol
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: tau
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: ssa
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: g

    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: tauLoc
    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: scaling

    integer  :: igpt, ilay, icol
    real(wp) :: wb, ssal, scaleTau
    do icol=1,ncol
      do ilay=1,nlay
        do igpt=1,ngpt
          ssal = ssa(igpt, ilay, icol)
          wb = ssal*(1._wp - g(igpt, ilay, icol)) / 2._wp
          scaleTau = (1._wp - ssal + wb )
          tauLoc(igpt, ilay, icol) = scaleTau * tau(igpt, ilay, icol)   ! Eq.15 of the paper
          !
          ! here scaling is used to store parameter wb/(1-w(1-b)) of Eq.21 of the Tang paper
          ! actually it is in line of parameter rescaling defined in Eq.7
          if (scaleTau < 1e-6_wp) then
            scaling(igpt, ilay, icol) = 1.0_wp
          else
            scaling(igpt, ilay, icol) = wb / scaleTau
          endif
        enddo
      enddo
    enddo
  end subroutine scaling_1rescl_safe
! -------------------------------------------------------------------------------------------------
!
! Similar to Longwave no-scattering tarnsport  (lw_transport_noscat)
!   a) adds adjustment factor based on cloud properties
!
!   implementation notice:
!       the adjustmentFactor computation can be skipped where Cn <= epsilon
!
! -------------------------------------------------------------------------------------------------
  subroutine lw_transport_1rescl(ngpt, nlay, top_at_1, &
                              trans, source_dn, source_up, &
                              radn_up, radn_dn, An, Cn,&
                              radn_up_Jac, radn_dn_Jac) bind(C, name="lw_transport_1rescl")
    integer,                          intent(in   ) :: ngpt, nlay ! Number of columns, layers, g-points
    logical(wl),                      intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: trans      ! transmissivity = exp_fast(-tau)
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: source_dn, &
                                                      source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay),   intent(in   ) :: An, Cn
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up_Jac ! Surface temperature Jacobians [W/m2-str/K]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn_Jac !Top level must set to 0

    ! Local variables
    integer :: ilev, igpt
    ! ---------------------------------------------------
    real(wp) :: adjustmentFactor
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      ! 1st Upward propagation
      do ilev = nlay, 1, -1
        radn_up    (:,ilev) = trans(:,ilev)*radn_up    (:,ilev+1) + source_up(:,ilev)
        radn_up_Jac(:,ilev) = trans(:,ilev)*radn_up_Jac(:,ilev+1)
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_dn(igpt,ilev) - &
                    trans(igpt,ilev)*source_dn(igpt,ilev) - source_up(igpt,ilev) )
            radn_up (igpt,ilev) = radn_up(igpt,ilev) + adjustmentFactor
          enddo
      end do
      ! 2nd Downward propagation
      do ilev = 1, nlay
        radn_dn    (:,ilev+1) = trans(:,ilev)*radn_dn    (:,ilev) + source_dn(:,ilev)
        radn_dn_Jac(:,ilev+1) = trans(:,ilev)*radn_dn_Jac(:,ilev)
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_up(igpt,ilev) - &
                    trans(igpt,ilev)*source_up(igpt,ilev) - source_dn(igpt,ilev) )

            radn_dn    (igpt,ilev+1) = radn_dn(igpt,ilev+1) + adjustmentFactor

            adjustmentFactor         = Cn(igpt,ilev)*An(igpt,ilev)*radn_up_Jac(igpt,ilev)
            radn_dn_Jac(igpt,ilev+1) = radn_dn_Jac(igpt,ilev+1) + adjustmentFactor
        enddo
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      ! Upward propagation
      do ilev = 1, nlay
        radn_up    (:,ilev+1) = trans(:,ilev) * radn_up    (:,ilev) +  source_up(:,ilev)
        radn_up_Jac(:,ilev+1) = trans(:,ilev) * radn_up_Jac(:,ilev)
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_dn(igpt,ilev+1) - &
                    trans(igpt,ilev)*source_dn(igpt,ilev) - source_up(igpt,ilev) )
            radn_up(igpt,ilev+1) = radn_up(igpt,ilev+1) + adjustmentFactor
        enddo
      end do

      ! 2st Downward propagation
      do ilev = nlay, 1, -1
        radn_dn    (:,ilev) = trans(:,ilev)*radn_dn    (:,ilev+1) + source_dn(:,ilev)
        radn_dn_Jac(:,ilev) = trans(:,ilev)*radn_dn_Jac(:,ilev+1)
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_up(igpt,ilev) - &
                    trans(igpt,ilev)*source_up(igpt,ilev) - source_dn(igpt,ilev) )
            radn_dn(igpt,ilev)  = radn_dn(igpt,ilev) + adjustmentFactor

            adjustmentFactor    = Cn(igpt,ilev)*An(igpt,ilev)*radn_up_Jac(igpt,ilev)
            radn_dn_Jac(igpt,ilev) = radn_dn_Jac(igpt,ilev) + adjustmentFactor
        enddo
      end do
    end if
  end subroutine lw_transport_1rescl

end module mo_rte_solver_kernels
