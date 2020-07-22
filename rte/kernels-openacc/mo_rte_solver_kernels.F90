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
  use mo_rte_kind, only: wp, wl
  implicit none
  private

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
contains
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
    real(wp), dimension(:,:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    real(wp), dimension(ngpt,nlay,ncol),   target              :: lev_source_dec, lev_source_inc
    real(wp), dimension(ngpt,nlay,ncol)                        :: lay_source
    real(wp), dimension(ngpt,ncol)                             :: sfc_src       ! Surface source function by g-point [W/m2]
    real(wp), dimension(ngpt ,ncol)                            :: sfc_src_Jac   ! Jacobian of surface source function by g-point [W/m2]
    real(wp), dimension(ngpt,nlay,ncol) :: tau_loc, &  ! path length (tau/mu)
                                        trans       ! transmissivity  = exp(-tau)
    real(wp), dimension(ngpt,nlay, ncol)  :: source_dn, source_up
    real(wp), dimension(ngpt,ncol)        :: source_sfc, sfc_albedo, source_sfcJac
    integer,  dimension(ngpt)             :: gpt_bands ! band number (1...16) for each g-point

    real(wp), parameter :: pi = acos(-1._wp)
    integer             :: ilay, ilev, icol, igpt, sfc_lay, top_level, ibnd

    ! ------------------------------------
    ! Which way is up?
    ! Level Planck sources for upward and downward radiation
    ! When top_at_1, lev_source_up => lev_source_dec
    !                lev_source_dn => lev_source_inc, and vice-versa

   !$acc data present(radn_dn, radn_up, radn_up_Jac)

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

    !$acc enter data create (gpt_bands, lev_source_dec, lev_source_inc, lay_source,  sfc_src, sfc_src_Jac) copyin(D, weight)

    !
    ! Compute the source function per g-point from source function per band
    !
    !$acc parallel loop
    do ibnd = 1, nbnd
      do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
        gpt_bands(igpt) = ibnd
      end do
    end do

    call lw_gpt_source_Jac(nbnd, ngpt, nlay, ncol, sfc_lay, gpt_bands, &
                  planck_frac, lay_source_bnd, lev_source_bnd, &
                  sfc_source_bnd, sfc_source_bnd_Jac, &
                  sfc_src, sfc_src_Jac, lay_source, lev_source_dec, lev_source_inc)
    !$acc exit data delete(planck_frac)

    !$acc enter data create (source_sfc, source_sfcJac, sfc_albedo) 

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        !
        ! Transport is for intensity
        !   convert flux at top of domain to intensity assuming azimuthal isotropy
        !
        radn_dn(igpt,top_level,icol) = radn_dn(igpt,top_level,icol)/(2._wp * pi * weight)
        !
        ! Surface albedo, surface source function
        !
        sfc_albedo(igpt,icol) = 1._wp - sfc_emis(igpt,icol)
        source_sfc(igpt,icol) = sfc_emis(igpt,icol) * sfc_src(igpt,icol)
      end do
    end do

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        source_sfcJac(igpt,icol) = sfc_emis(igpt,icol) * sfc_src_Jac(igpt,icol)
      end do
    end do


    ! NOTE: This kernel produces small differences between GPU and CPU
    ! implementations on Ascent with PGI, we assume due to floating point
    ! differences in the exp() function. These differences are small in the
    ! RFMIP test case (10^-6).

    !$acc enter data create(source_dn, source_up,tau_loc,trans)
    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          !
          ! Optical path and transmission, used in source function and transport calculations
          !
          tau_loc(igpt,ilay,icol) = tau(igpt,ilay,icol)*D(igpt,icol)
          trans  (igpt,ilay,icol) = exp(-tau_loc(igpt,ilay,icol))

          call lw_source_noscat_stencil(ngpt, nlay, ncol, igpt, ilay, icol,        &
                                        lay_source, lev_source_up, lev_source_dn,  &
                                        tau_loc, trans,                            &
                                        source_dn, source_up)
        end do
      end do
    end do

    !$acc exit data delete(lay_source, lev_source_dec, lev_source_inc, sfc_src, sfc_src_Jac)

    !
    ! Transport
    !

    call lw_transport_noscat(ngpt, nlay, ncol, top_at_1,  &
                             tau_loc, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                             radn_up, radn_dn, source_sfcJac, radn_up_Jac)
    !$acc exit data delete(source_dn, source_up, tau_loc, trans)

    !
    ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
    !
    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilev = 1, nlay+1
        do igpt = 1, ngpt
          radn_dn   (igpt,ilev,icol) = 2._wp * pi * weight * radn_dn   (igpt,ilev,icol)
          radn_up   (igpt,ilev,icol) = 2._wp * pi * weight * radn_up   (igpt,ilev,icol)
          radn_up_Jac(igpt,ilev,icol) = 2._wp * pi * weight * radn_up_Jac(igpt,ilev,icol)
        end do
      end do
    end do

    !$acc exit data delete(source_sfc, sfc_albedo, source_sfcJac, D, weight)

    !$acc end data

  end subroutine lw_solver_noscat
  ! ---------------------------------------------------------------
  !
  ! LW transport, no scattering, multi-angle quadrature
  !   Users provide a set of weights and quadrature angles
  !   Routine sums over single-angle solutions for each sets of angles/weights
  !
  ! ---------------------------------------------------------------
  
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
    integer,  dimension(2,nbnd),           intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  planck_frac   ! Planck fractions (fraction of band source associated with each g-point) at layers
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis            ! Surface emissivity      []
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    ! Outputs
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(out)   ::  flux_up      ! Radiances [W/m2-str]
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(inout) ::  flux_dn      ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt, nlay+1,     ncol),  intent(out)   ::  flux_up_Jac   ! surface temperature Jacobian of radiances [W/m2-str / K]                                        
    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt

    real(wp), dimension(:,:,:),  allocatable      :: rad_up, rad_dn ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:,:),  allocatable      :: rad_up_Jac      ! perturbed Fluxes per quad angle
    integer :: imu, icol, sfc_lay, igpt, ilay
    ! ------------------------------------
    !
    ! For the first angle output arrays store total flux
    !

    !$acc enter data copyin(Ds, weights) create (Ds_ngpt)

    !$acc data present(band_limits, tau, planck_frac, lay_source_bnd, lev_source_bnd, sfc_emis, sfc_source_bnd, sfc_source_bnd_Jac, flux_up, flux_dn, flux_up_Jac)

    !$acc  parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        Ds_ngpt(igpt, icol) = Ds(1)
      end do
    end do

    call lw_solver_noscat(nbnd, ngpt, nlay, ncol, top_at_1, &
                          Ds_ngpt, weights(1), &
                          band_limits, tau, planck_frac, &
                          lay_source_bnd, lev_source_bnd, &
                          sfc_source_bnd, sfc_emis, &
                          flux_up, flux_dn, sfc_source_bnd_Jac, flux_up_Jac)

    if (nmus > 1) then
      allocate( rad_up(ngpt, nlay+1, ncol) )
      allocate( rad_dn(ngpt, nlay+1, ncol) )
      allocate( rad_up_Jac(ngpt, nlay+1, ncol) )
      !$acc enter data create(rad_up,rad_dn,rad_up_Jac)
    end if 

    do imu = 2, nmus

      Ds_ngpt(:,:) = Ds(imu)

      call lw_solver_noscat(nbnd, ngpt, nlay, ncol, top_at_1, &
        Ds_ngpt, weights(imu), &
        band_limits, tau, planck_frac, &
        lay_source_bnd, lev_source_bnd, &
        sfc_source_bnd, sfc_emis, &
        rad_up, rad_dn, sfc_source_bnd_Jac, rad_up_Jac)

      flux_up = flux_up + rad_up
      flux_dn = flux_dn + rad_dn
      flux_up_Jac = flux_up_Jac + rad_up_Jac
    end do                      

    !$acc end data

    !$acc exit data delete(Ds, weights, Ds_ngpt)

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
    integer,  dimension(2,nbnd),           intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: planck_frac  ! Planck fractions (fraction of band source function associated with each g-point)
    real(wp), dimension(nbnd,nlay,  ncol), intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in   ) :: lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis            ! Surface emissivity      []
    ! Outputs
    real(wp), dimension(nlay+1,     ncol), intent(inout) :: flux_up      ! Broadband radiances [W/m2-str]
    real(wp), dimension(nlay+1,     ncol), intent(inout) :: flux_dn      ! Top level must contain incident flux boundary condition
    ! real(wp), dimension(nlay+1,     ncol), optional, &
    !                                       intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    real(wp), dimension(nlay+1,     ncol), intent(inout) ::  flux_up_Jac   ! surface temperature Jacobian of broadband radiances [W/m2-str / K]
    ! ------------------------------------
    ! Local variables. no col dependency
    real(wp), dimension(:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    real(wp), dimension(ngpt,nlay),   target              :: lev_source_dec, lev_source_inc
    real(wp), dimension(ngpt,nlay)                        :: lay_source
    real(wp), dimension(ngpt)                             :: sfc_src       ! Surface source function by g-point [W/m2]
    real(wp), dimension(ngpt)                             :: sfc_src_Jac   ! Jacobian of surface source function by g-point [W/m2]
    real(wp), dimension(ngpt,nlay+1)                      :: radn_up          ! Radiances per g-point [W/m2-str]
    real(wp), dimension(ngpt,nlay+1)                      :: radn_dn          ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay+1)                      :: radn_up_Jac       ! surface temperature Jacobian of g-point radiances [W/m2-str / K]
    real(wp), dimension(ngpt,nlay) :: tau_loc, &  ! path length (tau/mu)
                                        trans       ! transmissivity  = exp(-tau)
    real(wp), dimension(ngpt,nlay) :: source_dn, source_up
    real(wp), dimension(ngpt     ) :: source_sfc, sfc_albedo, source_sfcJac

    real(wp), parameter :: pi = acos(-1._wp)
    real(wp)  :: fac
    integer             :: ilev, icol, igpt, ilay, sfc_lay, top_level
    ! ------------------------------------
    ! Which way is up?
    ! Level Planck sources for upward and downward radiation
    ! When top_at_1, lev_source_up => lev_source_dec
    !                lev_source_dn => lev_source_inc, and vice-versa

    ! if(top_at_1) then
    !   top_level = 1
    !   sfc_lay   = nlay  ! the layer (not level) closest to surface
    !   lev_source_up => lev_source_dec
    !   lev_source_dn => lev_source_inc
    ! else
    !   top_level = nlay+1
    !   sfc_lay = 1
    !   lev_source_up => lev_source_inc
    !   lev_source_dn => lev_source_dec
    ! end if

    ! do icol = 1, ncol
    
    !   ! Apply boundary condition
    !   radn_dn(:,top_level) = inc_flux(:,icol)
    !   !
    !   ! Transport is for intensity
    !   !   convert flux at top of domain to intensity assuming azimuthal isotropy
    !   !
    !   radn_dn(:,top_level) = radn_dn(:,top_level)/(2._wp * pi * weight)

    !   !
    !   ! Optical path and transmission, used in source function and transport calculations
    !   !
    !   do ilay = 1, nlay
    !       tau_loc(:,ilay)  = tau(:,ilay,icol) * D(:,icol)
    !       trans(:,ilay)    = exp(-tau_loc(:,ilay)) 
    !   end do
      
    !   !
    !   ! Compute the source function per g-point from source function per band
    !   !

    !   call lw_gpt_source_Jac(nbnd, ngpt, nlay, sfc_lay, band_limits, &
    !                 planck_frac(:,:,icol), lay_source_bnd(:,:,icol), lev_source_bnd(:,:,icol), &
    !                 sfc_source_bnd(:,icol), sfc_source_bnd_Jac(:,icol), &
    !                 sfc_src, sfc_src_Jac, lay_source, lev_source_dec, lev_source_inc)
    !   !
    !   ! Source function for diffuse radiation
    !   !
    !   call lw_source_noscat(ngpt, nlay, &
    !                         lay_source, lev_source_up, lev_source_dn, &
    !                         tau_loc, trans, source_dn, source_up)
    !   !
    !   ! Surface albedo, surface source function
    !   !
    !   sfc_albedo     = 1._wp - sfc_emis(:,icol)
    !   source_sfc     = sfc_emis(:,icol) * sfc_src
    !   source_sfcJac  = sfc_emis(:,icol) * sfc_src_Jac
    !   !
    !   ! Transport
    !   !
    !   call lw_transport_noscat(ngpt, nlay, top_at_1,  &
    !                            tau_loc, trans, sfc_albedo, source_dn, source_up, source_sfc, &
    !                            radn_up, radn_dn, &
    !                            source_sfcJac, radn_up_Jac)
    !   !
    !   ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
    !   !
    !   fac         = 2._wp * pi * weight
    !   radn_dn     = fac * radn_dn   
    !   radn_up     = fac * radn_up   
    !   radn_up_Jac = fac * radn_up_Jac

    !   ! Compute broadband fluxes
    !   call sum_broadband_nocol(ngpt, nlay+1, radn_up, flux_up(:,icol) )
    !   call sum_broadband_nocol(ngpt, nlay+1, radn_dn, flux_dn(:,icol) )

    !   if (compute_Jac) then
    !     call sum_broadband_nocol(ngpt, nlay+1, radn_up_Jac, flux_up_Jac(:,icol) )
    !   end if

    ! end do  ! column loop


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
    integer,  dimension(2,nbnd),            intent(in   ) :: band_limits
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

    ! Ds_ngpt(:,:) = Ds(1)
  
    ! call lw_solver_noscat_broadband(nbnd, ngpt, nlay, ncol, top_at_1, &
    !                       Ds_ngpt, weights(1), inc_flux, &
    !                       band_limits, tau, planck_frac, &
    !                       lay_source_bnd, lev_source_bnd, &
    !                       sfc_source_bnd, sfc_emis, &
    !                       flux_up, flux_dn, sfc_source_bnd_Jac, flux_up_Jac, compute_Jac)

    ! if (nmus > 1) then
    !   allocate( radn_up(nlay+1, ncol) )
    !   allocate( radn_dn(nlay+1, ncol) )
    !   allocate( radn_up_Jac(nlay+1, ncol) )
    ! end if 

    ! do imu = 2, nmus

    !   Ds_ngpt(:,:) = Ds(imu)

    !   call lw_solver_noscat_broadband(nbnd, ngpt, nlay, ncol, top_at_1, &
    !     Ds_ngpt, weights(imu), inc_flux, &
    !     band_limits, tau, planck_frac, &
    !     lay_source_bnd, lev_source_bnd, &
    !     sfc_source_bnd, sfc_emis, &
    !     radn_up, radn_dn, sfc_source_bnd_Jac, radn_up_Jac, compute_Jac)

    !   flux_up = flux_up + radn_up
    !   flux_dn = flux_dn + radn_dn
    !   flux_up_Jac = flux_up_Jac + radn_up_Jac

    ! end do                      

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
    ! real(wp), dimension(ngpt,nlay,ncol),   intent(in   ) :: lay_source   ! Planck source at layer average temperature [W/m2]
    ! real(wp), dimension(ngpt,nlay,ncol), target, &
    !                                        intent(in   ) :: lev_source_inc, lev_source_dec
                                        ! Planck source at layer edge for radiation in increasing/decreasing ilay direction [W/m2]
                                        ! Includes spectral weighting that accounts for state-dependent frequency to g-space mapping
    integer,  dimension(2,nbnd),           intent(in   ) :: band_limits                                              
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in   ) ::  lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in   ) ::  lev_source_bnd ! Planck source function at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol),  intent(in   ) ::  sfc_source_bnd    ! Surface source function by band[W/m2]

    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis         ! Surface emissivity      []
    !real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_src          ! Surface source function [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                           intent(  out) :: flux_up   ! Fluxes [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                           intent(inout) :: flux_dn   ! Top level (= merge(1, nlay+1, top_at_1)
                                                                      ! must contain incident flux boundary condition
    ! ----------------------------------------------------------------------
    real(wp), dimension(ngpt,nlay,ncol) :: lev_source_inc, lev_source_dec
                                        ! Planck source at layer edge for radiation in increasing/decreasing ilay direction [W/m2]
                                        ! Includes spectral weighting that accounts for state-dependent frequency to g-space mapping
    real(wp), dimension(ngpt,nlay,ncol) :: lay_source

    real(wp), dimension(ngpt,ncol)       :: sfc_src          ! Surface source function [W/m2]

    real(wp), dimension(ngpt,nlay ,ncol ) :: Rdif, Tdif, gamma1, gamma2
    real(wp), dimension(ngpt,ncol       ) :: sfc_albedo
    real(wp), dimension(ngpt,nlay+1,ncol) :: lev_source
    real(wp), dimension(ngpt,nlay,ncol  ) :: source_dn, source_up
    real(wp), dimension(ngpt     ,ncol  ) :: source_sfc
    integer :: icol, sfc_lay, top_level, igpt, ibnd
    integer,  dimension(ngpt)             :: gpt_bands ! band number (1...16) for each g-point

    ! ------------------------------------

    if(top_at_1) then
      top_level = 1
      sfc_lay = nlay

    else
      top_level = nlay+1
      sfc_lay = 1
    end if

    ! ------------------------------------
    !$acc enter data create (sfc_src, lay_source, lev_source_dec, lev_source_inc)

    !$acc enter data copyin(sfc_emis, flux_dn)
    !$acc enter data create(gpt_bands,flux_up, Rdif, Tdif, gamma1, gamma2, sfc_albedo, lev_source, source_dn, source_up, source_sfc)

    !$acc parallel loop
    do ibnd = 1, nbnd
      do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
        gpt_bands(igpt) = ibnd
      end do
    end do

    call lw_gpt_source(nbnd, ngpt, nlay, ncol, sfc_lay, gpt_bands, &
                  planck_frac(:,:,:), lay_source_bnd(:,:,:), lev_source_bnd(:,:,:), &
                  sfc_source_bnd(:,:),  &
                  sfc_src, lay_source, lev_source_dec, lev_source_inc)

    !
    ! RRTMGP provides source functions at each level using the spectral mapping
    !   of each adjacent layer. Combine these for two-stream calculations
    !
    call lw_combine_sources(ngpt, nlay, ncol, top_at_1, &
                            lev_source_inc, lev_source_dec, &
                            lev_source)
    !
    ! Cell properties: reflection, transmission for diffuse radiation
    !   Coupling coefficients needed for source function
    !
    call lw_two_stream(ngpt, nlay, ncol, &
                       tau , ssa, g,     &
                       gamma1, gamma2, Rdif, Tdif)

    !
    ! Source function for diffuse radiation
    !
    call lw_source_2str(ngpt, nlay, ncol, top_at_1, &
                        sfc_emis, sfc_src, &
                        lay_source, lev_source, &
                        gamma1, gamma2, Rdif, Tdif, tau, &
                        source_dn, source_up, source_sfc)

    !$acc  parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        sfc_albedo(igpt,icol) = 1._wp - sfc_emis(igpt,icol)
      end do
    end do
    !
    ! Transport
    !
    call adding(ngpt, nlay, ncol, top_at_1,        &
                sfc_albedo,                        &
                Rdif, Tdif,                        &
                source_dn, source_up, source_sfc,  &
                flux_up, flux_dn)
    !$acc exit data delete(tau, ssa, g, sfc_emis)
    !$acc exit data delete(Rdif, Tdif, gamma1, gamma2, sfc_albedo, lev_source, source_dn, source_up, source_sfc)

    !$acc exit data delete (sfc_src, lay_source, lev_source_dec, lev_source_inc)
            
    !$acc exit data copyout(flux_up, flux_dn)
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
  subroutine sw_solver_noscat(ngpt, nlay, ncol, &
                              top_at_1, tau, mu0, flux_dir) bind (C, name="sw_solver_noscat")
    integer,                    intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt            ), intent(in   ) :: mu0          ! cosine of solar zenith angle
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: flux_dir     ! Direct-beam flux, spectral [W/m2]
                                                                          ! Top level must contain incident flux boundary condition
    integer :: igpt, ilev, icol
    real(wp) :: mu0_inv(ngpt)
    ! ------------------------------------
    ! ------------------------------------
    !$acc enter data copyin(tau, mu0) create(mu0_inv, flux_dir)
    !$acc parallel loop
    do igpt = 1, ngpt
      mu0_inv(igpt) = 1._wp/mu0(igpt)
    enddo
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.

    ! Downward propagation
    if(top_at_1) then
      ! For the flux at this level, what was the previous level, and which layer has the
      !   radiation just passed through?
      ! layer index = level index - 1
      ! previous level is up (-1)
      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          do ilev = 2, nlay+1
            flux_dir(igpt,ilev,icol) = flux_dir(igpt,ilev-1,icol) * exp(-tau(igpt,ilev,icol)*mu0_inv(igpt))
          end do
        end do
      end do
    else
      ! layer index = level index
      ! previous level is up (+1)
      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          do ilev = nlay, 1, -1
            flux_dir(igpt,ilev,icol) = flux_dir(igpt,ilev+1,icol) * exp(-tau(igpt,ilev,icol)*mu0_inv(igpt))
          end do
        end do
      end do
    end if
    !$acc exit data delete(tau, mu0, mu0_inv) copyout(flux_dir)
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
                                flux_up, flux_dn, flux_dir) bind (C, name="sw_solver_2stream")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(ngpt            ), intent(in   ) :: mu0     ! cosine of solar zenith angle
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_alb_dir, sfc_alb_dif
                                                                  ! Spectral albedo of surface to direct and diffuse radiation
    real(wp), dimension(ngpt,nlay+1,ncol), &
                                            intent(  out) :: flux_up ! Fluxes [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), &                        ! Downward fluxes contain boundary conditions
                                            intent(inout) :: flux_dn, flux_dir
    ! -------------------------------------------
    integer :: igpt, ilay, icol
    real(wp), dimension(ngpt,nlay,ncol) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    real(wp), dimension(ngpt,nlay,ncol) :: source_up, source_dn
    real(wp), dimension(ngpt     ,ncol) :: source_srf
    ! ------------------------------------
    !
    ! Cell properties: transmittance and reflectance for direct and diffuse radiation
    !
    !$acc enter data copyin(tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, flux_dn, flux_dir)
    !$acc enter data create(Rdif, Tdif, Rdir, Tdir, Tnoscat, source_up, source_dn, source_srf, flux_up)
    call sw_two_stream(ngpt, nlay, ncol, mu0, &
                        tau , ssa , g   ,      &
                        Rdif, Tdif, Rdir, Tdir, Tnoscat)
    call sw_source_2str(ngpt, nlay, ncol, top_at_1,       &
                        Rdir, Tdir, Tnoscat, sfc_alb_dir, &
                        source_up, source_dn, source_srf, flux_dir)
    call adding(ngpt, nlay, ncol, top_at_1,   &
                sfc_alb_dif, Rdif, Tdif,      &
                source_dn, source_up, source_srf, flux_up, flux_dn)
    !
    ! adding computes only diffuse flux; flux_dn is total
    !
    !$acc  parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay+1
        do igpt = 1, ngpt
          flux_dn(igpt,ilay,icol) = flux_dn(igpt,ilay,icol) + flux_dir(igpt,ilay,icol)
        end do
      end do
    end do
    !$acc exit data copyout(flux_up, flux_dn, flux_dir)
    !$acc exit data delete (tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, Rdif, Tdif, Rdir, Tdir, Tnoscat, source_up, source_dn, source_srf)

  end subroutine sw_solver_2stream

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
    ! mu0_inv = 1._wp/mu0
    ! ! Indexing into arrays for upward and downward propagation depends on the vertical
    ! !   orientation of the arrays (whether the domain top is at the first or last index)
    ! ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    ! ! Downward propagation
    ! if(top_at_1) then

    !   do icol = 1, ncol
    
    !     ! Apply boundary condition
    !     radn_dir(:,nlay+1) = inc_flux(:,icol)

    !     ! For the flux at this level, what was the previous level, and which layer has the
    !     !   radiation just passed through?
    !     ! layer index = level index - 1
    !     ! previous level is up (-1)
    !     do ilev = 2, nlay+1
    !       radn_dir(:,ilev) = radn_dir(:,ilev-1) * exp(-tau(:,ilev-1,icol)*mu0_inv(icol))
    !     end do

    !     ! Compute broadband fluxes
    !     call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )

    !   end do

    ! else

    !   ! Apply boundary condition
    !   radn_dir(:,1) = inc_flux(:,icol)

    !   do icol = 1, ncol

    !     ! layer index = level index
    !     ! previous level is up (+1)
    !     do ilev = nlay, 1, -1
    !       radn_dir(:,ilev) = radn_dir(:,ilev+1) * exp(-tau(:,ilev,icol)*mu0_inv(icol))
    !     end do

    !     ! Compute broadband fluxes
    !     call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )

    !   end do 

    ! end if
  end subroutine sw_solver_noscat_broadband
  ! -------------------------------------------------------------------------------------------------
  !
  ! Shortwave two-stream calculation:
  !   compute layer reflectance, transmittance
  !   compute solar source function for diffuse radiation
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------

    !
  !   Top-level shortwave kernels, return broadband fluxes
  !
  ! -------------------------------------------------------------------------------------------------
  !
  !   Extinction-only i.e. solar direct beam
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

    ! if(top_at_1) then
    !   top_level = 1
    ! else
    !   top_level = nlay+1
    ! end if

    ! do icol = 1, ncol

    !   ! Apply boundary condition
    !   radn_dir(:,top_level) = inc_flux(:,icol) * mu0(icol)
    !   radn_dn(:,top_level)  = inc_flux_dif(:,icol)

    !   !
    !   ! Cell properties: transmittance and reflectance for direct and diffuse radiation
    !   !
    !   call sw_two_stream(ngpt, nlay, mu0(icol),                                &
    !                      tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), &
    !                      Rdif, Tdif, Rdir, Tdir, Tnoscat)      
    !   !
    !   ! Direct-beam and source for diffuse radiation
    !   !
    !   call sw_source_2str(ngpt, nlay, top_at_1, Rdir, Tdir, Tnoscat, sfc_alb_dir(:,icol),&
    !                       source_up, source_dn, source_srf, radn_dir)

    !   !
    !   ! Transport
    !   !
    !   call adding(ngpt, nlay, top_at_1,            &
    !                  sfc_alb_dif(:,icol), Rdif, Tdif, &
    !                  source_dn, source_up, source_srf, radn_up, radn_dn)
    !   !
    !   ! adding computes only diffuse flux; flux_dn is total
    !   !
    !   radn_dn = radn_dn + radn_dir

    !   ! Compute broadband fluxes
    !   call sum_broadband_nocol(ngpt, nlay+1, radn_dir, flux_dir(:,icol) )
    !   call sum_broadband_nocol(ngpt, nlay+1, radn_up, flux_up(:,icol) )
    !   call sum_broadband_nocol(ngpt, nlay+1, radn_dn, flux_dn(:,icol) )
    ! end do

  end subroutine sw_solver_2stream_broadband

  ! -------------------------------------------------------------------------------------------------
  !
  !   Lower-level longwave kernels
  !
  ! ---------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption
  ! See Clough et al., 1992, doi: 10.1029/92JD01419, Eq 13
  ! This routine implements point-wise stencil, and has to be called in a loop
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_noscat_stencil(ngpt, nlay, ncol, igpt, ilay, icol,                   &
    lay_source, lev_source_up, lev_source_dn, tau, trans, &
    source_dn, source_up)
    !$acc routine seq
    !
    integer,                               intent(in)   :: ngpt, nlay, ncol
    integer,                               intent(in)   :: igpt, ilay, icol ! Working point coordinates
    real(wp), dimension(ngpt, nlay, ncol), intent(in)   :: lay_source,    & ! Planck source at layer center
                            lev_source_up, & ! Planck source at levels (layer edges),
                            lev_source_dn, & !   increasing/decreasing layer index
                            tau,           & ! Optical path (tau/mu)
                            trans            ! Transmissivity (exp(-tau))
    real(wp), dimension(ngpt, nlay, ncol), intent(inout):: source_dn, source_up
                                    ! Source function at layer edges
                                    ! Down at the bottom of the layer, up at the top
    ! --------------------------------
    real(wp), parameter  :: tau_thresh = sqrt(epsilon(tau))
    real(wp)             :: fact

    ! ---------------------------------------------------------------
    !
    ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
    !   is of order epsilon (smallest difference from 1. in working precision)
    !   Thanks to Peter Blossey
    !
    if(tau(igpt,ilay,icol) > tau_thresh) then
    fact = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
    else
    fact = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt,ilay,icol))
    end if
    !
    ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
    !
    source_dn(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source_dn(igpt,ilay,icol) + &
    2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source_dn(igpt,ilay,icol))
    source_up(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source_up(igpt,ilay,icol) + &
    2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source_up(igpt,ilay,icol))

  end subroutine lw_source_noscat_stencil
  ! ---------------------------------------------------------------
  !
  ! Driver function to compute LW source function for upward and downward emission
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_noscat(ngpt, nlay, ncol, lay_source, lev_source_up, lev_source_dn, tau, trans, &
                              source_dn, source_up) bind(C, name="lw_source_noscat")
    integer,                               intent(in) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt, nlay, ncol), intent(in) :: lay_source,    & ! Planck source at layer center
                                                         lev_source_up, & ! Planck source at levels (layer edges),
                                                         lev_source_dn, & !   increasing/decreasing layer index
                                                         tau,           & ! Optical path (tau/mu)
                                                         trans            ! Transmissivity (exp(-tau))
    real(wp), dimension(ngpt, nlay, ncol), intent(out):: source_dn, source_up
                                                                ! Source function at layer edges
                                                                ! Down at the bottom of the layer, up at the top
    ! --------------------------------
    integer :: igpt, ilay, icol
    ! ---------------------------------------------------------------
    !$acc  parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          call lw_source_noscat_stencil(ngpt, nlay, ncol, igpt, ilay, icol,        &
                                        lay_source, lev_source_up, lev_source_dn,  &
                                        tau, trans,                                &
                                        source_dn, source_up)
        end do
      end do
    end do

  end subroutine lw_source_noscat
  ! ---------------------------------------------------------------
  !
  ! ---------------------------------------------------------------
  !
  ! Longwave no-scattering transport
  !
  ! ---------------------------------------------------------------
  subroutine lw_transport_noscat(ngpt, nlay, ncol, top_at_1, &
                                 tau, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                                 radn_up, radn_dn, source_sfcJac, radn_up_Jac) bind(C, name="lw_transport_noscat")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: tau, &     ! Absorption optical thickness, pre-divided by mu []
                                                            trans      ! transmissivity = exp(-tau)
    real(wp), dimension(ngpt       ,ncol), intent(in   ) :: sfc_albedo ! Surface albedo
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: source_dn, &
                                                            source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt       ,ncol), intent(in   ) :: source_sfc ! Surface source function [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_dn ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(  out) :: radn_up ! Radiances [W/m2-str]
                                                                             ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt       ,ncol), intent(in )   :: source_sfcJac ! surface temperature Jacobian of surface source function [W/m2/K]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(out)   :: radn_up_Jac    ! surface temperature Jacobian of Radiances [W/m2-str / K]
    ! Local variables
    integer :: icol, ilev, igpt
    ! ---------------------------------------------------
    ! ---------------------------------------------------
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Downward propagation
          do ilev = 2, nlay+1
            radn_dn(igpt,ilev,icol) = trans(igpt,ilev-1,icol)*radn_dn(igpt,ilev-1,icol) + source_dn(igpt,ilev-1,icol)
          end do

          ! Surface reflection and emission
          radn_up   (igpt,nlay+1,icol) = radn_dn(igpt,nlay+1,icol)*sfc_albedo(igpt,icol) + source_sfc   (igpt,icol)
          radn_up_Jac(igpt,nlay+1,icol) = source_sfcJac(igpt,icol)

          ! Upward propagation
          do ilev = nlay, 1, -1
            radn_up   (igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up   (igpt,ilev+1,icol) + source_up(igpt,ilev,icol)
            radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev+1,icol)
          end do
        end do
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Downward propagation
          do ilev = nlay, 1, -1
            radn_dn(igpt,ilev,icol) = trans(igpt,ilev  ,icol)*radn_dn(igpt,ilev+1,icol) + source_dn(igpt,ilev,icol)
          end do

          ! Surface reflection and emission
          radn_up   (igpt,1,icol) = radn_dn(igpt,1,icol)*sfc_albedo(igpt,icol) + source_sfc   (igpt,icol)
          radn_up_Jac(igpt,1,icol) = source_sfcJac(igpt,icol)

          ! Upward propagation
          do ilev = 2, nlay+1
            radn_up   (igpt,ilev,icol) = trans(igpt,ilev-1,icol) * radn_up   (igpt,ilev-1,icol) +  source_up(igpt,ilev-1,icol)
            radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev-1,icol) * radn_up_Jac(igpt,ilev-1,icol)
          end do
        end do
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
  subroutine lw_two_stream(ngpt, nlay, ncol, tau, w0, g, &
                                gamma1, gamma2, Rdif, Tdif) bind(C, name="lw_two_stream")
    integer,                             intent(in)  :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt,nlay,ncol), intent(out) :: gamma1, gamma2, Rdif, Tdif

    ! -----------------------
    integer  :: igpt, ilay, icol

    ! Variables used in Meador and Weaver
    real(wp) :: k

    ! Ancillary variables
    real(wp) :: RT_term
    real(wp) :: exp_minusktau, exp_minus2ktau

    real(wp), parameter :: LW_diff_sec = 1.66  ! 1./cos(diffusivity angle)
    ! ---------------------------------
    ! ---------------------------------
    !$acc enter data copyin(tau, w0, g)
    !$acc enter data create(gamma1, gamma2, Rdif, Tdif)

    !$acc  parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          !
          ! Coefficients differ from SW implementation because the phase function is more isotropic
          !   Here we follow Fu et al. 1997, doi:10.1175/1520-0469(1997)054<2799:MSPITI>2.0.CO;2
          !   and use a diffusivity sec of 1.66
          !
          gamma1(igpt,ilay,icol)= LW_diff_sec * (1._wp - 0.5_wp * w0(igpt,ilay,icol) * (1._wp + g(igpt,ilay,icol))) ! Fu et al. Eq 2.9
          gamma2(igpt,ilay,icol)= LW_diff_sec *          0.5_wp * w0(igpt,ilay,icol) * (1._wp - g(igpt,ilay,icol))  ! Fu et al. Eq 2.10

          ! Written to encourage vectorization of exponential, square root
          ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
          !   k = 0 for isotropic, conservative scattering; this lower limit on k
          !   gives relative error with respect to conservative solution
          !   of < 0.1% in Rdif down to tau = 10^-9
          k = sqrt(max((gamma1(igpt,ilay,icol) - gamma2(igpt,ilay,icol)) * &
                       (gamma1(igpt,ilay,icol) + gamma2(igpt,ilay,icol)),  &
                       1.e-12_wp))
          exp_minusktau = exp(-tau(igpt,ilay,icol)*k)

          !
          ! Diffuse reflection and transmission
          !
          exp_minus2ktau = exp_minusktau * exp_minusktau

          ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
          RT_term = 1._wp / (k * (1._wp + exp_minus2ktau)  + &
                    gamma1(igpt,ilay,icol) * (1._wp - exp_minus2ktau) )

          ! Equation 25
          Rdif(igpt,ilay,icol) = RT_term * gamma2(igpt,ilay,icol) * (1._wp - exp_minus2ktau)

          ! Equation 26
          Tdif(igpt,ilay,icol) = RT_term * 2._wp * k * exp_minusktau
        end do
      end do
    end do
    !$acc exit data delete (tau, w0, g)
    !$acc exit data copyout(gamma1, gamma2, Rdif, Tdif)
  end subroutine lw_two_stream
  ! -------------------------------------------------------------------------------------------------
  !
  ! Source function combination
  ! RRTMGP provides two source functions at each level
  !   using the spectral mapping from each of the adjascent layers.
  !   Need to combine these for use in two-stream calculation.
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_combine_sources(ngpt, nlay, ncol, top_at_1, &
                                lev_src_inc, lev_src_dec, lev_source) bind(C, name="lw_combine_sources")
    integer,                                 intent(in ) :: ngpt, nlay, ncol
    logical(wl),                             intent(in ) :: top_at_1
    real(wp), dimension(ngpt, nlay  , ncol), intent(in ) :: lev_src_inc, lev_src_dec
    real(wp), dimension(ngpt, nlay+1, ncol), intent(out) :: lev_source

    integer :: igpt, ilay, icol
    ! ---------------------------------------------------------------
    ! ---------------------------------
    !$acc enter data copyin(lev_src_inc, lev_src_dec)
    !$acc enter data create(lev_source)

    !$acc  parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay+1
        do igpt = 1,ngpt
          if(ilay == 1) then
            lev_source(igpt, ilay, icol) =      lev_src_dec(igpt, ilay,   icol)
          else if (ilay == nlay+1) then
            lev_source(igpt, ilay, icol) =      lev_src_inc(igpt, ilay-1, icol)
          else
            lev_source(igpt, ilay, icol) = sqrt(lev_src_dec(igpt, ilay, icol) * &
                                                lev_src_inc(igpt, ilay-1, icol))
          end if
        end do
      end do
    end do
    !$acc exit data delete (lev_src_inc, lev_src_dec)
    !$acc exit data copyout(lev_source)
  end subroutine lw_combine_sources
  ! ---------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption
  !   This version straight from ECRAD
  !   Source is provided as W/m2-str; factor of pi converts to flux units
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_2str(ngpt, nlay, ncol, top_at_1,   &
                            sfc_emis, sfc_src,      &
                            lay_source, lev_source, &
                            gamma1, gamma2, rdif, tdif, tau, source_dn, source_up, source_sfc) &
                            bind (C, name="lw_source_2str")
    integer,                         intent(in) :: ngpt, nlay, ncol
    logical(wl),                     intent(in) :: top_at_1
    real(wp), dimension(ngpt      , ncol), intent(in) :: sfc_emis, sfc_src
    real(wp), dimension(ngpt, nlay, ncol), intent(in) :: lay_source,    & ! Planck source at layer center
                                                   tau,           & ! Optical depth (tau)
                                                   gamma1, gamma2,& ! Coupling coefficients
                                                   rdif, tdif       ! Layer reflectance and transmittance
    real(wp), dimension(ngpt, nlay+1, ncol), target, &
                                     intent(in)  :: lev_source       ! Planck source at layer edges
    real(wp), dimension(ngpt, nlay, ncol), intent(out) :: source_dn, source_up
    real(wp), dimension(ngpt      , ncol), intent(out) :: source_sfc      ! Source function for upward radation at surface

    integer             :: igpt, ilay, icol
    real(wp)            :: Z, Zup_top, Zup_bottom, Zdn_top, Zdn_bottom
    real(wp)            :: lev_source_bot, lev_source_top
    ! ---------------------------------------------------------------
    ! ---------------------------------
    !$acc enter data copyin(sfc_emis, sfc_src, lay_source, tau, gamma1, gamma2, rdif, tdif, lev_source)
    !$acc enter data create(source_dn, source_up, source_sfc)

    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          if (tau(igpt,ilay,ncol) > 1.0e-8_wp) then
            if(top_at_1) then
              lev_source_top = lev_source(igpt,ilay  ,ncol)
              lev_source_bot = lev_source(igpt,ilay+1,ncol)
            else
              lev_source_top = lev_source(igpt,ilay+1,ncol)
              lev_source_bot = lev_source(igpt,ilay  ,ncol)
            end if
            !
            ! Toon et al. (JGR 1989) Eqs 26-27
            !
            Z = (lev_source_bot-lev_source_top) / (tau(igpt,ilay,icol)*(gamma1(igpt,ilay,icol)+gamma2(igpt,ilay,icol)))
            Zup_top        =  Z + lev_source_top
            Zup_bottom     =  Z + lev_source_bot
            Zdn_top        = -Z + lev_source_top
            Zdn_bottom     = -Z + lev_source_bot
            source_up(igpt,ilay,icol) = pi * (Zup_top    - rdif(igpt,ilay,icol) * Zdn_top    - tdif(igpt,ilay,icol) * Zup_bottom)
            source_dn(igpt,ilay,icol) = pi * (Zdn_bottom - rdif(igpt,ilay,icol) * Zup_bottom - tdif(igpt,ilay,icol) * Zdn_top)
          else
            source_up(igpt,ilay,icol) = 0._wp
            source_dn(igpt,ilay,icol) = 0._wp
          end if
          if(ilay == 1) source_sfc(igpt,icol) = pi * sfc_emis(igpt,icol) * sfc_src(igpt,icol)
        end do
      end do
    end do
    !$acc exit data delete(sfc_emis, sfc_src, lay_source, tau, gamma1, gamma2, rdif, tdif, lev_source)
    !$acc exit data copyout(source_dn, source_up, source_sfc)

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
    subroutine sw_two_stream(ngpt, nlay, ncol, mu0, tau, w0, g, &
                                  Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream")
      integer,                             intent(in)  :: ngpt, nlay, ncol
      real(wp), dimension(ngpt),           intent(in)  :: mu0
      real(wp), dimension(ngpt,nlay,ncol), intent(in)  :: tau, w0, g
      real(wp), dimension(ngpt,nlay,ncol), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat

      ! -----------------------
      integer  :: igpt,ilay,icol

      ! Variables used in Meador and Weaver
      real(wp) :: gamma1, gamma2, gamma3, gamma4
      real(wp) :: alpha1, alpha2, k

      ! Ancillary variables
      real(wp) :: RT_term
      real(wp) :: exp_minusktau, exp_minus2ktau
      real(wp) :: k_mu, k_gamma3, k_gamma4
      real(wp) :: mu0_inv(ngpt)
      ! ---------------------------------
      ! ---------------------------------
      !$acc enter data copyin (mu0, tau, w0, g)
      !$acc enter data create(Rdif, Tdif, Rdir, Tdir, Tnoscat, mu0_inv)

      !$acc parallel loop
      do igpt = 1, ngpt
        mu0_inv(igpt) = 1._wp/mu0(igpt)
      enddo

      ! NOTE: this kernel appears to cause small (10^-6) differences between GPU
      ! and CPU. This *might* be floating point differences in implementation of
      ! the exp function.
      !$acc  parallel loop collapse(3)
      do icol = 1, ncol
        do ilay = 1, nlay
          do igpt = 1, ngpt
            ! Zdunkowski Practical Improved Flux Method "PIFM"
            !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
            !
            gamma1= (8._wp - w0(igpt,ilay,icol) * (5._wp + 3._wp * g(igpt,ilay,icol))) * .25_wp
            gamma2=  3._wp *(w0(igpt,ilay,icol) * (1._wp -         g(igpt,ilay,icol))) * .25_wp
            gamma3= (2._wp - 3._wp * mu0(igpt)  *                  g(igpt,ilay,icol) ) * .25_wp
            gamma4=  1._wp - gamma3

            alpha1 = gamma1 * gamma4 + gamma2 * gamma3           ! Eq. 16
            alpha2 = gamma1 * gamma3 + gamma2 * gamma4           ! Eq. 17
            ! Written to encourage vectorization of exponential, square root
            ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
            !   k = 0 for isotropic, conservative scattering; this lower limit on k
            !   gives relative error with respect to conservative solution
            !   of < 0.1% in Rdif down to tau = 10^-9
            k = sqrt(max((gamma1 - gamma2) * &
                         (gamma1 + gamma2),  &
                         1.e-12_wp))
            exp_minusktau = exp(-tau(igpt,ilay,icol)*k)
            !
            ! Diffuse reflection and transmission
            !
            exp_minus2ktau = exp_minusktau * exp_minusktau

            ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
            RT_term = 1._wp / (k      * (1._wp + exp_minus2ktau)  + &
                               gamma1 * (1._wp - exp_minus2ktau) )

            ! Equation 25
            Rdif(igpt,ilay,icol) = RT_term * gamma2 * (1._wp - exp_minus2ktau)

            ! Equation 26
            Tdif(igpt,ilay,icol) = RT_term * 2._wp * k * exp_minusktau

            !
            ! Transmittance of direct, unscattered beam. Also used below
            !
            Tnoscat(igpt,ilay,icol) = exp(-tau(igpt,ilay,icol)*mu0_inv(igpt))

            !
            ! Direct reflect and transmission
            !
            k_mu     = k * mu0(igpt)
            k_gamma3 = k * gamma3
            k_gamma4 = k * gamma4

            !
            ! Equation 14, multiplying top and bottom by exp(-k*tau)
            !   and rearranging to avoid div by 0.
            !
            RT_term =  w0(igpt,ilay,icol) * RT_term/merge(1._wp - k_mu*k_mu, &
                                                         epsilon(1._wp),    &
                                                         abs(1._wp - k_mu*k_mu) >= epsilon(1._wp))

            Rdir(igpt,ilay,icol) = RT_term  *                                    &
               ((1._wp - k_mu) * (alpha2 + k_gamma3)                  - &
                (1._wp + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau - &
                2.0_wp * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau  * Tnoscat(igpt,ilay,icol))

            !
            ! Equation 15, multiplying top and bottom by exp(-k*tau),
            !   multiplying through by exp(-tau/mu0) to
            !   prefer underflow to overflow
            ! Omitting direct transmittance
            !
            Tdir(igpt,ilay,icol) = &
                     -RT_term * ((1._wp + k_mu) * (alpha1 + k_gamma4) * Tnoscat(igpt,ilay,icol) - &
                                 (1._wp - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat(igpt,ilay,icol) - &
                                  2.0_wp * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau )

          end do
        end do
      end do
      !$acc exit data delete (mu0, tau, w0, g, mu0_inv)
      !$acc exit data copyout(Rdif, Tdif, Rdir, Tdir, Tnoscat)

    end subroutine sw_two_stream
  ! ---------------------------------------------------------------
  !
  ! Direct beam source for diffuse radiation in layers and at surface;
  !   report direct beam as a byproduct
  !
  subroutine sw_source_2str(ngpt, nlay, ncol, top_at_1, Rdir, Tdir, Tnoscat, sfc_albedo, &
                            source_up, source_dn, source_sfc, flux_dn_dir) bind(C, name="sw_source_2str")
    integer,                                 intent(in   ) :: ngpt, nlay, ncol
    logical(wl),                             intent(in   ) :: top_at_1
    real(wp), dimension(ngpt, nlay  , ncol), intent(in   ) :: Rdir, Tdir, Tnoscat ! Layer reflectance, transmittance for diffuse radiation
    real(wp), dimension(ngpt        , ncol), intent(in   ) :: sfc_albedo          ! surface albedo for direct radiation
    real(wp), dimension(ngpt, nlay  , ncol), intent(  out) :: source_dn, source_up
    real(wp), dimension(ngpt        , ncol), intent(  out) :: source_sfc          ! Source function for upward radation at surface
    real(wp), dimension(ngpt, nlay+1, ncol), intent(inout) :: flux_dn_dir ! Direct beam flux
                                                                    ! intent(inout) because top layer includes incident flux

    integer :: igpt, ilev, icol
    ! ---------------------------------
    ! ---------------------------------
    !$acc enter data copyin (Rdir, Tdir, Tnoscat, sfc_albedo, flux_dn_dir)
    !$acc enter data create(source_dn, source_up, source_sfc)

    if(top_at_1) then
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          do ilev = 1, nlay
            source_up(igpt,ilev,icol)     =    Rdir(igpt,ilev,icol) * flux_dn_dir(igpt,ilev,icol)
            source_dn(igpt,ilev,icol)     =    Tdir(igpt,ilev,icol) * flux_dn_dir(igpt,ilev,icol)
            flux_dn_dir(igpt,ilev+1,icol) = Tnoscat(igpt,ilev,icol) * flux_dn_dir(igpt,ilev,icol)
            if(ilev == nlay) source_sfc(igpt,icol) = flux_dn_dir(igpt,nlay+1,icol)*sfc_albedo(igpt,icol)
          end do
        end do
      end do
    else
      ! layer index = level index
      ! previous level is up (+1)
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          do ilev = nlay, 1, -1
            source_up(igpt,ilev,icol)   =    Rdir(igpt,ilev,icol) * flux_dn_dir(igpt,ilev+1,icol)
            source_dn(igpt,ilev,icol)   =    Tdir(igpt,ilev,icol) * flux_dn_dir(igpt,ilev+1,icol)
            flux_dn_dir(igpt,ilev,icol) = Tnoscat(igpt,ilev,icol) * flux_dn_dir(igpt,ilev+1,icol)
            if(ilev ==    1) source_sfc(igpt,icol) = flux_dn_dir(igpt,    1,icol)*sfc_albedo(igpt,icol)
          end do
        end do
      end do
    end if
    !$acc exit data copyout(source_dn, source_up, source_sfc, flux_dn_dir)
    !$acc exit data delete(Rdir, Tdir, Tnoscat, sfc_albedo)

  end subroutine sw_source_2str
! ---------------------------------------------------------------
!
! Transport of diffuse radiation through a vertically layered atmosphere.
!   Equations are after Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08)
!   This routine is shared by longwave and shortwave
!
! -------------------------------------------------------------------------------------------------
  subroutine adding(ngpt, nlay, ncol, top_at_1, &
                    albedo_sfc,           &
                    rdif, tdif,           &
                    src_dn, src_up, src_sfc, &
                    flux_up, flux_dn) bind(C, name="adding")
    integer,                               intent(in   ) :: ngpt, nlay, ncol
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt       ,ncol), intent(in   ) :: albedo_sfc
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: rdif, tdif
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: src_dn, src_up
    real(wp), dimension(ngpt       ,ncol), intent(in   ) :: src_sfc
    real(wp), dimension(ngpt,nlay+1,ncol), intent(  out) :: flux_up
    ! intent(inout) because top layer includes incident flux
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: flux_dn
    ! ------------------
    integer :: igpt, ilev, icol

    ! These arrays could be private per thread in OpenACC, with 1 dimension of size nlay (or nlay+1)
    ! However, current PGI (19.4) has a bug preventing it from properly handling such private arrays.
    ! So we explicitly create the temporary arrays of size nlay(+1) per each of the ngpt*ncol elements
    !
    real(wp), dimension(ngpt,nlay+1,ncol) :: albedo, &  ! reflectivity to diffuse radiation below this level
                                              ! alpha in SH08
                                   src        ! source of diffuse upwelling radiation from emission or
                                              ! scattering of direct beam
                                              ! G in SH08
    real(wp), dimension(ngpt,nlay  ,ncol) :: denom      ! beta in SH08
    ! ------------------
    ! ---------------------------------
    !
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    !
    !$acc enter data copyin(albedo_sfc, rdif, tdif, src_dn, src_up, src_sfc, flux_dn)
    !$acc enter data create(flux_up, albedo, src, denom)

    if(top_at_1) then
      !$acc parallel loop gang vector collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ilev = nlay + 1
          ! Albedo of lowest level is the surface albedo...
          albedo(igpt,ilev,icol)  = albedo_sfc(igpt,icol)
          ! ... and source of diffuse radiation is surface emission
          src(igpt,ilev,icol) = src_sfc(igpt,icol)

          !
          ! From bottom to top of atmosphere --
          !   compute albedo and source of upward radiation
          !
          do ilev = nlay, 1, -1
            denom(igpt,ilev,icol) = 1._wp/(1._wp - rdif(igpt,ilev,icol)*albedo(igpt,ilev+1,icol))    ! Eq 10
            albedo(igpt,ilev,icol) = rdif(igpt,ilev,icol) + &
                  tdif(igpt,ilev,icol)*tdif(igpt,ilev,icol) * albedo(igpt,ilev+1,icol) * denom(igpt,ilev,icol) ! Equation 9
            !
            ! Equation 11 -- source is emitted upward radiation at top of layer plus
            !   radiation emitted at bottom of layer,
            !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
            !
            src(igpt,ilev,icol) =  src_up(igpt, ilev, icol) + &
                           tdif(igpt,ilev,icol) * denom(igpt,ilev,icol) *       &
                             (src(igpt,ilev+1,icol) + albedo(igpt,ilev+1,icol)*src_dn(igpt,ilev,icol))
          end do

          ! Eq 12, at the top of the domain upwelling diffuse is due to ...
          ilev = 1
          flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! ... reflection of incident diffuse and
                                    src(igpt,ilev,icol)                                  ! emission from below

          !
          ! From the top of the atmosphere downward -- compute fluxes
          !
          do ilev = 2, nlay+1
            flux_dn(igpt,ilev,icol) = (tdif(igpt,ilev-1,icol)*flux_dn(igpt,ilev-1,icol) + &  ! Equation 13
                               rdif(igpt,ilev-1,icol)*src(igpt,ilev,icol) +       &
                               src_dn(igpt,ilev-1,icol)) * denom(igpt,ilev-1,icol)
            flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! Equation 12
                              src(igpt,ilev,icol)
          end do
        end do
      end do

    else

      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ilev = 1
          ! Albedo of lowest level is the surface albedo...
          albedo(igpt,ilev,icol)  = albedo_sfc(igpt,icol)
          ! ... and source of diffuse radiation is surface emission
          src(igpt,ilev,icol) = src_sfc(igpt,icol)

          !
          ! From bottom to top of atmosphere --
          !   compute albedo and source of upward radiation
          !
          do ilev = 1, nlay
            denom (igpt,ilev  ,icol) = 1._wp/(1._wp - rdif(igpt,ilev,icol)*albedo(igpt,ilev,icol))                ! Eq 10
            albedo(igpt,ilev+1,icol) = rdif(igpt,ilev,icol) + &
                               tdif(igpt,ilev,icol)*tdif(igpt,ilev,icol) * albedo(igpt,ilev,icol) * denom(igpt,ilev,icol) ! Equation 9
            !
            ! Equation 11 -- source is emitted upward radiation at top of layer plus
            !   radiation emitted at bottom of layer,
            !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
            !
            src(igpt,ilev+1,icol) =  src_up(igpt, ilev, icol) +  &
                             tdif(igpt,ilev,icol) * denom(igpt,ilev,icol) *       &
                             (src(igpt,ilev,icol) + albedo(igpt,ilev,icol)*src_dn(igpt,ilev,icol))
          end do

          ! Eq 12, at the top of the domain upwelling diffuse is due to ...
          ilev = nlay+1
          flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! ... reflection of incident diffuse and
                            src(igpt,ilev,icol)                          ! scattering by the direct beam below

          !
          ! From the top of the atmosphere downward -- compute fluxes
          !
          do ilev = nlay, 1, -1
            flux_dn(igpt,ilev,icol) = (tdif(igpt,ilev,icol)*flux_dn(igpt,ilev+1,icol) + &  ! Equation 13
                               rdif(igpt,ilev,icol)*src(igpt,ilev,icol) + &
                               src_dn(igpt, ilev, icol)) * denom(igpt,ilev,icol)
            flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! Equation 12
                              src(igpt,ilev,icol)

          end do
        end do
      end do
    end if
    !$acc exit data delete(albedo_sfc, rdif, tdif, src_dn, src_up, src_sfc, albedo, src, denom)
    !$acc exit data copyout(flux_up, flux_dn)
  end subroutine adding
  ! -------------------------------------------------------------------------------------------------
  !
  ! Planck sources by g-point from plank fraction and sources by band
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine lw_gpt_source_Jac_nocol(nbnd, ngpt, nlay, sfc_lay, gpt_bands, planck_frac, &
    lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_source_bnd_Jac, &     ! inputs: band source functions
    sfc_source, sfc_source_Jac,  lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

    integer,                          intent(in   ) :: nbnd, ngpt, nlay, sfc_lay
    integer,  dimension(ngpt),        intent(in)    :: gpt_bands ! band number (1...16) for each g-point
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

    !$acc parallel loop 
    do igpt = 1, ngpt
      sfc_source(igpt)     = planck_frac(igpt,sfc_lay) * sfc_source_bnd(gpt_bands(igpt))
      sfc_source_Jac(igpt) = planck_frac(igpt,sfc_lay) * (sfc_source_bnd(gpt_bands(igpt)) - sfc_source_bnd_Jac(gpt_bands(igpt)))
    end do

    !$acc parallel loop collapse(2)
    do ilay = 1, nlay
      do igpt = 1,ngpt 
        ! compute layer source irradiance for each g-point
        lay_source(igpt,ilay)       = planck_frac(igpt,ilay) * lay_source_bnd(gpt_bands(igpt),ilay)
        ! compute level source irradiance for each g-point, one each for upward and downward paths
        lev_source_dec(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(gpt_bands(igpt),ilay)
        lev_source_inc(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(gpt_bands(igpt),ilay+1)
      end do
    end do

  end subroutine lw_gpt_source_Jac_nocol


  pure subroutine lw_gpt_source_nocol(nbnd, ngpt, nlay, sfc_lay, gpt_bands, planck_frac, &
    lay_source_bnd, lev_source_bnd, sfc_source_bnd, &     ! inputs: band source functions
    sfc_source, lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

    integer,                          intent(in   ) :: nbnd, ngpt, nlay, sfc_lay
    integer,  dimension(ngpt),        intent(in)    :: gpt_bands ! band number (1...16) for each g-point
    real(wp), dimension(ngpt,nlay),   intent(in   ) :: planck_frac 
    real(wp), dimension(nbnd,nlay),   intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1), intent(in )   :: lev_source_bnd
    real(wp), dimension(nbnd),        intent(in )   :: sfc_source_bnd      ! Surface source by band
    ! outputs
    real(wp), dimension(ngpt     ),   intent(out)   :: sfc_source
    real(wp), dimension(ngpt,nlay),   intent(out)   :: lay_source
    real(wp), dimension(ngpt,nlay),   intent(out)   :: lev_source_dec, lev_source_inc

    integer             ::  ilay, igpt, ibnd, gptS, gptE

   !$acc parallel loop 
    do igpt = 1, ngpt
      sfc_source(igpt)     = planck_frac(igpt,sfc_lay) * sfc_source_bnd(gpt_bands(igpt))
    end do

    !$acc parallel loop collapse(2)
    do ilay = 1, nlay
      do igpt = 1 , ngpt 
        ! compute layer source irradiance for each g-point
        lay_source(igpt,ilay)       = planck_frac(igpt,ilay) * lay_source_bnd(gpt_bands(igpt),ilay)
        ! compute level source irradiance for each g-point, one each for upward and downward paths
        lev_source_dec(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(gpt_bands(igpt),ilay)
        lev_source_inc(igpt,ilay)   = planck_frac(igpt,ilay) * lev_source_bnd(gpt_bands(igpt),ilay+1)
      end do
    end do

  end subroutine lw_gpt_source_nocol
  ! ---------------------------------------------------------------

  ! -------------------------------------------------------------------------------------------------
  !
  ! Planck sources by g-point from plank fraction and sources by band
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_gpt_source_Jac(nbnd, ngpt, nlay, ncol, sfc_lay, gpt_bands, planck_frac, &
    lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_source_bnd_Jac, &     ! inputs: band source functions
    sfc_source, sfc_source_Jac,  lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

    integer,                                intent(in )   :: nbnd, ngpt, nlay, ncol, sfc_lay
    integer,  dimension(ngpt),              intent(in)    :: gpt_bands ! band number (1...16) for each g-point
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in )   :: planck_frac 
    real(wp), dimension(nbnd,nlay,  ncol),  intent(in )   :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),  intent(in )   :: lev_source_bnd
    real(wp), dimension(nbnd,       ncol),  intent(in )   :: sfc_source_bnd      ! Surface source by band
    real(wp), dimension(nbnd,       ncol),  intent(in )   :: sfc_source_bnd_Jac  ! Surface source by band using perturbed temperature
    ! outputs
    real(wp), dimension(ngpt,       ncol),  intent(out)   :: sfc_source, sfc_source_Jac ! Surface source by g-point and its Jacobian
    real(wp), dimension(ngpt,nlay,  ncol),  intent(out)   :: lay_source
    real(wp), dimension(ngpt,nlay,  ncol),  intent(out)   :: lev_source_dec, lev_source_inc

    integer             ::  ilay, icol, igpt, ibnd, gptS, gptE

  
    !$acc data present(gpt_bands,planck_frac,lay_source_bnd,lev_source_bnd,sfc_source_bnd,sfc_source_bnd_Jac)

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        sfc_source(igpt, icol)     = planck_frac(igpt,sfc_lay, icol) * sfc_source_bnd(gpt_bands(igpt), icol)
        sfc_source_Jac(igpt, icol) = planck_frac(igpt,sfc_lay, icol) * (sfc_source_bnd(gpt_bands(igpt), icol) - sfc_source_bnd_Jac(gpt_bands(igpt), icol))
      end do
    end do

    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1,ngpt 
          ! compute layer source irradiance for each g-point
          lay_source(igpt,ilay, icol)       = planck_frac(igpt,ilay, icol) * lay_source_bnd(gpt_bands(igpt),ilay, icol)
          ! compute level source irradiance for each g-point, one each for upward and downward paths
          lev_source_dec(igpt,ilay, icol)   = planck_frac(igpt,ilay, icol) * lev_source_bnd(gpt_bands(igpt),ilay, icol)
          lev_source_inc(igpt,ilay, icol)   = planck_frac(igpt,ilay, icol) * lev_source_bnd(gpt_bands(igpt),ilay+1, icol)
        end do
      end do
    end do
    
    !$acc end data

  end subroutine lw_gpt_source_Jac

 ! -------------------------------------------------------------------------------------------------
  !
  ! Planck sources by g-point from plank fraction and sources by band
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine lw_gpt_source(nbnd, ngpt, nlay, ncol, sfc_lay, gpt_bands, planck_frac, &
    lay_source_bnd, lev_source_bnd, sfc_source_bnd, &     ! inputs: band source functions
    sfc_source,  lay_source, lev_source_dec, lev_source_inc)  ! outputs: g-point source functions

    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol, sfc_lay
    integer,  dimension(ngpt),             intent(in)    :: gpt_bands ! band number (1...16) for each g-point
    real(wp), dimension(ngpt,nlay,ncol),   intent(in   ) :: planck_frac 
    real(wp), dimension(nbnd,nlay,ncol),   intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in )   :: lev_source_bnd
    real(wp), dimension(nbnd,ncol),        intent(in )   :: sfc_source_bnd      ! Surface source by band
    ! outputs
    real(wp), dimension(ngpt,ncol     ),   intent(out)   :: sfc_source ! Surface source by g-point and its Jacobian
    real(wp), dimension(ngpt,nlay,ncol),   intent(out)   :: lay_source
    real(wp), dimension(ngpt,nlay,ncol),   intent(out)   :: lev_source_dec, lev_source_inc

    integer             ::  ilay, icol, igpt, ibnd, gptS, gptE

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        sfc_source(igpt, icol)     = planck_frac(igpt,sfc_lay, icol) * sfc_source_bnd(gpt_bands(igpt), icol)
      end do
    end do

    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1,ngpt 
          ! compute layer source irradiance for each g-point
          lay_source(igpt,ilay, icol)       = planck_frac(igpt,ilay, icol) * lay_source_bnd(gpt_bands(igpt),ilay, icol)
          ! compute level source irradiance for each g-point, one each for upward and downward paths
          lev_source_dec(igpt,ilay, icol)   = planck_frac(igpt,ilay, icol) * lev_source_bnd(gpt_bands(igpt),ilay, icol)
          lev_source_inc(igpt,ilay, icol)   = planck_frac(igpt,ilay, icol) * lev_source_bnd(gpt_bands(igpt),ilay+1, icol)
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

    !$acc loop vector
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

    !$acc kernels
    flux_dn(:,  top_level, :)  = inc_flux(:,:)
    !$acc end kernels
  end subroutine apply_BC
  ! ---------------------
  pure subroutine apply_BC_old(ngpt, nlay, ncol, top_at_1, inc_flux, flux_dn) bind (C, name="apply_BC_old")
  integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
  logical(wl),                           intent( in) :: top_at_1
  real(wp), dimension(ngpt, ncol      ), intent( in) :: inc_flux         ! Flux at top of domain
  real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below

  !   Upper boundary condition
  !$acc kernels
  if(top_at_1) then
    flux_dn(:,  1, :)  = inc_flux(:,:)
  else
    flux_dn(:,  nlay+1, :)  = inc_flux(:,:) 
  end if 
  !$acc end kernels
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

    !$acc kernels

    if(top_at_1) then
      do igpt = 1, ngpt
          flux_dn(igpt, 1, :)  = inc_flux(igpt,:) * factor
      end do
    else
      do igpt = 1, ngpt
          flux_dn(igpt, nlay+1, 1:ncol)  = inc_flux(igpt,:) * factor
      end do
    end if
    !$acc end kernels

  end subroutine apply_BC_factor
  ! --------------------- 
  pure subroutine apply_BC_0(ngpt, nlay, ncol, top_at_1, flux_dn) bind (C, name="apply_BC_0")
    integer,                               intent( in) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent( in) :: top_at_1
    real(wp), dimension(ngpt,nlay+1,ncol), intent(out) :: flux_dn          ! Flux to be used as input to solvers below

    !$acc kernels

    if(top_at_1) then
      flux_dn(1:ngpt,      1, 1:ncol)  = 0._wp
    else
      flux_dn(1:ngpt, nlay+1, 1:ncol)  = 0._wp
    end if
    !$acc end kernels

  end subroutine apply_BC_0
! -------------------------------------------------------------------------------------------------
!
! Similar to Longwave no-scattering (lw_solver_noscat)
!   a) relies on rescaling of the optical parameters based on asymetry factor and single scattering albedo
!       scaling can be computed  by scaling_1rescl
!   b) adds adustment term based on cloud properties (lw_transport_1rescl)
!      adustment terms is computed based on solution of the Tang equations
!      for "linear-in-tau" internal source (not in the paper)
!
!   Attention:
!      use must prceompute scaling before colling the function
!
!   Implemented based on the paper
!   Tang G, et al, 2018: https://doi.org/10.1175/JAS-D-18-0014.1
!
! -------------------------------------------------------------------------------------------------
  subroutine lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, D , band_limits, &
                            tau, scaling, planck_frac, &
                            lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
                            radn_up, radn_dn, &
                            sfc_source_bnd_Jac, radn_up_Jac, radn_dn_Jac) bind(C, name="lw_solver_1rescl")
    integer,                               intent(in   ) :: nbnd, ngpt, nlay, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: D            ! secant of propagation angle  []
    integer,  dimension(2,nbnd),           intent(in   ) :: band_limits
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: scaling
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: planck_frac  ! Planck fractions (fraction of band source function associated with each g-point)
    real(wp), dimension(nbnd,nlay,  ncol), intent(in   ) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol), intent(in   ) :: lev_source_bnd      ! Planck source at layers and levels by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd      ! Surface source function by band [W/m2]
    real(wp), dimension(nbnd,       ncol), intent(in   ) :: sfc_source_bnd_Jac  ! Jacobian of surface source function by band[W/m2]
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_emis            ! Surface emissivity      []
    ! Outputs
    real(wp), dimension(ngpt,nlay+1,ncol), intent(  out) :: radn_up      ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_dn      ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay+1,ncol), intent(  out) :: radn_up_Jac  ! Surface Temperature Jacobians [W/m2-str/K]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(  out) :: radn_dn_Jac  ! Top level set to 0
    ! Local variables, WITH g-point dependency
    real(wp), dimension(ngpt,nlay,ncol) :: tau_loc, &  ! path length (tau/mu)
                                             trans       ! transmissivity  = exp(-tau)
    real(wp), dimension(ngpt,nlay,ncol) :: source_dn, source_up
    real(wp), dimension(ngpt,nlay,ncol) :: lay_source      
    real(wp), dimension(ngpt,     ncol) :: source_sfc, sfc_albedo
    real(wp), dimension(ngpt,     ncol) :: source_sfcJac
    real(wp), dimension(ngpt,     ncol) :: sfc_src      ! Surface source function [W/m2]
    real(wp), dimension(ngpt,     ncol) :: sfc_src_Jac   ! Surface Temperature Jacobian source function [W/m2/K]

    real(wp), dimension(:,:,:),         contiguous, pointer :: lev_source_up, lev_source_dn ! Mapping increasing/decreasing indicies to up/down
    real(wp), dimension(ngpt,nlay,ncol),   target           :: lev_source_dec, lev_source_inc
    integer,  dimension(ngpt)               :: gpt_bands ! band number (1...16) for each g-point
    real(wp), parameter :: pi = acos(-1._wp)
    real(wp), parameter :: tau_thresh = sqrt(epsilon(tau))
    integer             :: ilev, icol, igpt, ilay, sfc_lay, top_level, ibnd
    real(wp), dimension(ngpt,nlay,ncol) :: An, Cn
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

    !$acc enter data create(tau_loc,trans, source_dn, source_up, lay_source, source_sfc, source_sfcJac, sfc_src, sfc_src_Jac, sfc_albedo, lev_source_dec, lev_source_inc)

    !$acc enter data create(An, Cn)
    !$acc enter data attach(lev_source_up,lev_source_dn)

    !$acc parallel loop
    do ibnd = 1, nbnd
      do igpt = band_limits(1,ibnd), band_limits(2,ibnd)
        gpt_bands(igpt) = ibnd
      end do
    end do

    call lw_gpt_source_Jac(nbnd, ngpt, nlay, ncol, sfc_lay, gpt_bands, &
              planck_frac, lay_source_bnd, lev_source_bnd, &
              sfc_source_bnd, sfc_source_bnd_Jac, &
              sfc_src, sfc_src_Jac, lay_source, lev_source_dec, lev_source_inc)


    ! NOTE: This kernel produces small differences between GPU and CPU
    ! implementations on Ascent with PGI, we assume due to floating point
    ! differences in the exp() function. These differences are small in the
    ! RFMIP test case (10^-6).
    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilev = 1, nlay
        do igpt = 1, ngpt
          !
          ! Optical path and transmission, used in source function and transport calculations
          !
          tau_loc(igpt,ilev,icol) = tau(igpt,ilev,icol)*D(igpt,icol)
          trans  (igpt,ilev,icol) = exp(-tau_loc(igpt,ilev,icol))
          ! here scaling is used to store parameter wb/[(]1-w(1-b)] of Eq.21 of the Tang's paper
          ! explanation of factor 0.4 note A of Table
          Cn(igpt,ilev,icol) = 0.4_wp*scaling(igpt,ilev,icol)
          An(igpt,ilev,icol) = (1._wp-trans(igpt,ilev,icol)*trans(igpt,ilev,icol))

          ! initialize radn_dn_Jac
          radn_dn_Jac(igpt,ilev,icol) = 0._wp
        end do
      end do
    end do

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
      !
      ! Surface albedo, surface source function
      !
        sfc_albedo   (igpt,icol) = 1._wp - sfc_emis(igpt,icol)
        source_sfc   (igpt,icol) = sfc_emis(igpt,icol) * sfc_src   (igpt,icol)
        source_sfcJac(igpt,icol) = sfc_emis(igpt,icol) * sfc_src_Jac(igpt,icol)
      end do
    end do

    !
    ! Source function for diffuse radiation
    !
    call lw_source_noscat(ngpt, nlay, ncol, &
                          lay_source, lev_source_up, lev_source_dn, &
                          tau_loc, trans, source_dn, source_up)

    !
    ! Transport
    !
    !  compute no-scattering fluxes
    call lw_transport_noscat(ngpt, nlay, ncol, top_at_1,  &
                             tau_loc, trans, sfc_albedo, source_dn, source_up, source_sfc, &
                             radn_up, radn_dn,&
                             source_sfcJac, radn_up_Jac)
    !  make adjustment
    call lw_transport_1rescl(ngpt, nlay, ncol, top_at_1,  &
                             tau_loc, trans, &
                             sfc_albedo, source_dn, source_up, &
                             radn_up, radn_dn, An, Cn, radn_up_Jac, radn_dn_Jac)

    !$acc exit data copyout(radn_dn,radn_up)


    !$acc exit data detach(lev_source_up,lev_source_dn)

    !$acc exit data delete(An, Cn)

    !$acc exit data delete(tau_loc,trans, source_dn, source_up, lay_source, source_sfc, source_sfcJac, sfc_src, sfc_src_Jac, sfc_albedo, lev_source_dec, lev_source_inc)
                          

  end subroutine lw_solver_1rescl
! -------------------------------------------------------------------------------------------------
!
!  Similar to lw_solver_noscat_GaussQuad.
!    It is main solver to use the Tang approximation for fluxes
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
    integer,  dimension(2,nbnd),            intent(in   ) :: band_limits
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

    integer :: imu, top_level, igpt, ilev, icol
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
    ! store TOA flux
    fluxTOA = flux_dn(1:ngpt, top_level, 1:ncol)

    Ds_ngpt(:,:) = Ds(1)
    weight = 2._wp*pi*weights(1)
    ! Transport is for intensity
    !   convert flux at top of domain to intensity assuming azimuthal isotropy
    !
    radn_dn(1:ngpt, top_level, 1:ncol) = fluxTOA(1:ngpt, 1:ncol) / weight
    call lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, Ds_ngpt, band_limits, &
                            tauLoc, scaling, planck_frac, &
                            lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
                            flux_up, flux_dn, &
                            sfc_source_bnd_Jac, flux_up_Jac, flux_dn_Jac)

    !$acc  parallel loop collapse(3)
    do icol = 1, ncol
      do ilev = 1, nlay+1
        do igpt = 1, ngpt
          flux_up    (igpt,ilev,icol) = weight*flux_up    (igpt,ilev,icol)
          flux_dn    (igpt,ilev,icol) = weight*flux_dn    (igpt,ilev,icol)
          flux_up_Jac(igpt,ilev,icol) = weight*flux_up_Jac(igpt,ilev,icol)
          flux_dn_Jac(igpt,ilev,icol) = weight*flux_dn_Jac(igpt,ilev,icol)
        enddo
      enddo
    enddo

    do imu = 2, nmus
      Ds_ngpt(:,:) = Ds(imu)
      weight = 2._wp*pi*weights(imu)
      radn_dn(1:ngpt, top_level, 1:ncol)  = fluxTOA(1:ngpt, 1:ncol) / weight
      call lw_solver_1rescl(nbnd, ngpt, nlay, ncol, top_at_1, Ds_ngpt, band_limits, &
              tauLoc, scaling, planck_frac, &
              lay_source_bnd, lev_source_bnd, sfc_source_bnd, sfc_emis, &
              flux_up, flux_dn, &
              sfc_source_bnd_Jac, flux_up_Jac, flux_dn_Jac)
      !$acc  parallel loop collapse(3)
      do icol = 1, ncol
        do ilev = 1, nlay+1
          do igpt = 1, ngpt
            flux_up    (igpt,ilev,icol) = flux_up    (igpt,ilev,icol) + weight*radn_up    (igpt,ilev,icol)
            flux_dn    (igpt,ilev,icol) = flux_dn    (igpt,ilev,icol) + weight*radn_dn    (igpt,ilev,icol)
            flux_up_Jac(igpt,ilev,icol) = flux_up_Jac(igpt,ilev,icol) + weight*radn_up_Jac(igpt,ilev,icol)
            flux_dn_Jac(igpt,ilev,icol) = flux_dn_Jac(igpt,ilev,icol) + weight*radn_dn_Jac(igpt,ilev,icol)
          enddo
        enddo
      enddo

    end do
   !$acc exit data copyout(flux_up_Jac,flux_dn_Jac)
   !$acc exit data copyout(flux_up,flux_dn)

  end subroutine lw_solver_1rescl_GaussQuad
! -------------------------------------------------------------------------------------------------
!
!  Computes Tang scaling of layer optical thickness and scaling parameter
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
    !$acc enter data copyin(tau, ssa, g)
    !$acc enter data create(tauLoc, scaling)
    !$acc parallel loop collapse(3)
    do icol=1,ncol
      do ilay=1,nlay
        do igpt=1,ngpt
          ssal = ssa(igpt, ilay, icol)
          wb = ssal*(1._wp - g(igpt, ilay, icol)) / 2._wp
          scaleTau = (1._wp - ssal + wb )

          tauLoc(igpt, ilay, icol) = scaleTau * tau(igpt, ilay, icol) ! Eq.15 of the paper
          !
          ! here scaling is used to store parameter wb/[1-w(1-b)] of Eq.21 of the Tang's paper
          ! actually it is in line of parameter rescaling defined in Eq.7
          ! potentialy if g=ssa=1  then  wb/scaleTau = NaN
          ! it should not happen
          scaling(igpt, ilay, icol) = wb / scaleTau
        enddo
      enddo
    enddo
    !$acc exit data copyout(tauLoc, scaling)
    !$acc exit data delete(tau, ssa, g)
  end subroutine scaling_1rescl
! -------------------------------------------------------------------------------------------------
!
!  Computes Tang scaling of layer optical thickness and scaling parameter
!  Safe implementation
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
    !$acc enter data copyin(tau, ssa, g)
    !$acc enter data create(tauLoc, scaling)
    !$acc parallel loop collapse(3)
    do icol=1,ncol
      do ilay=1,nlay
        do igpt=1,ngpt
          ssal = ssa(igpt, ilay, icol)
          wb = ssal*(1._wp - g(igpt, ilay, icol)) / 2._wp
          scaleTau = (1._wp - ssal + wb )

          tauLoc(igpt, ilay, icol) = scaleTau * tau(igpt, ilay, icol) ! Eq.15 of the paper
          !
          ! here scaling is used to store parameter wb/[1-w(1-b)] of Eq.21 of the Tang's paper
          ! actually it is in line of parameter rescaling defined in Eq.7
          if (scaleTau < 1e-6_wp) then
              scaling(igpt, ilay, icol) = 1.0_wp
          else
              scaling(igpt, ilay, icol) = wb / scaleTau
          endif
        enddo
      enddo
    enddo
    !$acc exit data copyout(tauLoc, scaling)
    !$acc exit data delete(tau, ssa, g)
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
  subroutine lw_transport_1rescl(ngpt, nlay, ncol, top_at_1, &
                                 tau, trans, sfc_albedo, source_dn, source_up, &
                                 radn_up, radn_dn, An, Cn,&
                                 radn_up_Jac, radn_dn_Jac) bind(C, name="lw_transport_1rescl")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: tau, &     ! Absorption optical thickness, pre-divided by mu []
                                                       trans      ! transmissivity = exp(-tau)
    real(wp), dimension(ngpt       ,ncol), intent(in   ) :: sfc_albedo ! Surface albedo
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: source_dn, &
                                                            source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: An, Cn
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_up_Jac ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_dn_Jac !Top level must contain incident flux boundary condition
    ! Local variables
    integer :: ilev, igpt, icol
    ! ---------------------------------------------------
    real(wp) :: adjustmentFactor
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      ! Downward propagation
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! 1st Upward propagation
          do ilev = nlay, 1, -1
            radn_up   (igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up   (igpt,ilev+1,icol) + source_up(igpt,ilev,icol)
            radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev+1,icol)

            adjustmentFactor = Cn(igpt,ilev,icol)*&
                   ( An(igpt,ilev,icol)*radn_dn(igpt,ilev,icol) - &
                     source_dn(igpt,ilev,icol)  *trans(igpt,ilev,icol ) - &
                     source_up(igpt,ilev,icol))
            radn_up(igpt,ilev,icol) = radn_up(igpt,ilev,icol) + adjustmentFactor
          enddo
          ! 2nd Downward propagation
          do ilev = 1, nlay
            radn_dn   (igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_dn   (igpt,ilev,icol) + source_dn(igpt,ilev,icol)
            radn_dn_Jac(igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_dn_Jac(igpt,ilev,icol)
            adjustmentFactor = Cn(igpt,ilev,icol)*( &
                An(igpt,ilev,icol)*radn_up(igpt,ilev,icol) - &
                source_up(igpt,ilev,icol)*trans(igpt,ilev,icol) - &
                source_dn(igpt,ilev,icol) )
            radn_dn(igpt,ilev+1,icol)    = radn_dn(igpt,ilev+1,icol) + adjustmentFactor

            adjustmentFactor             = Cn(igpt,ilev,icol)*An(igpt,ilev,icol)*radn_up_Jac(igpt,ilev,icol)
            radn_dn_Jac(igpt,ilev+1,icol) = radn_dn_Jac(igpt,ilev+1,icol) + adjustmentFactor
          enddo
        enddo
      enddo
    else
      !$acc  parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Upward propagation
          do ilev = 1, nlay
            radn_up   (igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_up   (igpt,ilev,icol) +  source_up(igpt,ilev,icol)
            radn_up_Jac(igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev,icol)
            adjustmentFactor = Cn(igpt,ilev,icol)*&
                   ( An(igpt,ilev,icol)*radn_dn(igpt,ilev+1,icol) - &
                     source_dn(igpt,ilev,icol) *trans(igpt,ilev ,icol) - &
                     source_up(igpt,ilev,icol))
            radn_up(igpt,ilev+1,icol) = radn_up(igpt,ilev+1,icol) + adjustmentFactor
          end do
          ! 2st Downward propagation
          do ilev = nlay, 1, -1
            radn_dn   (igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_dn   (igpt,ilev+1,icol) + source_dn(igpt,ilev,icol)
            radn_dn_Jac(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_dn_Jac(igpt,ilev+1,icol)
            adjustmentFactor = Cn(igpt,ilev,icol)*( &
                    An(igpt,ilev,icol)*radn_up(igpt,ilev,icol) - &
                    source_up(igpt,ilev,icol)*trans(igpt,ilev ,icol ) - &
                    source_dn(igpt,ilev,icol) )
            radn_dn(igpt,ilev,icol)    = radn_dn(igpt,ilev,icol) + adjustmentFactor

            adjustmentFactor           = Cn(igpt,ilev,icol)*An(igpt,ilev,icol)*radn_up_Jac(igpt,ilev,icol)
            radn_dn_Jac(igpt,ilev,icol) = radn_dn_Jac(igpt,ilev,icol) + adjustmentFactor
          end do
        enddo
      enddo
    end if
  end subroutine lw_transport_1rescl
end module mo_rte_solver_kernels
