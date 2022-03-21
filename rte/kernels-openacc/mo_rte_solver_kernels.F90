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
  use mo_fluxes_broadband_kernels, only : sum_broadband
  use mo_rte_kind, only: wp, wl
  use mo_rte_rrtmgp_config, only: compute_Jac, use_Pade_source
  implicit none
  private

  interface apply_BC
    module procedure apply_BC, apply_BC_old, apply_BC_nocol, apply_BC_factor, apply_BC_0
  end interface apply_BC

  public :: apply_BC, &
            lw_solver_noscat, lw_solver_noscat_GaussQuad, lw_solver_2stream,  &
            sw_solver_noscat,                             sw_solver_2stream

  ! These routines don't really need to be visible but making them so is useful for testing.
  public :: lw_source_noscat, lw_combine_sources, &
            lw_source_2str, &
            lw_two_stream, sw_two_stream, &
            adding

  real(wp), parameter :: pi = acos(-1._wp)


! #ifdef NGPT 
! integer, parameter :: ngpt = NGPT
! #else
! #define ngpt ngpt_in
! #endif

#ifdef NGPT_SW 
integer, parameter :: ngpt_sw = NGPT_SW
#else
#define ngpt_sw ngpt_sw_in
#endif

#ifdef NGPT_LW 
integer, parameter :: ngpt_lw = NGPT_LW
#else
#define ngpt_lw ngpt_lw_in
#endif


#ifdef NLAY 
integer, parameter :: nlay = NLAY
#else
#define nlay nlay_in
#endif

#ifdef DOUBLE_PRECISION
  real(wp), parameter :: k_min = 1.e-12_wp
#else 
  real(wp), parameter :: k_min = 1.e-4_wp 
#endif

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
  subroutine lw_solver_noscat(ngpt_lw_in, nlay_in, ncol, top_at_1, nmus, D, weight, inc_flux, &
                              tau, lay_source, lev_source, &
                              sfc_emis, sfc_source, &
                              flux_up, flux_dn, &
                              sfc_source_Jac, flux_up_Jac, &
                              do_rescaling, ssa, g, &
                              save_gpt_flux, radn_up, radn_dn, radn_up_Jac) bind(C, name="lw_solver_noscat")
    integer,                                intent(in   ) ::  ngpt_lw_in, nlay_in, ncol ! Number of bands, g-points, layers, columns
    logical(wl),                            intent(in   ) ::  top_at_1
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(ngpt_lw,       ncol),  intent(in   ) ::  D            ! secant of propagation angle  []
    real(wp),                               intent(in   ) ::  weight       ! quadrature weight
    real(wp), dimension(ngpt_lw,ncol),         intent(in   ) ::  inc_flux        ! incident flux at domain top [W/m2] (ngpts, ncol)
    real(wp), dimension(ngpt_lw,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt_lw,nlay,  ncol),  intent(in   ) ::  lay_source      ! Planck source at layer average temperature [W/m2]
    real(wp), dimension(ngpt_lw,nlay+1,ncol),  intent(in   ) ::  lev_source      ! Planck source at layer edges [W/m2]
    real(wp), dimension(ngpt_lw,       ncol),  intent(in   ) ::  sfc_emis        ! Surface emissivity      []
    real(wp), dimension(ngpt_lw,       ncol),  intent(in   ) ::  sfc_source      ! Surface source function  [W/m2]
    ! Outputs
    real(wp), dimension(ngpt_lw,nlay+1,ncol),  intent(out)   :: radn_up, radn_dn    ! Radiances per g-point [W/m2-str]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   :: flux_up      ! Broadband fluxes [W/m2]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   :: flux_dn      ! Top level must contain incident flux boundary condition
    !
    ! Optional variables - arrays aren't referenced if corresponding logical  == False
    !
    real(wp), dimension(:,:,:),             intent(out)   :: radn_up_Jac ! surface temperature Jacobian of radiance [W/m2-str / K] 
    real(wp), dimension(:,:),               intent(out)   :: flux_up_Jac 
    real(wp), dimension(:,:),               intent(in   ) :: sfc_source_Jac  ! Jacobian of surface source function  [W/m2/K] (ngpt,ncol)
    logical(wl),                            intent(in   ) :: do_rescaling
    real(wp), dimension(:,:,:),             intent(in   ) :: ssa, g    ! single-scattering albedo, asymmetry parameter] (ngpt,nlay,ncol)
    logical(wl),                            intent(in   ) :: save_gpt_flux
    ! ------------------------------------
    ! Local variables
    real(wp), dimension(ngpt_lw,nlay,ncol)   :: tau_loc, &  ! path length (tau/mu)
                                             trans       ! transmissivity  = exp(-tau)
    real(wp), dimension(ngpt_lw,nlay, ncol)  :: source_dn!, source_up
    ! real(wp), dimension(ngpt_lw,      ncol)  :: source_sfc, source_sfcJac, sfc_albedo
    real(wp), parameter :: pi = acos(-1._wp)
    real(wp)            :: fac, bb_flux_up, bb_flux_dn
    integer             :: ilay, ilev, icol, igpt, sfc_level, top_level
    ! Used when approximating scattering
    real(wp), dimension(:,:,:), allocatable :: An, Cn
    ! real(wp) :: wb, ssal, scaleTau

    ! ------------------------------------
    ! Where it the top of atmosphere
    if(top_at_1) then
      top_level = 1
      sfc_level = nlay+1
    else
      top_level = nlay+1
      sfc_level = 1
    end if

    if (do_rescaling) then
      allocate(An(ngpt_lw,nlay,ncol), Cn(ngpt_lw,nlay,ncol))
      !$acc enter data create(An, Cn)
    end if

    fac  = 2._wp * pi * weight

    ! Combined source and transport 
    associate(source_up=>tau_loc)
    !$acc enter data create (tau_loc, trans) 
    !$acc parallel default(present)
    !$acc loop collapse(2) 
    do icol = 1, ncol
      do igpt = 1, ngpt_lw
        !
        ! Transport is for intensity
        !   convert flux at top of domain to intensity assuming azimuthal isotropy
        !
        ! radn_dn(igpt,top_level,icol) = radn_dn(igpt,top_level,icol)/(2._wp * pi * weight)
        radn_dn(igpt,top_level,icol) = inc_flux(igpt,icol) / fac
      end do
    end do

    !$acc loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt_lw
          if(do_rescaling) then
            call scaling_1rescl(ngpt_lw, nlay, ncol, igpt, ilay, icol, &
                                D, tau, ssa, g, trans, tau_loc, Cn, An)
            ! ssal = ssa(igpt, ilay, icol)
            ! wb = ssal*(1._wp - g(igpt, ilay, icol)) * 0.5_wp
            ! scaleTau = (1._wp - ssal + wb )
            ! ! here wb/scaleTau is parameter wb/(1-w(1-b)) of Eq.21 of the Tang paper
            ! ! actually it is in line of parameter rescaling defined in Eq.7
            ! ! potentialy if g=ssa=1  then  wb/scaleTau = NaN
            ! ! it should not happen because g is never 1 in atmospheres
            ! ! explanation of factor 0.4 note A of Table
            ! Cn(igpt,ilay,icol) = 0.4_wp*wb/scaleTau
            ! ! Eq.15 of the paper, multiplied by path length
            ! tau_loc(igpt,ilay,icol) = tau(igpt,ilay,icol)*D(igpt,icol)*scaleTau
            ! trans  (igpt,ilay,icol) = exp(-tau_loc(igpt,ilay,icol))
            ! An     (igpt,ilay,icol) = (1._wp-trans(igpt,ilay,icol)**2)
        
          else
            !
            ! Optical path and transmission, used in source function and transport calculations
            !
            tau_loc(igpt,ilay,icol) = tau(igpt,ilay,icol)*D(igpt,icol)
            trans  (igpt,ilay,icol) = exp(-tau_loc(igpt,ilay,icol))
          end if
        end do
      end do
    end do
    !$acc end parallel 

    !
    ! Transport down, combined with source computation
    !   
    if(do_rescaling) then
      !$acc enter data create (source_dn)
      call lw_sources_transport_noscat_dn(ngpt_lw, nlay, ncol, top_at_1, &
                                    lay_source, lev_source, tau_loc, trans, &
                                    source_dn, radn_dn) 
    else
      call lw_source_transport_noscat_dn(ngpt_lw, nlay, ncol, top_at_1, &
                                    lay_source, lev_source, tau_loc, trans, &
                                    radn_dn) 
    end if

    !
    ! Surface reflection and emission
    !
    !$acc parallel loop collapse(2) default(present)
    do icol = 1, ncol
      do igpt = 1, ngpt_lw
      ! Surface reflection and emission                                          
        radn_up (igpt,sfc_level,icol)  = radn_dn(igpt,sfc_level,icol) *  &
          ! albedo
          (1-sfc_emis(igpt,icol)) + (sfc_emis(igpt,icol) * sfc_source(igpt,icol))
        if (compute_Jac) radn_up_Jac(igpt,sfc_level,icol)  =  sfc_emis(igpt,icol) * sfc_source_Jac(igpt,icol)
        end do
    end do

    !
    ! Transport up, or up and down again if using rescaling
    !
    if(do_rescaling) then
      call lw_transport_1rescl(ngpt_lw, nlay, ncol, top_at_1, trans, &
                               source_dn, source_up,              &
                               radn_up, radn_dn, An, Cn,          &
                               radn_up_Jac) 
      !$acc exit data delete(source_dn)       
    else
      call lw_transport_noscat_up(ngpt_lw, nlay, ncol, top_at_1, &
                                  trans, source_up, radn_up, radn_up_Jac) 
    end if
    !$acc exit data delete(tau_loc, trans)
    end associate

    !
    ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
    !

    ! !$acc parallel loop collapse(3) default(present)
    ! do icol = 1, ncol
    !   do ilev = 1, nlay+1
    !     do igpt = 1, ngpt_lw
    !       radn_dn    (igpt,ilev,icol) = fac * radn_dn    (igpt,ilev,icol)
    !       radn_up    (igpt,ilev,icol) = fac * radn_up    (igpt,ilev,icol)
    !       if (compute_Jac) radn_up_Jac(igpt,ilev,icol) = fac * radn_up_Jac(igpt,ilev,icol)
    !     end do
    !   end do
    ! end do

    if (nmus==1) then
      !$acc data copyout(flux_up, flux_dn)
      !$acc parallel loop gang worker collapse(2) default(present) 
      do icol = 1, ncol
        do ilev = 1, nlay+1
          bb_flux_up = 0.0_wp
          bb_flux_dn = 0.0_wp
          !$acc loop vector reduction(+:bb_flux_up,bb_flux_dn)
          do igpt = 1, ngpt_lw
            bb_flux_up = bb_flux_up + radn_up(igpt, ilev, icol)
            bb_flux_dn = bb_flux_dn + radn_dn(igpt, ilev, icol)
          end do
          flux_up(ilev, icol) = fac*bb_flux_up
          flux_dn(ilev, icol) = fac*bb_flux_dn
        end do
      end do
      !$acc end data
      ! call sum_broadband_fac(ngpt_lw, nlay+1, ncol, fac, radn_up, flux_up)
      ! call sum_broadband_fac(ngpt_lw, nlay+1, ncol, fac, radn_dn, flux_dn)
      if (compute_Jac)  then
        call sum_broadband_fac(ngpt_lw, nlay+1, ncol, fac, radn_up_Jac, flux_up_Jac)
      end if
    else 
      !$acc parallel loop collapse(3) default(present)
      do icol = 1, ncol
        do ilev = 1, nlay+1
          do igpt = 1, ngpt_lw
            radn_dn    (igpt,ilev,icol) = fac * radn_dn    (igpt,ilev,icol)
            radn_up    (igpt,ilev,icol) = fac * radn_up    (igpt,ilev,icol)
            if (compute_Jac) radn_up_Jac(igpt,ilev,icol) = fac * radn_up_Jac(igpt,ilev,icol)
          end do
        end do
      end do

    end if

  end subroutine lw_solver_noscat
  ! ---------------------------------------------------------------
  !
  ! LW transport, no scattering, multi-angle quadrature
  !   Users provide a set of weights and quadrature angles
  !   Routine sums over single-angle solutions for each sets of angles/weights
  !
  ! ---------------------------------------------------------------
  
  subroutine lw_solver_noscat_GaussQuad(ngpt, nlay, ncol, top_at_1, nmus, Ds, weights, inc_flux, &
                                   tau, lay_source, lev_source, &
                                   sfc_emis, sfc_source, &
                                   flux_up, flux_dn, &
                                   sfc_source_Jac, flux_up_Jac, &
                                   do_rescaling, ssa, g, &
                                   save_gpt_flux, flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac) bind(C, name="lw_solver_noscat_GaussQuad")
    integer,                                intent(in   ) ::  ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                            intent(in   ) ::  top_at_1
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(nmus),              intent(in   ) ::  Ds, weights  ! quadrature secants, weights
    real(wp), dimension(ngpt,ncol),         intent(in   ) ::  inc_flux    ! incident flux at domain top [W/m2] (ngpts, ncol)
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau          ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  lay_source      ! Planck source at layer average temperature [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(in   ) ::  lev_source      ! Planck source at layer edges [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis        ! Surface emissivity      []
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_source      ! Surface source function by band [W/m2]
    ! Outputs
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_up      ! Broadband fluxes [W/m2]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_dn      ! Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(out  ) ::  flux_up_gpt, flux_dn_gpt
    !
    ! Optional variables - arrays aren't referenced if corresponding logical  == False
    !
    real(wp), dimension(:,:),               intent(out  ) :: flux_up_Jac 
    real(wp), dimension(:,:,:),             intent(out  ) :: flux_up_gpt_Jac 
    real(wp), dimension(:,:),               intent(in   ) :: sfc_source_Jac  ! Jacobian of surface source function  [W/m2/K] (ngpt,ncol)
    logical(wl),                            intent(in   ) :: do_rescaling
    real(wp), dimension(:,:,:),             intent(in   ) :: ssa, g    ! single-scattering albedo, asymmetry parameter] (ngpt,nlay,ncol)
    logical(wl),                            intent(in   ) :: save_gpt_flux
    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt
    real(wp), dimension(:,:,:),  allocatable      :: radn_up, radn_dn ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:,:),  allocatable      :: radn_up_Jac      ! perturbed Fluxes per quad angle
    integer :: imu, icol, sfc_lay, igpt, ilev

    !
    ! For the first angle output arrays store total flux
    !

    !$acc data copyin(Ds, weights) create (Ds_ngpt)
    !$acc  parallel loop collapse(2) default(present)
    do icol = 1, ncol
      do igpt = 1, ngpt
        Ds_ngpt(igpt, icol) = Ds(1)
      end do
    end do

    call lw_solver_noscat(ngpt, nlay, ncol, top_at_1, &
                          nmus, Ds_ngpt, weights(1), inc_flux, &
                          tau, lay_source, lev_source,&
                          sfc_emis, sfc_source,  &
                          flux_up, flux_dn,      &
                          sfc_source_Jac, flux_up_Jac, &
                          do_rescaling, ssa, g,&
                          save_gpt_flux, flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac)

    if (nmus > 1) then
      allocate( radn_up(ngpt, nlay+1, ncol) )
      allocate( radn_dn(ngpt, nlay+1, ncol) )
      if (compute_Jac) allocate( radn_up_Jac(ngpt, nlay+1, ncol) )
      !$acc enter data create(radn_up,radn_dn)
      !$acc enter data create(radn_up_Jac) if (compute_Jac)
      do imu = 2, nmus

        !$acc  parallel loop collapse(2) default(present)
        do icol = 1, ncol
          do igpt = 1, ngpt
            Ds_ngpt(igpt, icol) = Ds(imu)
          end do
        end do

        call lw_solver_noscat(ngpt, nlay, ncol, top_at_1, &
                              nmus, Ds_ngpt, weights(imu), inc_flux, &
                              tau, lay_source, lev_source, &
                              sfc_emis, sfc_source, &
                              flux_up, flux_dn, sfc_source_Jac, flux_up_Jac, &
                              do_rescaling, ssa, g,&
                              save_gpt_flux, radn_up, radn_dn, radn_up_Jac )

        !$acc parallel loop collapse(3) default(present)
        do icol = 1, ncol
          do ilev = 1, nlay+1
            do igpt = 1, ngpt
              flux_up_gpt(igpt,ilev,icol)     = flux_up_gpt(igpt,ilev,icol) + radn_up(igpt,ilev,icol) 
              flux_dn_gpt(igpt,ilev,icol)     = flux_dn_gpt(igpt,ilev,icol) + radn_dn(igpt,ilev,icol) 
              if (compute_Jac) flux_up_gpt_Jac(igpt,ilev,icol) = flux_up_gpt_Jac(igpt,ilev,icol) + radn_up_Jac(igpt,ilev,icol) 
            end do
          end do
        end do

      end do        

      call sum_broadband(ngpt, nlay+1, ncol, flux_up_gpt, flux_up)
      call sum_broadband(ngpt, nlay+1, ncol, flux_dn_gpt, flux_dn)
      if (compute_Jac)  call sum_broadband(ngpt, nlay+1, ncol, flux_up_gpt_Jac, flux_up_Jac)

      !$acc exit data delete(radn_up,radn_dn)
      !$acc exit data delete(radn_up_Jac) if(compute_Jac)

    end if 

    !$acc end data
  end subroutine lw_solver_noscat_GaussQuad


  ! -------------------------------------------------------------------------------------------------
  !
  ! Longwave two-stream calculation:
  !   combine RRTMGP-specific sources at levels
  !   compute layer reflectance, transmittance
  !   compute total source function at levels using linear-in-tau
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine lw_solver_2stream ( ngpt, nlay, ncol, top_at_1, inc_flux, &
                              tau, ssa, g, &
                              lay_source, lev_source, &
                              sfc_emis, sfc_source, &
                              flux_up, flux_dn, flux_up_gpt, flux_dn_gpt) bind(C, name="lw_solver_2stream")
    integer,                               intent(in   )  :: ngpt, nlay, ncol ! Number of g-points, layers, columns
    logical(wl),                           intent(in   )  :: top_at_1
    real(wp), dimension(ngpt,ncol),        intent(in   ) ::  inc_flux        ! incident flux at domain top [W/m2] (ngpts, ncol)
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   )  :: tau, & ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  lay_source  ! Planck source at layer average temperature [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(in   ) ::  lev_source  ! Planck source at layer edges [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis    ! Surface emissivity      []
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_source  ! Surface source function [W/m2]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_up      ! Broadband fluxes [W/m2-str]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_dn      ! 
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(out)   ::  flux_up_gpt, flux_dn_gpt ! G-point fluxes [W/m2]

    real(wp), dimension(ngpt,nlay ,ncol ) :: Rdif, Tdif, gamma1, gamma2
    real(wp), dimension(ngpt,ncol       ) :: sfc_albedo
     ! Planck source at layer edge for radiation in increasing/decreasing ilay direction [W/m2]
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

    !$acc enter data copyin(sfc_emis)
    !$acc enter data create(flux_up, flux_dn, Rdif, Tdif, gamma1, gamma2, sfc_albedo, source_dn, source_up, source_sfc)

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
                        sfc_emis, sfc_source, &
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
                flux_up_gpt, flux_dn_gpt)
    !$acc exit data delete(tau, ssa, g, sfc_emis)
    !$acc exit data delete(Rdif, Tdif, gamma1, gamma2, sfc_albedo, source_dn, source_up, source_sfc)
            
    call sum_broadband(ngpt, nlay+1, ncol, flux_up_gpt, flux_up)
    call sum_broadband(ngpt, nlay+1, ncol, flux_dn_gpt, flux_dn)

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
                              top_at_1, tau, mu0, flux_dir, flux_dir_bb) bind (C, name="sw_solver_noscat")
    integer,                    intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   )   :: tau          ! Absorption optical thickness []
    real(wp), dimension(ncol            ),  intent(in   )   :: mu0          ! cosine of solar zenith angle
    real(wp), dimension(     nlay+1,ncol),  intent(out)     :: flux_dir_bb  ! Direct-beam flux, broadband [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(inout)   :: flux_dir     ! Direct-beam flux, spectral [W/m2]
                                                                          ! Top level must contain incident flux boundary condition
    integer :: igpt, ilev, icol
    real(wp) :: mu0_inv(ncol)
    ! ------------------------------------
    ! ------------------------------------
    !$acc enter data copyin(tau, mu0) create(mu0_inv, flux_dir, flux_dir_bb)
    !$acc parallel loop
    do icol = 1, ncol
      mu0_inv(icol) = 1._wp/mu0(icol)
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
            flux_dir(igpt,ilev,icol) = flux_dir(igpt,ilev-1,icol) * exp(-tau(igpt,ilev,icol)*mu0_inv(icol))
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
            flux_dir(igpt,ilev,icol) = flux_dir(igpt,ilev+1,icol) * exp(-tau(igpt,ilev,icol)*mu0_inv(icol))
          end do
        end do
      end do
    end if

    call sum_broadband(ngpt, nlay+1, ncol, flux_dir, flux_dir_bb)

    !$acc exit data delete(tau, mu0, mu0_inv) copyout(flux_dir, flux_dir_bb)
    
  end subroutine sw_solver_noscat
  ! -------------------------------------------------------------------------------------------------
  !
  ! Shortwave two-stream calculation:
  !   compute layer reflectance, transmittance
  !   compute solar source function for diffuse radiation
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  
  subroutine sw_solver_2stream (ngpt_sw_in, nlay_in, ncol, top_at_1, &
                                inc_flux, inc_flux_dif,     &
                                tau, ssa, g, mu0,           &
                                sfc_alb_dir, sfc_alb_dif,   &
                                flux_up, flux_dn, flux_dir, &
                                radn_up, radn_dn, radn_dir) bind (C, name="sw_solver_2stream")
                                ! save_gpt_flux, radn_up, radn_dn, radn_dir) bind (C, name="sw_solver_2stream")

    integer,                               intent(in   ) :: ngpt_sw_in, nlay_in, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt_sw,       ncol), intent(in   ) :: inc_flux, inc_flux_dif   ! incident flux at top of domain [W/m2] (ngpt, ncol)
    real(wp), dimension(ngpt_sw,nlay,  ncol), intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(ncol            ), intent(in   ) :: mu0     ! cosine of solar zenith angle
    real(wp), dimension(ngpt_sw,       ncol), intent(in   ) :: sfc_alb_dir, sfc_alb_dif
                                                                  ! Spectral albedo of surface to direct and diffuse radiation
    real(wp), dimension(     nlay+1,ncol),  intent(out) :: flux_up, flux_dn, flux_dir ! Broadband fluxes  [W/m2]
    ! logical(wl),                            intent(in ) :: save_gpt_flux
    real(wp), dimension(ngpt_sw,nlay+1,ncol), optional, &
                                            intent(  out) :: radn_up, radn_dn, radn_dir
    ! -------------------------------------------
    integer :: igpt, ilay, icol, top_level
    real(wp), dimension(ngpt_sw,nlay,ncol) :: Rdif, Tdif!, Rdir, Tdir, Tnoscat
    real(wp), dimension(ngpt_sw,nlay,ncol) :: source_up, source_dn
    real(wp), dimension(ngpt_sw,     ncol) :: source_srf
    real(wp) :: bb_flux_dn, bb_flux_dir, bb_flux_up
    logical(wl) :: save_gpt_flux = .false.
    ! ------------------------------------

    ! Apply boundary condition
    if(top_at_1) then
      top_level = 1
    else
      top_level = nlay+1
    end if

    if (.not. present(radn_dir)) stop 'spectral fluxes need to be provided when using openACC'

    !$acc parallel loop collapse(2) default(present)
    do icol = 1, ncol
      do igpt = 1, ngpt_sw
        radn_dir(igpt,top_level, icol)  = inc_flux(igpt,icol) * mu0(icol)
        radn_dn(igpt, top_level, icol)  = inc_flux_dif(igpt,icol)
      end do
    end do

    !
    ! Cell properties: transmittance and reflectance for diffuse radiation
    ! Direct-beam radiation and source for diffuse radiation

    if(top_at_1) then
      call sw_dif_and_source_and_adding(ngpt_sw, nlay, ncol, top_at_1, mu0, sfc_alb_dif, &
                            tau, ssa, g,                                  &
                            radn_up, radn_dn, radn_dir)                     
    else 
      !$acc        data create(   Rdif, Tdif, source_up, source_dn, source_srf)
      !$omp target data map(alloc:Rdif, Tdif, source_up, source_dn, source_srf)
      call sw_dif_and_source(ngpt_sw, nlay, ncol, top_at_1, mu0, sfc_alb_dif, &
                            tau, ssa, g,                                  &
                            Rdif, Tdif, source_dn, source_up, source_srf, radn_dir)                     

      call adding(ngpt_sw, nlay, ncol, top_at_1,   &
                  sfc_alb_dif, Rdif, Tdif,      &
                  source_dn, source_up, source_srf, radn_up, radn_dn)
      !$acc        end data
      !$omp end target data
    end if


    ! Final loop to compute fluxes

    !$acc data copyout (flux_up, flux_dn, flux_dir)
      
    !$acc parallel default(present)
    !$acc loop gang
    do icol = 1, ncol
      !$acc loop worker
      do ilay = 1, nlay+1
        bb_flux_dn = 0.0_wp
        bb_flux_up = 0.0_wp
        bb_flux_dir = 0.0_wp
        !$acc loop vector  reduction(+:bb_flux_up,bb_flux_dn,bb_flux_dir)
        do igpt = 1, ngpt_sw
          ! adding computes only diffuse flux; flux_dn is total
          ! The addition is more efficient to do for broadband fluxes if only those are needed
          if (save_gpt_flux) radn_dn(igpt, ilay, icol) = radn_dn(igpt, ilay, icol) + radn_dir(igpt,ilay, icol)

          ! Compute broadband fluxes
          bb_flux_dir = bb_flux_dir + radn_dir(igpt, ilay, icol)
          bb_flux_dn = bb_flux_dn + radn_dn(igpt, ilay, icol)
          bb_flux_up = bb_flux_up + radn_up(igpt, ilay, icol)

        end do
        flux_dir(ilay,icol) = bb_flux_dir
        flux_dn(ilay,icol) = bb_flux_dn
        flux_up(ilay,icol) = bb_flux_up
         ! adding computes only diffuse flux; flux_dn is total
        if (.not. (save_gpt_flux)) flux_dn(ilay, icol) = flux_dn(ilay, icol) + flux_dir(ilay, icol)
      end do
    end do
    !$acc end parallel

    !$acc end data
    
  end subroutine sw_solver_2stream

  ! subroutine sw_solver_2stream(ngpt_sw_in, nlay_in, ncol, top_at_1, &
  !                                inc_flux, inc_flux_dif,     &
  !                                tau, ssa, g, mu0,           &
  !                                sfc_alb_dir, sfc_alb_dif,   &
  !                                flux_up, flux_dn, flux_dir, &
  !                               radn_up, radn_dn, radn_dir ) 
  !   integer,                               intent(in   ) :: ngpt_sw_in, nlay_in, ncol ! Number of columns, layers, g-points
  !   logical(wl),                           intent(in   ) :: top_at_1
  !   real(wp), dimension(ngpt_sw,       ncol), intent(in   ) :: inc_flux, inc_flux_dif     ! incident flux at top of domain [W/m2] (ngpt, ncol)
  !   real(wp), dimension(ngpt_sw,nlay,ncol), intent(in   ) :: tau, &  ! Optical thickness,
  !                                                           ssa, &  ! single-scattering albedo,
  !                                                           g       ! asymmetry parameter []
  !   real(wp), dimension(ncol            ), intent(in   ) :: mu0     ! cosine of solar zenith angle
  !   real(wp), dimension(ngpt_sw,     ncol), intent(in   ) :: sfc_alb_dir, sfc_alb_dif
  !                                                                   ! Spectral albedo of surface to direct and diffuse radiation
  !                                                           ! Broadband fluxes  [W/m2]
  !   real(wp), dimension(nlay+1,ncol),       intent(inout) :: flux_up, flux_dn, flux_dir
  !   ! logical(wl),                            intent(in ) :: save_gpt_flux
  !   real(wp), dimension(ngpt_sw, nlay+1, ncol),intent(out), optional :: radn_up, radn_dn, radn_dir   ! G-point fluxes
  !   ! -------------------------------------------
  !   real(wp)  :: bb_flux_up, bb_flux_dn, bb_flux_dir, Rdir, Tdir, Tnoscat
  !   integer :: icol, igpt, ilay, top_level, ilay2
  !   real(wp), dimension(nlay)   :: Rdif, Tdif
  !   real(wp), dimension(nlay+1)    :: albedo!, &  ! reflectivity to diffuse radiation below this level
  !                                             ! alpha in SH08
  !   real(wp), dimension(nlay)    :: denom      ! beta in SH08
  !   real(wp), dimension(nlay+1)    :: source ! source of diffuse upwelling radiation from emission or
  !                                         ! scattering of direct beam. G in SH08
  !   real(wp), dimension(nlay)     :: source_dn

  !   ! Temporary variables for sw_two_stream_nocol
  !   real(wp) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
  !   real(wp) :: RT_term, exp_minusktau, exp_minus2ktau, k_mu

  !   ! ------------------------------------

  !   ! Where it the top of atmosphere: at index 1 if top_at_1 true, otherwise nlay+1
  !   top_level = MERGE(1, nlay+1, top_at_1)

  !   !$acc enter data create(flux_up, flux_dn, flux_dir)

  !   !$acc kernels
  !   flux_up = 0.0_wp
  !   flux_dn = 0.0_wp
  !   flux_dir = 0.0_wp
  !   !$acc end kernels


  !   !$acc parallel loop collapse(2) default(present) private(albedo, denom,  source, source_dn, Rdif, Tdif)
  !   do icol = 1, ncol
  !      do igpt = 1, ngpt_sw
      

  !       ! Apply boundary condition
  !       radn_dir(igpt,top_level, icol)  = inc_flux(igpt,icol) * mu0(icol)
  !       radn_dn(igpt, top_level, icol)  = inc_flux_dif(igpt,icol)


  !       !$acc loop seq
  !       do ilay = 1, nlay

  !         !
  !         ! Cell properties: transmittance and reflectance for direct and diffuse radiation
  !         !  
          
  !         ! Zdunkowski Practical Improved Flux Method "PIFM"
  !         !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
  !         !
  !         gamma1= (8._wp - ssa(igpt,ilay,icol) * (5._wp + 3._wp * g(igpt,ilay,icol))) * .25_wp
  !         gamma2=  3._wp *(ssa(igpt,ilay,icol) * (1._wp -         g(igpt,ilay,icol))) * .25_wp
  !         gamma3= (2._wp - 3._wp * mu0(icol)  *                  g(igpt,ilay,icol) ) * .25_wp
  !         gamma4=  1._wp - gamma3

  !         alpha1 = gamma1 * gamma4 + gamma2 * gamma3           ! Eq. 16
  !         alpha2 = gamma1 * gamma3 + gamma2 * gamma4           ! Eq. 17

  !         k = sqrt(max((gamma1 - gamma2) * &
  !                       (gamma1 + gamma2),  k_min)) !  1.e-12_wp))
  !         exp_minusktau = exp(-tau(igpt,ilay,icol)*k)
  !         !
  !         ! Transmittance of direct, unscattered beam. Also used below
  !         !
  !         Tnoscat = exp(-tau(igpt,ilay,icol)*(1._wp/mu0(icol)))

  !         ! Diffuse reflection and transmission
  !         !
  !         exp_minus2ktau = exp_minusktau * exp_minusktau

  !         ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
  !         RT_term = 1._wp / (k      * (1._wp + exp_minus2ktau)  + &
  !                             gamma1 * (1._wp - exp_minus2ktau) )

  !         ! Equation 25
  !         Rdif(ilay) = RT_term * gamma2 * (1._wp - exp_minus2ktau)

  !         ! Equation 26
  !         Tdif(ilay) = RT_term * 2._wp * k * exp_minusktau

  !         !
  !         ! Direct reflect and transmission
  !         !
  !         k_mu     = k * mu0(icol)
  !         gamma3  = k * gamma3
  !         !
  !         ! Equation 14, multiplying top and bottom by exp(-k*tau)
  !         !   and rearranging to avoid div by 0.
  !         !
  !         RT_term =  ssa(igpt,ilay,icol) * RT_term/merge(1._wp - k_mu*k_mu, &
  !                                                         epsilon(1._wp),    &
  !                                                         abs(1._wp - k_mu*k_mu) >= epsilon(1._wp))

  !         Rdir  = RT_term  *                                    &
  !             ((1._wp - k_mu) * (alpha2 + gamma3)                  - &
  !             (1._wp + k_mu) * (alpha2 - gamma3) * exp_minus2ktau - &
  !             2.0_wp * (gamma3 - alpha2 * k_mu)  * exp_minusktau  * Tnoscat)

  !         !
  !         ! Equation 15, multiplying top and bottom by exp(-k*tau),
  !         !   multiplying through by exp(-tau/mu0) to prefer underflow to overflow
  !         ! Omitting direct transmittance
  !         !
  !         gamma4 = k * gamma4
  !         Tdir = &
  !                     -RT_term * ((1._wp + k_mu) * (alpha1 + gamma4) * Tnoscat - &
  !                                 (1._wp - k_mu) * (alpha1 - gamma4) * exp_minus2ktau * Tnoscat - &
  !                                 2.0_wp * (gamma4 + alpha1 * k_mu)  * exp_minusktau )

  !         source(ilay)     =    Rdir * radn_dir(igpt,ilay,icol)
  !         source_dn(ilay)  =    Tdir * radn_dir(igpt,ilay,icol)
  !         radn_dir(igpt,ilay+1,icol) = Tnoscat * radn_dir(igpt,ilay,icol)                    
  !       end do

  !       !
  !       ! Direct-beam and source for diffuse radiation + Transport
  !       !                 
  !       ! ADDING code
  !       ilay = nlay+1
  !       ! Albedo of lowest level is the surface albedo...
  !       albedo(ilay)  = sfc_alb_dif(igpt,icol)
  !       ! ... and source of diffuse radiation is surface emission
  !       source(ilay) = radn_dir(igpt,ilay,icol)*sfc_alb_dir(igpt,icol)

  !       !
  !       ! From bottom to top of atmosphere --
  !       !   compute albedo and source of upward radiation
  !       !
  !       !$acc loop seq
  !       do ilay = nlay, 1, -1
  !         ilay2 = ilay+1
  !         denom(ilay) = 1._wp/(1._wp - Rdif(ilay)*albedo(ilay2))    ! Eq 10
  !         albedo(ilay) = Rdif(ilay) + &
  !               Tdif(ilay)*Tdif(ilay) * albedo(ilay2) * denom(ilay) ! Equation 9

  !         !
  !         ! Equation 11 -- source is emitted upward radiation at top of layer plus
  !         !   radiation emitted at bottom of layer,
  !         !   transmitted through the layer and reflected from layers below (Tdiff*source*albedo)
  !         !
  !         source(ilay) =  source(ilay) + Tdif(ilay) * denom(ilay) * & 
  !                         (source(ilay2) + albedo(ilay2)*source_dn(ilay))
  !       end do
  !       ! Eq 12, at the top of the domain upwelling diffuse is due to ...
  !       radn_up(igpt,1,icol) = radn_dn(igpt,1,icol)* albedo(1) + & ! ... reflection of incident diffuse and
  !                                 source(1)                                  ! emission from below
  !       !
  !       ! From the top of the atmosphere downward -- compute fluxes
  !       ! Computationally heavy part
  !       ! 
  !       !$acc loop seq                         
  !       do ilay = 2, nlay+1
  !         ilay2 = ilay-1
  !         radn_dn(igpt,ilay,icol) = (Tdif(ilay2)*radn_dn(igpt,ilay2,icol) + &  ! Equation 13
  !                           Rdif(ilay2)*source(ilay) + source_dn(ilay2)) * denom(ilay2)
  !         radn_up(igpt,ilay,icol) = radn_dn(igpt,ilay,icol) * albedo(ilay) + source(ilay) ! Equation 12

  !         !$acc atomic
  !         flux_dir(ilay2,icol) = flux_dir(ilay2,icol) + radn_dir(igpt,ilay2,icol)
  !         !$acc atomic
  !         flux_dn(ilay2,icol) = flux_dn(ilay2,icol) + (radn_dn(igpt,ilay2,icol) + radn_dir(igpt,ilay2,icol))
  !         !$acc atomic
  !         flux_up(ilay2,icol) = flux_up(ilay2,icol) + radn_up(igpt,ilay2,icol)

  !       end do
  !       ilay = nlay+1
  !       !$acc atomic
  !       flux_dir(ilay,icol) = flux_dir(ilay,icol) + radn_dir(igpt,ilay,icol)
  !       !$acc atomic
  !       flux_dn(ilay,icol) = flux_dn(ilay,icol) + (radn_dn(igpt,ilay,icol) + radn_dir(igpt,ilay,icol))
  !       !$acc atomic
  !       flux_up(ilay,icol) = flux_up(ilay,icol) + radn_up(igpt,ilay,icol)
  !     end do

      
  !   end do

  !   !$acc exit data copyout(flux_up, flux_dn, flux_dir) delete(Rdif, albedo,denom,source,source_dn)

  ! end subroutine sw_solver_2stream


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
                              lay_source, lev_source, tau, trans, &
                              source_dn, source_up)
    !$acc routine seq
    !
    integer,                               intent(in)   :: ngpt, nlay, ncol
    integer,                               intent(in)   :: igpt, ilay, icol ! Working point coordinates
    real(wp), dimension(ngpt, nlay, ncol), intent(in)   :: lay_source,    & ! Planck source at layer center
                                                          tau,           &  ! Optical path (tau/mu)
                                                          trans             ! Transmissivity (exp(-tau))
    real(wp), dimension(ngpt,nlay+1,ncol), intent(in)   :: lev_source       ! Planck source at layer interfaces
                                                      
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
    source_dn(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                        2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
    source_up(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                        2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))

  end subroutine lw_source_noscat_stencil
  ! ---------------------------------------------------------------
  !
  ! Driver function to compute LW source function for upward and downward emission
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_noscat(ngpt, nlay, ncol, lay_source, lev_source, tau, trans, &
                              source_dn, source_up) bind(C, name="lw_source_noscat")
    integer,                               intent(in) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt, nlay, ncol), intent(in) :: lay_source,    & ! Planck source at layer center
                                                         tau,           & ! Optical path (tau/mu)
                                                         trans            ! Transmissivity (exp(-tau))
    real(wp), dimension(ngpt,nlay+1,ncol), intent(in) :: lev_source ! Planck source at levels (layer edges),
    real(wp), dimension(ngpt, nlay, ncol), intent(out):: source_dn, source_up
                                                                ! Source function at layer edges
                                                                ! Down at the bottom of the layer, up at the top
    ! --------------------------------
    integer :: igpt, ilay, icol
    real(wp), parameter  :: tau_thresh = sqrt(epsilon(tau))
    real(wp)             :: fact
    ! ---------------------------------------------------------------

    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! call lw_source_noscat_stencil(ngpt, nlay, ncol, igpt, ilay, icol,        &
          !                               lay_source, lev_source,  &
          !                               tau, trans,                                &
          !                               source_dn, source_up)

          if(tau(igpt,ilay,icol) > tau_thresh) then
            fact = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
          else
            fact = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt,ilay,icol))
          end if
          !
          ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
          !
          source_dn(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                              2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
          source_up(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                              2._wp * fact * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
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
    real(wp), dimension(:,:,:),             intent(out)   :: radn_up_Jac    ! surface temperature Jacobian of Radiances [W/m2-str / K]
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
          if (compute_Jac) radn_up_Jac(igpt,nlay+1,icol) = source_sfcJac(igpt,icol)

          ! Upward propagation
          do ilev = nlay, 1, -1
            radn_up   (igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up   (igpt,ilev+1,icol) + source_up(igpt,ilev,icol)
            if (compute_Jac) radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev+1,icol)
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
          if (compute_Jac) radn_up_Jac(igpt,1,icol) = source_sfcJac(igpt,icol)

          ! Upward propagation
          do ilev = 2, nlay+1
            radn_up   (igpt,ilev,icol) = trans(igpt,ilev-1,icol) * radn_up   (igpt,ilev-1,icol) +  source_up(igpt,ilev-1,icol)
            if (compute_Jac) radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev-1,icol) * radn_up_Jac(igpt,ilev-1,icol)
          end do
        end do
      end do
    end if

  end subroutine lw_transport_noscat

  ! ---------------------------------------------------------------
  !
  ! Longwave no-scattering transport downward, with source computation inlined to improve efficiency
  !
  ! ---------------------------------------------------------------
  !  pure subroutine lw_source_transport_noscat_dn(ngpt, nlay, ncol, top_at_1,  &  ! inputs
  !                   lay_source, lev_source,  tau, trans, &                ! inputs
  !                   source_up, radn_dn)                                   ! outputs
  !   integer,                                  intent(in)    :: ngpt, nlay, ncol
  !   logical(wl),                              intent(in)    :: top_at_1
  !   real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: lay_source
  !   real(wp), dimension(ngpt, nlay+1, ncol),  intent(in)    :: lev_source
  !   real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: tau,        & ! Optical path (tau/mu)
  !                                                             trans         ! Transmissivity (exp_fast(-tau))
  !   real(wp), dimension(ngpt,nlay,ncol),      intent(out)   :: source_up  ! Upward source at top of the layer
  !   real(wp), dimension(ngpt, nlay+1, ncol),  intent(inout) :: radn_dn    ! Top level must contain incident flux boundary condition
  !   ! --------------------------------
  !   integer                             :: igpt, ilay, icol
  !   real(wp)                            :: coeff  
  !   real(wp)                            :: source_dn ! Downward source at bottom of layer
  !   real(wp), parameter                 :: tau_thresh = sqrt(epsilon(tau))
  !   ! ---------------------------------------------------

  !   if(top_at_1) then
  !     !
  !     ! Top of domain is index 1
  !     !
  !     !$acc parallel loop collapse(2) default(present)
  !     do icol = 1, ncol
  !       do igpt = 1, ngpt
  !         ! not vectorized
  !         do ilay = 1, nlay
  !           ! Compute upward and downward source at layer edges 
  !           ! Downward source is only needed when computing downward transport, and can therefore be a local scalar
  !           if (use_Pade_source) then
  !             ! Alternative to avoid the conditional
  !             ! Equation below uses a Pade approximant for the linear-in-tau solution for the effective Planck function
  !             ! See Clough et al., 1992, doi:10.1029/92JD01419, Eq 15
  !             coeff = 0.2_wp * tau(igpt,ilay,icol)
  !             source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
  !                       coeff*lev_source(igpt,ilay,icol))   / (1 + coeff)
  !             source_dn             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
  !                       coeff*lev_source(igpt,ilay+1,icol)) / (1 + coeff)
  !           else 
  !             if(tau(igpt, ilay, icol) > tau_thresh) then
  !               coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
  !             else
  !               coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
  !             end if
  !             ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
  !             source_dn = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
  !                                   2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
  !             source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
  !                                   2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
  !           end if
  !           ! Downward propagation
  !           radn_dn(igpt,ilay+1,icol) = trans(igpt,ilay,icol)*radn_dn(igpt,ilay,icol) + source_dn
  !         end do
  !       end do
  !     end do
  !   else
  !     !
  !     ! Top of domain is index nlay+1
  !     !
  !     !$acc  parallel loop collapse(2) default(present)
  !     do icol = 1, ncol
  !       do igpt = 1, ngpt
  !         ! Downward propagation
  !         do ilay = nlay, 1, -1
  !           if (use_Pade_source) then
  !             ! Alternative to avoid the conditional
  !             coeff = 0.2_wp * tau(igpt,ilay,icol)
  !             source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
  !                       coeff*lev_source(igpt,ilay+1,icol))   / (1 + coeff)
  !             source_dn             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
  !                       coeff*lev_source(igpt,ilay,icol)) / (1 + coeff)
  !           else 
  !             if(tau(igpt, ilay, icol) > tau_thresh) then
  !               coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
  !             else
  !               coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
  !             end if
  !             ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
  !             source_dn = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
  !                                   2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
  !             source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
  !                                   2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
  !           end if
  !           radn_dn(igpt,ilay,icol) = trans(igpt,ilay  ,icol)*radn_dn(igpt,ilay+1,icol) + source_dn
  !         end do
  !       end do
  !     end do
  !   end if

  ! end subroutine lw_source_transport_noscat_dn
     pure subroutine lw_source_transport_noscat_dn(ngpt, nlay, ncol, top_at_1,  &  ! inputs
                    lay_source, lev_source,  tau, trans,  radn_dn)    
    integer,                                  intent(in)    :: ngpt, nlay, ncol
    logical(wl),                              intent(in)    :: top_at_1
    real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: lay_source
    real(wp), dimension(ngpt, nlay+1, ncol),  intent(in)    :: lev_source
    real(wp), dimension(ngpt, nlay,   ncol),  intent(inout)    :: tau ! Optical path (tau/mu) on input, source_up on output
    real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: trans         ! Transmissivity (exp_fast(-tau))
    ! real(wp), dimension(ngpt,nlay,ncol),      intent(out)   :: source_up  ! Upward source at top of the layer
    real(wp), dimension(ngpt, nlay+1, ncol),  intent(inout) :: radn_dn    ! Top level must contain incident flux boundary condition
    ! --------------------------------
    integer                             :: igpt, ilay, icol
    real(wp)                            :: coeff  
    real(wp)                            :: source_dn ! Downward source at bottom of layer
    real(wp), parameter                 :: tau_thresh = sqrt(epsilon(tau))
    ! ---------------------------------------------------
    
    associate(source_up=>tau)
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      !$acc parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! not vectorized
          do ilay = 1, nlay
            ! Compute upward and downward source at layer edges 
            ! Downward source is only needed when computing downward transport, and can therefore be a local scalar
            if (use_Pade_source) then
              ! Alternative to avoid the conditional
              ! Equation below uses a Pade approximant for the linear-in-tau solution for the effective Planck function
              ! See Clough et al., 1992, doi:10.1029/92JD01419, Eq 15
              coeff = 0.2_wp * tau(igpt,ilay,icol)
              source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay,icol))   / (1 + coeff)
              source_dn             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay+1,icol)) / (1 + coeff)
            else 
              if(tau(igpt, ilay, icol) > tau_thresh) then
                coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
              else
                coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
              end if
              ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
              source_dn = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
              source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
            end if
            ! Downward propagation
            radn_dn(igpt,ilay+1,icol) = trans(igpt,ilay,icol)*radn_dn(igpt,ilay,icol) + source_dn
          end do
        end do
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      !$acc  parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Downward propagation
          do ilay = nlay, 1, -1
            if (use_Pade_source) then
              ! Alternative to avoid the conditional
              coeff = 0.2_wp * tau(igpt,ilay,icol)
              source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay+1,icol))   / (1 + coeff)
              source_dn             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay,icol)) / (1 + coeff)
            else 
              if(tau(igpt, ilay, icol) > tau_thresh) then
                coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
              else
                coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
              end if
              ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
              source_dn = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
              source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
            end if
            radn_dn(igpt,ilay,icol) = trans(igpt,ilay  ,icol)*radn_dn(igpt,ilay+1,icol) + source_dn
          end do
        end do
      end do
    end if
  end associate 

  end subroutine lw_source_transport_noscat_dn

  pure subroutine lw_sources_transport_noscat_dn(ngpt, nlay, ncol, top_at_1,  &  ! inputs
                    lay_source, lev_source,  tau, trans, &                ! inputs
                    source_dn, radn_dn)                        ! outputs
    integer,                                  intent(in)    :: ngpt, nlay, ncol
    logical(wl),                              intent(in)    :: top_at_1
    real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: lay_source
    real(wp), dimension(ngpt, nlay+1, ncol),  intent(in)    :: lev_source
    real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: tau,        & ! Optical path (tau/mu)
                                                              trans         ! Transmissivity (exp_fast(-tau))
    ! real(wp), dimension(ngpt,nlay,ncol),      intent(out)   :: source_up  ! Upward source at top of the layer,
    real(wp), dimension(ngpt,nlay,ncol),      intent(out)   :: source_dn  ! downward at bottom
    real(wp), dimension(ngpt, nlay+1, ncol),  intent(inout) :: radn_dn    ! Top level must contain incident flux boundary condition
    ! --------------------------------
    integer                             :: igpt, ilay, icol
    real(wp)                            :: coeff  
    real(wp), parameter                 :: tau_thresh = sqrt(epsilon(tau))
    ! ---------------------------------------------------
    associate(source_up=>tau)
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      !$acc parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! not vectorized
          do ilay = 1, nlay
            ! Compute upward and downward source at layer edges 
            ! Downward source is only needed when computing downward transport, and can therefore be a local scalar
            if (use_Pade_source) then
              ! Alternative to avoid the conditional
              ! Equation below uses a Pade approximant for the linear-in-tau solution for the effective Planck function
              ! See Clough et al., 1992, doi:10.1029/92JD01419, Eq 15
              coeff = 0.2_wp * tau(igpt,ilay,icol)
              source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay,icol))   / (1 + coeff)
              source_dn(igpt,ilay,icol)             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay+1,icol)) / (1 + coeff)
            else 
              if(tau(igpt, ilay, icol) > tau_thresh) then
                coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
              else
                coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
              end if
              ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
              source_dn(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
              source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
            end if
            ! Downward propagation
            radn_dn(igpt,ilay+1,icol) = trans(igpt,ilay,icol)*radn_dn(igpt,ilay,icol) + source_dn(igpt,ilay,icol)
          end do
        end do
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      !$acc  parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Downward propagation
          do ilay = nlay, 1, -1
            if (use_Pade_source) then
              ! Alternative to avoid the conditional
              coeff = 0.2_wp * tau(igpt,ilay,icol)
              source_up(igpt,ilay,icol)  = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay+1,icol))   / (1 + coeff)
              source_dn(igpt,ilay,icol)             = (1.0_wp-trans(igpt,ilay,icol)) * (lay_source(igpt,ilay,icol) + &
                        coeff*lev_source(igpt,ilay,icol)) / (1 + coeff)
            else 
              if(tau(igpt, ilay, icol) > tau_thresh) then
                coeff = (1._wp - trans(igpt,ilay,icol))/tau(igpt,ilay,icol) - trans(igpt,ilay,icol)
              else
                coeff = tau(igpt, ilay,icol) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay,icol))
              end if
              ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
              source_dn(igpt,ilay,icol) = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay,icol))
              source_up(igpt,ilay,icol)    = (1._wp - trans(igpt,ilay,icol)) * lev_source(igpt,ilay+1,icol) + &
                                    2._wp * coeff * (lay_source(igpt,ilay,icol) - lev_source(igpt,ilay+1,icol))
            end if
            radn_dn(igpt,ilay,icol) = trans(igpt,ilay  ,icol)*radn_dn(igpt,ilay+1,icol) + source_dn(igpt,ilay,icol)
          end do
        end do
      end do
    end if
    end associate
  end subroutine lw_sources_transport_noscat_dn

  pure subroutine lw_transport_noscat_up(ngpt, nlay, ncol, top_at_1,  & 
                    trans, source_up, radn_up, radn_up_Jac)
    integer,                                  intent(in)    :: ngpt, nlay, ncol
    logical(wl),                              intent(in)    :: top_at_1
    real(wp), dimension(ngpt, nlay,   ncol),  intent(in)    :: trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt,nlay,ncol),      intent(in)    :: source_up 
    real(wp), dimension(ngpt, nlay+1, ncol),  intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(:,:,:),               intent(inout) :: radn_up_Jac ! surface temperature Jacobian of Radiances [W/m2-str / K]
    ! --------------------------------
    integer                             :: igpt, ilay, icol
    ! ---------------------------------------------------

    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      !$acc parallel loop collapse(2)  default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
         ! Upward propagation
          do ilay = nlay, 1, -1
            radn_up    (igpt,ilay,icol) = trans(igpt,ilay,icol)*radn_up    (igpt,ilay+1,icol) + source_up(igpt,ilay,icol)
            if (compute_Jac) radn_up_Jac(igpt,ilay,icol) = trans(igpt,ilay,icol)*radn_up_Jac(igpt,ilay+1,icol)
          end do
        end do
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      !$acc  parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Upward propagation
          do ilay = 2, nlay+1
            radn_up   (igpt,ilay,icol) = trans(igpt,ilay-1,icol) * radn_up   (igpt,ilay-1,icol) +  source_up(igpt,ilay-1,icol)
            radn_up_Jac(igpt,ilay,icol) = trans(igpt,ilay-1,icol) * radn_up_Jac(igpt,ilay-1,icol)
          end do
        end do
      end do
    end if

  end subroutine lw_transport_noscat_up

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
                       k_min))
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
                            gamma1, gamma2, Rdif, Tdif, tau, source_dn, source_up, source_sfc) &
                            bind (C, name="lw_source_2str")
    integer,                         intent(in) :: ngpt, nlay, ncol
    logical(wl),                     intent(in) :: top_at_1
    real(wp), dimension(ngpt      , ncol), intent(in) :: sfc_emis, sfc_src
    real(wp), dimension(ngpt, nlay, ncol), intent(in) :: lay_source,    & ! Planck source at layer center
                                                   tau,           & ! Optical depth (tau)
                                                   gamma1, gamma2,& ! Coupling coefficients
                                                   Rdif, Tdif       ! Layer reflectance and transmittance
    real(wp), dimension(ngpt, nlay+1, ncol), target, &
                                     intent(in)  :: lev_source       ! Planck source at layer edges
    real(wp), dimension(ngpt, nlay, ncol), intent(out) :: source_dn, source_up
    real(wp), dimension(ngpt      , ncol), intent(out) :: source_sfc      ! Source function for upward radation at surface

    integer             :: igpt, ilay, icol
    real(wp)            :: Z, Zup_top, Zup_bottom, Zdn_top, Zdn_bottom
    real(wp)            :: lev_source_bot, lev_source_top
    ! ---------------------------------------------------------------
    ! ---------------------------------
    !$acc enter data copyin(sfc_emis, sfc_src, lay_source, tau, gamma1, gamma2, Rdif, Tdif, lev_source)
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
            source_up(igpt,ilay,icol) = pi * (Zup_top    - Rdif(igpt,ilay,icol) * Zdn_top    - Tdif(igpt,ilay,icol) * Zup_bottom)
            source_dn(igpt,ilay,icol) = pi * (Zdn_bottom - Rdif(igpt,ilay,icol) * Zup_bottom - Tdif(igpt,ilay,icol) * Zdn_top)
          else
            source_up(igpt,ilay,icol) = 0._wp
            source_dn(igpt,ilay,icol) = 0._wp
          end if
          if(ilay == 1) source_sfc(igpt,icol) = pi * sfc_emis(igpt,icol) * sfc_src(igpt,icol)
        end do
      end do
    end do
    !$acc exit data delete(sfc_emis, sfc_src, lay_source, tau, gamma1, gamma2, Rdif, Tdif, lev_source)
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
  subroutine sw_two_stream(ngpt_sw_in, nlay_in, ncol, mu0, tau, w0, g, &
                                  Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream")
    integer,                             intent(in)  :: ngpt_sw_in, nlay_in, ncol
    real(wp), dimension(ncol),           intent(in)  :: mu0
    real(wp), dimension(ngpt_sw,nlay,ncol), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt_sw,nlay,ncol), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat

    ! -----------------------
    integer  :: igpt,ilay,icol

    ! Variables used in Meador and Weaver
    real(wp) :: gamma1, gamma2, gamma3, gamma4
    real(wp) :: alpha1, alpha2, k

    ! Ancillary variables
    real(wp) :: RT_term
    real(wp) :: exp_minusktau, exp_minus2ktau
    real(wp) :: k_mu !, k_gamma3, k_gamma4
    real(wp) :: k_gamma3, k_gamma4  ! Need to be in double precision
    real(wp) :: mu0_inv(ncol)
    ! ---------------------------------
    ! ---------------------------------

    !$acc data create(mu0_inv)

    !$acc parallel default(present)
    !$acc loop
    do icol = 1, ncol
      mu0_inv(icol) = 1._wp/mu0(icol)
    enddo

    !$acc loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt_sw
          ! Zdunkowski Practical Improved Flux Method "PIFM"
          !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
          !
          gamma1= (8._wp - w0(igpt,ilay,icol) * (5._wp + 3._wp * g(igpt,ilay,icol))) * .25_wp
          gamma2=  3._wp *(w0(igpt,ilay,icol) * (1._wp -         g(igpt,ilay,icol))) * .25_wp
          gamma3= (2._wp - 3._wp * mu0(icol)  *                  g(igpt,ilay,icol) ) * .25_wp
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
                        k_min)) !  1.e-12_wp))
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
          Tnoscat(igpt,ilay,icol) = exp(-tau(igpt,ilay,icol)*mu0_inv(icol))

          !
          ! Direct reflect and transmission
          !
          k_mu     = k * mu0(icol)
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
          !   multiplying through by exp(-tau/mu0) to prefer underflow to overflow
          ! Omitting direct transmittance
          !
          Tdir(igpt,ilay,icol) = &
                    -RT_term * ((1._wp + k_mu) * (alpha1 + k_gamma4) * Tnoscat(igpt,ilay,icol) - &
                                (1._wp - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat(igpt,ilay,icol) - &
                                2.0_wp * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau )

        end do
      end do
    end do
    !$acc end parallel
    !$acc end data

  end subroutine sw_two_stream

  ! ---------------------------------------------------------------
  !
  ! Direct beam source for diffuse radiation in layers and at surface;
  !   report direct beam as a byproduct
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine sw_dif_and_source(ngpt, nlay, ncol, top_at_1, mu0, sfc_albedo, &
                                tau, w0, g,                                      &
                                Rdif, Tdif, source_dn, source_up, source_sfc,    &
                                flux_dn_dir) bind (C, name="sw_source_dir")
    integer,                               intent(in   ) :: ngpt, nlay, ncol
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt            ), intent(in   ) :: mu0
    real(wp), dimension(ngpt,       ncol), intent(in   ) :: sfc_albedo          ! surface albedo for direct radiation
    real(wp), dimension(ngpt,nlay,  ncol), intent(in   ) :: tau, w0, g
    real(wp), dimension(ngpt,nlay,  ncol), target, &
                                           intent(  out) :: Rdif, Tdif, source_dn, source_up
    real(wp), dimension(ngpt,       ncol), intent(  out) :: source_sfc ! Source function for upward radation at surface
    real(wp), dimension(ngpt,nlay+1,ncol), target, &
                                           intent(inout) :: flux_dn_dir ! Direct beam flux

    ! -----------------------
    integer  :: icol, ilay, igpt

    ! Variables used in Meador and Weaver
    real(wp) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2


    ! Ancillary variables
    real(wp) :: k, exp_minusktau, k_mu, k_gamma3, k_gamma4
    real(wp) :: RT_term, exp_minus2ktau
    real(wp) :: Rdir, Tdir, Tnoscat, inc_flux
    integer  :: lay_index, inc_index, trans_index
    real(wp) :: tau_s, w0_s, g_s, mu0_s
    ! ---------------------------------
    !$acc  parallel loop collapse(2)
    !$omp target teams distribute parallel do simd collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt
        do ilay = 1, nlay
          if(top_at_1) then
            lay_index   = ilay
            inc_index   = lay_index
            trans_index = lay_index+1
          else
            lay_index   = nlay-ilay+1
            inc_index   = lay_index+1
            trans_index = lay_index
          end if
          inc_flux = flux_dn_dir(igpt,inc_index,icol)
          !
          ! Scalars
          !
          tau_s = tau(igpt,lay_index,icol)
          w0_s  = w0 (igpt,lay_index,icol)
          g_s   = g  (igpt,lay_index,icol)
          mu0_s = mu0(icol)
          !
          ! Zdunkowski Practical Improved Flux Method "PIFM"
          !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
          !
          gamma1 = (8._wp - w0_s * (5._wp + 3._wp * g_s)) * .25_wp
          gamma2 =  3._wp *(w0_s * (1._wp -         g_s)) * .25_wp
          gamma3 = (2._wp - 3._wp * mu0_s *         g_s ) * .25_wp
          gamma4 =  1._wp - gamma3
          alpha1 = gamma1 * gamma4 + gamma2 * gamma3           ! Eq. 16
          alpha2 = gamma1 * gamma3 + gamma2 * gamma4           ! Eq. 17
          !
          ! Direct reflect and transmission
          !
          ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
          !   k = 0 for isotropic, conservative scattering; this lower limit on k
          !   gives relative error with respect to conservative solution
          !   of < 0.1% in Rdif down to tau = 10^-9
          k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min))
          k_mu     = k * mu0_s
          k_gamma3 = k * gamma3
          k_gamma4 = k * gamma4
          exp_minusktau = exp(-tau_s*k)
          exp_minus2ktau = exp_minusktau * exp_minusktau

          ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
          RT_term = 1._wp / (k      * (1._wp + exp_minus2ktau)  + &
                             gamma1 * (1._wp - exp_minus2ktau) )
          ! Equation 25
          Rdif(igpt,lay_index,icol) = RT_term * gamma2 * (1._wp - exp_minus2ktau)

          ! Equation 26
          Tdif(igpt,lay_index,icol) = RT_term * 2._wp * k * exp_minusktau
          !
          ! Equation 14, multiplying top and bottom by exp(-k*tau)
          !   and rearranging to avoid div by 0.
          !
          RT_term =  w0_s * RT_term/merge(1._wp - k_mu*k_mu, &
                                          epsilon(1._wp),    &
                                          abs(1._wp - k_mu*k_mu) >= epsilon(1._wp))

          !
          ! Transmittance of direct, unscattered beam.
          !
          Tnoscat = exp(-tau_s/mu0_s)
          Rdir = RT_term  *                                            &
              ((1._wp - k_mu) * (alpha2 + k_gamma3)                  - &
               (1._wp + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau - &
               2.0_wp * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * Tnoscat)
          !
          ! Equation 15, multiplying top and bottom by exp(-k*tau),
          !   multiplying through by exp(-tau/mu0) to
          !   prefer underflow to overflow
          ! Omitting direct transmittance
          !
          Tdir = -RT_term *                                                             &
                ((1._wp + k_mu) * (alpha1 + k_gamma4)                  * Tnoscat - &
                 (1._wp - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat - &
                 2.0_wp * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau)
          ! Final check that Rdir + Tdir <= 1
          Rdir    = max(0.0_wp, min(Rdir, (1.0_wp - Tnoscat       ) ))
          Tdir    = max(0.0_wp, min(Tdir, (1.0_wp - Tnoscat - Rdir) ))

          source_up  (igpt,lay_index,  icol) =    Rdir * inc_flux
          source_dn  (igpt,lay_index,  icol) =    Tdir * inc_flux
          flux_dn_dir(igpt,trans_index,icol) = Tnoscat * inc_flux
        end do
        source_sfc(igpt,icol) = flux_dn_dir(igpt,trans_index,icol)*sfc_albedo(igpt,icol)
      end do
    end do
  end subroutine sw_dif_and_source

  subroutine sw_dif_and_source_and_adding(ngpt_sw_in, nlay_in, ncol, top_at_1, mu0, sfc_albedo, &
                                tau, w0, g,                                      &
                                flux_up, flux_dn, flux_dn_dir)
    integer,                               intent(in   ) :: ngpt_sw_in, nlay_in, ncol
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt_sw            ), intent(in   ) :: mu0
    real(wp), dimension(ngpt_sw,       ncol), intent(in   ) :: sfc_albedo          ! surface albedo for direct radiation
    real(wp), dimension(ngpt_sw,nlay,  ncol), intent(in   ) :: tau, w0, g
    real(wp), dimension(ngpt_sw,nlay+1,ncol), intent(  out) :: flux_up
    real(wp), dimension(ngpt_sw,nlay+1,ncol), intent(inout) :: flux_dn
    real(wp), dimension(ngpt_sw,nlay+1,ncol), intent(inout) :: flux_dn_dir ! Direct beam flux

    ! -----------------------
    real(wp), dimension(ngpt_sw, nlay, ncol) :: Rdif, Tdif!, source_dn, source_up
    ! real(wp) :: source_sfc ! Source function for upward radation at surface
    integer  :: icol, ilay, igpt

    ! Variables used in Meador and Weaver
    real(wp) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2

    ! Ancillary variables
    real(wp) :: k, exp_minusktau, k_mu, k_gamma3, k_gamma4
    real(wp) :: RT_term, exp_minus2ktau
    real(wp) :: Rdir, Tdir, Tnoscat, inc_flux
    integer  :: lay_index, inc_index, trans_index
    real(wp) :: tau_s, w0_s, g_s, mu0_s
    ! adding
    real(wp), dimension(ngpt_sw,nlay+1,ncol) :: & ! albedo, &  ! reflectivity to diffuse radiation below this level
                                              ! alpha in SH08
                                   src        ! source of diffuse upwelling radiation from emission or
                                              ! scattering of direct beam
                                              ! G in SH08
    real(wp), dimension(ngpt_sw,nlay  ,ncol) :: denom      ! beta in SH08
    ! ---------------------------------
    
    if(.not. top_at_1) stop "sw_dif_and_source_and_adding currently only works when top_at_1"

    associate (albedo=>flux_up, source_dn=>flux_dn, source_up=>src)
    !$acc        data create(   Rdif, Tdif, src, denom)
    !$omp target data map(alloc:Rdif, Tdif, src, denom)

    !$acc  parallel loop collapse(2)
    !$omp target teams distribute parallel do simd collapse(2)
    do icol = 1, ncol
      do igpt = 1, ngpt_sw
        do ilay = 1, nlay
          inc_flux = flux_dn_dir(igpt,ilay,icol)
          !
          ! Scalars
          !
          tau_s = tau(igpt,ilay,icol)
          w0_s  = w0 (igpt,ilay,icol)
          g_s   = g  (igpt,ilay,icol)
          mu0_s = mu0(icol)
          !
          ! Zdunkowski Practical Improved Flux Method "PIFM"
          !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
          !
          gamma1 = (8._wp - w0_s * (5._wp + 3._wp * g_s)) * .25_wp
          gamma2 =  3._wp *(w0_s * (1._wp -         g_s)) * .25_wp
          gamma3 = (2._wp - 3._wp * mu0_s *         g_s ) * .25_wp
          gamma4 =  1._wp - gamma3
          alpha1 = gamma1 * gamma4 + gamma2 * gamma3           ! Eq. 16
          alpha2 = gamma1 * gamma3 + gamma2 * gamma4           ! Eq. 17
          !
          ! Direct reflect and transmission
          !
          ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
          !   k = 0 for isotropic, conservative scattering; this lower limit on k
          !   gives relative error with respect to conservative solution
          !   of < 0.1% in Rdif down to tau = 10^-9
          k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min))
          k_mu     = k * mu0_s
          k_gamma3 = k * gamma3
          k_gamma4 = k * gamma4
          exp_minusktau = exp(-tau_s*k)
          exp_minus2ktau = exp_minusktau * exp_minusktau

          ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
          RT_term = 1._wp / (k      * (1._wp + exp_minus2ktau)  + &
                             gamma1 * (1._wp - exp_minus2ktau) )
          ! Equation 25
          Rdif(igpt,ilay,icol) = RT_term * gamma2 * (1._wp - exp_minus2ktau)

          ! Equation 26
          Tdif(igpt,ilay,icol) = RT_term * 2._wp * k * exp_minusktau
          !
          ! Equation 14, multiplying top and bottom by exp(-k*tau)
          !   and rearranging to avoid div by 0.
          !
          RT_term =  w0_s * RT_term/merge(1._wp - k_mu*k_mu, &
                                          epsilon(1._wp),    &
                                          abs(1._wp - k_mu*k_mu) >= epsilon(1._wp))

          !
          ! Transmittance of direct, unscattered beam.
          !
          Tnoscat = exp(-tau_s/mu0_s)
          Rdir = RT_term  *                                            &
              ((1._wp - k_mu) * (alpha2 + k_gamma3)                  - &
               (1._wp + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau - &
               2.0_wp * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau * Tnoscat)
          !
          ! Equation 15, multiplying top and bottom by exp(-k*tau),
          !   multiplying through by exp(-tau/mu0) to
          !   prefer underflow to overflow
          ! Omitting direct transmittance
          !
          Tdir = -RT_term *                                                             &
                ((1._wp + k_mu) * (alpha1 + k_gamma4)                  * Tnoscat - &
                 (1._wp - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat - &
                 2.0_wp * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau)
          ! Final check that Rdir + Tdir <= 1
          Rdir    = max(0.0_wp, min(Rdir, (1.0_wp - Tnoscat       ) ))
          Tdir    = max(0.0_wp, min(Tdir, (1.0_wp - Tnoscat - Rdir) ))

          source_up  (igpt,ilay,  icol) =    Rdir * inc_flux
          ! source_dn  (igpt,ilay,  icol) =    Tdir * inc_flux
          source_dn  (igpt,ilay+1,  icol) =    Tdir * inc_flux
          flux_dn_dir(igpt,ilay+1,icol) = Tnoscat * inc_flux
        end do
        ! source_sfc = flux_dn_dir(igpt,nlay+1,icol)*sfc_albedo(igpt,icol)

        ! ADDING
        ilay = nlay + 1
        ! Albedo of lowest level is the surface albedo...
        albedo(igpt,ilay,icol)  = sfc_albedo(igpt,icol)
        ! ... and source of diffuse radiation is surface emission
        ! src(igpt,ilay,icol) = source_sfc
        src(igpt,ilay,icol) = flux_dn_dir(igpt,nlay+1,icol)*sfc_albedo(igpt,icol)

        !
        ! From bottom to top of atmosphere --
        !   compute albedo and source of upward radiation
        !
        do ilay = nlay, 1, -1
          denom(igpt,ilay,icol) = 1._wp/(1._wp - Rdif(igpt,ilay,icol)*albedo(igpt,ilay+1,icol))    ! Eq 10
          albedo(igpt,ilay,icol) = Rdif(igpt,ilay,icol) + &
                Tdif(igpt,ilay,icol)*Tdif(igpt,ilay,icol) * albedo(igpt,ilay+1,icol) * denom(igpt,ilay,icol) ! Equation 9
          !
          ! Equation 11 -- source is emitted upward radiation at top of layer plus
          !   radiation emitted at bottom of layer,
          !   transmitted through the layer and reflected from layers below (Tdiff*src*albedo)
          !
          src(igpt,ilay,icol) =  source_up(igpt, ilay, icol) + &
                        Tdif(igpt,ilay,icol) * denom(igpt,ilay,icol) *       &
                        (src(igpt,ilay+1,icol) + albedo(igpt,ilay+1,icol)*source_dn(igpt,ilay+1,icol))
                        ! (src(igpt,ilay+1,icol) + albedo(igpt,ilay+1,icol)*source_dn(igpt,ilay,icol))

        end do

        ! Eq 12, at the top of the domain upwelling diffuse is due to ...
        ilay = 1
        flux_up(igpt,ilay,icol) = flux_dn(igpt,ilay,icol) * albedo(igpt,ilay,icol) + & ! ... reflection of incident diffuse and
                                  src(igpt,ilay,icol)                                  ! emission from below

        !
        ! From the top of the atmosphere downward -- compute fluxes
        !
        do ilay = 2, nlay+1
          flux_dn(igpt,ilay,icol) = (Tdif(igpt,ilay-1,icol)*flux_dn(igpt,ilay-1,icol) + &  ! Equation 13
                              Rdif(igpt,ilay-1,icol)*src(igpt,ilay,icol) +       &
                              source_dn(igpt,ilay,icol)) * denom(igpt,ilay-1,icol)
                              ! source_dn(igpt,ilay-1,icol)) * denom(igpt,ilay-1,icol)
          flux_up(igpt,ilay,icol) = flux_dn(igpt,ilay,icol) * albedo(igpt,ilay,icol) + & ! Equation 12
                            src(igpt,ilay,icol)
        end do
      end do
    end do

    !$acc        end data
    !$omp end target data
    end associate

  end subroutine sw_dif_and_source_and_adding


  ! ---------------------------------------------------------------
  !
  ! Transport of diffuse radiation through a vertically layered atmosphere.
  !   Equations are after Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08)
  !   This routine is shared by longwave and shortwave
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine adding(ngpt_sw_in, nlay_in, ncol, top_at_1, &
                    albedo_sfc,           &
                    Rdif, Tdif,           &
                    src_dn, src_up, src_sfc, &
                    flux_up, flux_dn) bind(C, name="adding")
    integer,                              intent(in   ) :: ngpt_sw_in, nlay_in, ncol
    logical(wl),                           intent(in   ) :: top_at_1
    real(wp), dimension(ngpt_sw       ,ncol), intent(in   ) :: albedo_sfc
    real(wp), dimension(ngpt_sw,nlay  ,ncol), intent(in   ) :: Rdif, Tdif
    real(wp), dimension(ngpt_sw,nlay  ,ncol), intent(in   ) :: src_dn, src_up
    real(wp), dimension(ngpt_sw       ,ncol), intent(in   ) :: src_sfc
    real(wp), dimension(ngpt_sw,nlay+1,ncol), intent(  out) :: flux_up
    ! intent(inout) because top layer includes incident flux
    real(wp), dimension(ngpt_sw,nlay+1,ncol), intent(inout) :: flux_dn
    ! ------------------
    integer :: igpt, ilev, icol

    ! These arrays could be private per thread in OpenACC, with 1 dimension of size nlay (or nlay+1)
    ! However, current PGI (19.4) has a bug preventing it from properly handling such private arrays.
    ! So we explicitly create the temporary arrays of size nlay(+1) per each of the ngpt*ncol elements
    !
    real(wp), dimension(ngpt_sw,nlay+1,ncol) :: albedo, &  ! reflectivity to diffuse radiation below this level
                                              ! alpha in SH08
                                   src        ! source of diffuse upwelling radiation from emission or
                                              ! scattering of direct beam
                                              ! G in SH08
    real(wp), dimension(ngpt_sw,nlay  ,ncol) :: denom      ! beta in SH08
    ! ------------------
    ! ---------------------------------
    !
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    !

    !$acc data present(albedo_sfc, Rdif, Tdif, src_dn, src_up, src_sfc, flux_up, flux_dn)

    !$acc enter data create(albedo, src, denom)

    if(top_at_1) then
      !$acc parallel loop gang vector collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt_sw
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
            denom(igpt,ilev,icol) = 1._wp/(1._wp - Rdif(igpt,ilev,icol)*albedo(igpt,ilev+1,icol))    ! Eq 10
            albedo(igpt,ilev,icol) = Rdif(igpt,ilev,icol) + &
                  Tdif(igpt,ilev,icol)*Tdif(igpt,ilev,icol) * albedo(igpt,ilev+1,icol) * denom(igpt,ilev,icol) ! Equation 9
            !
            ! Equation 11 -- source is emitted upward radiation at top of layer plus
            !   radiation emitted at bottom of layer,
            !   transmitted through the layer and reflected from layers below (Tdiff*src*albedo)
            !
            src(igpt,ilev,icol) =  src_up(igpt, ilev, icol) + &
                           Tdif(igpt,ilev,icol) * denom(igpt,ilev,icol) *       &
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
            flux_dn(igpt,ilev,icol) = (Tdif(igpt,ilev-1,icol)*flux_dn(igpt,ilev-1,icol) + &  ! Equation 13
                               Rdif(igpt,ilev-1,icol)*src(igpt,ilev,icol) +       &
                               src_dn(igpt,ilev-1,icol)) * denom(igpt,ilev-1,icol)
            flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! Equation 12
                              src(igpt,ilev,icol)
          end do
        end do
      end do

    else

      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt_sw
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
            denom(igpt,ilev  ,icol) = 1._wp/(1._wp - Rdif(igpt,ilev,icol)*albedo(igpt,ilev,icol))                ! Eq 10
            albedo(igpt,ilev+1,icol) = Rdif(igpt,ilev,icol) + &
                               Tdif(igpt,ilev,icol)*Tdif(igpt,ilev,icol) * albedo(igpt,ilev,icol) * denom(igpt,ilev,icol) ! Equation 9
            !
            ! Equation 11 -- source is emitted upward radiation at top of layer plus
            !   radiation emitted at bottom of layer,
            !   transmitted through the layer and reflected from layers below (Tdiff*src*albedo)
            !
            src(igpt,ilev+1,icol) =  src_up(igpt, ilev, icol) +  &
                             Tdif(igpt,ilev,icol) * denom(igpt,ilev,icol) *       &
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
            flux_dn(igpt,ilev,icol) = (Tdif(igpt,ilev,icol)*flux_dn(igpt,ilev+1,icol) + &  ! Equation 13
                               Rdif(igpt,ilev,icol)*src(igpt,ilev,icol) + &
                               src_dn(igpt, ilev, icol)) * denom(igpt,ilev,icol)
            flux_up(igpt,ilev,icol) = flux_dn(igpt,ilev,icol) * albedo(igpt,ilev,icol) + & ! Equation 12
                              src(igpt,ilev,icol)

          end do
        end do
      end do
    end if
    !$acc exit data delete(albedo, src, denom)
    !$acc end data

  end subroutine adding

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
!  Computes Tang scaling of layer optical thickness and scaling parameter
!    unsafe if ssa*g =1.
!
! ---------------------------------------------------------------
  pure subroutine scaling_1rescl(ngpt, nlay, ncol, igpt, ilay, icol, &
                                D, tau, ssa, g, &
                                trans, tau_loc, Cn, An)
    !$acc routine seq
    integer ,                              intent(in)    :: ngpt, nlay, ncol
    integer ,                              intent(in)    :: igpt, ilay, icol
    real(wp), dimension(ngpt,       ncol), intent(in)    :: D
    real(wp), dimension(ngpt, nlay, ncol), intent(in)    :: tau, ssa, g
    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: trans, tau_loc, Cn, An

    real(wp) :: wb, ssal, scaleTau

    ssal = ssa(igpt, ilay, icol)
    wb = ssal*(1._wp - g(igpt, ilay, icol)) * 0.5_wp
    scaleTau = (1._wp - ssal + wb )
    ! here wb/scaleTau is parameter wb/(1-w(1-b)) of Eq.21 of the Tang paper
    ! actually it is in line of parameter rescaling defined in Eq.7
    ! potentialy if g=ssa=1  then  wb/scaleTau = NaN
    ! it should not happen because g is never 1 in atmospheres
    ! explanation of factor 0.4 note A of Table
    Cn(igpt,ilay,icol) = 0.4_wp*wb/scaleTau
    ! Eq.15 of the paper, multiplied by path length
    tau_loc(igpt,ilay,icol) = tau(igpt,ilay,icol)*D(igpt,icol)*scaleTau
    trans  (igpt,ilay,icol) = exp(-tau_loc(igpt,ilay,icol))
    An     (igpt,ilay,icol) = (1._wp-trans(igpt,ilay,icol)**2)


  end subroutine scaling_1rescl
! -------------------------------------------------------------------------------------------------
!
! Similar to Longwave no-scattering transport  (lw_transport_noscat)
!   a) adds adjustment factor based on cloud properties
!
!   implementation notice:
!       the adjustmentFactor computation can be skipped where Cn <= epsilon
!
! -------------------------------------------------------------------------------------------------
  subroutine lw_transport_1rescl(ngpt, nlay, ncol, top_at_1, &
                                 trans, source_dn, source_up, &
                                 radn_up, radn_dn, An, Cn,&
                                 radn_up_Jac) bind(C, name="lw_transport_1rescl")
    integer,                               intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                           intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: trans      ! transmissivity = exp(-tau)
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: source_dn, &
                                                            source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1,ncol), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay  ,ncol), intent(in   ) :: An, Cn
    real(wp), dimension(:,:,:),             intent(inout) :: radn_up_Jac ! Radiances [W/m2-str]
    !
    ! We could in principle compute a downwelling Jacobian too, but it's small
    !   (only a small proportion of LW is scattered) and it complicates code and the API,
    !   so we will not
    !
    ! Local variables
    integer :: ilev, igpt, icol
    ! ---------------------------------------------------
    real(wp) :: adjustmentFactor
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      ! Downward propagation
      !$acc  parallel loop collapse(2) default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! 1st Upward propagation
          do ilev = nlay, 1, -1
            adjustmentFactor = Cn(igpt,ilev,icol)*&
                   ( An(igpt,ilev,icol)*radn_dn(igpt,ilev,icol) - &
                     source_dn(igpt,ilev,icol)  *trans(igpt,ilev,icol ) - &
                     source_up(igpt,ilev,icol))
            radn_up(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up   (igpt,ilev+1,icol) + source_up(igpt,ilev,icol) + adjustmentFactor
            if (compute_Jac) radn_up_Jac(igpt,ilev,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev+1,icol)
          enddo
          ! 2nd Downward propagation
          do ilev = 1, nlay
            adjustmentFactor = Cn(igpt,ilev,icol)*( &
              An(igpt,ilev,icol)*radn_up(igpt,ilev,icol) - source_up(igpt,ilev,icol)*trans(igpt,ilev,icol) - source_dn(igpt,ilev,icol) )
            radn_dn(igpt,ilev+1,icol)    = trans(igpt,ilev,icol)*radn_dn   (igpt,ilev,icol) + source_dn(igpt,ilev,icol) + adjustmentFactor
          enddo
        enddo
      enddo
    else
      !$acc  parallel loop collapse(2)  default(present)
      do icol = 1, ncol
        do igpt = 1, ngpt
          ! Upward propagation
          do ilev = 1, nlay
            adjustmentFactor = Cn(igpt,ilev,icol)*&
                   ( An(igpt,ilev,icol)*radn_dn(igpt,ilev+1,icol) - &
                     source_dn(igpt,ilev,icol) *trans(igpt,ilev ,icol) - &
                     source_up(igpt,ilev,icol))
            radn_up(igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_up(igpt,ilev,icol) + source_up(igpt,ilev,icol) + adjustmentFactor
            if (compute_Jac) radn_up_Jac(igpt,ilev+1,icol) = trans(igpt,ilev,icol)*radn_up_Jac(igpt,ilev,icol)
          end do
          ! 2st Downward propagation
          do ilev = nlay, 1, -1
            adjustmentFactor = Cn(igpt,ilev,icol)*( &
                An(igpt,ilev,icol)*radn_up(igpt,ilev,icol) - source_up(igpt,ilev,icol)*trans(igpt,ilev,icol ) - source_dn(igpt,ilev,icol) )
            radn_dn(igpt,ilev,icol)  = trans(igpt,ilev,icol)*radn_dn(igpt,ilev+1,icol) + source_dn(igpt,ilev,icol) + adjustmentFactor
          end do
        enddo
      enddo
    end if
  end subroutine lw_transport_1rescl

pure subroutine sum_broadband_fac(ngpt, nlev, ncol, fac, spectral_flux, broadband_flux)
  integer,                               intent(in ) :: ngpt, nlev, ncol
  real(wp),                               intent(in ) :: fac
  real(wp), dimension(ngpt, nlev, ncol), intent(in ) :: spectral_flux
  real(wp), dimension(nlev, ncol),       intent(out) :: broadband_flux
  integer  :: igpt, ilev, icol
  real(wp) :: bb_flux_s

  !$acc data copyout(broadband_flux)
  !$acc parallel loop gang worker collapse(2) default(present)
  do icol = 1, ncol
    do ilev = 1, nlev

      bb_flux_s = 0.0_wp
      !$acc loop vector reduction(+:bb_flux_s)
      do igpt = 1, ngpt
        bb_flux_s = bb_flux_s + spectral_flux(igpt, ilev, icol)
      end do
     broadband_flux(ilev, icol) = fac*bb_flux_s
    end do
  end do
  !$acc end data

end subroutine sum_broadband_fac

end module mo_rte_solver_kernels
