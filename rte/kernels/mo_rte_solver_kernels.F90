! This code is !!a refactored version!! of Radiative Transfer for Energetics (RTE)
! RTE is Developed Robert Pincus and Eli Mlawer (rrtmgp@aer.com)
! 
! !! Refactoring effort by Peter Ukkonen (peterukk@gmail.com) 
! !! This version uses a different dimension order (g-points first) and also includes
! !! other changes which aim to increase efficiency. See https://doi.org/10.1029/2020MS002226
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
  use mo_rte_kind, only: wp, dp, sp, wl
  use mo_fluxes_broadband_kernels, only : sum_broadband, sum_broadband_nocol
  use mo_rte_rrtmgp_config, only: compute_Jac, use_Pade_source
  ! TEMP. CODE FOR ML EXPERIMENTS
  use mod_network,      only: network_type, output_sgemm_flat_byrows                             
  ! TEMP. CODE FOR ML EXPERIMENTS

#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop
#endif
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
            lw_source_2str, sw_source_2str, &
            lw_two_stream, sw_two_stream, &
            adding


  real(wp), parameter :: pi = acos(-1._wp)
  
#ifdef NGPT 
integer, parameter :: ngpt = NGPT
#else
#define ngpt ngpt_in
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
  ! real(wp), parameter :: k_min = 1.e-3_wp 
  ! real(wp), parameter :: k_min = 1.e4_wp * epsilon(1._wp)
#endif

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
#else
#define exp_fast exp
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
  subroutine lw_solver_noscat(ngpt_in, nlay_in, ncol, top_at_1, nmus, D, weight, inc_flux, &
                              tau, lay_source, lev_source, &
                              sfc_emis, sfc_source, &
                              flux_up, flux_dn, &
                              sfc_source_Jac, flux_up_Jac, &
                              do_rescaling, ssa, g, &
                              save_gpt_flux, flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac) bind(C, name="lw_solver_noscat")
    integer,                                intent(in   ) ::  ngpt_in, nlay_in, ncol ! Number of g-points, layers, columns
    logical(wl),                            intent(in   ) ::  top_at_1 ! 
    integer,                                intent(in   ) ::  nmus         ! number of quadrature angles
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  D               ! secant of propagation angle  []
    real(wp),                               intent(in   ) ::  weight          ! quadrature weight
    real(wp), dimension(ngpt,ncol),         intent(in   ) ::  inc_flux        ! incident flux at domain top [W/m2] (ngpts, ncol)
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  tau             ! Absorption optical thickness []
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) ::  lay_source      ! Planck source at layer average temperature [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(in   ) ::  lev_source      ! Planck source at layer edges [W/m2]
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_emis        ! Surface emissivity      []
    real(wp), dimension(ngpt,       ncol),  intent(in   ) ::  sfc_source      ! Surface source function by band [W/m2]
    ! Outputs
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_up      ! Broadband fluxes [W/m2-str]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_dn      ! Top level must contain incident flux boundary condition
    !
    ! Optional variables - arrays aren't referenced if corresponding logical  == False
    !
    real(wp), dimension(:,:),               intent(out  ) :: flux_up_Jac 
    real(wp), dimension(:,:),               intent(in   ) :: sfc_source_Jac  ! Jacobian of surface source function  [W/m2/K] (ngpt,ncol)
    logical(wl),                            intent(in   ) :: do_rescaling
    real(wp), dimension(:,:,:),             intent(in   ) :: ssa, g    ! single-scattering albedo, asymmetry parameter] (ngpt,nlay,ncol)
    logical(wl),                            intent(in   ) :: save_gpt_flux
    real(wp), dimension(:,:,:), contiguous, target,    &
                                            intent(out  ) :: flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac
    ! ------------------------------------
    ! Local variables. no col dependency
    real(wp), dimension(:,:), contiguous, pointer   ::  radn_up, radn_dn, radn_up_Jac ! Radiances per g-point [W/m2-str]
    real(wp), dimension(ngpt,nlay+1),     target    ::  radn_dn_arr, radn_up_arr
    real(wp), dimension(ngpt,nlay+1),     target    ::  radn_up_Jac_arr   ! surface temperature Jacobian of g-point radiances [W/m2-str / K]
    real(wp), dimension(ngpt,nlay)                  ::  tau_loc      ! path length (tau/mu)
    real(wp), dimension(ngpt,nlay)                  ::  trans        ! transmissivity  = exp_fast(-tau)
    real(wp), dimension(ngpt,nlay)                  ::  source_up, source_dn

    real(wp), parameter :: pi = acos(-1._wp)
    real(wp)            :: fac, sums_up(4), sums_dn(4)
    integer             :: ilev, icol, igpt, ilay, top_level, sfc_level
    ! Used when approximating scattering
    real(wp), dimension(:,:), allocatable :: An, Cn
    real(wp) :: wb, ssal, scaleTau
    ! ------------------------------------

#ifdef USE_TIMING
    ret =  gptlstart('lw_solver_noscat')
#endif
    ! Where it the top: at index 1 if top_at_1 true, otherwise nlay+1
    if(top_at_1) then
      top_level = 1
      sfc_level = nlay+1
    else
      top_level = nlay+1
      sfc_level = 1
    end if

    if (do_rescaling) then
      allocate(An(ngpt,nlay), Cn(ngpt,nlay))
    end if

    if (.not.(save_gpt_flux)) then ! fluxes by g-point not needed, use local 2D arrays instead
      radn_dn => radn_dn_arr
      radn_up => radn_up_arr
      if (compute_Jac) radn_up_Jac => radn_up_Jac_arr
    end if

    do icol = 1, ncol

      if (save_gpt_flux) then
        radn_dn => flux_dn_gpt(:,:,icol)
        radn_up => flux_up_gpt(:,:,icol)
        if (compute_Jac) radn_up_Jac => flux_up_gpt_Jac(:,:,icol)
      end if

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
      if (do_rescaling) then
        !
        ! The scaling and scaleTau terms are independent of propagation
        !   angle D and could be pre-computed if several values of D are used
        ! We re-compute them here to keep not have to localize memory use
        !
        do ilay = 1, nlay
          do igpt = 1, ngpt
            ssal = ssa(igpt,ilay,icol)
            wb = ssal*(1._wp - g(igpt,ilay,icol)) * 0.5_wp
            scaleTau = (1._wp - ssal + wb)
            ! here wb/scaleTau is parameter wb/(1-w(1-b)) of Eq.21 of the Tang paper
            ! actually it is in line of parameter rescaling defined in Eq.7
            ! potentialy if g=ssa=1  then  wb/scaleTau = NaN
            ! it should not happen because g is never 1 in atmospheres
            ! explanation of factor 0.4 note A of Table
            Cn(igpt,ilay) = 0.4_wp*wb/scaleTau
            ! Eq.15 of the paper, multiplied by path length
            tau_loc(igpt,ilay) = tau(igpt,ilay,icol)*D(igpt,icol)*scaleTau
          end do
          trans  (:,ilay) = exp(-tau_loc(:,ilay))
          An(:,ilay) = (1._wp-trans(:,ilay)**2)
        end do
      else
        do ilay = 1, nlay
          tau_loc(:,ilay)  = tau(:,ilay,icol) * D(:,icol)
          trans(:,ilay)    = exp_fast(-tau_loc(:,ilay)) 
        end do
      end if
#ifdef USE_TIMING
    ret =  gptlstop('compute_trans_exp()')
#endif  

#ifdef USE_TIMING
    ret =  gptlstart('lw_source_transport_noscat')
#endif  
      ! !
      ! ! Source function for diffuse radiation, plus transport without scattering
      ! !
      ! call lw_sources_transport_noscat_dn(ngpt, nlay, top_at_1, &
      !                     lay_source(:,:,icol), lev_source(:,:,icol), &
      !                     tau_loc, trans, source_up, source_dn, radn_dn) 
      !

      ! Source function for diffuse radiation
      !
      call lw_source_noscat(ngpt, nlay, &
                            lay_source(:,:,icol), lev_source(:,:,icol), &
                            tau_loc, trans, source_dn, source_up)
      !
      ! Transport down
      !
      call lw_transport_noscat_dn(ngpt, nlay, top_at_1, trans, source_dn, radn_dn)
 
#ifdef USE_TIMING
    ret =  gptlstop('lw_source_transport_noscat')
#endif
      ! Surface reflection and emission                                     albedo
      radn_up (:,sfc_level)                     = radn_dn(:,sfc_level)*(1-sfc_emis(:,icol)) + sfc_emis(:,icol) *  sfc_source(:,icol)
      if (compute_Jac) radn_up_Jac(:,sfc_level) = sfc_emis(:,icol) * sfc_source_Jac(:,icol)

      !
      ! Transport up, or up and down again if using rescaling
      !
      if(do_rescaling) then
        call lw_transport_1rescl(ngpt, nlay, top_at_1, trans, source_dn, source_up, &
                                radn_up, radn_dn, An, Cn, radn_up_Jac) 
      else
        call lw_transport_noscat_up(ngpt, nlay, top_at_1, trans, & 
                                source_up, radn_up, radn_up_Jac)  
      end if
  
      !
      ! Convert intensity to flux assuming azimuthal isotropy and quadrature weight
      !
      fac         = 2._wp * pi * weight
      if (nmus/=1) then
        radn_dn     = fac * radn_dn   
        radn_up     = fac * radn_up   
        if (compute_Jac) radn_up_Jac = fac * radn_up_Jac
      end if

#ifdef USE_TIMING
    ret =  gptlstart('spectral_reduction')
#endif
      ! Inline the computation of broadband fluxes
      if (nmus==1) then ! ..but only if the number of quadrature angles is 1, otherwise do this within lw_solver_noscat_GaussQuad

        ! flux_up(:,icol) = sum(radn_up, 1)
        ! flux_dn(:,icol) = sum(radn_dn, 1)
        if (mod(ngpt,4) == 0)  then
          do ilay = 1, nlay+1
            sums_up = 0.0_wp
            sums_dn = 0.0_wp
            do igpt = 1, ngpt, 4
              sums_up(1) = sums_up(1) + fac*radn_up(igpt,   ilay); sums_up(2) = sums_up(2) + fac*radn_up(igpt+1, ilay)
              sums_up(3) = sums_up(3) + fac*radn_up(igpt+2, ilay); sums_up(4) = sums_up(4) + fac*radn_up(igpt+3, ilay)

              sums_dn(1) = sums_dn(1) + fac*radn_dn(igpt,   ilay); sums_dn(2) = sums_dn(2) + fac*radn_dn(igpt+1, ilay)
              sums_dn(3) = sums_dn(3) + fac*radn_dn(igpt+2, ilay); sums_dn(4) = sums_dn(4) + fac*radn_dn(igpt+3, ilay)
            end do
            flux_up(ilay,icol) = sums_up(1) + sums_up(2) + sums_up(3) + sums_up(4)
            flux_dn(ilay,icol) = sums_dn(1) + sums_dn(2) + sums_dn(3) + sums_dn(4)
          end do
        else 
          flux_up(:,icol) = sum(radn_up, 1)
          flux_dn(:,icol) = sum(radn_dn, 1)
        end if
        if (compute_Jac) flux_up_Jac(:,icol) = sum(radn_up_Jac, 1)
      end if
#ifdef USE_TIMING
    ret =  gptlstop('spectral_reduction')
#endif
    end do  ! column loop

#ifdef USE_TIMING
    ret =  gptlstop('lw_solver_noscat')
#endif

  end subroutine lw_solver_noscat

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
    !
    ! Optional variables - arrays aren't referenced if corresponding logical  == False
    !
    real(wp), dimension(:,:),               intent(out  ) :: flux_up_Jac 
    real(wp), dimension(:,:),               intent(in   ) :: sfc_source_Jac  ! Jacobian of surface source function  [W/m2/K] (ngpt,ncol)
    logical(wl),                            intent(in   ) :: do_rescaling
    real(wp), dimension(:,:,:),             intent(in   ) :: ssa, g    ! single-scattering albedo, asymmetry parameter] (ngpt,nlay,ncol)
    logical(wl),                            intent(in   ) :: save_gpt_flux
    real(wp), dimension(:,:,:), contiguous, target,    &
                                            intent(out  ) :: flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac    ! Local variables
    real(wp), dimension(ngpt, ncol)             :: Ds_ngpt
    real(wp), dimension(:,:,:),  allocatable      :: radn_up, radn_dn ! Fluxes per quad angle  (nlay+1, ncol)
    real(wp), dimension(:,:,:),  allocatable      :: radn_up_Jac      ! perturbed Fluxes per quad angle
    integer :: imu, icol, igpt, ilay
    ! ------------------------------------
    !
    ! For the first angle output arrays store total flux
    !

    Ds_ngpt(:,:) = Ds(1)
  
    call lw_solver_noscat(ngpt, nlay, ncol, top_at_1, &
                          nmus, Ds_ngpt, weights(1), inc_flux, &
                          tau, lay_source, lev_source,&
                          sfc_emis, sfc_source,  &
                          flux_up, flux_dn,      &
                          ! optional variables
                          sfc_source_Jac, flux_up_Jac, &
                          do_rescaling, ssa, g,&
                          save_gpt_flux, flux_up_gpt, flux_dn_gpt, flux_up_gpt_Jac)

    if (nmus > 1) then
       ! if nmus is 1 the broadband fluxes are inlined in lw_solver_noscat,
      ! but for more angles we only want to do the reduction once
      ! in this case we need gpt fluxes (save_gpt_flux is on) and we do the reduction below
      ! we also need local arrays
      allocate( radn_up(ngpt, nlay+1, ncol) )
      allocate( radn_dn(ngpt, nlay+1, ncol) )
      if (compute_Jac) allocate( radn_up_Jac(ngpt, nlay+1, ncol) )

      do imu = 2, nmus

        Ds_ngpt(:,:) = Ds(imu)

        call lw_solver_noscat(ngpt, nlay, ncol, top_at_1, &
          nmus, Ds_ngpt, weights(imu), inc_flux, &
          tau, lay_source, lev_source, &
          sfc_emis, sfc_source, &
          flux_up, flux_dn, sfc_source_Jac, flux_up_Jac, &
          do_rescaling, ssa, g,&
          save_gpt_flux, radn_up, radn_dn, radn_up_Jac )

        flux_up_gpt = flux_up_gpt  + radn_up
        flux_dn_gpt = flux_dn_gpt  + radn_dn
        if (compute_Jac) flux_up_gpt_Jac = flux_up_gpt_Jac + radn_up_Jac
      end do      
      
      call sum_broadband(ngpt, nlay+1, ncol, flux_up_gpt, flux_up)
      call sum_broadband(ngpt, nlay+1, ncol, flux_dn_gpt, flux_dn)
      if (compute_Jac)  call sum_broadband(ngpt, nlay+1, ncol, flux_up_gpt_Jac, flux_up_Jac)
    end if


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
    real(wp), dimension(ngpt,ncol),         intent(in   ) ::  sfc_source  ! Surface source function [W/m2]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_up      ! Broadband fluxes [W/m2-str]
    real(wp), dimension(nlay+1,     ncol),  intent(out)   ::  flux_dn      ! 
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(out)   ::  flux_up_gpt, flux_dn_gpt ! G-point fluxes [W/m2]
    ! ----------------------------------------------------------------------
    integer :: icol, top_level
    real(wp), dimension(ngpt,nlay  ) :: Rdif, Tdif, gamma1, gamma2
    real(wp), dimension(ngpt       ) :: sfc_albedo
    real(wp), dimension(ngpt,nlay  ) :: source_dn, source_up
    real(wp), dimension(ngpt       ) :: source_sfc
    ! ------------------------------------

    top_level = MERGE(1, nlay+1, top_at_1)

    do icol = 1, ncol
            ! Apply boundary condition
      flux_dn_gpt(:,top_level,icol) = inc_flux(:,icol)

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
                          sfc_emis(:,icol), sfc_source(:,icol), &
                          lay_source(:,:,icol), lev_source(:,:,icol), &
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
                  flux_up_gpt(:,:,icol), flux_dn_gpt(:,:,icol))

      call sum_broadband_nocol(ngpt, nlay+1, flux_up_gpt(:,:,icol), flux_up(:,icol) )
      call sum_broadband_nocol(ngpt, nlay+1, flux_dn_gpt(:,:,icol), flux_dn(:,icol) )           
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
                              top_at_1, tau, mu0, flux_dir, flux_dir_bb) bind (C, name="sw_solver_noscat")
    integer,                    intent(in   ) :: ngpt, nlay, ncol ! Number of columns, layers, g-points
    logical(wl),                intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   )   :: tau          ! Absorption optical thickness []
    real(wp), dimension(ncol            ),  intent(in   )   :: mu0          ! cosine of solar zenith angle
    real(wp), dimension     (nlay+1,ncol),  intent(out)     :: flux_dir_bb  ! Direct-beam flux, broadband [W/m2]
    real(wp), dimension(ngpt,nlay+1,ncol),  intent(inout)   :: flux_dir     ! Direct-beam flux, spectral [W/m2]
                                                                          ! Top level must contain incident flux boundary condition
    integer :: igpt, ilev, icol
    real(wp), dimension(ncol) :: mu0_inv
    ! ------------------------------------
    mu0_inv = 1._wp/mu0
    ! Indexing into arrays for upward and downward propagation depends on the vertical
    !   orientation of the arrays (whether the domain top is at the first or last index)
    ! We write the loops out explicitly so compilers will have no trouble optimizing them.
    ! Downward propagation
    do icol = 1, ncol
      if(top_at_1) then
        ! For the flux at this level, what was the previous level, and which layer has the
        !   radiation just passed through?
        ! layer index = level index - 1
        ! previous level is up (-1)
        do ilev = 2, nlay+1
          flux_dir(:,ilev,icol) = flux_dir(:,ilev-1,icol) * exp_fast(-tau(:,ilev-1,icol)*mu0_inv(icol))
        end do
      else
        ! layer index = level index
        ! previous level is up (+1)
        do ilev = nlay, 1, -1
          flux_dir(:,ilev,icol) = flux_dir(:,ilev+1,icol) * exp_fast(-tau(:,ilev,icol)*mu0_inv(icol))
        end do
      end if
      ! Compute broadband fluxes
      call sum_broadband_nocol(ngpt, nlay+1, flux_dir, flux_dir_bb(:,icol) )
    end do
  end subroutine sw_solver_noscat
  ! -------------------------------------------------------------------------------------------------
  !
  ! Shortwave two-stream calculation:
  !   compute layer reflectance, transmittance
  !   compute solar source function for diffuse radiation
  !   transport
  !
  ! -------------------------------------------------------------------------------------------------
  subroutine sw_solver_2stream(ngpt_in, nlay_in, ncol, top_at_1, &
                                 inc_flux, inc_flux_dif,     &
                                 tau, ssa, g, mu0,           &
                                 sfc_alb_dir, sfc_alb_dif,   &
                                 flux_up, flux_dn, flux_dir, &
                                 flux_up_gpt, flux_dn_gpt, flux_dir_gpt, &
                                 reftrans_variables, neural_net & !!TEMPORARY CODE FOR ML EXPERIMENTS!!
                                 ) 
    integer,                                intent(in   ) :: ngpt_in, nlay_in, ncol ! Number of columns, layers, g-points
    logical(wl),                            intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,       ncol),  intent(in   ) :: inc_flux, inc_flux_dif     ! incident flux at top of domain [W/m2] (ngpt, ncol)
    real(wp), dimension(ngpt,nlay,  ncol),  intent(in   ) :: tau, &  ! Optical thickness,
                                                            ssa, &  ! single-scattering albedo,
                                                            g       ! asymmetry parameter []
    real(wp), dimension(            ncol),  intent(in   ) :: mu0     ! cosine of solar zenith angle
    real(wp), dimension(ngpt,       ncol),  intent(in   ) :: sfc_alb_dir, sfc_alb_dif
                                                                    ! Spectral albedo of surface to direct and diffuse radiation
    real(wp), dimension(nlay+1,ncol),       intent(out) :: flux_up, flux_dn, flux_dir ! Broadband fluxes  [W/m2]
    real(wp), dimension(ngpt, nlay+1, ncol), optional,target,    &  ! G-point fluxes - optional output
                                            intent(out) :: flux_up_gpt, flux_dn_gpt, flux_dir_gpt 
    real(wp), dimension(ngpt, nlay, ncol, 4), optional, target, intent(inout) :: reftrans_variables !!TEMP. CODE FOR ML EXPERIMENTS      
    type(network_type),                      optional,          intent(in)    :: neural_net 
                                 
    ! -------------------------------------------
    real(wp), dimension(:,:), contiguous, pointer   ::  radn_up, radn_dn, radn_dir             ! G-point fluxes [W/m2], local array
    real(wp), dimension(ngpt,nlay+1),     target    ::  radn_up_arr, radn_dn_arr, radn_dir_arr ! G-point fluxes [W/m2], pointer
    integer :: icol, igpt, ilay, top_level, j
    real(wp)            :: sums_dir(4), sums_up(4), sums_dn(4)

    !real(wp), dimension(ngpt,nlay) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    !!TEMPORARY CODE FOR ML EXPERIMENTS!!
    real(wp), dimension(ngpt,nlay) :: Tnoscat
    real(wp), dimension(ngpt,nlay), target :: Rdif_arr, Tdif_arr, Rdir_arr, Tdir_arr
    real(wp), dimension(:,:), contiguous, pointer   :: Rdif, Tdif, Rdir, Tdir
    ! real(wp), dimension(ngpt,nlay,ncol), target :: Rdif_col, Tdif_col, Rdir_col, Tdir_col, Tnoscat_col
    real(sp), dimension(:,:,:), allocatable, target :: reftrans_variables_nocol
    real(sp), dimension(:,:),   contiguous, pointer     :: nn_output   
    !!TEMPORARY CODE FOR ML, EXPERIMENTS!!
    real(wp), dimension(ngpt,nlay) :: source_up, source_dn
    real(wp), dimension(ngpt     ) :: source_srf
    logical(wl) :: save_gpt_flux = .false.
    logical(wl) :: compare_reftrans = .false.
    real(wp), dimension(:,:,:,:), allocatable :: reftrans_true, reftrans_pred
    ! ------------------------------------

    if (compare_reftrans) then
      allocate(reftrans_true(ngpt,nlay,ncol,4), reftrans_pred(ngpt,nlay,ncol,4))
    end if

    top_level = MERGE(1, nlay+1, top_at_1)

    if (present(flux_up_gpt)) save_gpt_flux = .true.

    if (.not.(save_gpt_flux)) then ! fluxes by g-point not needed, use local 2D arrays instead
      radn_up => radn_up_arr
      radn_dn => radn_dn_arr
      radn_dir => radn_dir_arr
    end if

    if (present(neural_net) .and. .not. present(reftrans_variables)) then 
      allocate(reftrans_variables_nocol(ngpt,nlay,4))
      call C_F_POINTER (C_LOC(reftrans_variables_nocol), nn_output, [ngpt*nlay,4])
    end if
#ifdef USE_TIMING
    ret =  gptlstart('sw_2stream')
#endif 
    do icol = 1, ncol

      !!TEMPORARY CODE FOR ML EXPERIMENTS!!
      if (present(neural_net)) then
#ifndef DOUBLE_PRECISION
        Rdif => reftrans_variables_nocol(:,:,1)
        Tdif => reftrans_variables_nocol(:,:,2)
        Rdir => reftrans_variables_nocol(:,:,3)
        Tdir => reftrans_variables_nocol(:,:,4)
#endif
      else
        Rdif => Rdif_arr
        Tdif => Tdif_arr
        Rdir => Rdir_arr
        Tdir => Tdir_arr
      end if
      if (present(reftrans_variables)) then ! inout reftrans, assumed kernel not called with neural_net
        Rdif => reftrans_variables(:,:,icol,1)
        Tdif => reftrans_variables(:,:,icol,2)
        Rdir => reftrans_variables(:,:,icol,3)
        Tdir => reftrans_variables(:,:,icol,4)
      end if
      !!TEMPORARY CODE FOR ML EXPERIMENTS!!

      if (save_gpt_flux) then
        radn_up => flux_up_gpt(:,:,icol)
        radn_dn => flux_dn_gpt(:,:,icol)
        radn_dir => flux_dir_gpt(:,:,icol)
      end if

      ! Apply boundary condition
      radn_dir(:,top_level) = inc_flux(:,icol) * mu0(icol)
      radn_dn(:,top_level)  = inc_flux_dif(:,icol)
      !
      ! Cell properties: transmittance and reflectance for direct and diffuse radiation
      !
! #ifdef USE_TIMING
!     ret =  gptlstart('sw_two_stream')
! #endif

      ! if (present(neural_net)) then
      !   Tnoscat = exp_fast(-tau(:,:,icol)*(1/mu0(icol)))
      !  ! call NN code to predict other reftrans variables
      !  call predict_nn_reftrans(nlay, ngpt, &
      !           neural_net,         &
      !           tau(:,:,icol), ssa(:,:,icol), g(:,:,icol), Tnoscat, mu0(icol), &
      !           nn_output)

      !   if (compare_reftrans) then
      !     call sw_two_stream(ngpt, nlay, mu0(icol),                                &
      !     tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), &
      !     Rdif_arr, Tdif_arr, Rdir_arr, Tdir_arr, Tnoscat)
      !     Rdif_arr = min(max(0.0_wp, Rdif_arr),1.0_wp)
      !     Tdif_arr = min(max(0.0_wp, Tdif_arr),1.0_wp)
      !     Rdir_arr = min(max(0.0_wp, Rdir_arr),1.0_wp)
      !     Tdir_arr = min(max(0.0_wp, Tdir_arr),1.0_wp)
      !     reftrans_true(:,:,icol,1) = Rdif_arr
      !     reftrans_true(:,:,icol,2) = Tdif_arr
      !     reftrans_true(:,:,icol,3) = Rdir_arr
      !     reftrans_true(:,:,icol,4) = Tdir_arr
      !     reftrans_pred(:,:,icol,1) = Rdif
      !     reftrans_pred(:,:,icol,2) = Tdif
      !     reftrans_pred(:,:,icol,3) = Rdir
      !     reftrans_pred(:,:,icol,4) = Tdir
      !   ! if (icol < 3) then 
      !   !   print *, "pred", Rdif(1,1), Tdif(1,1), Rdir(1,1), Tdir(1,1)
      !   !   print *, "pred1", Rdif(ngpt,1), Tdif(ngpt,1), Rdir(ngpt,1), Tdir(ngpt,1)
      !   !   print *, "pred2", Rdif(ngpt,nlay), Tdif(ngpt,nlay), Rdir(ngpt,nlay), Tdir(ngpt,nlay)

      !     ! print *, "true", Rdif_arr(1,1), Tdif_arr(1,1), Rdir_arr(1,1), Tdir_arr(1,1)
      !     ! print *, "true1", Rdif_arr(ngpt,1), Tdif_arr(ngpt,1), Rdir_arr(ngpt,1), Tdir_arr(ngpt,1)
      !     ! print *, "true2", Rdif_arr(ngpt,nlay), Tdif_arr(ngpt,nlay), Rdir_arr(ngpt,nlay), Tdir_arr(ngpt,nlay)
      !   ! end if
      !   ! Rdif = Rdif_arr
      !   ! Tdif = Tdif_arr
      !   ! Rdir = Rdir_arr
      !   ! Tdir = Tdir_arr
      !   end if

      ! else
      !   call sw_two_stream(ngpt, nlay, mu0(icol),                                &
      !                     tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), &
      !                     Rdif, Tdif, Rdir, Tdir, Tnoscat)
      !   ! Rdif = min(max(0.0_wp, Rdif),1.0_wp)
      !   ! Tdif = min(max(0.0_wp, Tdif),1.0_wp)
      !   ! Rdir = min(max(0.0_wp, Rdir),1.0_wp)
      !   ! Tdir = min(max(0.0_wp, Tdir),1.0_wp)
      ! end if
      ! end if     
! #ifdef USE_TIMING
!     ret =  gptlstop('sw_two_stream')
! #endif    
      !
      ! Direct-beam and source for diffuse radiation
      !
! #ifdef USE_TIMING
!     ret =  gptlstart('sw_source_2str')
! #endif
!       call sw_source_2str(ngpt, nlay, top_at_1, Rdir, Tdir, Tnoscat, sfc_alb_dir(:,icol),&
!                           source_up, source_dn, source_srf, radn_dir)
! #ifdef USE_TIMING
!     ret =  gptlstop('sw_source_2str')
! #endif
#ifdef USE_TIMING
    ret =  gptlstart('sw_two_stream_source')
#endif 
      call sw_two_stream_source(ngpt, nlay, top_at_1, mu0(icol),                                &
      tau (:,:,icol), ssa (:,:,icol), g(:,:,icol), sfc_alb_dir(:,icol), &
      Rdif, Tdif, source_up, source_dn, radn_dir, source_srf)     
#ifdef USE_TIMING
    ret =  gptlstop('sw_two_stream_source')
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

#ifdef USE_TIMING
    ret =  gptlstart('sum_broadband_nocol')
#endif  
      ! Compute broadband fluxes
      ! Here doing the reduction manually for different fluxes within a single loop, and combining this  
      ! with unrolling the inner loop, can greatly improve instruction-level parallelism
      if (mod(ngpt,4) == 0)  then
        do ilay = 1, nlay+1
          sums_up = 0.0_wp; sums_dn = 0.0_wp; sums_dir = 0.0_wp
          do igpt = 1, ngpt, 4

            ! sums_up(1) = sums_up(1) + radn_up(igpt,   ilay); sums_up(2) = sums_up(2) + radn_up(igpt+1, ilay)
            ! sums_up(3) = sums_up(3) + radn_up(igpt+2, ilay); sums_up(4) = sums_up(4) + radn_up(igpt+3, ilay)

            ! sums_dir(1) = sums_dir(1) + radn_dir(igpt,   ilay); sums_dir(2) = sums_dir(2) + radn_dir(igpt+1, ilay)
            ! sums_dir(3) = sums_dir(3) + radn_dir(igpt+2, ilay); sums_dir(4) = sums_dir(4) + radn_dir(igpt+3, ilay)

            ! radn_dn(igpt, ilay) = radn_dn(igpt, ilay) + radn_dir(igpt, ilay)
            ! radn_dn(igpt+1, ilay) = radn_dn(igpt+1, ilay) + radn_dir(igpt+1, ilay)
            ! radn_dn(igpt+2, ilay) = radn_dn(igpt+2, ilay) + radn_dir(igpt+2, ilay)
            ! radn_dn(igpt+3, ilay) = radn_dn(igpt+3, ilay) + radn_dir(igpt+3, ilay)

            ! sums_dn(1) = sums_dn(1) + radn_dn(igpt,   ilay); sums_dn(2) = sums_dn(2) + radn_dn(igpt+1, ilay)
            ! sums_dn(3) = sums_dn(3) + radn_dn(igpt+2, ilay); sums_dn(4) = sums_dn(4) + radn_dn(igpt+3, ilay)
            do j = 1,  4
              ! Upward flux
              sums_up(j) = sums_up(j) + radn_up(igpt+(j-1), ilay)
              ! Downward direct flux
              sums_dir(j) = sums_dir(j) + radn_dir(igpt+(j-1), ilay)
  
              if (save_gpt_flux) then
                ! adding computes only diffuse flux; flux_dn is total
                radn_dn(igpt+(j-1), ilay) = radn_dn(igpt+(j-1), ilay) + radn_dir(igpt+(j-1), ilay)
                ! Downward total flux
                sums_dn(j) = sums_dn(j) + radn_dn(igpt+(j-1), ilay)
              else
                sums_dn(j) = sums_dn(j) + radn_dn(igpt+(j-1), ilay) + radn_dir(igpt+(j-1), ilay)
              end if
            end do
          end do
          flux_up(ilay,icol) = sums_up(1) + sums_up(2) + sums_up(3) + sums_up(4)
          flux_dn(ilay,icol) = sums_dn(1) + sums_dn(2) + sums_dn(3) + sums_dn(4)
          flux_dir(ilay,icol) = sums_dir(1) + sums_dir(2) + sums_dir(3) + sums_dir(4)
        end do
      else 
        radn_dn = radn_dn + radn_dir
        flux_dir(:,icol) = sum(radn_dir, 1)
        flux_up(:,icol) = sum(radn_up, 1)
        flux_dn(:,icol) = sum(radn_dn, 1)
      end if
#ifdef USE_TIMING
    ret =  gptlstop('sum_broadband_nocol')
#endif    
    end do
#ifdef USE_TIMING
    ret =  gptlstop('sw_2stream')
#endif 
    ! if (compare_reftrans) then
    !     print *, "mae Rdif",  mae_3d(reftrans_true(:,:,:,1),reftrans_pred(:,:,:,1))
    !     print *, "Tdif",      mae_3d(reftrans_true(:,:,:,2),reftrans_pred(:,:,:,2))
    !     print *, "Rdir",      mae_3d(reftrans_true(:,:,:,3),reftrans_pred(:,:,:,3))
    !     print *, "Tdir",      mae_3d(reftrans_true(:,:,:,4),reftrans_pred(:,:,:,4))

    !     print *, "bias Rdif",  mean_3d(reftrans_true(:,:,:,1)) - mean_3d(reftrans_pred(:,:,:,1))
    !     print *, "Tdif",      mean_3d(reftrans_true(:,:,:,2)) - mean_3d(reftrans_pred(:,:,:,2))
    !     print *, "Rdir",      mean_3d(reftrans_true(:,:,:,3)) - mean_3d(reftrans_pred(:,:,:,3))
    !     print *, "Tdir",      mean_3d(reftrans_true(:,:,:,4)) - mean_3d(reftrans_pred(:,:,:,4))

    !     print *, "maxdiff Rdif",  maxval(abs(reftrans_true(:,:,:,1)) - reftrans_pred(:,:,:,1))
    !     print *, "Tdif",      maxval(abs(reftrans_true(:,:,:,2)) - reftrans_pred(:,:,:,2))
    !     print *, "Rdir",      maxval(abs(reftrans_true(:,:,:,3)) - reftrans_pred(:,:,:,3))
    !     print *, "Tdir",      maxval(abs(reftrans_true(:,:,:,4)) - reftrans_pred(:,:,:,4))

    !     print *, "maxvals..", maxval(reftrans_true(:,:,:,1)), maxval(reftrans_true(:,:,:,2)),&
    !      maxval(reftrans_true(:,:,:,3)), maxval(reftrans_true(:,:,:,4))
    ! end if

  end subroutine sw_solver_2stream
  ! -------------------------------------------------------------------------------------------------
  !
  !   Lower-level longwave kernels
  !
  ! -------------------------------------------------------------------------------------------------
  !
  ! Compute LW source function for upward and downward emission at levels using linear-in-tau assumption
  ! See Clough et al., 1992, doi: 10.1029/92JD01419, Eq 15
  !
  ! ---------------------------------------------------------------
  ! subroutine lw_source_noscat(ngpt, nlay, lay_source, lev_source_up, lev_source_dn, tau, trans, &
  !                             source_dn, source_up) bind(C, name="lw_source_noscat")
  !   integer,                         intent(in) :: ngpt, nlay
  !   real(wp), dimension(ngpt, nlay), intent(in) :: lay_source, & ! Planck source at layer center
  !                                                  lev_source_up, & ! Planck source at levels (layer edges),
  !                                                  lev_source_dn, & !   increasing/decreasing layer index
  !                                                  tau,        & ! Optical path (tau/mu)
  !                                                  trans         ! Transmissivity (exp_fast(-tau))
  !   real(wp), dimension(ngpt, nlay), intent(out):: source_dn, source_up
  !                                                                  ! Source function at layer edges
  !                                                                  ! Down at the bottom of the layer, up at the top
  !   ! --------------------------------
  !   integer             :: igpt, ilay
  !   real(wp)            :: fact
  !   real(wp), parameter :: tau_thresh = sqrt(epsilon(tau))
  !   ! ---------------------------------------------------------------
  !   do ilay = 1, nlay
  !     do igpt = 1, ngpt
  !     !
  !     ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
  !     !   is of order epsilon (smallest difference from 1. in working precision)
  !     !   Thanks to Peter Blossey
  !     !
  !     if(tau(igpt, ilay) > tau_thresh) then
  !       fact = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
  !     else
  !       fact = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
  !     end if
  !     !
  !     ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
  !     !
  !     source_dn(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_dn(igpt,ilay) + &
  !                             2._wp * fact * (lay_source(igpt,ilay) - lev_source_dn(igpt,ilay))
  !     source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source_up(igpt,ilay  ) + &
  !                             2._wp * fact * (lay_source(igpt,ilay) - lev_source_up(igpt,ilay))
  !     end do
  !   end do
    
  ! end subroutine lw_source_noscat
  subroutine lw_source_noscat(ngpt, nlay, lay_source, lev_source, tau, trans, &
                              source_dn, source_up) bind(C, name="lw_source_noscat")
    integer,                         intent(in) :: ngpt, nlay
    real(wp), dimension(ngpt, nlay), intent(in) :: lay_source, & ! Planck source at layer center
                                                   tau,        & ! Optical path (tau/mu)
                                                   trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt,nlay+1), intent(in )   :: lev_source
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
      ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
      source_dn(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay+1) + &
                            2._wp * fact * (lay_source(igpt,ilay) - lev_source(igpt,ilay+1))
      source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay) + &
                            2._wp * fact * (lay_source(igpt,ilay) - lev_source(igpt,ilay))
      end do
    end do
    
  end subroutine lw_source_noscat
  ! ---------------------------------------------------------------
  !
  ! Longwave no-scattering transport downward, with source computation inlined to improve efficiency
  !
  ! ---------------------------------------------------------------
  subroutine lw_source_transport_noscat_dn(ngpt_in, nlay_in, top_at_1,  &  ! inputs
                    lay_source, lev_source,  tau, trans, &                ! inputs
                    source_up, radn_dn)                                   ! outputs
    integer,                          intent(in   ) :: ngpt_in, nlay_in
    logical(wl),                      intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay),   intent(in )   :: lay_source
    real(wp), dimension(ngpt,nlay+1), intent(in )   :: lev_source
    real(wp), dimension(ngpt, nlay),  intent(in)    :: tau,        & ! Optical path (tau/mu)
                                                      trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt, nlay),  intent(out)   :: source_up !  Down at the bottom of the layer, up at the top
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    ! Top level must contain incident flux boundary condition
    ! --------------------------------
    integer                         :: igpt, ilay, ibnd
    real(wp)                        :: coeff
    real(wp)                        :: source_dn
    real(wp), parameter             :: tau_thresh = sqrt(epsilon(tau))
    ! ---------------------------------------------------------------

    if(top_at_1) then

      do ilay = 1, nlay
        !dir$ vector aligned
        do igpt = 1,ngpt
          if (use_Pade_source) then
            ! Alternative to avoid the conditional (which may or may not have performance penalties, depending on the platform)
            ! Equation below uses a Pade approximant for the linear-in-tau solution for the effective Planck function
            ! See Clough et al., 1992, doi:10.1029/92JD01419, Eq 15
            coeff = 0.2_wp * tau(igpt,ilay)
            source_up(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay))   / (1 + coeff)
            source_dn             = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay+1)) / (1 + coeff)
          else
            !
            ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
            !   is of order epsilon (smallest difference from 1. in working precision)
            !   Thanks to Peter Blossey
            !
            if(tau(igpt, ilay) > tau_thresh) then
              coeff = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              coeff = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if
            
            ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
            source_dn           = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay+1) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay+1))
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay))
          end if
          ! Compute downward radiance
          radn_dn(igpt,ilay+1) = trans(igpt,ilay)*radn_dn(igpt,ilay) + source_dn
        end do ! gpt
      end do ! lay
      

    else  ! Top of domain is index nlay+1

      do ilay = nlay, 1, -1
        !dir$ vector aligned
        do igpt = 1,ngpt
          if (use_Pade_source) then
            ! Use Pade approximant
            coeff = 0.2_wp * tau(igpt,ilay)
            source_up(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay+1)) / (1 + coeff)
            source_dn             = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay))   / (1 + coeff)
          else
            if(tau(igpt, ilay) > tau_thresh) then
              coeff = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              coeff = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if
            
            source_dn           = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay))
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay+1) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay+1))
          end if
          radn_dn(igpt,ilay) = trans(igpt,ilay)*radn_dn(igpt,ilay+1) + source_dn
        end do
      end do
      
    end if

  end subroutine lw_source_transport_noscat_dn

  subroutine lw_sources_transport_noscat_dn(ngpt_in, nlay_in, top_at_1,  &  ! inputs
                    lay_source, lev_source,  tau, trans, &                ! inputs
                    source_up, source_dn, radn_dn)                                   ! outputs
    integer,                          intent(in   ) :: ngpt_in, nlay_in
    logical(wl),                      intent(in   ) :: top_at_1
    real(wp), dimension(ngpt,nlay),   intent(in )   :: lay_source
    real(wp), dimension(ngpt,nlay+1), intent(in )   :: lev_source
    real(wp), dimension(ngpt, nlay),  intent(in)    :: tau,        & ! Optical path (tau/mu)
                                                      trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt, nlay),  intent(out)   :: source_up, source_dn !  Down at the bottom of the layer, up at the top
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    ! Top level must contain incident flux boundary condition
    ! --------------------------------
    integer                         :: igpt, ilay, ibnd
    real(wp)                        :: coeff
    real(wp), parameter             :: tau_thresh = sqrt(epsilon(tau))
    ! ---------------------------------------------------------------

    if(top_at_1) then

      do ilay = 1, nlay
        !dir$ vector aligned
        do igpt = 1,ngpt
          if (use_Pade_source) then
            ! Alternative to avoid the conditional (which may or may not have performance penalties, depending on the platform)
            ! Equation below uses a Pade approximant for the linear-in-tau solution for the effective Planck function
            ! See Clough et al., 1992, doi:10.1029/92JD01419, Eq 15
            coeff = 0.2_wp * tau(igpt,ilay)
            source_up(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay))   / (1 + coeff)
            source_dn(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay+1)) / (1 + coeff)
          else
            !
            ! Weighting factor. Use 2nd order series expansion when rounding error (~tau^2)
            !   is of order epsilon (smallest difference from 1. in working precision)
            !   Thanks to Peter Blossey
            !
            if(tau(igpt, ilay) > tau_thresh) then
              coeff = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              coeff = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if
            
            ! Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
            source_dn(igpt,ilay)  = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay+1) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay+1))
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay))
          end if
          ! Compute downward radiance
          radn_dn(igpt,ilay+1) = trans(igpt,ilay)*radn_dn(igpt,ilay) + source_dn(igpt,ilay)
        end do ! gpt
      end do ! lay
      

    else  ! Top of domain is index nlay+1

      do ilay = nlay, 1, -1
        !dir$ vector aligned
        do igpt = 1,ngpt
          if (use_Pade_source) then
            ! Use Pade approximant
            coeff = 0.2_wp * tau(igpt,ilay)
            source_up(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay+1)) / (1 + coeff)
            source_dn(igpt,ilay)  = (1.0_wp-trans(igpt,ilay)) * (lay_source(igpt,ilay) + coeff*lev_source(igpt,ilay))   / (1 + coeff)
          else
            if(tau(igpt, ilay) > tau_thresh) then
              coeff = (1._wp - trans(igpt,ilay))/tau(igpt,ilay) - trans(igpt,ilay)
            else
              coeff = tau(igpt, ilay) * (0.5_wp - 1._wp/3._wp*tau(igpt, ilay))
            end if
            
            source_dn(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay))
            source_up(igpt,ilay) = (1._wp - trans(igpt,ilay)) * lev_source(igpt,ilay+1) + &
                                  2._wp * coeff * (lay_source(igpt,ilay) - lev_source(igpt,ilay+1))
          end if
          radn_dn(igpt,ilay) = trans(igpt,ilay)*radn_dn(igpt,ilay+1) + source_dn(igpt,ilay)
        end do
      end do
      
    end if

  end subroutine lw_sources_transport_noscat_dn

  subroutine lw_transport_noscat_up(ngpt_in, nlay_in, top_at_1,  & 
                    trans, source_up, radn_up, radn_up_Jac)  
    integer,                          intent(in   ) :: ngpt_in, nlay_in
    logical(wl),                      intent(in   ) :: top_at_1
    real(wp), dimension(ngpt, nlay),  intent(in)    :: trans         ! Transmissivity (exp_fast(-tau))
    real(wp), dimension(ngpt, nlay),  intent(in)    :: source_up
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(:,:),         intent(inout) :: radn_up_Jac ! surface temperature Jacobian of Radiances [W/m2-str / K]
    ! ---------------------------------------------------------------
    integer                         :: ilay
    ! ---------------------------------------------------------------

    if(top_at_1) then

      ! Upward propagation
      do ilay = nlay, 1, -1
        radn_up (:,ilay)                     = trans(:,ilay  )*radn_up    (:,ilay+1) + source_up(:,ilay)
        if (compute_Jac) radn_up_Jac(:,ilay) = trans(:,ilay  )*radn_up_Jac(:,ilay+1)
      end do

    else  ! Top of domain is index nlay+1

      ! Upward propagation
      do ilay = 2, nlay+1
        radn_up(:,ilay)                      = trans(:,ilay-1)*radn_up    (:,ilay-1) + source_up(:,ilay-1)
        if (compute_Jac) radn_up_Jac(:,ilay) = trans(:,ilay-1)*radn_up_Jac(:,ilay-1)
      end do

    end if

  end subroutine lw_transport_noscat_up

  subroutine lw_transport_noscat_dn(ngpt_in, nlay_in, top_at_1,     &
                                   trans, source_dn, radn_dn) bind(C, name="lw_transport_noscat_dn")
    integer,                          intent(in   ) :: ngpt_in, nlay_in ! Number of columns, layers, g-points
    logical(wl),                      intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: trans      ! transmissivity = exp(-tau)
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: source_dn  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    ! Radiances [W/m2-str] Top level must contain incident flux boundary condition

    ! ---------------------------------------------------
    ! Local variables
    integer :: ilev
    ! ---------------------------------------------------
    if(top_at_1) then
      !
      ! Top of domain is index 1
      !
      do ilev = 2, nlay+1
        radn_dn(:,ilev) = trans(:,ilev-1)*radn_dn(:,ilev-1) + source_dn(:,ilev-1)
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      do ilev = nlay, 1, -1
        radn_dn(:,ilev) = trans(:,ilev  )*radn_dn(:,ilev+1) + source_dn(:,ilev)
      end do
    end if
  end subroutine lw_transport_noscat_dn
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
                           k_min))
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
    integer,                           intent(in ) :: nlay, ngpt
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
    integer,                          intent(in) :: ngpt, nlay
    logical(wl),                      intent(in) :: top_at_1
    real(wp), dimension(ngpt      ),  intent(in) :: sfc_emis, sfc_src
    real(wp), dimension(ngpt, nlay),  intent(in) :: tau,           & ! Optical depth (tau)
                                                   gamma1, gamma2,& ! Coupling coefficients
                                                   rdif, tdif       ! Layer reflectance and transmittance
    real(wp), dimension(ngpt, nlay),  intent(in)  :: lay_source       ! Planck source at layer mean temp. (not used)                                                                         
    real(wp), dimension(ngpt, nlay+1), target, &
                                      intent(in)  :: lev_source       ! Planck source at layer edges
    real(wp), dimension(ngpt, nlay),  intent(out) :: source_dn, source_up
    real(wp), dimension(ngpt      ),  intent(out) :: source_sfc      ! Source function for upward radation at surface
    ! Local variables
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

    source_sfc(:) = pi * sfc_emis(:) * sfc_src(:)

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
!   subroutine sw_two_stream(ngpt_in, nlay_in, mu0, tau, w0, g, &
!                                 Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream")
!     integer,                        intent(in)  :: ngpt_in, nlay_in
!     real(wp),                       intent(in)  :: mu0
!     real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
!     real(wp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
!     ! -----------------------
!     integer  :: i, j

!     ! Variables used in Meador and Weaver
!     real(wp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha2, k
!     ! Ancillary variables
!     real(wp), dimension(ngpt) :: exp_minusktau, exp_minus2ktau, RT_term
!     real(wp) :: k_gamma3, k_mu, k_mu2, mu0_inv
!     real(wp) :: term1,term2,term3
!     ! double precision
!     real(dp) :: k_gamma4, alpha1(ngpt)

!     ! ---------------------------------
!     mu0_inv = 1._wp/mu0

!     !
!     ! Transmittance of direct, unscattered beam. Also used below
!     !
! #ifdef USE_TIMING
!     ret =  gptlstart('tnoscat')
! #endif
!     Tnoscat = exp_fast(-tau*mu0_inv)
! #ifdef USE_TIMING
!     ret =  gptlstop('tnoscat')
! #endif
!     do j = 1, nlay
!       !$OMP SIMD
!       do i = 1, ngpt
!         ! Zdunkowski Practical Improved Flux Method "PIFM"
!         !  (Zdunkowski et al., 1980;  Contributions to Atmowpheric Physics 53, 147-66)
!         !
!         gamma1(i)= (8._wp - w0(i,j) * (5._wp + 3._wp * g(i,j))) * .25_wp
!         gamma2(i)=  3._wp *(w0(i,j) * (1._wp -         g(i,j))) * .25_wp
!         gamma3(i)= (2._wp - 3._wp * mu0 *              g(i,j) ) * .25_wp
!         gamma4(i)=  1._wp - gamma3(i)

!         alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
!         alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17

!         k(i) = sqrt(max((gamma1(i) - gamma2(i)) * (gamma1(i) + gamma2(i)),  k_min))

!       end do
!       exp_minusktau(:) = exp_fast(-tau(:,j)*k(:))
!       !
!       ! Diffuse reflection and transmission
!       !
!       !$OMP SIMD
!       do i = 1, ngpt
!         exp_minus2ktau(i)  = exp_minusktau(i) * exp_minusktau(i)

!         ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
!         RT_term(i) = 1._wp / (k(i) * (1._wp + exp_minus2ktau(i)) + gamma1(i) * (1._wp - exp_minus2ktau(i)) )

!         ! Equation 25
!         Rdif(i,j) = RT_term(i) * gamma2(i) * (1._wp - exp_minus2ktau(i))

!         ! Equation 26
!         Tdif(i,j) = RT_term(i) * 2._wp * k(i) * exp_minusktau(i)
!       ! end do

!       ! !
!       ! ! Transmittance of direct, unscattered beam. Also used below
!       ! !
!       ! Tnoscat(:,j) = exp_fast(-tau(:,j)*mu0_inv)
!       !
!       ! Direct reflect and transmission
!       !
!       !$OMP SIMD
!       ! do i = 1, ngpt
!         k_mu     = k(i) * mu0
!         k_mu2    = k_mu*k_mu
!         k_gamma3 = k(i) * gamma3(i)
!         k_gamma4 = k(i) * gamma4(i)
!         !
!         ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
!         !   and rearranging to avoid div by 0.         
!         RT_term(i) =  w0(i,j) *  &
!         RT_term(i) / merge(1._wp - k_mu2, epsilon(1._wp), abs(1._wp - k_mu2) >= epsilon(1._wp))
!         !  --> divide by (1 - kmu2) when (1-kmu2)> eps, otherwise divide by eps

!         Rdir(i,j) = RT_term(i)  *                              &
!                 (   (1._dp - k_mu) * (alpha2(i) + k_gamma3) -  &
!                    (1._dp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i) - &
!              2.0_wp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) * Tnoscat(i,j)  )

!         ! term1 = (1._dp - k_mu) * (alpha2(i) + k_gamma3)
!         ! term2 = (1._dp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i)
!         ! term3 =  2.0_wp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) 

!         ! Rdir(i,j) = RT_term(i)  *                              &
!         !         (  term1 -  &
!         !            term2  - &
!         !            term3 * Tnoscat(i,j)  )

!         ! temp(i) =  (1._sp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i)
!         ! this term must be in dp
!         ! Rdir still having problems (too large) if k_gammas are sp, even after fixing RT_Term

!         !
!         ! Equation 15, multiplying top and bottom by exp(-k*tau),
!         !   multiplying through by exp(-tau/mu0) to
!         !   prefer underflow to overflow
!         ! Omitting direct transmittance
!         ! !
!         ! Tdir(i,j) = -RT_term(i) *                                                                 &
!         !             ((1._dp + k_mu) * (alpha1(i) + k_gamma4)                     * Tnoscat(i,j) - &
!         !              (1._dp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i) * Tnoscat(i,j) - &
!         !              2.0_wp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))
!         term1 = (1._dp + k_mu) * (alpha1(i) + k_gamma4) 
!         term2 = (1._dp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i)
!         term3 = 2.0_wp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i)
!         Tdir(i,j) = -RT_term(i) *                                                                 &
!                     ( (term1-term2)* Tnoscat(i,j) - term3)     
!       end do
!     end do

!   end subroutine sw_two_stream

  ! most accurate full double precision version
  pure subroutine sw_two_stream(ngpt_in, nlay_in, mu0, tau, w0, g, &
                                Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream")
    integer,                        intent(in)  :: ngpt_in, nlay_in
    real(wp),                       intent(in)  :: mu0
    real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
    ! -----------------------
    integer  :: i, j

    ! Variables used in Meador and Weaver
    real(dp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
    ! Ancillary variables
    real(dp), dimension(ngpt) :: exp_minusktau, exp_minus2ktau, RT_term
    real(dp) :: k_gamma3, k_gamma4, k_mu, k_mu2, mu0_inv
    ! ---------------------------------
    mu0_inv = 1._wp/mu0

    do j = 1, nlay
      do i = 1, ngpt
        ! Zdunkowski Practical Improved Flux Method "PIFM"
        !  (Zdunkowski et al., 1980;  Contributions to Atmodpheric Physics 53, 147-66)
        !
        gamma1(i)= (8._dp - w0(i,j) * (5._dp + 3._dp * g(i,j))) * .25_dp
        gamma2(i)=  3._dp *(w0(i,j) * (1._dp -         g(i,j))) * .25_dp
        gamma3(i)= (2._dp - 3._dp * mu0 *              g(i,j) ) * .25_dp
        gamma4(i)=  1._dp - gamma3(i)

        alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
        alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17

        k(i) = sqrt(max((gamma1(i) - gamma2(i)) * (gamma1(i) + gamma2(i)),  1.e-12_wp))

      end do
      exp_minusktau(:) = exp_fast(-tau(:,j)*k(:))
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
      Tnoscat(:,j) = exp_fast(-tau(:,j)*mu0_inv)
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
                     2.0_dp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))
      end do
    end do

  end subroutine sw_two_stream

    ! pure subroutine sw_two_stream(ngpt_in, nlay_in, mu0, tau, w0, g, &
  !                               Rdif, Tdif, Rdir, Tdir, Tnoscat) bind (C, name="sw_two_stream")
  !   integer,                        intent(in)  :: ngpt_in, nlay_in
  !   real(wp),                       intent(in)  :: mu0
  !   real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
  !   real(wp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
  !   ! -----------------------
  !   integer  :: i, j

  !   ! Variables used in Meador and Weaver
  !   real(wp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
  !   ! Ancillary variables
  !   real(wp), dimension(ngpt) :: exp_minusktau, exp_minus2ktau, RT_term
  !   real(wp) :: k_gamma3, k_gamma4, k_mu, k_mu2, mu0_inv
  !   ! ---------------------------------
  !   mu0_inv = 1._wp/mu0

  !   do j = 1, nlay
  !     do i = 1, ngpt
  !       ! Zdunkowski Practical Improved Flux Method "PIFM"
  !       !  (Zdunkowski et al., 1980;  Contributions to Atmowpheric Physics 53, 147-66)
  !       !
  !       gamma1(i)= (8._wp - w0(i,j) * (5._wp + 3._wp * g(i,j))) * .25_wp
  !       gamma2(i)=  3._wp *(w0(i,j) * (1._wp -         g(i,j))) * .25_wp
  !       gamma3(i)= (2._wp - 3._wp * mu0 *              g(i,j) ) * .25_wp
  !       gamma4(i)=  1._wp - gamma3(i)

  !       alpha1(i) = gamma1(i) * gamma4(i) + gamma2(i) * gamma3(i)           ! Eq. 16
  !       alpha2(i) = gamma1(i) * gamma3(i) + gamma2(i) * gamma4(i)           ! Eq. 17

  !       k(i) = sqrt(max((gamma1(i) - gamma2(i)) * (gamma1(i) + gamma2(i)),  k_min))

  !     end do
  !     exp_minusktau(:) = exp_fast(-tau(:,j)*k(:))
  !     !
  !     ! Diffuse reflection and transmission
  !     !
  !     do i = 1, ngpt
  !       exp_minus2ktau(i)  = exp_minusktau(i) * exp_minusktau(i)

  !       ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
  !       RT_term(i) = 1._wp / (k(i) * (1._wp + exp_minus2ktau(i)) + gamma1(i) * (1._wp - exp_minus2ktau(i)) )

  !       ! Equation 25
  !       Rdif(i,j) = RT_term(i) * gamma2(i) * (1._wp - exp_minus2ktau(i))

  !       ! Equation 26
  !       Tdif(i,j) = RT_term(i) * 2._wp * k(i) * exp_minusktau(i)
  !     end do

  !     !
  !     ! Transmittance of direct, unscattered beam. Also used below
  !     !
  !     Tnoscat(:,j) = exp_fast(-tau(:,j)*mu0_inv)
  !     !
  !     ! Direct reflect and transmission
  !     !
  !     do i = 1, ngpt
  !       k_mu     = k(i) * mu0
  !       k_mu2    = k_mu*k_mu
  !       k_gamma3 = k(i) * gamma3(i)
  !       k_gamma4 = k(i) * gamma4(i)
  !       !
  !       ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
  !       !   and rearranging to avoid div by 0.         
  !       RT_term(i) =  w0(i,j) *  &
  !       RT_term(i) / merge(1._wp - k_mu2, epsilon(1._wp), abs(1._wp - k_mu2) >= epsilon(1._wp))
  !       !  --> divide by (1 - kmu2) when (1-kmu2)> eps, otherwise divide by eps

  !       Rdir(i,j) = RT_term(i)  *                              &
  !               (   (1._wp - k_mu) * (alpha2(i) + k_gamma3) -  &
  !                  (1._wp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i) - &
  !            2.0_wp * (k_gamma3 - alpha2(i) * k_mu)  * exp_minusktau (i) * Tnoscat(i,j)  )

  !       ! temp(i) =  (1._sp + k_mu) * (alpha2(i) - k_gamma3) * exp_minus2ktau(i)
  !       ! this term must be in wp
  !       ! Rdir still having problems (too large) if k_gammas are sp, even after fixing RT_Term

  !       !
  !       ! Equation 15, multiplying top and bottom by exp(-k*tau),
  !       !   multiplying through by exp(-tau/mu0) to
  !       !   prefer underflow to overflow
  !       ! Omitting direct transmittance
  !       !
  !       Tdir(i,j) = -RT_term(i) *                                                                 &
  !                   ((1._wp + k_mu) * (alpha1(i) + k_gamma4)                     * Tnoscat(i,j) - &
  !                    (1._wp - k_mu) * (alpha1(i) - k_gamma4) * exp_minus2ktau(i) * Tnoscat(i,j) - &
  !                    2.0_wp * (k_gamma4 + alpha1(i) * k_mu)  * exp_minusktau (i))
  !     end do
  !   end do

  ! end subroutine sw_two_stream

  pure subroutine sw_two_stream_source(ngpt_in, nlay_in, top_at_1, mu0, tau, w0, g, sfc_albedo, &
                                Rdif, Tdif, source_up, source_dn, flux_dn_dir, source_sfc)
    integer,                        intent(in)  :: ngpt_in, nlay_in
    logical(wl),                    intent(in)  :: top_at_1
    real(wp),                       intent(in)  :: mu0
    real(wp), dimension(ngpt,nlay), intent(in)  :: tau, w0, g
    real(wp), dimension(ngpt),      intent(in)  :: sfc_albedo
    real(wp), dimension(ngpt,nlay), intent(out) :: Rdif, Tdif,  source_up, source_dn
    real(wp), dimension(ngpt,nlay+1), target, intent(inout) :: flux_dn_dir
    real(wp), dimension(ngpt),      intent(out) :: source_sfc

    ! -----------------------
    integer  :: igpt, ilev, j

    ! Variables used in Meador and Weaver
    real(wp), dimension(ngpt) :: gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, k
    ! Ancillary variables
    real(wp), dimension(ngpt) :: exp_minusktau, Tnoscat
    real(wp) :: k_gamma3, k_gamma4, k_mu, k_mu2, mu0_inv
    real(wp) :: Rdir, Tdir, exp_minus2ktau, RT_term
    real(wp), pointer, contiguous, dimension(:) :: dir_flux_inc, dir_flux_trans

    ! ---------------------------------
    mu0_inv = 1._wp/mu0

    do j = 1, nlay
      if(top_at_1) then
        ilev      =  j
        dir_flux_inc   => flux_dn_dir(:,ilev  )
        dir_flux_trans => flux_dn_dir(:,ilev+1)
      else
        ilev      =  nlay-j+1
        dir_flux_inc   => flux_dn_dir(:,ilev+1)
        dir_flux_trans => flux_dn_dir(:,ilev  )
      end if

      !
      ! Transmittance of direct, unscattered beam. Also used below
      !
      Tnoscat(:) = exp_fast(-tau(:,ilev)*mu0_inv)

      !$OMP SIMD
      do igpt = 1, ngpt
        ! Zdunkowski Practical Improved Flux Method "PIFM"
        !  (Zdunkowski et al., 1980;  Contributions to Atmowpheric Physics 53, 147-66)
        !
        gamma1(igpt)= (8._wp - w0(igpt,ilev) * (5._wp + 3._wp * g(igpt,ilev))) * .25_wp
        gamma2(igpt)=  3._wp *(w0(igpt,ilev) * (1._wp -         g(igpt,ilev))) * .25_wp
        gamma3(igpt)= (2._wp - 3._wp * mu0 *              g(igpt,ilev) ) * .25_wp
        gamma4(igpt)=  1._wp - gamma3(igpt)

        alpha1(igpt) = gamma1(igpt) * gamma4(igpt) + gamma2(igpt) * gamma3(igpt)           ! Eq. 16
        alpha2(igpt) = gamma1(igpt) * gamma3(igpt) + gamma2(igpt) * gamma4(igpt)           ! Eq. 17

        k(igpt) = sqrt(max((gamma1(igpt) - gamma2(igpt)) * (gamma1(igpt) + gamma2(igpt)),  k_min))

      end do
      exp_minusktau(:) = exp_fast(-tau(:,ilev)*k(:))
      !
      ! Diffuse reflection and transmission
      !
      !$OMP SIMD
      do igpt = 1, ngpt
        exp_minus2ktau  = exp_minusktau(igpt) * exp_minusktau(igpt)

        ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
        RT_term = 1._wp / (k(igpt) * (1._wp + exp_minus2ktau) + gamma1(igpt) * (1._wp - exp_minus2ktau) )

        ! Equation 25
        Rdif(igpt,ilev) = RT_term * gamma2(igpt) * (1._wp - exp_minus2ktau)

        ! Equation 26
        Tdif(igpt,ilev) = RT_term * 2._wp * k(igpt) * exp_minusktau(igpt)

        k_mu     = k(igpt) * mu0
        k_mu2    = k_mu*k_mu
        k_gamma3 = k(igpt) * gamma3(igpt)
        k_gamma4 = k(igpt) * gamma4(igpt)
        !
        ! Equation 14, multiplying top and bottom by exp_fast(-k*tau)
        !   and rearranging to avoid div by 0.         
        RT_term =  w0(igpt,ilev) *  &
        RT_term / merge(1._wp - k_mu2, epsilon(1._wp), abs(1._wp - k_mu2) >= epsilon(1._wp))
        !  --> divide by (1 - kmu2) when (1-kmu2)> eps, otherwise divide by eps

        Rdir = RT_term  *                              &
                (   (1._wp - k_mu) * (alpha2(igpt) + k_gamma3) -  &
                   (1._wp + k_mu) * (alpha2(igpt) - k_gamma3) * exp_minus2ktau - &
             2.0_wp * (k_gamma3 - alpha2(igpt) * k_mu)  * exp_minusktau (igpt) * Tnoscat(igpt)  )
        !
        ! Equation 15, multiplying top and bottom by exp(-k*tau),
        !   multiplying through by exp(-tau/mu0) to
        !   prefer underflow to overflow
        ! Omitting direct transmittance
        !
        Tdir = -RT_term *                                                             &
                    ((1._wp + k_mu) * (alpha1(igpt) + k_gamma4)                     * Tnoscat(igpt) - &
                     (1._wp - k_mu) * (alpha1(igpt) - k_gamma4) * exp_minus2ktau * Tnoscat(igpt) - &
                     2.0_wp * (k_gamma4 + alpha1(igpt) * k_mu)  * exp_minusktau (igpt))

        source_up  (igpt,ilev) =   Rdir    *   dir_flux_inc(igpt) 
        source_dn  (igpt,ilev) =   Tdir    *   dir_flux_inc(igpt)
        dir_flux_trans(igpt) =   Tnoscat(igpt) * dir_flux_inc(igpt)
      end do
    end do

    source_sfc(:) = dir_flux_trans(:)*sfc_albedo(:)


  end subroutine sw_two_stream_source

  ! ---------------------------------------------------------------
  !
  ! Direct beam source for diffuse radiation in layers and at surface;
  !   report direct beam as a byproduct
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine sw_source_2str(ngpt_in, nlay_in, top_at_1, Rdir, Tdir, Tnoscat, sfc_albedo, &
                            source_up, source_dn, source_sfc, flux_dn_dir) bind(C, name="sw_source_2str")
    integer,                           intent(in   ) :: ngpt_in, nlay_in
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
  pure subroutine adding(ngpt_in, nlay_in, top_at_1, &
                  albedo_sfc,           &
                  rdif, tdif,           &
                  src_dn, src_up, src_sfc, &
                  flux_up, flux_dn) bind(C, name="adding")
    integer,                          intent(in   ) :: ngpt_in, nlay_in
    logical(wl),                      intent(in   ) :: top_at_1
    real(wp), dimension(ngpt       ), intent(in   ) :: albedo_sfc
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: rdif, tdif
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: src_dn, src_up
    real(wp), dimension(ngpt       ), intent(in   ) :: src_sfc
    real(wp), dimension(ngpt,nlay+1), intent(  out) :: flux_up
    ! intent(inout) because top layer includes incident flux
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: flux_dn
    ! ------------------
    integer :: ilev, igpt
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
        do igpt = 1, ngpt
          denom(igpt, ilev) = 1._wp/(1._wp - rdif(igpt,ilev)*albedo(igpt,ilev+1))                 ! Eq 10
          albedo(igpt,ilev) = rdif(igpt,ilev) + &
                          tdif(igpt,ilev)*tdif(igpt,ilev) * albedo(igpt,ilev+1) * denom(igpt,ilev) ! Equation 9
          !
          ! Equation 11 -- source is emitted upward radiation at top of layer plus
          !   radiation emitted at bottom of layer,
          !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
          !
          src(igpt,ilev) =  src_up(igpt, ilev) + &
                        tdif(igpt,ilev) * denom(igpt,ilev) *       &
                          (src(igpt,ilev+1) + albedo(igpt,ilev+1)*src_dn(igpt,ilev))
        end do
      end do
      ! Eq 12, at the top of the domain upwelling diffuse is due to ...
      ilev = 1
      flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! ... reflection of incident diffuse and
                        src(:,ilev)                          ! emission from below
      !
      ! From the top of the atmosphere downward -- compute fluxes
      !
      do ilev = 2, nlay+1
        do igpt = 1, ngpt
          flux_dn(igpt,ilev) = (tdif(igpt,ilev-1)*flux_dn(igpt,ilev-1) + &  ! Equation 13
                            rdif(igpt,ilev-1)*src(igpt,ilev) +       &
                            src_dn(igpt,ilev-1)) * denom(igpt,ilev-1)
          flux_up(igpt,ilev) = flux_dn(igpt,ilev) * albedo(igpt,ilev) + & ! Equation 12
                            src(igpt,ilev)
        end do
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
        do igpt = 1, ngpt
          denom(igpt, ilev  ) = 1._wp/(1._wp - rdif(igpt,ilev)*albedo(igpt,ilev))                ! Eq 10
          albedo(igpt,ilev+1) = rdif(igpt,ilev) + &
                            tdif(igpt,ilev)*tdif(igpt,ilev) * albedo(igpt,ilev) * denom(igpt,ilev) ! Equation 9
          !
          ! Equation 11 -- source is emitted upward radiation at top of layer plus
          !   radiation emitted at bottom of layer,
          !   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
          !
          src(igpt,ilev+1) =  src_up(igpt, ilev) +  &
                          tdif(igpt,ilev) * denom(igpt,ilev) *       &
                          (src(igpt,ilev) + albedo(igpt,ilev)*src_dn(igpt,ilev))
        end do
      end do
      ! Eq 12, at the top of the domain upwelling diffuse is due to ...
      ilev = nlay+1
      flux_up(:,ilev) = flux_dn(:,ilev) * albedo(:,ilev) + & ! ... reflection of incident diffuse and
                        src(:,ilev)                          ! scattering by the direct beam below
      !
      ! From the top of the atmosphere downward -- compute fluxes
      !
      do ilev = nlay, 1, -1
        do igpt = 1, ngpt
          flux_dn(igpt,ilev) = (tdif(igpt,ilev)*flux_dn(igpt,ilev+1) + &  ! Equation 13
                            rdif(igpt,ilev)*src(igpt,ilev) + &
                            src_dn(igpt, ilev)) * denom(igpt,ilev)
          flux_up(igpt,ilev) = flux_dn(igpt,ilev) * albedo(igpt,ilev) + & ! Equation 12
                            src(igpt,ilev)
        end do
      end do
    end if
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
                                radn_up_Jac) bind(C, name="lw_transport_1rescl")
    integer,                          intent(in   ) :: ngpt, nlay ! Number of columns, layers, g-points
    logical(wl),                      intent(in   ) :: top_at_1   !
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: trans      ! transmissivity = exp_fast(-tau)
    real(wp), dimension(ngpt,nlay  ), intent(in   ) :: source_dn, &
                                                      source_up  ! Diffuse radiation emitted by the layer
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_up    ! Radiances [W/m2-str]
    real(wp), dimension(ngpt,nlay+1), intent(inout) :: radn_dn    !Top level must contain incident flux boundary condition
    real(wp), dimension(ngpt,nlay),   intent(in   ) :: An, Cn
    real(wp), dimension(:,:),         intent(inout) :: radn_up_Jac ! Surface temperature Jacobians [W/m2-str/K]
    !
    ! We could in principle compute a downwelling Jacobian too, but it's small
    !   (only a small proportion of LW is scattered) and it complicates code and the API,
    !   so we will not
    !
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
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_dn(igpt,ilev) - &
                    trans(igpt,ilev)*source_dn(igpt,ilev) - source_up(igpt,ilev) )  
            radn_up (igpt,ilev) = trans(igpt,ilev)*radn_up (igpt,ilev+1) + source_up(igpt,ilev) + adjustmentFactor
            if (compute_Jac) radn_up_Jac(igpt,ilev) = trans(igpt,ilev)*radn_up_Jac(igpt,ilev+1)
        enddo
      end do
      ! 2nd Downward propagation
      do ilev = 1, nlay
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_up(igpt,ilev) - &
                    trans(igpt,ilev)*source_up(igpt,ilev) - source_dn(igpt,ilev) )
            radn_dn    (igpt,ilev+1) = trans(igpt,ilev)*radn_dn(igpt,ilev) + source_dn(igpt,ilev) + adjustmentFactor
        enddo
      end do
    else
      !
      ! Top of domain is index nlay+1
      !
      ! Upward propagation
      do ilev = 1, nlay
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_dn(igpt,ilev+1) - &
                    trans(igpt,ilev)*source_dn(igpt,ilev) - source_up(igpt,ilev) )
            radn_up(igpt,ilev+1) = trans(igpt,ilev) * radn_up(igpt,ilev) +  source_up(igpt,ilev) + adjustmentFactor
            if (compute_Jac) radn_up_Jac(igpt,ilev+1) = trans(igpt,ilev) * radn_up_Jac(igpt,ilev)
        enddo
      end do

      ! 2st Downward propagation
      do ilev = nlay, 1, -1
        do igpt=1,ngpt
            adjustmentFactor = Cn(igpt,ilev)*( An(igpt,ilev)*radn_up(igpt,ilev) - &
                    trans(igpt,ilev)*source_up(igpt,ilev) - source_dn(igpt,ilev) )
            radn_dn(igpt,ilev) = trans(igpt,ilev)*radn_dn(igpt,ilev+1) + source_dn(igpt,ilev) + adjustmentFactor
        enddo
      end do
    end if
  end subroutine lw_transport_1rescl

  pure subroutine sw_layer_props_sources_2str(ngpt, nlay, top_at_1,  &
                            mu0, tau, w0, g, sfc_albedo, &
                            source_up, source_dn, source_sfc, flux_dn_dir, Rdif, Tdif) bind(C, name="sw_layer_props_sources_2str")
    integer,                                 intent(in   ) :: ngpt, nlay
    logical(wl),                             intent(in   ) :: top_at_1
    real(wp),                               intent(in   ) :: mu0
    real(wp), dimension(ngpt, nlay  ), intent(in   ) :: tau, w0, g
    real(wp), dimension(ngpt        ), intent(in   ) :: sfc_albedo        ! surface albedo for direct radiation
    real(wp), dimension(ngpt, nlay  ), intent(out  ) :: source_dn, source_up
    real(wp), dimension(ngpt        ), intent(out  ) :: source_sfc        ! Source function for upward radiation at surface
    real(wp), dimension(ngpt, nlay+1), intent(inout) :: flux_dn_dir       ! Direct beam flux
                                                                          ! intent(inout) because top layer includes incident flux
    real(wp), dimension(ngpt, nlay  ), intent(out  ) :: Rdif, Tdif

    integer  :: igpt, ilev
    real(wp) :: Rdir, Tdir, Tnoscat
    ! ---------------------------------
    if(top_at_1) then
      do ilev = 1, nlay
        do igpt = 1, ngpt
          call sw_two_stream_scalar(mu0,                                       &
                                     tau (igpt,ilev), w0  (igpt,ilev), g(igpt,ilev), &
                                     Rdif(igpt,ilev), Tdif(igpt,ilev),               &
                                     Rdir, Tdir, Tnoscat)
          source_up  (igpt,ilev) =   Rdir    * flux_dn_dir(igpt,ilev)
          source_dn  (igpt,ilev) =   Tdir    * flux_dn_dir(igpt,ilev)
          flux_dn_dir(igpt,ilev+1) = Tnoscat * flux_dn_dir(igpt,ilev)
        end do
      end do
      source_sfc(:) = flux_dn_dir(:,nlay+1)*sfc_albedo(:)

    else
      ! layer index = level index
      ! previous level is up (+1)
      do igpt = 1, ngpt
        do ilev = nlay, 1, -1
          call sw_two_stream_scalar(mu0,                                       &
                                     tau (igpt,ilev), w0  (igpt,ilev), g(igpt,ilev), &
                                     Rdif(igpt,ilev), Tdif(igpt,ilev),               &
                                     Rdir, Tdir, Tnoscat)
          source_up  (igpt,ilev) = Rdir    * flux_dn_dir(igpt,ilev+1)
          source_dn  (igpt,ilev) = Tdir    * flux_dn_dir(igpt,ilev+1)
          flux_dn_dir(igpt,ilev) = Tnoscat * flux_dn_dir(igpt,ilev+1)
!          call sw_source_2str_scalar(Rdir, Tdir, Tnoscat,      &
!                                     flux_dn_dir(igpt,ilev+1), &
!                                     source_up  (igpt,ilev  ), &
!                                     source_dn  (igpt,ilev  ), &
!                                     flux_dn_dir(igpt,ilev  ) )

        end do
        source_sfc(igpt) = flux_dn_dir(igpt,    1)*sfc_albedo(igpt)
      end do
    end if
  end subroutine sw_layer_props_sources_2str

  elemental subroutine sw_two_stream_scalar(mu0, tau, w0, g, &
                                    Rdif, Tdif, Rdir, Tdir, Tnoscat)
    !$acc routine seq
    real(wp), intent(in)  :: mu0, tau, w0, g
    real(wp), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat

    ! -----------------------

    ! Variables used in Meador and Weaver
    real(wp) :: gamma1, gamma2, gamma3, gamma4
    real(wp) :: alpha1, alpha2, k

    ! Ancillary variables
    real(wp) :: RT_term
    real(wp) :: exp_minusktau, exp_minus2ktau
    real(wp) :: k_mu, k_gamma3, k_gamma4
    real(wp) :: mu0_inv
    ! ---------------------------------
    ! ---------------------------------
    mu0_inv = 1._wp/mu0

    ! Zdunkowski Practical Improved Flux Method "PIFM"
    !  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
    !
    gamma1= (8._wp - w0 * (5._wp + 3._wp * g)) * .25_wp
    gamma2=  3._wp *(w0 * (1._wp -         g)) * .25_wp
    gamma3= (2._wp - 3._wp * mu0         * g ) * .25_wp
    gamma4=  1._wp - gamma3

    alpha1 = gamma1 * gamma4 + gamma2 * gamma3           ! Eq. 16
    alpha2 = gamma1 * gamma3 + gamma2 * gamma4           ! Eq. 17
    ! Written to encourage vectorization of exponential, square root
    ! Eq 18;  k = SQRT(gamma1**2 - gamma2**2), limited below to avoid div by 0.
    !   k = 0 for isotropic, conservative scattering; this lower limit on k
    !   gives relative error with respect to conservative solution
    !   of < 0.1% in Rdif down to tau = 10^-9
    k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), k_min))
    exp_minusktau = exp(-tau*k)
    !
    ! Diffuse reflection and transmission
    !
    exp_minus2ktau = exp_minusktau * exp_minusktau

    ! Refactored to avoid rounding errors when k, gamma1 are of very different magnitudes
    RT_term = 1._wp / (k      * (1._wp + exp_minus2ktau)  + &
                      gamma1 * (1._wp - exp_minus2ktau) )

    ! Equation 25
    Rdif = RT_term * gamma2 * (1._wp - exp_minus2ktau)

    ! Equation 26
    Tdif = RT_term * 2._wp * k * exp_minusktau

    !
    ! Transmittance of direct, unscattered beam. Also used below
    !
    Tnoscat = exp(-tau*mu0_inv)

    !
    ! Direct reflect and transmission
    !
    k_mu     = k * mu0
    k_gamma3 = k * gamma3
    k_gamma4 = k * gamma4

    !
    ! Equation 14, multiplying top and bottom by exp(-k*tau)
    !   and rearranging to avoid div by 0.
    !
    RT_term =  w0 * RT_term/merge(1._wp - k_mu*k_mu, &
                                  epsilon(1._wp),    &
                                  abs(1._wp - k_mu*k_mu) >= epsilon(1._wp))

    Rdir = RT_term  *                                    &
      ((1._wp - k_mu) * (alpha2 + k_gamma3)                  - &
        (1._wp + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau - &
        2.0_wp * (k_gamma3 - alpha2 * k_mu)  * exp_minusktau  * Tnoscat)
    !
    ! Equation 15, multiplying top and bottom by exp(-k*tau),
    !   multiplying through by exp(-tau/mu0) to prefer underflow to overflow
    ! Omitting direct transmittance
    !
    Tdir = -RT_term * ((1._wp + k_mu) * (alpha1 + k_gamma4) * Tnoscat - &
                      (1._wp - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * Tnoscat - &
                        2.0_wp * (k_gamma4 + alpha1 * k_mu)  * exp_minusktau )

  end subroutine sw_two_stream_scalar

  subroutine predict_nn_reftrans(nlay, ngpt, &
                                neural_net,         &
                                tau, ssa, g, Tnoscat, mu0, &
                                nn_output &
                                )
    use, intrinsic :: ISO_C_BINDING
    integer,                              intent(in)    :: ngpt,nlay
    type(network_type),                   intent(in)    :: neural_net 
    real(wp), dimension(ngpt,nlay), intent(in)          :: tau, & ! Inputs: optical thickness,
                                                          ssa, &  ! single-scattering albedo,
                                                          g,   &  ! asymmetry parameter,
                                                          Tnoscat ! no scattering transmittance
    real(wp),                            intent(in)    :: mu0    ! cosine of solar zenith angle
    ! real(wp), dimension(ngpt,nlay), &                ! outputs
    !                             intent(out), target     :: Rdif,Tdif,Rdir,Tdir
    real(sp), dimension(ngpt*nlay,4), intent(out)      :: nn_output
    ! local vars
    integer :: nbatch, i, ivar, ilay, igpt, j, k
    real(sp), dimension(ngpt*nlay,5)    :: nn_input

    ! after log scaling tau: min max scaling 
    ! real(sp), dimension(5)    :: xmin =  (/ -20.723267, 0.0, 0.0, 0.0, 0.0 /)
    ! real(sp), dimension(5)    :: xmax =  (/ 9.0,  1.0,  1.0,  1.0, 1.0 /)

    ! sqrt4
    real(sp), dimension(5)    :: xmin =  (/ 0.0,   0.0, 0.0, 0.0,  0.0 /)
    real(sp), dimension(5)    :: xmax =  (/ 13.05, 1.0, 1.0, 1.0,  1.0 /)

    ! real(sp), dimension(:), contiguous, pointer     :: tau_1D, ssa_1D, g_1D
    
#ifdef USE_TIMING
    ret =  gptlstart('prep_input')
#endif
    
      nbatch = ngpt*nlay
#ifdef USE_TIMING
    ret =  gptlstart('log')
#endif
      ! nn_input(:,1) = (log(reshape(tau,(/nbatch/))) - xmin(1))  / (xmax(1) - xmin(1))
    ! call C_F_POINTER (C_LOC(tau), tau_1D, [nbatch])
    ! call C_F_POINTER (C_LOC(ssa), ssa_1D, [nbatch])
    ! call C_F_POINTER (C_LOC(g),   g_1D, [nbatch])

    ! nn_input(:,1) =  reshape(tau,(/nbatch/))
    nn_input(:,1) =  sqrt(sqrt(reshape(tau,(/nbatch/)))) / xmax(1)
#ifdef USE_TIMING
    ret =  gptlstop('log')
#endif
      nn_input(:,2) = reshape(ssa, (/nbatch/))
      ! nn_input(:,3) = reshape(g, (/nbatch/)) / xmax(3)
      nn_input(:,3) = reshape(g, (/nbatch/))
      nn_input(:,4) = mu0
      nn_input(:,5) = reshape(Tnoscat, (/nbatch/))

#ifdef USE_TIMING
    ret =  gptlstop('prep_input')
#endif

      ! do i = 1,5
      !   print *," min max inp ", i, ":", minval(nn_input(:,i)), maxval(nn_input(:,i))
      ! end do

#ifdef USE_TIMING
    ret =  gptlstart('kernel')
#endif
    call neural_net % output_sgemm_flat_byrows(size(nn_input,2), size(nn_output,2), nbatch, nn_input, nn_output)
    
#ifdef USE_TIMING
    ret =  gptlstop('kernel')
#endif
#ifdef USE_TIMING
    ret =  gptlstart('postproc')
#endif
    do ivar = 1, 4
      do i = 1, nbatch
        ! nn_output(i,ivar) = nn_output(i,ivar) * ystds(ivar)  + ymeans(ivar)
        nn_output(i,ivar) = nn_output(i,ivar)**2
      end do
    end do

    ! nn_output = min(1.0_wp, nn_output)
#ifdef USE_TIMING
    ret =  gptlstop('postproc')
#endif
    ! print *,"min,max Rdif", minval(nn_output(:,1)), maxval(nn_output(:,1))
    ! print *,"min,max Tdif", minval(nn_output(:,2)), maxval(nn_output(:,2))
    ! print *,"min,max  Rdir", minval(nn_output(:,3)), maxval(nn_output(:,3))
    ! print *,"min,max  Tdir", minval(nn_output(:,4)), maxval(nn_output(:,4))

  end subroutine predict_nn_reftrans


  function mean_2d(x2) result(mean2)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x2
    real(wp) :: mean2
    
    mean2 = sum(x2) / (size(x2))
  end function mean_2d

  function mae(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:,:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1,dim=1),size(x1,dim=2)) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff)/size(diff)
  end function mae

  function mean_3d(x2) result(mean2)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x2
    real(wp) :: mean2
    
    mean2 = sum(x2) / (size(x2))
  end function mean_3d

  function mae_3d(x1,x2) result(res)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x1,x2
    real(wp) :: res
    real(wp), dimension(size(x1,dim=1),size(x1,dim=2),size(x1,dim=3)) :: diff 
    
    diff = abs(x1 - x2)
    res = sum(diff)/size(diff)
  end function mae_3d

end module mo_rte_solver_kernels
