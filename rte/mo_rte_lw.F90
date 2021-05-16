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
!  Contains a single routine to compute direct and diffuse fluxes of solar radiation given
!    atmospheric optical properties, spectrally-resolved
!    information about vertical ordering
!    internal Planck source functions, defined per g-point on the same spectral grid at the atmosphere
!    boundary conditions: surface emissivity defined per band
!    optionally, a boundary condition for incident diffuse radiation
!    optionally, an integer number of angles at which to do Gaussian quadrature if scattering is neglected
!
! If optical properties are supplied via class ty_optical_props_1scl (absorption optical thickenss only)
!    then an emission/absorption solver is called
!    If optical properties are supplied via class ty_optical_props_2str fluxes are computed via
!    two-stream calculations and adding.
!
! It is the user's responsibility to ensure that emissivity is on the same
!   spectral grid as the optical properties.
!
! Final output is via user-extensible ty_fluxes which must reduce the detailed spectral fluxes to
!   whatever summary the user needs.
!
! The routine does error checking and choses which lower-level kernel to invoke based on
!   what kinds of optical properties are supplied
!
! -------------------------------------------------------------------------------------------------
module mo_rte_lw
  use mo_rte_kind,          only: wp, wl
  use mo_rte_rrtmgp_config, only: check_extents, check_values, compute_Jac
  use mo_rte_util_array,    only: any_vals_less_than, any_vals_outside, extents_are

  use mo_optical_props,     only: ty_optical_props, &
                              ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_source_functions,   &
                            only: ty_source_func_lw
  use mo_fluxes,            only: ty_fluxes_broadband, ty_fluxes_flexible
  use mo_rte_solver_kernels, &
                            only: apply_BC, lw_solver_noscat, lw_solver_noscat_GaussQuad, &
                            lw_solver_2stream

  implicit none
  private

  public :: rte_lw
contains
  ! --------------------------------------------------
  !
  ! Interface using only optical properties and source functions as inputs; fluxes as outputs.
  !
  ! --------------------------------------------------
  function rte_lw(optical_props, top_at_1, &
                  sources, sfc_emis,       &
                  fluxes,                  &
                  inc_flux, n_gauss_angles, use_2stream, &
                  lw_Ds, flux_up_Jac, flux_dn_Jac) result(error_msg)
    class(ty_optical_props_arry), intent(in   ) :: optical_props     ! Array of ty_optical_props. This type is abstract
                                                                     ! and needs to be made concrete, either as an array
                                                                     ! (class ty_optical_props_arry) or in some user-defined way
    logical,                  intent(in   ) :: top_at_1          ! Is the top of the domain at index 1?
                                                                     ! (if not, ordering is bottom-to-top)
    type(ty_source_func_lw),      intent(in   ) :: sources
    real(wp), dimension(:,:),     intent(in   ) :: sfc_emis    ! emissivity at surface [] (nband, ncol)
    class(ty_fluxes_flexible),   intent(inout) :: fluxes      ! Array of ty_fluxes. Default computes broadband fluxes at all levels
                                                               !   if output arrays are defined. Can be extended per user desires.
    real(wp), dimension(:,:),   &
              target, optional, intent(in   ) :: inc_flux    ! incident flux at domain top [W/m2] (ngpts, ncol)
    integer,          optional, intent(in   ) :: n_gauss_angles ! Number of angles used in Gaussian quadrature
                                                                ! (no-scattering solution)
    logical,          optional, intent(in   ) :: use_2stream    ! When 2-stream parameters (tau/ssa/g) are provided, use 2-stream methods
                                                                ! Default is to use re-scaled longwave transport
    real(wp), dimension(:,:),   &
                      optional,   intent(in   ) :: lw_Ds          ! linear fit to column transmissivity (ngpts,ncol)
    real(wp), dimension(:,:),   &
                target, optional, intent(inout) :: flux_up_Jac    ! surface temperature flux  Jacobian [W/m2/K] (nlev+1, ncol)
    real(wp), dimension(:,:),   &
                target, optional, intent(inout) :: flux_dn_Jac    ! surface temperature flux  Jacobian [W/m2/K] (nlev+1, ncol)
    ! logical,          optional, intent(in   ) :: save_gpt_flux    ! compute fluxes at g-points, not only broadband fluxes

    character(len=128)                        :: error_msg   ! If empty, calculation was successful
    ! --------------------------------
    !
    ! Local variables
    !
    integer :: ncol, nlay, ngpt, nband
    integer :: n_quad_angs
    integer :: icol, iband, igpt
    real(wp) :: lw_Ds_wt
    logical :: using_2stream, do_gpt_flux
    integer, dimension(2,optical_props%get_nband())   :: band_limits
    real(wp), dimension(:,:), contiguous, pointer     :: inc_flux_toa
    real(wp), dimension(:,:), allocatable, target     :: inc_flux_zero

    real(wp), dimension(:,:,:), allocatable,  target   :: gpt_flux_up, gpt_flux_dn, gpt_flux_upJac
    real(wp), dimension(:,:), allocatable             :: sfc_emis_gpt
    real(wp), dimension(:,:), allocatable             :: flux_upJac


    ! --------------------------------------------------
    !
    ! Weights and angle secants for first order (k=1) Gaussian quadrature.
    !   Values from Table 2, Clough et al, 1992, doi:10.1029/92JD01419
    !   after Abramowitz & Stegun 1972, page 921
    !
    integer,  parameter :: max_gauss_pts = 4
    real(wp), parameter,                         &
      dimension(max_gauss_pts, max_gauss_pts) :: &
        gauss_Ds  = RESHAPE([1.66_wp,               0._wp,         0._wp,         0._wp, &  ! Diffusivity angle, not Gaussian angle
                             1.18350343_wp, 2.81649655_wp,         0._wp,         0._wp, &
                             1.09719858_wp, 1.69338507_wp, 4.70941630_wp,         0._wp, &
                             1.06056257_wp, 1.38282560_wp, 2.40148179_wp, 7.15513024_wp], &
                            [max_gauss_pts, max_gauss_pts]),              &
        gauss_wts = RESHAPE([0.5_wp,          0._wp,           0._wp,           0._wp, &
                             0.3180413817_wp, 0.1819586183_wp, 0._wp,           0._wp, &
                             0.2009319137_wp, 0.2292411064_wp, 0.0698269799_wp, 0._wp, &
                             0.1355069134_wp, 0.2034645680_wp, 0.1298475476_wp, 0.0311809710_wp], &
                             [max_gauss_pts, max_gauss_pts])


    ! ------------------------------------------------------------------------------------
    !
    ! Error checking
    !   if inc_flux is present it has the right dimensions, is positive definite
    !
    ! --------------------------------
    ncol  = optical_props%get_ncol()
    nlay  = optical_props%get_nlay()
    ngpt  = optical_props%get_ngpt()
    nband = optical_props%get_nband()
    band_limits = optical_props%get_band_lims_gpoint()
    
    error_msg = ""

    ! ------------------------------------------------------------------------------------
    !
    ! Error checking -- consistency of sizes and validity of values
    !
    ! --------------------------------
    if(.not. fluxes%are_desired()) then
      error_msg = "rte_lw: no space allocated for fluxes"
      return
    end if
    ! if (present(flux_up_Jac)) then
    !   ! The optional argument flux_up_Jac can't be passed directly to the kernels, 
    !   ! because they have C-binds which are incompatible with optional/allocatable
    !   compute_Jac = .true.
    !   if(.not. extents_are(flux_up_Jac, nlay+1, ncol)) &
    !     error_msg = "rte_lw: flux Jacobian inconsistently sized"
    ! else
    !   compute_Jac = .false.
    ! endif
    if (compute_Jac) then
     if (.not. (present(flux_up_Jac))) error_msg ="rte_lw: compute_Jac=true but Jacobian arrays not provided"
     if(.not. extents_are(flux_up_Jac, nlay+1, ncol)) error_msg = "rte_lw: flux Jacobian inconsistently sized"
    end if
    !
    ! Source functions
    !
    if (check_extents) then
      if(any([sources%get_ncol(), sources%get_nlay(), sources%get_ngpt()]  /= [ncol, nlay, ngpt])) &
        error_msg = "rte_lw: sources and optical properties inconsistently sized"
    end if

    ! Also need to validate

    if (check_extents) then
      !
      ! Surface emissivity
      !
      if(.not. extents_are(sfc_emis, nband, ncol)) &
        error_msg = "rte_lw: sfc_emis inconsistently sized"
      !
      ! Incident flux, if present
      !
      if(present(inc_flux)) then
        if(.not. extents_are(inc_flux, ngpt, ncol)) &
          error_msg = "rte_lw: inc_flux inconsistently sized"
      end if
    end if


    if(check_values) then
      if(any_vals_outside(sfc_emis, 0._wp, 1._wp)) &
        error_msg = "rte_lw: sfc_emis has values < 0 or > 1"
      if(present(inc_flux)) then
        if(any_vals_less_than(inc_flux, 0._wp)) &
          error_msg = "rte_lw: inc_flux has values < 0"
      end if
      if(present(n_gauss_angles)) then
        if(n_gauss_angles > max_gauss_pts) &
          error_msg = "rte_lw: asking for too many quadrature points for no-scattering calculation"
        if(n_gauss_angles < 1) &
          error_msg = "rte_lw: have to ask for at least one quadrature point for no-scattering calculation"
      end if
    end if
    if(len_trim(error_msg) > 0) return

    !
    ! Number of quadrature points for no-scattering calculation
    !
    n_quad_angs = 1
    if(present(n_gauss_angles)) n_quad_angs = n_gauss_angles
    !
    ! Optionally - use 2-stream methods when low-order scattering properties are provided?
    !
    using_2stream = .false.
    if(present(use_2stream)) using_2stream = use_2stream

    !
    ! Optionally - output spectral fluxes, not only broadband fluxes?
    ! 
    !
    do_gpt_flux = .false.
    if(fluxes%are_desired_gpt()) do_gpt_flux = .true. 

    ! Allocate spectral fluxes if they are needed but not already allocated for flux derived type:
    ! When using two-stream solution or GPU acceleration they are always allocated, regardless if user wants them

    ! GPU acceleration
#ifdef USE_OPENACC 
    do_gpt_flux = .true.
#endif

    !
    ! Checking that optional arguements are consistent with one another and with optical properties
    !
    select type (optical_props)
      class is (ty_optical_props_1scl)
        if (using_2stream) &
          error_msg = "rte_lw: can't use two-stream methods with only absorption optical depth"
        if (present(lw_Ds)) then
          if(.not. extents_are(lw_Ds, ncol, ngpt)) &
            error_msg = "rte_lw: lw_Ds inconsistently sized"
          if(any_vals_less_than(lw_Ds, 1._wp)) &
            error_msg = "rte_lw: one or more values of lw_Ds < 1."
          if(n_quad_angs /= 1) &
            error_msg = "rte_lw: providing lw_Ds incompatible with specifying n_gauss_angles"
        end if
      class is (ty_optical_props_2str)
        if (present(lw_Ds)) &
          error_msg = "rte_lw: lw_Ds not valid input for _2str class"
        if (using_2stream .and. n_quad_angs /= 1) &
          error_msg = "rte_lw: using_2stream=true incompatible with specifying n_gauss_angles"
        if (using_2stream .and. (present(flux_up_Jac) .or. present(flux_up_Jac))) &
          error_msg = "rte_lw: can't provide Jacobian of fluxes w.r.t surface temperature with 2-stream"
        ! if (.not. using_2stream .and. .not.(allocated(optical_props%ssa) .and. allocated(optical_props%g))) &
        !   error_msg = "rte_lw: can't use re-scaled no-scattering solution when ssa and g not provided"
        ! Broadband kernels not implemented for calculations with scattering
        if (using_2stream) do_gpt_flux = .true.
      class default
        call stop_on_err("rte_lw: lw_solver(...ty_optical_props_nstr...) not yet implemented")
    end select
    if(len_trim(error_msg) > 0) return

    !
    ! Ensure values of tau, ssa, and g are reasonable if using scattering
    !
    if(check_values) error_msg =  optical_props%validate()

    if(len_trim(error_msg) > 0) then
      if(len_trim(optical_props%get_name()) > 0) &
        error_msg = trim(optical_props%get_name()) // ': ' // trim(error_msg)
      return
    end if

    ! Now allocate spectral fluxes if they are needed
    if (do_gpt_flux) then
      if (.not. fluxes%are_desired_gpt()) then ! ..and not already allocated
        allocate(gpt_flux_up (ngpt, nlay+1, ncol), gpt_flux_dn(ngpt, nlay+1, ncol))
        !$acc enter data create(gpt_flux_up, gpt_flux_dn)
        fluxes%gpt_flux_up => gpt_flux_up(:,:,:)
        fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:)
        if (compute_Jac) then
          allocate(gpt_flux_upJac(ngpt, nlay+1, ncol))
          !$acc enter data create(gpt_flux_upJac)
          fluxes%gpt_flux_up_Jac => gpt_flux_upJac(:,:,:)
        end if
      end if
      !$acc enter data attach(fluxes%gpt_flux_up, fluxes%gpt_flux_dn)
      !$acc enter data attach(fluxes%gpt_flux_up_Jac) if(compute_Jac)
    end if

    ! ------------------------------------------------------------------------------------
    !
    !    Lower boundary condition -- expand surface emissivity by band to gpoints
    !
    allocate(sfc_emis_gpt(ngpt,         ncol))
    !$acc enter data copyin(band_limits) create(sfc_emis_gpt)

    if (compute_Jac) then
      allocate(flux_upJac  (nlay+1,       ncol))
      !$acc enter data create(flux_upJac) 
    end if

    call expand(nband, ngpt, ncol, band_limits, sfc_emis, sfc_emis_gpt)

    if(present(inc_flux)) then
      !$acc enter data copyin(inc_flux)
      inc_flux_toa => inc_flux
    else
      allocate(inc_flux_zero(ngpt, ncol))
      !$acc enter data create(inc_flux_zero)
      !$acc parallel loop collapse(2)
      do icol = 1, ncol
        do igpt = 1, ngpt
          inc_flux_zero(igpt,icol) = 0.0_wp
        end do
      end do
      inc_flux_toa => inc_flux_zero
    end if
    !$acc enter data attach(inc_flux_toa) 

    
    ! Compute the radiative transfer...

    select type (optical_props)
      class is (ty_optical_props_1scl)
        !
        ! No scattering two-stream calculation
        !
      if (present(lw_Ds)) then
        call lw_solver_noscat(ngpt, nlay, ncol, logical(top_at_1, wl), &
                        1, lw_Ds,                               &
                        gauss_wts(1,1), inc_flux_toa,           &
                        optical_props%tau,                      &
                        sources%lay_source, sources%lev_source, &
                        sfc_emis_gpt, sources%sfc_source,       &
                        fluxes%flux_up, fluxes%flux_dn,         &
                        sources%sfc_source_Jac, flux_upJac,     &
                        logical(.false., wl),  optical_props%tau, optical_props%tau, & 
                        ! do_rescaling is false
                        logical(do_gpt_flux, wl), fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_up_Jac)

      else

        call lw_solver_noscat_GaussQuad(ngpt, nlay, ncol, logical(top_at_1, wl), &
                        n_quad_angs, gauss_Ds(1:n_quad_angs,n_quad_angs), &
                        gauss_wts(1:n_quad_angs,n_quad_angs), inc_flux_toa, &
                        optical_props%tau,                      &
                        sources%lay_source, sources%lev_source, &
                        sfc_emis_gpt, sources%sfc_source,       &
                        fluxes%flux_up, fluxes%flux_dn,         &
                        sources%sfc_source_Jac, flux_upJac,     &
                        logical(.false., wl),  optical_props%tau, optical_props%tau, &
                        ! do_rescaling is false
                        logical(do_gpt_flux, wl), fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_up_Jac)
      end if

    class is (ty_optical_props_2str)

      if (using_2stream) then
        !
        ! two-stream calculation with scattering
        !
        error_msg =  optical_props%validate()
        if(len_trim(error_msg) > 0) return

        call lw_solver_2stream(ngpt, nlay, ncol, logical(top_at_1, wl), inc_flux_toa,  &
                    optical_props%tau, optical_props%ssa, optical_props%g,      &
                    sources%lay_source, sources%lev_source, &
                    sfc_emis_gpt, sources%sfc_source,       &
                    fluxes%flux_up, fluxes%flux_dn,         &
                    fluxes%gpt_flux_up, fluxes%gpt_flux_dn)                        
      else
        !
        ! Re-scaled solution to account for scattering
        !
        !$acc enter data copyin(optical_props%tau, optical_props%ssa, optical_props%g)
        call lw_solver_noscat_GaussQuad(ngpt, nlay, ncol, logical(top_at_1, wl), &
                        n_quad_angs, gauss_Ds(1:n_quad_angs,n_quad_angs), &
                        gauss_wts(1:n_quad_angs,n_quad_angs), inc_flux_toa, &
                        optical_props%tau,                      &
                        sources%lay_source, sources%lev_source, &
                        sfc_emis_gpt, sources%sfc_source,       &
                        fluxes%flux_up, fluxes%flux_dn,         &
                        sources%sfc_source_Jac, flux_upJac,     &
                        logical(.true., wl),  optical_props%ssa, optical_props%g, &
                        ! do_rescaling is false
                        logical(do_gpt_flux, wl), fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_up_Jac)
          !$acc exit data delete(optical_props%tau,  optical_props%ssa, optical_props%g)                       
      endif

      class is (ty_optical_props_nstr)
      !
      ! n-stream calculation
      !
      error_msg = 'lw_solver(...ty_optical_props_nstr...) not yet implemented'
    end select

    ! if (do_gpt_flux) then
    !   !$acc exit data delete(gpt_flux_dn, gpt_flux_up)
    !   deallocate(gpt_flux_up, gpt_flux_dn)
    !   if (compute_Jac) then
    !     !$acc exit data delete(gpt_flux_dnJac)
    !     deallocate(gpt_flux_dnJac)
    !   end if
    ! end if
  
    if (error_msg /= '') return

    !$acc exit data detach(inc_flux_toa) delete(inc_flux_zero, inc_flux_toa, inc_flux, flux_upJac, sfc_emis_gpt)
    deallocate(sfc_emis_gpt, inc_flux_zero)

  end function rte_lw
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Expand from band to g-point dimension, transpose dimensions (ncol) -> (ncol,ngpt)
  !
  subroutine expand(nband, ngpt, ncol, band_limits,arr_in, arr_out)
    integer,                          intent(in)  :: nband, ngpt, ncol
    integer,  dimension(2,nband),     intent(in)  :: band_limits
    real(wp), dimension(nband,ncol),  intent(in)  :: arr_in  ! (ncol)
    real(wp), dimension(ngpt,ncol),   intent(out) :: arr_out ! (ngpt, ncol)
    ! -------------
    integer :: icol, iband, igpt

    !$acc data present(arr_in, arr_out, band_limits)
    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do iband = 1, nband
        do igpt = band_limits(1, iband), band_limits(2, iband)
          arr_out(igpt, icol) = arr_in(iband,icol)
        end do
      end do
    end do
    !$acc end data
  end subroutine expand
  !--------------------------------------------------------------------------------------------------------------------

end module mo_rte_lw
