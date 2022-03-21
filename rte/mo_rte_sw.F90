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
!    atmospheric optical properties on a spectral grid
!    information about vertical ordering
!    boundary conditions
!      solar zenith angle, spectrally-resolved incident colimated flux, surface albedos for direct and diffuse radiation
!    optionally, a boundary condition for incident diffuse radiation
!
! It is the user's responsibility to ensure that boundary conditions (incident fluxes, surface albedos) are on the same
!   spectral grid as the optical properties.
!
! Final output is via user-extensible ty_fluxes which must reduce the detailed spectral fluxes to
!   whatever summary the user needs.
!
! The routine does error checking and choses which lower-level kernel to invoke based on
!   what kinds of optical properties are supplied
!
! -------------------------------------------------------------------------------------------------
module mo_rte_sw
  use mo_rte_kind,          only: wp, wl
  use mo_rte_rrtmgp_config, only: check_extents, check_values
  use mo_rte_util_array,    only: any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props,     only: ty_optical_props, &
                              ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_fluxes,            only: ty_fluxes, ty_fluxes_broadband, ty_fluxes_flexible
  use mo_rte_solver_kernels, &
                            only: apply_BC, sw_solver_noscat, sw_solver_2stream
  use mo_fluxes_broadband_kernels, only : sum_broadband, sum_broadband_nocol
  implicit none
  private

  public :: rte_sw

contains
  ! --------------------------------------------------

  function rte_sw(atmos, top_at_1,                 &
                  mu0, inc_flux,                   &
                  sfc_alb_dir_gpt, sfc_alb_dif_gpt,        &
                  fluxes, inc_flux_dif &
                  ) result(error_msg)
    class(ty_optical_props_arry), intent(in   ) :: atmos           ! Optical properties provided as arrays
    logical,                      intent(in   ) :: top_at_1        ! Is the top of the domain at index 1?
                                                                   ! (if not, ordering is bottom-to-top)
    real(wp), dimension(:),       intent(in   ) :: mu0             ! cosine of solar zenith angle (ncol)
    real(wp), dimension(:,:),     intent(in   ) :: inc_flux,    &  ! incident flux at top of domain [W/m2] (ngpt, ncol)
                                                  !  sfc_alb_dir, &  ! surface albedo for direct and
                                                  !  sfc_alb_dif     ! diffuse radiation (nband, ncol)
                                                  sfc_alb_dir_gpt, &  ! surface albedo for direct and
                                                  sfc_alb_dif_gpt     ! diffuse radiation (ngpt, ncol)
    class(ty_fluxes_flexible),   intent(inout) :: fluxes                 ! Array of ty_fluxes. Default computes broadband fluxes at all levels
    real(wp), dimension(:,:), optional, contiguous, target, &
                                  intent(in   ) :: inc_flux_dif    ! incident diffuse flux at top of domain [W/m2] (ngpt, ncol)
    character(len=128)                          :: error_msg       ! If empty, calculation was successful
    ! --------------------------------
    !
    ! Local variables
    !
    integer :: ncol, nlay, ngpt, nband
    integer :: icol, igpt, ret
    logical :: do_gpt_flux = .false.
    ! integer, dimension(2,atmos%get_nband())   :: band_limits
    real(wp), dimension(:,:,:), allocatable, target :: gpt_flux_up, gpt_flux_dn, gpt_flux_dir
    ! Surface albedos expanded to g-points (now done outside RTE)
    ! real(wp), dimension(:,:),   allocatable :: sfc_alb_dir_gpt, sfc_alb_dif_gpt ! 
    real(wp), dimension(:,:), contiguous, pointer     :: inc_diff_flux
    real(wp), dimension(:,:), allocatable, target     :: inc_flux_zero
    ! ------------------------------------------------------------------------------------
    ncol  = atmos%get_ncol()
    nlay  = atmos%get_nlay()
    ngpt  = atmos%get_ngpt()
    nband = atmos%get_nband()
    ! band_limits = atmos%get_band_lims_gpoint()
    error_msg = ""

    ! ------------------------------------------------------------------------------------
    !
    ! Error checking -- consistency of sizes and validity of values
    !
    ! --------------------------------
    if(.not. fluxes%are_desired()) then
      error_msg = "rte_sw: no space allocated for fluxes"
      return
    end if

    !
    ! Sizes of input arrays
    !
    if(check_extents) then
      if(.not. extents_are(mu0, ncol)) &
        error_msg = "rte_sw: mu0 inconsistently sized"
      if(.not. extents_are(inc_flux, ngpt, ncol)) &
        error_msg = "rte_sw: inc_flux inconsistently sized"
      !if(.not. extents_are(sfc_alb_dir, nband, ncol)) &
      if(.not. extents_are(sfc_alb_dir_gpt, ngpt, ncol)) &
        error_msg = "rte_sw: sfc_alb_dir inconsistently sized"
      !if(.not. extents_are(sfc_alb_dif, nband, ncol)) &
      if(.not. extents_are(sfc_alb_dif_gpt, ngpt, ncol)) &
        error_msg = "rte_sw: sfc_alb_dif inconsistently sized"
      if(present(inc_flux_dif)) then
        if(.not. extents_are(inc_flux_dif, ngpt, ncol)) &
          error_msg = "rte_sw: inc_flux_dif inconsistently sized"
      end if
    end if

    !
    ! Values of input arrays 
    !
    if(check_values) then
      if(any_vals_outside(mu0, 0._wp, 1._wp)) &
        error_msg = "rte_sw: one or more mu0 <= 0 or > 1"
      if(any_vals_less_than(inc_flux, 0._wp)) &
        error_msg = "rte_sw: one or more inc_flux < 0"
      ! if(any_vals_outside(sfc_alb_dir,  0._wp, 1._wp)) &
      if(any_vals_outside(sfc_alb_dir_gpt,  0._wp, 1._wp)) &
        error_msg = "rte_sw: sfc_alb_dir out of bounds [0,1]"
      !if(any_vals_outside(sfc_alb_dif,  0._wp, 1._wp)) &
      if(any_vals_outside(sfc_alb_dif_gpt,  0._wp, 1._wp)) &
        error_msg = "rte_sw: sfc_alb_dif out of bounds [0,1]"
      if(present(inc_flux_dif)) then
        if(any_vals_less_than(inc_flux_dif, 0._wp)) &
          error_msg = "rte_sw: one or more inc_flux_dif < 0"
      end if
    end if

    if(len_trim(error_msg) > 0) then
      if(len_trim(atmos%get_name()) > 0) &
        error_msg = trim(atmos%get_name()) // ': ' // trim(error_msg)
      return
    end if

    if(len_trim(error_msg) > 0) return

    !
    ! Ensure values of tau, ssa, and g are reasonable
    !
    if(check_values) error_msg =  atmos%validate()
    if(len_trim(error_msg) > 0) return

    !
    ! Optionally - output spectral fluxes, not only broadband fluxes?
    ! 
    !
    do_gpt_flux = fluxes%are_desired_gpt()

    if (.not. do_gpt_flux) then  ! If not desired (and already allocated), g-point flux arrays will still be needed if..
      select type (atmos)
        class is (ty_optical_props_1scl) ! .. doing no-scattering computations
          allocate(gpt_flux_dir(ngpt, nlay+1, ncol))
          !$acc enter data create(gpt_flux_dir)
          fluxes%gpt_flux_dn_dir => gpt_flux_dir(:,:,:)
           !$acc enter data attach(fluxes%gpt_flux_dn_dir)
        class is (ty_optical_props_2str)
          ! ...or doing two-stream scattering calculations using GPU kernels
#ifdef USE_OPENACC
          do_gpt_flux = .true.
          allocate(gpt_flux_up (ngpt, nlay+1, ncol), gpt_flux_dn(ngpt, nlay+1, ncol), gpt_flux_dir(ngpt, nlay+1, ncol))
          !$acc enter data create(gpt_flux_up, gpt_flux_dn, gpt_flux_dir)
          fluxes%gpt_flux_up => gpt_flux_up(:,:,:)
          fluxes%gpt_flux_dn => gpt_flux_dn(:,:,:)
          fluxes%gpt_flux_dn_dir => gpt_flux_dir(:,:,:)
          !$acc enter data attach(fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_dn_dir)
#endif
      end select
    end if

    ! ------------------------------------------------------------------------------------

    !   ---------------------- NOW DONE OUTSIDE RTE  ----------------------
    ! Lower boundary condition -- expand surface albedos by band to gpoints
    ! allocate(sfc_alb_dir_gpt(ngpt, ncol), sfc_alb_dif_gpt(ngpt, ncol))
    ! !$acc enter data create(sfc_alb_dir_gpt, sfc_alb_dif_gpt) copyin(band_limits)
    ! call expand(nband, ngpt, ncol, band_limits, sfc_alb_dir, sfc_alb_dir_gpt)
    ! call expand(nband, ngpt, ncol, band_limits, sfc_alb_dif, sfc_alb_dif_gpt)
    !   ---------------------- NOW DONE OUTSIDE RTE  ----------------------


    ! Boundary conditions - for computations with scattering these are passed to the kernel
    ! 
    if(present(inc_flux_dif)) then
      !$acc enter data copyin(inc_flux_dif)
      inc_diff_flux => inc_flux_dif
    else 
      allocate(inc_flux_zero(ngpt, ncol))
      !$acc enter data create(inc_flux_zero)
      !$acc parallel loop collapse(2) present(inc_flux_zero)
      do icol = 1, ncol
        do igpt = 1, ngpt
          inc_flux_zero(igpt,icol) = 0.0_wp
        end do
      end do
      inc_diff_flux => inc_flux_zero
    end if
    !$acc enter data attach(inc_diff_flux)

    ! ------------------------------------------------------------------------------------
    !
    ! Compute the radiative transfer...
    !
    !
      select type (atmos)
        class is (ty_optical_props_1scl)
          !
          ! Direct beam only - no diffuse flux
          !
          !$acc enter data copyin(inc_flux)
          call apply_BC(ngpt, nlay, ncol, logical(top_at_1, wl),  inc_flux, mu0, gpt_flux_dir)
          !$acc exit data delete(inc_flux)
          call sw_solver_noscat(ngpt, nlay, ncol, logical(top_at_1, wl), &
                                atmos%tau, mu0,                          &
                                fluxes%flux_dn_dir, fluxes%gpt_flux_dn_dir)

        class is (ty_optical_props_2str)
          !
          ! two-stream calculation with scattering
          !
          if (do_gpt_flux) then
            call sw_solver_2stream(ngpt, nlay, ncol, logical(top_at_1, wl), &
                                  inc_flux, inc_diff_flux,                 &
                                  atmos%tau, atmos%ssa, atmos%g, mu0,      &
                                  sfc_alb_dir_gpt, sfc_alb_dif_gpt,        &
                                  fluxes%flux_up, fluxes%flux_dn, fluxes%flux_dn_dir, &
                                  fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_dn_dir)
          else
            call sw_solver_2stream(ngpt, nlay, ncol, logical(top_at_1, wl), &
                                  inc_flux, inc_diff_flux,                 &
                                  atmos%tau, atmos%ssa, atmos%g, mu0,      &
                                  sfc_alb_dir_gpt, sfc_alb_dif_gpt,        &
                                  fluxes%flux_up, fluxes%flux_dn, fluxes%flux_dn_dir)
          end if

        class is (ty_optical_props_nstr)
          !
          ! n-stream calculation
          !
          ! not yet implemented so fail
          !
          error_msg = 'sw_solver(...ty_optical_props_nstr...) not yet implemented'
      end select
    !
    ! ------------------------------------------------------------------------------------
    !
    if (error_msg /= '') return

    !$acc exit data detach(fluxes%gpt_flux_up, fluxes%gpt_flux_dn, fluxes%gpt_flux_dn_dir)
    !$acc exit data delete(gpt_flux_up, gpt_flux_dn, gpt_flux_dir)

    if(.not. present(inc_flux_dif)) then
      !$acc exit data delete(inc_flux_zero)
      deallocate(inc_flux_zero)
    end if

    ! !$acc exit data delete(sfc_alb_dir_gpt, sfc_alb_dif_gpt, band_limits)

  end function rte_sw
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Expand from band to g-point dimension
  !
  subroutine expand(nband, ngpt, ncol, band_limits, arr_in, arr_out)
    integer,                          intent(in)  :: ncol, nband, ngpt
    integer,  dimension(2,nband),     intent(in)  :: band_limits
    real(wp), dimension(nband,ncol),  intent(in)  :: arr_in  ! (nband, ncol)
    real(wp), dimension(ngpt,ncol),   intent(out) :: arr_out ! (ngpt, ncol)
    ! -------------
    integer :: icol, iband, igpt

    !$acc parallel loop collapse(2) default(present)
    do icol = 1, ncol
      do iband = 1, nband
        do igpt = band_limits(1, iband), band_limits(2, iband)
          arr_out(igpt, icol) = arr_in(iband,icol)
        end do
      end do
    end do

  end subroutine expand
  
end module mo_rte_sw
