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
! Kernels for computing broadband fluxes by summing over all elements in the spectral dimension
!
! -------------------------------------------------------------------------------------------------
module mo_fluxes_broadband_kernels
  use, intrinsic :: iso_c_binding
  use mo_rte_kind, only: wp
  implicit none
  private
  public :: sum_broadband, sum_broadband_nocol, net_broadband

  interface net_broadband
    module procedure net_broadband_full, net_broadband_precalc
  end interface net_broadband
contains
  ! ----------------------------------------------------------------------------
    !
    ! Spectral reduction over all points
    !
  pure subroutine sum_broadband(ngpt, nlev, ncol, spectral_flux, broadband_flux) bind(C, name="sum_broadband")
    integer,                               intent(in ) :: ngpt, nlev, ncol
    real(wp), dimension(ngpt, nlev, ncol), intent(in ) :: spectral_flux
    real(wp), dimension(nlev, ncol),       intent(out) :: broadband_flux
    integer  :: igpt, ilev, icol
    real(wp) :: bb_flux_s

    !$acc enter data create(broadband_flux)
    !$acc parallel loop gang worker collapse(2) default(present)
    do icol = 1, ncol
      do ilev = 1, nlev

        bb_flux_s = 0.0_wp
        !$acc loop vector reduction(+:bb_flux_s)
        do igpt = 1, ngpt
          bb_flux_s = bb_flux_s + spectral_flux(igpt, ilev, icol)
        end do
       broadband_flux(ilev, icol) = bb_flux_s
      end do
    end do
    !$acc exit data copyout(broadband_flux)

  end subroutine sum_broadband

  subroutine sum_broadband_nocol(ngpt, nlev, spectral_flux, broadband_flux)
    !$acc routine worker
    integer,                         intent(in ) :: nlev, ngpt
    real(wp), dimension(ngpt, nlev), intent(in ) :: spectral_flux
    real(wp), dimension(nlev),       intent(out) :: broadband_flux
    integer  :: igpt, ilev
    real(wp) :: bb_flux_s

    !$acc loop worker
    do ilev = 1, nlev
      bb_flux_s = 0.0_wp
      !$acc loop vector reduction(+:bb_flux_s)
      do igpt = 1, ngpt
        bb_flux_s = bb_flux_s + spectral_flux(igpt, ilev)
      end do
      broadband_flux(ilev) = bb_flux_s
    end do

  end subroutine sum_broadband_nocol
  ! ----------------------------------------------------------------------------
  !
  ! Net flux: Spectral reduction over all points
  !
  pure subroutine net_broadband_full(ngpt, nlev, ncol, spectral_flux_dn, spectral_flux_up, broadband_flux_net) &
    bind(C, name="net_broadband_full")
    integer,                               intent(in ) :: ngpt, nlev, ncol
    real(wp), dimension(ngpt, nlev, ncol), intent(in ) :: spectral_flux_dn, spectral_flux_up
    real(wp), dimension(nlev, ncol),       intent(out) :: broadband_flux_net

    broadband_flux_net = sum((spectral_flux_dn-spectral_flux_up),1)
      
  end subroutine net_broadband_full
  ! ----------------------------------------------------------------------------
  !
  ! Net flux when bradband flux up and down are already available
  !
  pure subroutine net_broadband_precalc(nlev, ncol, flux_dn, flux_up, broadband_flux_net) &
    bind(C, name="net_broadband_precalc")
    integer,                         intent(in ) :: nlev,  ncol
    real(wp), dimension(nlev,  ncol), intent(in ) :: flux_dn, flux_up
    real(wp), dimension(nlev,  ncol), intent(out) :: broadband_flux_net

    broadband_flux_net = flux_dn - flux_up

  end subroutine net_broadband_precalc
  ! ----------------------------------------------------------------------------
end module mo_fluxes_broadband_kernels
