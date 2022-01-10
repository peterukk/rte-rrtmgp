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
  public :: sum_broadband, sum_broadband_nocol, sums_broadband_fac, net_broadband

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

    broadband_flux  = sum(spectral_flux, 1)

  end subroutine sum_broadband

  pure subroutine sum_broadband_nocol(ngpt, nlev, spectral_flux, broadband_flux) bind (C, name="sum_broadband_nocol")
    integer,                         intent(in ) :: nlev, ngpt
    real(wp), dimension(ngpt, nlev), intent(in ) :: spectral_flux
    real(wp), dimension(nlev),       intent(out) :: broadband_flux

    broadband_flux  = sum(spectral_flux, 1)

  end subroutine sum_broadband_nocol

  pure subroutine sums_broadband_fac(ngpt, nlev, fac, radn_up, radn_dn, flux_up, flux_dn) &
    bind (C, name="sums_broadband_fac")
    integer,                          intent(in ) :: nlev, ngpt
    real(wp),                         intent(in) :: fac
    real(wp), dimension(ngpt, nlev),  intent(in ) :: radn_up, radn_dn
    real(wp), dimension(nlev),        intent(out) :: flux_up, flux_dn
    integer :: ilev, igpt
    real(wp) :: sums_dn(4), sums_up(4)

    if (mod(ngpt,4) == 0)  then
      do ilev = 1, nlev
        sums_up = 0.0_wp
        sums_dn = 0.0_wp
        do igpt = 1, ngpt, 4
          sums_up(1) = sums_up(1) + fac*radn_up(igpt, ilev)
          sums_up(2) = sums_up(2) + fac*radn_up(igpt+1, ilev)
          sums_up(3) = sums_up(3) + fac*radn_up(igpt+2, ilev)
          sums_up(4) = sums_up(4) + fac*radn_up(igpt+3, ilev)

          sums_dn(1) = sums_dn(1) + fac*radn_dn(igpt, ilev)
          sums_dn(2) = sums_dn(2) + fac*radn_dn(igpt+1, ilev)
          sums_dn(3) = sums_dn(3) + fac*radn_dn(igpt+2, ilev)
          sums_dn(4) = sums_dn(4) + fac*radn_dn(igpt+3, ilev)
        end do
        flux_up(ilev) = sums_up(1) + sums_up(2) + sums_up(3) + sums_up(4)
        flux_dn(ilev) = sums_dn(1) + sums_dn(2) + sums_dn(3) + sums_dn(4)
      end do
    else
      flux_up = fac*sum(radn_up,1)
      flux_dn = fac*sum(radn_dn,1)
    end if

  end subroutine sums_broadband_fac
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
