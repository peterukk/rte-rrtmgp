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
! Kernels for arrays of optical properties:
!   delta-scaling
!   adding two sets of properties
!   extracting subsets
!   validity checking
!
! -------------------------------------------------------------------------------------------------

module mo_optical_props_kernels
  use, intrinsic :: iso_c_binding
  use mo_rte_kind, only: wp, wl
  implicit none

  public
  interface delta_scale_2str_kernel
    module procedure delta_scale_2str_f_k, delta_scale_2str_k
  end interface

  interface extract_subset
    module procedure extract_subset_dim1_3d!, extract_subset_dim2_4d
    module procedure extract_subset_absorption_tau
  end interface extract_subset

  real(wp), parameter, private :: eps = 3.0_wp*tiny(1.0_wp)
contains
  ! -------------------------------------------------------------------------------------------------
  !
  ! Delta-scaling, provided only for two-stream properties at present
  !
  ! -------------------------------------------------------------------------------------------------
  ! Delta-scale two-stream optical properties
  !   user-provided value of f (forward scattering)
  !
  pure subroutine delta_scale_2str_f_k(ngpt, nlay, ncol, tau, ssa, g, f) &
      bind(C, name="delta_scale_2str_f_k")
    integer,                               intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) ::  tau, ssa, g
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) ::  f

    real(wp) :: wf
    integer  :: igpt, ilay ,icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          wf = ssa(igpt,ilay,icol) * f(igpt,ilay,icol)
          tau(igpt,ilay,icol) = (1._wp - wf) * tau(igpt,ilay,icol)
          ssa(igpt,ilay,icol) = (ssa(igpt,ilay,icol) - wf) /  max(eps,(1.0_wp - wf))
          g  (igpt,ilay,icol) = (g  (igpt,ilay,icol) - f(igpt,ilay,icol)) / &
                                        max(eps,(1._wp - f(igpt,ilay,icol)))
        end do
      end do
    end do

  end subroutine delta_scale_2str_f_k
  ! ---------------------------------
  ! Delta-scale
  !   f = g*g
  !
  pure subroutine delta_scale_2str_k(ngpt, nlay, ncol, tau, ssa, g) &
      bind(C, name="delta_scale_2str_k")
    integer,                               intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) ::  tau, ssa, g

    real(wp) :: f, wf
    integer  :: igpt, ilay ,icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          f  = g  (igpt,ilay,icol) * g  (igpt,ilay,icol)
          wf = ssa(igpt,ilay,icol) * f
          tau(igpt,ilay,icol) = (1._wp - wf) * tau(igpt,ilay,icol)
          ssa(igpt,ilay,icol) = (ssa(igpt,ilay,icol) - wf) /  max(eps,(1.0_wp - wf))
          g  (igpt,ilay,icol) = (g  (igpt,ilay,icol) -  f) /  max(eps,(1.0_wp -  f))
        end do
      end do
    end do

  end subroutine delta_scale_2str_k
  ! -------------------------------------------------------------------------------------------------
  !
  ! Addition of optical properties: the first set are incremented by the second set.
  !
  !   There are three possible representations of optical properties (scalar = optical depth only;
  !   two-stream = tau, single-scattering albedo, and asymmetry factor g, and
  !   n-stream = tau, ssa, and phase function moments p.) Thus we need nine routines, three for
  !   each choice of representation on the left hand side times three representations of the
  !   optical properties to be added.
  !
  !   There are two sets of these nine routines. In the first the two sets of optical
  !   properties are defined at the same spectral resolution. There is also a set of routines
  !   to add properties defined at lower spectral resolution to a set defined at higher spectral
  !   resolution (adding properties defined by band to those defined by g-point)
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine increment_1scalar_by_1scalar(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2) bind(C, name="increment_1scalar_by_1scalar")
    integer,                              intent(in  ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2

    integer  :: igpt, ilay ,icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
        end do
      end do
    end do
  end subroutine increment_1scalar_by_1scalar
  ! ---------------------------------
  ! increment 1scalar by 2stream
  pure subroutine increment_1scalar_by_2stream(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2, ssa2) bind(C, name="increment_1scalar_by_2stream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2

    integer  :: igpt, ilay ,icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + &
                                 tau2(igpt,ilay,icol) * (1._wp - ssa2(igpt,ilay,icol))
        end do
      end do
    end do
  end subroutine increment_1scalar_by_2stream
  ! ---------------------------------
  ! increment 1scalar by nstream
  pure subroutine increment_1scalar_by_nstream(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2, ssa2) bind(C, name="increment_1scalar_by_nstream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2

    integer  :: igpt, ilay ,icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + &
                                 tau2(igpt,ilay,icol) * (1._wp - ssa2(igpt,ilay,icol))
        end do
      end do
    end do
  end subroutine increment_1scalar_by_nstream
  ! ---------------------------------
  ! ---------------------------------
  ! increment 2stream by 1scalar
  pure subroutine increment_2stream_by_1scalar(ngpt, nlay, ncol, &
                                               tau1, ssa1,       &
                                               tau2) bind(C, name="increment_2stream_by_1scalar")
    integer,                              intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2

    integer  :: igpt, ilay ,icol
    real(wp) :: tau12

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          ssa1(igpt,ilay,icol) = tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
          ! g is unchanged
        end do
      end do
    end do
  end subroutine increment_2stream_by_1scalar
  ! ---------------------------------
  ! increment 2stream by 2stream
  pure subroutine increment_2stream_by_2stream(ngpt, nlay, ncol, &
                                               tau1, ssa1, g1,   &
                                               tau2, ssa2, g2) bind(C, name="increment_2stream_by_2stream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1, g1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2, g2

    integer :: igpt, ilay ,icol
    real(wp) :: tau12, tauscat12

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! t=tau1 + tau2
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          ! w=(tau1*ssa1 + tau2*ssa2) / t
          tauscat12 = tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
                      tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol)
          g1(igpt,ilay,icol) = &
            (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * g1(igpt,ilay,icol) + &
             tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol) * g2(igpt,ilay,icol)) &
              / max(eps,tauscat12)
          ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
        end do
      end do
    end do
  end subroutine increment_2stream_by_2stream
  ! ---------------------------------
  ! increment 2stream by nstream
  pure subroutine increment_2stream_by_nstream(ngpt, nlay, ncol, nmom2, &
                                               tau1, ssa1, g1,          &
                                               tau2, ssa2, p2) bind(C, name="increment_2stream_by_nstream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol, nmom2
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1, g1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2
    real(wp), dimension(nmom2, &
                        ngpt,nlay,ncol), intent(in   ) :: p2

    integer  :: igpt, ilay ,icol
    real(wp) :: tau12, tauscat12

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! t=tau1 + tau2
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          ! w=(tau1*ssa1 + tau2*ssa2) / t
          tauscat12 = &
             tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
             tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol)
          g1(igpt,ilay,icol) = &
            (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * g1(   igpt,ilay,icol)+ &
             tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol) * p2(1, igpt,ilay,icol)) / max(eps,tauscat12)
          ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
        end do
      end do
    end do
  end subroutine increment_2stream_by_nstream
  ! ---------------------------------
  ! ---------------------------------
  ! increment nstream by 1scalar
  pure subroutine increment_nstream_by_1scalar(ngpt, nlay, ncol, &
                                               tau1, ssa1,       &
                                               tau2) bind(C, name="increment_nstream_by_1scalar")
    integer,                              intent(in   ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2

    integer  :: igpt, ilay ,icol
    real(wp) :: tau12

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          ssa1(igpt,ilay,icol) = tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
          ! p is unchanged
        end do
      end do
    end do
  end subroutine increment_nstream_by_1scalar
  ! ---------------------------------
  ! increment nstream by 2stream
  pure subroutine increment_nstream_by_2stream(ngpt, nlay, ncol, nmom1, &
                                               tau1, ssa1, p1,          &
                                               tau2, ssa2, g2) bind(C, name="increment_nstream_by_2stream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol, nmom1
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nmom1, &
                        ngpt,nlay,ncol), intent(inout) :: p1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2, g2

    integer  :: igpt, ilay ,icol
    real(wp) :: tau12, tauscat12
    real(wp), dimension(nmom1) :: temp_moms ! TK
    integer  :: imom  !TK

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          tauscat12 = &
             tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
             tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol)
          !
          ! Here assume Henyey-Greenstein
          !
          temp_moms(1) = g2(igpt,ilay,icol)
          do imom = 2, nmom1
            temp_moms(imom) = temp_moms(imom-1) * g2(igpt,ilay,icol)
          end do
          p1(1:nmom1, igpt,ilay,icol) = &
              (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * p1(1:nmom1, igpt,ilay,icol) + &
               tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol) * temp_moms(1:nmom1)  ) / max(eps,tauscat12)
          ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
        end do
      end do
    end do
  end subroutine increment_nstream_by_2stream
  ! ---------------------------------
  ! increment nstream by nstream
  pure subroutine increment_nstream_by_nstream(ngpt, nlay, ncol, nmom1, nmom2, &
                                               tau1, ssa1, p1,                 &
                                               tau2, ssa2, p2) bind(C, name="increment_nstream_by_nstream")
    integer,                              intent(in   ) :: ngpt, nlay, ncol, nmom1, nmom2
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nmom1, &
                        ngpt,nlay,ncol), intent(inout) :: p1
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau2, ssa2
    real(wp), dimension(nmom2, &
                        ngpt,nlay,ncol), intent(in   ) :: p2

    integer  :: igpt, ilay ,icol, mom_lim
    real(wp) :: tau12, tauscat12

    mom_lim = min(nmom1, nmom2)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau12 = tau1(igpt,ilay,icol) + tau2(igpt,ilay,icol)
          tauscat12 = &
             tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
             tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol)
          !
          ! If op2 has more moments than op1 these are ignored;
          !   if it has fewer moments the higher orders are assumed to be 0
          !
          p1(1:mom_lim, igpt,ilay,icol) = &
              (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * p1(1:mom_lim, igpt,ilay,icol) + &
               tau2(igpt,ilay,icol) * ssa2(igpt,ilay,icol) * p2(1:mom_lim, igpt,ilay,icol)) / max(eps,tauscat12)
          ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
          tau1(igpt,ilay,icol) = tau12
        end do
      end do
    end do
  end subroutine increment_nstream_by_nstream
  ! -------------------------------------------------------------------------------------------------
  !
  ! Incrementing when the second set of optical properties is defined at lower spectral resolution
  !   (e.g. by band instead of by gpoint)
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine inc_1scalar_by_1scalar_bybnd(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2,             &
                                               nbnd, gpt_lims) bind(C, name="inc_1scalar_by_1scalar_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer :: ibnd, igpt, ilay, icol
    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
          end do
        end do
      end do
    end do
  end subroutine inc_1scalar_by_1scalar_bybnd
  ! ---------------------------------
  ! increment 1scalar by 2stream
  pure subroutine inc_1scalar_by_2stream_bybnd(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2, ssa2,       &
                                               nbnd, gpt_lims) bind(C, name="inc_1scalar_by_2stream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer :: ibnd, igpt, ilay, icol

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol) * (1._wp - ssa2(ibnd,ilay,icol))
          end do
        end do
      end do
    end do
  end subroutine inc_1scalar_by_2stream_bybnd
  ! ---------------------------------
  ! increment 1scalar by nstream
  pure subroutine inc_1scalar_by_nstream_bybnd(ngpt, nlay, ncol, &
                                               tau1,             &
                                               tau2, ssa2,       &
                                               nbnd, gpt_lims) bind(C, name="inc_1scalar_by_nstream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer :: ibnd, igpt, ilay, icol
    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau1(igpt,ilay,icol) = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol) * (1._wp - ssa2(ibnd,ilay,icol))
          end do
        end do
      end do
    end do
  end subroutine inc_1scalar_by_nstream_bybnd

    ! ---------------------------------
  ! increment 2stream by 1scalar
  pure subroutine inc_2stream_by_1scalar_bybnd(ngpt, nlay, ncol, &
                                               tau1, ssa1,       &
                                               tau2,             &
                                               nbnd, gpt_lims) bind(C, name="inc_2stream_by_1scalar_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd
    real(wp) :: tau12

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            ssa1(igpt,ilay,icol) = tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
            ! g is unchanged
          end do
        end do
      end do
    end do
  end subroutine inc_2stream_by_1scalar_bybnd
  ! ---------------------------------
  ! increment 2stream by 2stream
  pure subroutine inc_2stream_by_2stream_bybnd(ngpt, nlay, ncol, &
                                               tau1, ssa1, g1,   &
                                               tau2, ssa2, g2,   &
                                               nbnd, gpt_lims) bind(C, name="inc_2stream_by_2stream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1, g1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2, g2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd
    real(wp) :: tau12, tauscat12

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            ! t=tau1 + tau2
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            ! w=(tau1*ssa1 + tau2*ssa2) / t
            tauscat12 = &
               tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol)
            g1(igpt,ilay,icol) = &
              (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * g1(igpt,ilay,icol) + &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol) * g2(ibnd,ilay,icol)) / max(eps,tauscat12)
            ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
          end do
        end do
      end do
    end do
  end subroutine inc_2stream_by_2stream_bybnd
  ! ---------------------------------
  ! increment 2stream by nstream
  pure subroutine inc_2stream_by_nstream_bybnd(ngpt, nlay, ncol, nmom2, &
                                               tau1, ssa1, g1,          &
                                               tau2, ssa2, p2,          &
                                               nbnd, gpt_lims) bind(C, name="inc_2stream_by_nstream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nmom2, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1, g1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2
    real(wp), dimension(nmom2, &
                        nbnd,nlay,ncol), intent(in   ) :: p2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd
    real(wp) :: tau12, tauscat12

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            ! t=tau1 + tau2
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            ! w=(tau1*ssa1 + tau2*ssa2) / t
            tauscat12 = &
               tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol)
            g1(igpt,ilay,icol) = &
              (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * g1(   igpt,ilay,icol)+ &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol) * p2(1, ibnd,ilay,icol)) / max(eps,tauscat12)
            ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
          end do
        end do
      end do
    end do
  end subroutine inc_2stream_by_nstream_bybnd
  ! ---------------------------------
  ! ---------------------------------
  ! increment nstream by 1scalar
  pure subroutine inc_nstream_by_1scalar_bybnd(ngpt, nlay, ncol, &
                                               tau1, ssa1,       &
                                               tau2,             &
                                               nbnd, gpt_lims) bind(C, name="inc_nstream_by_1scalar_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd
    real(wp) :: tau12

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            ssa1(igpt,ilay,icol) = tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
            ! p is unchanged
          end do
        end do
      end do
    end do
  end subroutine inc_nstream_by_1scalar_bybnd
  ! ---------------------------------
  ! increment nstream by 2stream
  pure subroutine inc_nstream_by_2stream_bybnd(ngpt, nlay, ncol, nmom1, &
                                               tau1, ssa1, p1,          &
                                               tau2, ssa2, g2,          &
                                               nbnd, gpt_lims) bind(C, name="inc_nstream_by_2stream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nmom1, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nmom1, &
                        ngpt,nlay,ncol), intent(inout) :: p1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2, g2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd
    real(wp) :: tau12, tauscat12
    real(wp), dimension(nmom1) :: temp_moms ! TK
    integer  :: imom  !TK

    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            tauscat12 = &
               tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol)
            !
            ! Here assume Henyey-Greenstein
            !
            temp_moms(1) = g2(ibnd,ilay,icol)
            do imom = 2, nmom1
              temp_moms(imom) = temp_moms(imom-1) * g2(ibnd,ilay,icol)
            end do
            p1(1:nmom1, igpt,ilay,icol) = &
                (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * p1(1:nmom1, igpt,ilay,icol) + &
                 tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol) * temp_moms(1:nmom1)  ) / max(eps,tauscat12)
            ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
          end do
        end do
      end do
    end do
  end subroutine inc_nstream_by_2stream_bybnd
  ! ---------------------------------
  ! increment nstream by nstream
  pure subroutine inc_nstream_by_nstream_bybnd(ngpt, nlay, ncol, nmom1, nmom2, &
                                               tau1, ssa1, p1,                 &
                                               tau2, ssa2, p2,                 &
                                               nbnd, gpt_lims) bind(C, name="inc_nstream_by_nstream_bybnd")
    integer,                             intent(in   ) :: ngpt, nlay, ncol, nmom1, nmom2, nbnd
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau1, ssa1
    real(wp), dimension(nmom1, &
                        ngpt,nlay,ncol), intent(inout) :: p1
    real(wp), dimension(nbnd,nlay,ncol), intent(in   ) :: tau2, ssa2
    real(wp), dimension(nmom2, &
                        nbnd,nlay,ncol), intent(in   ) :: p2
    integer,  dimension(2,nbnd),         intent(in   ) :: gpt_lims ! Starting and ending gpoint for each band

    integer  :: igpt, ilay ,icol, ibnd, mom_lim
    real(wp) :: tau12, tauscat12

    mom_lim = min(nmom1, nmom2)
    do icol = 1, ncol
      do ilay = 1, nlay
        do ibnd = 1, nbnd
          do igpt = gpt_lims(1, ibnd), gpt_lims(2, ibnd)
            tau12 = tau1(igpt,ilay,icol) + tau2(ibnd,ilay,icol)
            tauscat12 = &
               tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) + &
               tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol)
            !
            ! If op2 has more moments than op1 these are ignored;
            !   if it has fewer moments the higher orders are assumed to be 0
            !
            p1(1:mom_lim, igpt,ilay,icol) = &
                (tau1(igpt,ilay,icol) * ssa1(igpt,ilay,icol) * p1(1:mom_lim, igpt,ilay,icol) + &
                 tau2(ibnd,ilay,icol) * ssa2(ibnd,ilay,icol) * p2(1:mom_lim, ibnd,ilay,icol)) / max(eps,tauscat12)
            ssa1(igpt,ilay,icol) = tauscat12 / max(eps,tau12)
            tau1(igpt,ilay,icol) = tau12
          end do
        end do
      end do
    end do
  end subroutine inc_nstream_by_nstream_bybnd
  ! -------------------------------------------------------------------------------------------------
  !
  ! Subsetting, meaning extracting some portion of the 3D domain
  !
  ! -------------------------------------------------------------------------------------------------
  pure subroutine extract_subset_dim1_3d(ngpt, nlay, ncol, array_in, colS, colE, array_out) &
    bind (C, name="extract_subset_dim1_3d")
    integer,                             intent(in ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(in ) :: array_in
    integer,                             intent(in ) :: colS, colE
    real(wp), dimension(ngpt,nlay,colE-colS+1), intent(out) :: array_out

    integer :: igpt, ilay ,icol
    do icol = colS, colE
      do ilay = 1, nlay
        do igpt = 1, ngpt
          array_out(igpt, ilay, icol-colS+1) = array_in(igpt, ilay ,icol)
        end do
      end do
    end do

  end subroutine extract_subset_dim1_3d
  ! ! ---------------------------------
  ! pure subroutine extract_subset_dim2_4d(nmom, ngpt, nlay, ncol, array_in, colS, colE, array_out) &
  !   bind (C, name="extract_subset_dim2_4d")
  !   integer,                                  intent(in ) :: nmom, ngpt, nlay, ncol
  !   real(wp), dimension(nmom,ngpt,nlay,ncol), intent(in ) :: array_in
  !   integer,                                  intent(in ) :: colS, colE
  !   real(wp), dimension(nmom,colE-colS+1,&
  !                                 nlay,ngpt), intent(out) :: array_out

  !   integer :: igpt, ilay ,icol, imom

  !   do icol = colS, colE
  !     do ilay = 1, nlay
  !       do igpt = 1, ngpt
  !         do imom = 1, nmom
  !           array_out(imom, igpt, ilay, icol-colS+1) = array_in(imom, igpt, ilay ,icol)
  !         end do
  !       end do
  !     end do
  !   end do

  ! end subroutine extract_subset_dim2_4d
  ! ---------------------------------
  !
  ! Extract the absorption optical thickness which requires mulitplying by 1 - ssa
  !
  pure subroutine extract_subset_absorption_tau(ngpt, nlay, ncol, tau_in, ssa_in, &
                                                colS, colE, tau_out)              &
    bind (C, name="extract_subset_absorption_tau")
    integer,                             intent(in ) :: ngpt, nlay, ncol
    real(wp), dimension(ngpt,nlay,ncol), intent(in ) :: tau_in, ssa_in
    integer,                             intent(in ) :: colS, colE
    real(wp), dimension(ngpt,nlay, colE-colS+1), intent(out) :: tau_out

    integer :: igpt, ilay ,icol

    do icol = colS, colE
      do ilay = 1, nlay
        do igpt = 1, ngpt
          tau_out(igpt, ilay, icol-colS+1) = &
            tau_in(igpt, ilay ,icol) * (1._wp - ssa_in(igpt, ilay ,icol))
        end do
      end do
    end do

  end subroutine extract_subset_absorption_tau
end module mo_optical_props_kernels
