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
! Compute output quantities from RTE based on spectrally-resolved flux profiles
!    This module contains an abstract class and a broadband implmentation that sums over all spectral points
!    The abstract base class defines the routines that extenstions must implement: reduce() and are_desired()
!    The intent is for users to extend it as required, using mo_flxues_broadband as an example
!
! -------------------------------------------------------------------------------------------------
module mo_fluxes
  use mo_rte_kind,       only: wp
  use mo_rte_util_array, only: extents_are
  use mo_optical_props,  only: ty_optical_props
  use mo_fluxes_broadband_kernels, &
                         only: sum_broadband, net_broadband
  implicit none
  private
  ! -----------------------------------------------------------------------------------------------
  !
  ! Abstract base class
  !   reduce() function accepts spectral flux profiles, computes desired outputs
  !   are_desired() returns a logical - does it makes sense to invoke reduce()?
  !
  ! -----------------------------------------------------------------------------------------------
  type, abstract, public :: ty_fluxes
  contains
    procedure(reduce_abstract),      deferred, public :: reduce
    procedure(are_desired_abstract), deferred, public :: are_desired
  end type ty_fluxes
  ! -----------------------------------------------------------------------------------------------
  !
  ! Class implementing broadband integration for the complete flux profile
  !   Data components are pointers so results can be written directly into memory
  !
  ! -----------------------------------------------------------------------------------------------
  type, extends(ty_fluxes), public :: ty_fluxes_broadband
    real(wp), dimension(:,:), pointer :: flux_up => NULL(), flux_dn => NULL()
    real(wp), dimension(:,:), pointer :: flux_net => NULL()    ! Net (down - up)
    real(wp), dimension(:,:), pointer :: flux_dn_dir => NULL() ! Direct flux down
  contains
    procedure, public :: reduce      => reduce_broadband
    procedure, public :: are_desired => are_desired_broadband
  end type ty_fluxes_broadband
  ! -----------------------------------------------------------------------------------------------

