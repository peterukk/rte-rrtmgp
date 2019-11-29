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
! Encapsulate source function arrays for longwave/lw/internal sources
!    and shortwave/sw/external source.
!
! -------------------------------------------------------------------------------------------------
module mo_source_functions
  use mo_rte_kind,      only: wp
  use mo_optical_props, only: ty_optical_props
  implicit none
  ! -------------------------------------------------------------------------------------------------
  !
  ! Type for longwave sources: computed at layer center, at layer edges using
  !   spectral mapping in each direction separately, and at the surface
  !
  type, extends(ty_optical_props), public :: ty_source_func_lw
    real(wp), allocatable, dimension(:,:,:) :: lay_source,     & ! Planck source at layer average temperature
                                                                 ! [W/m2] (ncol, nlay, ngpt)
                                               lev_source_inc, &  ! Planck source at layer edge,
                                               lev_source_dec, &  ! [W/m2] (ncol, nlay+1, ngpt)
                                                                  ! in increasing/decreasing ilay direction
                                                                  ! Includes spectral weighting that accounts for state-dependent
                                                                  ! frequency to g-space mapping
                                               planck_frac       ! (ncol, nlay, ngpt)
    real(wp), allocatable, dimension(:,:  ) :: sfc_source
  contains
    generic,   public :: alloc => alloc_lw, copy_and_alloc_lw
    procedure, private:: alloc_lw
    procedure, private:: copy_and_alloc_lw
    procedure, public :: is_allocated => is_allocated_lw
    procedure, public :: finalize => finalize_lw
    procedure, public :: get_subset => get_subset_range_lw
    procedure, public :: get_ncol => get_ncol_lw
    procedure, public :: get_nlay => get_nlay_lw
    ! validate?
  end type ty_source_func_lw
  ! -------------------------------------------------------------------------------------------------
  !
  ! Type for shortave sources: top-of-domain spectrally-resolved flux
  !
  type, extends(ty_optical_props), public :: ty_source_func_sw
    real(wp), allocatable, dimension(:,:  ) :: toa_source
  contains
    generic,   public :: alloc => alloc_sw, copy_and_alloc_sw
    procedure, private:: alloc_sw
    procedure, private:: copy_and_alloc_sw
    procedure, public :: is_allocated => is_allocated_sw
    procedure, public :: finalize => finalize_sw
    procedure, public :: get_subset => get_subset_range_sw
    procedure, public :: get_ncol => get_ncol_sw
    ! validate?
  end type ty_source_func_sw
  ! -------------------------------------------------------------------------------------------------
contains
  ! ------------------------------------------------------------------------------------------
  !
  !  Routines for initialization, validity checking, finalization
  !
  ! ------------------------------------------------------------------------------------------
  !
  ! Longwave
  !
  ! ------------------------------------------------------------------------------------------
  pure function is_allocated_lw(this)
    class(ty_source_func_lw), intent(in) :: this
    logical                              :: is_allocated_lw

