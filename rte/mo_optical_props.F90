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
! Encapsulate optical properties defined on a spectral grid of N bands.
!   The bands are described by their limiting wavenumbers. They need not be contiguous or complete.
!   A band may contain more than one spectral sub-point (g-point) in which case a mapping must be supplied.
!   A name may be provided and will be prepended to error messages.
!   The base class (ty_optical_props) encapsulates only this spectral discretization and must be initialized
!      with the spectral information before use.
!
!   Optical properties may be represented as arrays with dimensions ncol, nlay, ngpt
!   (abstract class ty_optical_props_arry).
!   The type holds arrays depending on how much information is needed
!   There are three possibilites
!      ty_optical_props_1scl holds absorption optical depth tau, used in calculations accounting for extinction and emission
!      ty_optical_props_2str holds extincion optical depth tau, single-scattering albedo ssa, and
!        asymmetry parameter g. These fields are what's needed for two-stream calculations.
!      ty_optical_props_nstr holds extincion optical depth tau, single-scattering albedo ssa, and
!        phase function moments p with leading dimension nmom. These fields are what's needed for multi-stream calculations.
!   These classes must be allocated before use. Initialization and allocation can be combined.
!   The classes have a validate() function that checks all arrays for valid values (e.g. tau > 0.)
!
! Optical properties can be delta-scaled (though this is currently implemented only for two-stream arrays)
!
! Optical properties can increment or "add themselves to" a set of properties represented with arrays
!   as long as both sets have the same underlying band structure. Properties defined by band
!   may be added to properties defined by g-point; the same value is assumed for all g-points with each band.
!
! Subsets of optical properties held as arrays may be extracted along the column dimension.
!
! -------------------------------------------------------------------------------------------------
module mo_optical_props
  use mo_rte_kind,              only: wp
  use mo_rte_util_array,        only: any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props_kernels, only: &
        increment_1scalar_by_1scalar, increment_1scalar_by_2stream, increment_1scalar_by_nstream, &
        increment_2stream_by_1scalar, increment_2stream_by_2stream, increment_2stream_by_nstream, &
        increment_nstream_by_1scalar, increment_nstream_by_2stream, increment_nstream_by_nstream, &
        inc_1scalar_by_1scalar_bybnd, inc_1scalar_by_2stream_bybnd, inc_1scalar_by_nstream_bybnd, &
        inc_2stream_by_1scalar_bybnd, inc_2stream_by_2stream_bybnd, inc_2stream_by_nstream_bybnd, &
        inc_nstream_by_1scalar_bybnd, inc_nstream_by_2stream_bybnd, inc_nstream_by_nstream_bybnd, &
        delta_scale_2str_kernel, &
        extract_subset
  implicit none
  integer, parameter :: name_len = 32
  ! -------------------------------------------------------------------------------------------------
  !
  ! Base class for optical properties
  !   Describes the spectral discretization including the wavenumber limits
  !   of each band (spectral region) and the mapping between g-points and bands
  !
  ! -------------------------------------------------------------------------------------------------
  type, public :: ty_optical_props
    integer,  dimension(:,:), allocatable :: band2gpt       ! (begin g-point, end g-point) = band2gpt(2,band)
    integer,  dimension(:),   allocatable :: gpt2band       ! band = gpt2band(g-point)
    real(wp), dimension(:,:), allocatable :: band_lims_wvn  ! (upper and lower wavenumber by band) = band_lims_wvn(2,band)
    character(len=name_len)               :: name = ""
  contains
    generic,   public  :: init => init_base, init_base_from_copy
    procedure, private :: init_base
    procedure, private :: init_base_from_copy
    procedure, public  :: is_initialized => is_initialized_base
    procedure, private :: is_initialized_base
    procedure, public  :: finalize => finalize_base
    procedure, private :: finalize_base
    procedure, public  :: get_nband
    procedure, public  :: get_ngpt
    procedure, public  :: get_gpoint_bands
    procedure, public  :: convert_band2gpt
    procedure, public  :: convert_gpt2band
    procedure, public  :: get_band_lims_gpoint
    procedure, public  :: get_band_lims_wavenumber
    procedure, public  :: get_band_lims_wavelength
    procedure, public  :: bands_are_equal
    procedure, public  :: gpoints_are_equal
    procedure, public  :: expand
    procedure, public  :: set_name
    procedure, public  :: get_name
  end type
  !----------------------------------------------------------------------------------------
  !
  ! Optical properties as arrays, normally dimensioned ncol, nlay, ngpt/nbnd
  !   The abstract base class for arrays defines what procedures will be available
  !   The optical depth field is also part of the abstract base class, since
  !    any representation of values as arrays needs an optical depth field
  !
  ! -------------------------------------------------------------------------------------------------
  type, extends(ty_optical_props), abstract, public :: ty_optical_props_arry
    real(wp), dimension(:,:,:), allocatable :: tau ! optical depth (ncol, nlay, ngpt)
  contains
    procedure, public  :: get_ncol
    procedure, public  :: get_nlay
    !
    ! Increment another set of values
    !
    procedure, public  :: increment

