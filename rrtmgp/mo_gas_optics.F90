! This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
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
! -------------------------------------------------------------------------------------------------
module mo_gas_optics
  use mo_rte_kind,           only: wp, sp
  use mo_source_functions,   only: ty_source_func_lw
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_optical_props,      only: ty_optical_props, ty_optical_props_arry
  use mod_network,           only: network_type


  type, abstract, extends(ty_optical_props), public :: ty_gas_optics
  contains
    generic,   public :: gas_optics => gas_optics_int, gas_optics_ext !, gas_optics_ext_nn
    !
    ! Deferred procedures -- each must be implemented in each child class with
    !   arguments following the abstract interface (defined below)
    !
    ! gas_optics_int and gas_optics_ext should be declared private in concrete classes
    !    but need to be visible in the abstract type or the interfaces won't be inherited
    ! See https://software.intel.com/en-us/forums/intel-fortran-compiler-for-linux-and-mac-os-x/topic/681705
    !
    procedure(gas_optics_int_abstract),     deferred  :: gas_optics_int
    procedure(gas_optics_ext_abstract),     deferred  :: gas_optics_ext
    ! procedure(gas_optics_ext_nn_abstract),  deferred  :: gas_optics_ext_nn
    procedure(logical_abstract), deferred, public     :: source_is_internal
    procedure(logical_abstract), deferred, public     :: source_is_external
    procedure(real_abstract),    deferred, public     :: get_press_min
    procedure(real_abstract),    deferred, public     :: get_press_max
    procedure(real_abstract),    deferred, public     :: get_temp_min
    procedure(real_abstract),    deferred, public     :: get_temp_max
  end type
  !
  ! Interfaces for the methods to be implemented
  !
  abstract interface
    !--------------------------------------------------------------------------------------------------------------------
    !
    ! Compute gas optical depth given temperature, pressure, and composition
    !
    function gas_optics_ext_abstract(this,                         &
                                     play, plev, tlay, gas_desc,   & ! mandatory inputs
                                     optical_props, toa_src,       & ! mandatory outputs
                                     col_dry, neural_nets) result(error_msg)      ! optional input
      import ty_gas_optics, wp, sp, ty_gas_concs, ty_optical_props_arry, network_type
      class(ty_gas_optics), intent(in) :: this
      real(wp), dimension(:,:), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                 plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                                 tlay      ! layer temperatures [K]; (nlay,ncol)
      type(ty_gas_concs),       intent(in   ) :: gas_desc  ! Gas volume mixing ratios
      class(ty_optical_props_arry),  &
                                intent(inout) :: optical_props
      real(wp), dimension(:,:), intent(  out) :: toa_src     ! Incoming solar irradiance(ncol,ngpt)
      character(len=128)                      :: error_msg
      ! Optional inputs
      real(wp), dimension(:,:), intent(in   ), &
                             optional, target :: col_dry ! Column dry amount; dim(nlay,ncol)
    ! Optional input: neural network model (uses NN kernel if present)
                             type(network_type), dimension(2), intent(in), optional      :: neural_nets ! Planck fraction model, optical depth model 
    end function gas_optics_ext_abstract


    function gas_optics_int_abstract(this,                             &
                                     play, plev, tlay, tsfc, gas_desc, &
                                     optical_props, sources,           &
                                     col_dry, tlev, neural_nets       &
                                      ) result(error_msg)
      import ty_gas_optics, wp, sp, ty_gas_concs, ty_optical_props_arry, ty_source_func_lw, network_type
      class(ty_gas_optics),     intent(in   ) :: this
      real(wp), dimension(:,:), intent(in   ) :: play, &   ! layer pressures [Pa, mb]; (nlay,ncol)
                                                 plev, &   ! level pressures [Pa, mb]; (nlay+1,ncol)
                                                 tlay      ! layer temperatures [K]; (nlay,ncol)
      real(wp), dimension(:),   intent(in   ) :: tsfc      ! surface skin temperatures [K]; (ncol)
      type(ty_gas_concs),       intent(in   ) :: gas_desc  ! Gas volume mixing ratios
      class(ty_optical_props_arry),  &
                                intent(inout) :: optical_props ! Optical properties
      class(ty_source_func_lw    ),  &
                                intent(inout) :: sources       ! Planck sources
      character(len=128)                      :: error_msg
      real(wp), dimension(:,:), intent(in   ), &
                            optional, target :: col_dry, &  ! Column dry amount; dim(nlay,ncol)
                                                   tlev        ! level temperatures [K]l (nlay+1,ncol)
    ! Optional input: neural network model (uses NN kernel if present)
      type(network_type), dimension(2), intent(in), optional      :: neural_nets ! Planck fraction model, optical depth model                   
    end function gas_optics_int_abstract

    !--------------------------------------------------------------------------------------------------------------------
    function logical_abstract(this)
      import ty_gas_optics
      class(ty_gas_optics),     intent(in   ) :: this
      logical                                 :: logical_abstract
    end function logical_abstract
    !--------------------------------------------------------------------------------------------------------------------
    function real_abstract(this)
      import ty_gas_optics, wp
      class(ty_gas_optics),     intent(in   ) :: this
      real(wp)                                :: real_abstract
    end function real_abstract
    !--------------------------------------------------------------------------------------------------------------------
  end interface
end module mo_gas_optics
