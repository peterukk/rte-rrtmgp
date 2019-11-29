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
  use mo_rte_kind,      only: wp, wl
  use mo_rte_util_array,only: any_vals_less_than, any_vals_outside, extents_are
  use mo_optical_props, only: ty_optical_props, &
                              ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  use mo_source_functions,   &
                        only: ty_source_func_lw
  use mo_fluxes,        only: ty_fluxes
  use mo_rte_solver_kernels, &
                        only: apply_BC, lw_solver_noscat_GaussQuad, lw_solver_2stream
  implicit none
  private

