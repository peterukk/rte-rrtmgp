! This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
!
! Contacts: Robert Pincus and Eli Mlawer
! email:  rrtmgp@aer.com
!
! Copyright 2020,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
! -------------------------------------------------------------------------------------------------
!
! Control over input sanitization in Fortan front-end
!   Module variables can be changed only by calling one of the included subroutine
!
! -------------------------------------------------------------------------------------------------
module mo_rte_rrtmgp_config

  use mo_rte_kind, only: wl
  implicit none
  private

  logical(wl), protected, public :: check_extents = .true.
  logical(wl), protected, public :: check_values  = .false.

  ! RTE-Longwave options
  ! Compute surface temperature Jacobian of fluxes?
  logical(wl), parameter, public :: compute_Jac = .false.
  ! Use Pade approximant in the computation of LW upward and downward source to avoid conditional for low tau?
  logical(wl), parameter, public :: use_Pade_source = .false.

  ! ------------------------ FOR RRTMGP-NN ------------------------ 
  ! The NN models use ALL RRTMGP gases as input. In the shortwave, these are: h2o, o3, co2, n2o, ch4 
  ! In the longwave: h2o, o3, co2, n2o, ch4, cfc11, cfc12, co, ccl4, cfc22, hfc143a, hfc125, hfc23, hfc32, hfc134a, cf4
  ! Are all of these gases provided in the gas concentrations (ty_gas_concs)? 
  ! If so, we can use a faster kernel for computing NN inputs
  logical(wl), protected, public :: nn_all_gases_exist  = .false.
  !
  ! How to handle a required gas species if nn_all_gases_exist = .false. and a gas is missing?
  ! We can either use a concentration of zero, or a reference gas concentration. These reference values
  ! are scalar, i.e. height independent, and stored in rrtmgp_ref_concentrations for each greenhouse gas 
  ! except H2O and O3 and for three different scenarios (present-day, pre-industrial or future). 
  ! Configure using the integer variable scenario_index: 
  ! Set 0 for zero, 1 for present-day, 2 for pre-industrial or 3 for future reference value.  
  integer, protected, public    :: nn_scenario_index     = 0  ! must be 0, 1, 2, or 3
  ! --------------------------------------------------------------- 

  interface rte_rrtmgp_config_checks
    module procedure rte_rrtmgp_config_checks_each, rte_rrtmgp_config_checks_all
  end interface
  public :: rte_rrtmgp_config_checks
contains
  ! --------------------------------------------------------------
  subroutine rte_rrtmgp_config_checks_each(extents, values)
    logical(wl), intent(in) :: extents, values

    check_extents = extents
    check_values  = values
  end subroutine rte_rrtmgp_config_checks_each
  ! --------------------------------------------------------------
  subroutine rte_rrtmgp_config_checks_all(do_checks)
    logical(wl), intent(in) :: do_checks

    check_extents = do_checks
    check_values  = do_checks
  end subroutine rte_rrtmgp_config_checks_all
  ! --------------------------------------------------------------
end module mo_rte_rrtmgp_config
