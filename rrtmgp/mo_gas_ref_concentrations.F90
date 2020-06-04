! gas_ref_concentrations code is part of RRTM for GCM Applications - Parallel - Neural Networks (RRTMGP-NN)
!
! Contacts: Peter Ukkonen, Robert Pincus and Eli Mlawer
! email:  rrtmgp@aer.com
!
! Copyright 2015-2018,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
! -------------------------------------------------------------------------------------------------
! Reference concentrations for RRTMGP long-wave gases which are used in the neural network code
! if user has not provided them at runtime. Missing gases can either be set to zero, pre-industrial,
! present-day or future concentration: or modify these tables for custom values.
! -------------------------------------------------------------------------------------------------
module mo_gas_ref_concentrations
  use mo_rte_kind, only: wp
  use mo_rrtmgp_util_string, only: lower_case

  implicit none
  private
  public :: get_ref_vmr

contains
  ! -----------------------------------------

  function get_ref_vmr(iexp, gas, array) result(error_msg)
    integer,                  intent(in)  :: iexp
    character(len=*),         intent(in ) :: gas
    real(wp), dimension(:,:), intent(out) :: array
    character(len=128)                    :: error_msg
    ! ---------------------
    real(wp) :: vmr
    integer :: ilay, icol, igas, find_gas
    real(wp), dimension(14,3)     :: ref_conc_arrays

    character(32), dimension(14)  :: stored_gases = &
    [character(len=32) ::'co2', 'n2o', 'co', 'ch4', &
    'ccl4',   'cfc11', 'cfc12', 'cfc22',  'hfc143a',   &
    'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4']
    ! ---------------------
  ! For each gas, three reference values of mole fraction are stored: 
  !     Present-day,        pre-industrial, and future
    ref_conc_arrays = transpose(reshape( &
       [397.5470E-6_wp,     284.3170E-6_wp,     1066.850E-6_wp, &   ! co2
        326.9880E-9_wp,     273.0211E-9_wp,     389.3560E-9_wp, &   ! n2o
        1.200000E-7_wp,     1.000000E-8_wp,     1.800000E-7_wp, &   ! co
        1831.471E-9_wp,     808.2490E-9_wp,     2478.709E-9_wp, &   ! ch4
        83.06993E-12_wp,    0.0250004E-12_wp,   6.082623E-12_wp,&   ! ccl4
        233.0799E-12_wp,    0.0000000E-12_wp,   57.17037E-12_wp,&   ! cfc11
        520.5810E-12_wp,    0.0000000E-12_wp,   221.1720E-12_wp,&   ! cfc12 
        229.5421E-12_wp,    0.0000000E-12_wp,   0.856923E-12_wp,&   ! cfc22 = hcfc22 ?
        15.25278E-12_wp,    0.0000000E-12_wp,   713.8991E-12_wp,&   ! hfc143a
        15.35501E-12_wp,    0.0000000E-12_wp,   966.1801E-12_wp,&   ! hfc125
        26.89044E-12_wp,    0.0000000E-12_wp,   24.61550E-12_wp,&   ! hfc23
        8.336969E-12_wp,    0.0002184E-12_wp,   0.046355E-12_wp,&   ! hfc32
        80.51573E-12_wp,    0.0000000E-12_wp,   421.3692E-12_wp,&   ! hfc134a
        81.09249E-12_wp,    34.050000E-12_wp,   126.5040E-12_wp],&  ! cf4
                            [3,14]))
    error_msg = ''

    find_gas = -1
    do igas = 1, size(stored_gases)
      if (lower_case(trim(stored_gases(igas))) == lower_case(trim(gas))) then
        find_gas = igas
      end if
    end do

    if (find_gas == -1) then
      error_msg = 'gas_ref_concs-get_ref_vmr; gas ' // trim(gas) // ' not found'
    end if

    if(error_msg /= "") return

    vmr = ref_conc_arrays(find_gas, iexp)
    do concurrent(icol = 1 : size(array,2))
      do concurrent (ilay = 1 : size(array,1))
        array(ilay,icol) = vmr
      end do
    end do
  end function get_ref_vmr

end module mo_gas_ref_concentrations
