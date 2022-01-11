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
  use mo_rte_kind, only: sp
  use mo_rrtmgp_util_string, only: lower_case

  implicit none
  private
  public :: get_ref_vmr

contains
  ! -----------------------------------------

  function get_ref_vmr(iexp, gas, vmr) result(error_msg)
    integer,                  intent(in)  :: iexp
    character(len=*),         intent(in ) :: gas
    real(sp),                 intent(out) :: vmr
    ! real(sp), dimension(:,:), intent(out) :: array
    character(len=128)                    :: error_msg
    ! ---------------------
    ! real(sp) :: vmr
    integer :: ilay, icol, igas, find_gas
    real(sp), dimension(14,3)     :: ref_conc_arrays

    character(32), dimension(14)  :: stored_gases = &
    [character(len=32) ::'co2', 'n2o', 'co', 'ch4', &
    'ccl4',   'cfc11', 'cfc12', 'cfc22',  'hfc143a',   &
    'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4']
    ! ---------------------
  ! For each gas, three reference values of mole fraction are stored: 
  !     Present-day,        pre-industrial, and future
    ref_conc_arrays = transpose(reshape( &
       [397.5470E-6_sp,     284.3170E-6_sp,     1066.850E-6_sp, &   ! co2
        326.9880E-9_sp,     273.0211E-9_sp,     389.3560E-9_sp, &   ! n2o
        1.200000E-7_sp,     1.000000E-8_sp,     1.800000E-7_sp, &   ! co
        1831.471E-9_sp,     808.2490E-9_sp,     2478.709E-9_sp, &   ! ch4
        83.06993E-12_sp,    0.0250004E-12_sp,   6.082623E-12_sp,&   ! ccl4
        233.0799E-12_sp,    0.0000000E-12_sp,   57.17037E-12_sp,&   ! cfc11
        520.5810E-12_sp,    0.0000000E-12_sp,   221.1720E-12_sp,&   ! cfc12 
        229.5421E-12_sp,    0.0000000E-12_sp,   0.856923E-12_sp,&   ! cfc22 = hcfc22 ?
        15.25278E-12_sp,    0.0000000E-12_sp,   713.8991E-12_sp,&   ! hfc143a
        15.35501E-12_sp,    0.0000000E-12_sp,   966.1801E-12_sp,&   ! hfc125
        26.89044E-12_sp,    0.0000000E-12_sp,   24.61550E-12_sp,&   ! hfc23
        8.336969E-12_sp,    0.0002184E-12_sp,   0.046355E-12_sp,&   ! hfc32
        80.51573E-12_sp,    0.0000000E-12_sp,   421.3692E-12_sp,&   ! hfc134a
        81.09249E-12_sp,    34.050000E-12_sp,   126.5040E-12_sp],&  ! cf4
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
    print *, "VMR is:", vmr
    ! !$acc parallel loop collapse(2) copyin(vmr) present(array)
    ! do icol = 1, size(array,2)
    !   do ilay = 1, size(array,1)
    !     array(ilay,icol) = vmr
    !   end do
    ! end do
  end function get_ref_vmr

end module mo_gas_ref_concentrations
