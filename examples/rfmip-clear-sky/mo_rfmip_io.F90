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
! This module reads an example file containing atomspheric conditions (temperature, pressure, gas concentrations)
!   and surface properties (emissivity, temperature), defined on nlay layers across a set of ncol columns subject to
!   nexp perturbations, and returns them in data structures suitable for use in rte and rrtmpg. The input data
!   are partitioned into a user-specified number of blocks.
! For the moment only quantities relevant to longwave calculations are provided.
!
! The example files comes from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
!   The protocol for this experiment allows for different specifications of which gases to consider:
! all gases, (CO2, CH4, N2O) + {CFC11eq; CFC12eq + HFC-134eq}. Ozone is always included
! The protocol does not specify the treatmet of gases like CO
!
! -------------------------------------------------------------------------------------------------
module mo_rfmip_io
  use mo_rte_kind,      only: wp, sp, dp
  use mo_gas_concentrations, &
                        only: ty_gas_concs
  use mo_rrtmgp_util_string, &
                        only: lower_case, string_in_array, string_loc_in_array
  use mo_simple_netcdf, only: read_field, write_field, get_dim_size
  use netcdf
  implicit none
  interface unblock_and_write
    module procedure unblock_and_write_2D, unblock_and_write_3D, unblock_and_write_4D_dp, unblock_and_write_4D_sp
  end interface

  private
  public :: read_kdist_gas_names, determine_gas_names, read_size, read_and_block_pt, &
            read_and_block_sw_bc, read_and_block_lw_bc, read_and_block_gases_ty, unblock
  public :: unblock_and_write

  integer :: ncol_l = 0, nlay_l = 0, nexp_l = 0 ! Local copies
contains
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Find the size of the problem: columns, layers, perturbations (experiments)
  !
  subroutine read_size(fileName, ncol, nlay, nexp)
    character(len=*),          intent(in   ) :: fileName
    integer,         optional, intent(  out) :: ncol, nlay, nexp
    ! ---------------------------
    integer :: ncid
    ! ---------------------------
    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_size: can't find file " // trim(fileName))

    ncol = get_dim_size(ncid, 'site')
    nlay = get_dim_size(ncid, 'layer')
    nexp = get_dim_size(ncid, 'expt')
    if(get_dim_size(ncid, 'level') /= nlay+1) call stop_on_err("read_size: number of levels should be nlay+1")
    ncid = nf90_close(ncid)

    ncol_l = ncol
    nlay_l = nlay
    nexp_l = nexp
  end subroutine read_size
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Return layer and level pressures and temperatures as arrays dimensioned (ncol, nlay/+1, nblocks)
  !   Input arrays are dimensioned (nlay/+1, ncol, nexp)
  !   Output arrays are allocated within this routine
  !
  subroutine read_and_block_pt(fileName, blocksize, &
                               p_lay, p_lev, t_lay, t_lev)
    character(len=*),           intent(in   ) :: fileName
    integer,                    intent(in   ) :: blocksize
    real(wp), dimension(:,:,:), allocatable, & ! [nlay/+1, blocksize, nblocks]
                                intent(  out) :: p_lay, p_lev, t_lay, t_lev
    ! ---------------------------
    integer :: ncid, varid, ndims
    integer :: b, nblocks
    real(wp), dimension(:,:,:), allocatable :: temp3d
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("read_and_block_pt: Haven't read problem size yet.")
    if(mod(ncol_l*nexp_l, blocksize) /= 0 ) call stop_on_err("read_and_block_pt: number of columns doesn't fit evenly into blocks.")
    nblocks = (ncol_l*nexp_l)/blocksize
    allocate(p_lay(nlay_l, blocksize,   nblocks), t_lay(nlay_l, blocksize,   nblocks), &
             p_lev(nlay_l+1, blocksize, nblocks))

    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_pt: can't find file " // trim(fileName))
    !
    ! Read p, T data; reshape to suit RRTMGP dimensions
    !

    ! pres and temp can be 1D (nlay), 2D, (nlay, ncol) or 3D (nlay, ncol, nexp), check for dimensions
    if(nf90_inq_varid(ncid, "pres_layer", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "pres_layer")
    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "pres_layer")


    if (ndims == 3) then ! (nlay, ncol, nexp)
      temp3d = reshape(       read_field(ncid, "pres_layer", nlay_l,   ncol_l, nexp_l), &
                    shape = [nlay_l, blocksize, nblocks])

    else if (ndims == 2) then ! (nlay, ncol)
      temp3d = reshape(spread(read_field(ncid, "pres_layer", nlay_l,   ncol_l), dim = 3, ncopies = nexp_l), &
                     shape = [nlay_l, blocksize, nblocks])

    else if (ndims == 1) then  ! (nlay)
      temp3d = reshape( spread(spread(read_field(ncid, "pres_layer", nlay_l), dim = 2, ncopies = ncol_l), dim = 3, ncopies = nexp_l), &
                    shape = [nlay_l, blocksize, nblocks])
    end if 

    do b = 1, nblocks
      p_lay(:,:,b) = temp3d(:,:,b)
    end do

    deallocate(temp3d)
    

    if(nf90_inq_varid(ncid, "pres_level", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "pres_level")

    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "pres_level")

    if (ndims == 3) then ! (nlay, ncol, nexp)
      temp3d = reshape(       read_field(ncid, "pres_level", nlay_l+1, ncol_l, nexp_l), &
                    shape = [nlay_l+1, blocksize, nblocks])
    else if (ndims == 2) then ! (nlay, ncol)
      temp3d = reshape(spread(read_field(ncid, "pres_level", nlay_l+1, ncol_l),  dim = 3, ncopies = nexp_l), &
                      shape = [nlay_l+1, blocksize, nblocks])
    else if (ndims == 1) then
      temp3d = reshape( spread(spread(read_field(ncid, "pres_level", nlay_l+1), dim = 2, ncopies = ncol_l), dim = 3, ncopies = nexp_l), &
      shape = [nlay_l+1, blocksize, nblocks])
    end if

    do b = 1, nblocks
      p_lev(:,:,b) = temp3d(:,:,b)
    end do

    deallocate(temp3d)


    if(nf90_inq_varid(ncid, "temp_layer", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "temp_layer")

    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "temp_layer")

    if (ndims == 3) then ! (nlay, ncol, nexp)
      temp3d = reshape(       read_field(ncid, "temp_layer", nlay_l, ncol_l, nexp_l), &
                    shape = [nlay_l, blocksize, nblocks])
    else if (ndims == 2) then ! (nlay, ncol)
      temp3d = reshape(spread(read_field(ncid, "temp_layer", nlay_l, ncol_l),  dim = 3, ncopies = nexp_l), &
                      shape = [nlay_l, blocksize, nblocks])
    else
      call stop_on_err("temp_layer needs to be either 2D or 3D")
    
    end if

    do b = 1, nblocks
      t_lay(:,:,b) = temp3d(:,:,b)
    end do

    deallocate(temp3d)


    ! temp3d = reshape(       read_field(ncid, "temp_layer", nlay_l,   ncol_l, nexp_l), &
    !                  shape = [nlay_l, blocksize, nblocks])
    ! do b = 1, nblocks
    !   t_lay(:,:,b) = temp3d(:,:,b)
    ! end do

    ! deallocate(temp3d)


    if(nf90_inq_varid(ncid, "temp_level", varid) /= NF90_NOERR) then
      print *, "can't find variable temp_level, returning array with shape (1,1,1)"
      allocate(t_lev(1,1,1))
    else

      allocate(t_lev(nlay_l+1, blocksize, nblocks))

      if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "temp_layer")

      if (ndims == 3) then ! (nlay, ncol, nexp)
        temp3d = reshape(       read_field(ncid, "temp_level", nlay_l+1, ncol_l, nexp_l), &
                      shape = [nlay_l+1, blocksize, nblocks])
      else if (ndims == 2) then ! (nlay, ncol)
        temp3d = reshape(spread(read_field(ncid, "temp_level", nlay_l+1, ncol_l),  dim = 3, ncopies = nexp_l), &
                        shape = [nlay_l+1, blocksize, nblocks])
      else
        call stop_on_err("temp_level needs to be either 2D or 3D")
      end if

      do b = 1, nblocks
        t_lev(:,:,b) = temp3d(:,:,b)
      end do
      deallocate(temp3d)
    end if

    ncid = nf90_close(ncid)

  end subroutine read_and_block_pt
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Read and reshape shortwave boundary conditions
  !
  subroutine read_and_block_sw_bc(fileName, blocksize, &
                               surface_albedo, total_solar_irradiance, solar_zenith_angle)
    character(len=*),           intent(in   ) :: fileName
    integer,                    intent(in   ) :: blocksize
    real(wp), dimension(:,:), allocatable, &
                                intent(  out) :: surface_albedo, total_solar_irradiance, solar_zenith_angle
    ! ---------------------------
    integer :: ncid
    integer :: nblocks
    real(wp), dimension(ncol_l, nexp_l) :: temp2D
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("read_and_block_sw_bc: Haven't read problem size yet.")
    if(mod(ncol_l*nexp_l, blocksize) /= 0 ) call stop_on_err("read_and_block_sw_bc: number of columns doesn't fit evenly into blocks.")
    nblocks = (ncol_l*nexp_l)/blocksize
    !
    ! Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    !

    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_sw_bc: can't find file " // trim(fileName))

    temp2D(1:ncol_l,1:nexp_l) = spread(read_field(ncid, "surface_albedo",          ncol_l), dim=2, ncopies=nexp_l)
    surface_albedo         = reshape(temp2D, shape = [blocksize, nblocks])

    temp2D(1:ncol_l,1:nexp_l) = spread(read_field(ncid, "total_solar_irradiance",  ncol_l), dim=2, ncopies=nexp_l)
    total_solar_irradiance = reshape(temp2D, shape = [blocksize, nblocks])

    temp2D(1:ncol_l,1:nexp_l) = spread(read_field(ncid, "solar_zenith_angle",      ncol_l), dim=2, ncopies=nexp_l)
    solar_zenith_angle     = reshape(temp2d, shape = [blocksize, nblocks])

    ncid = nf90_close(ncid)
  end subroutine read_and_block_sw_bc
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Read and reshape longwave boundary conditions
  !
  subroutine read_and_block_lw_bc(fileName, blocksize, &
                                  surface_emissivity, surface_temperature)
    character(len=*),           intent(in   ) :: fileName
    integer,                    intent(in   ) :: blocksize
    real(wp), dimension(:,:), allocatable, &
                                intent(  out) :: surface_emissivity, surface_temperature
    ! ---------------------------
    integer :: ncid, varid, ndims
    integer :: nblocks
    ! real(wp), dimension(ncol_l, nexp_l) :: temp2D ! Required to make gfortran 8 work, not sure why
    real(wp), dimension(:,:), allocatable :: temp2D

    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) &
      call stop_on_err("read_and_block_lw_bc: Haven't read problem size yet.")
    if(mod(ncol_l*nexp_l, blocksize) /= 0 ) &
      call stop_on_err("read_and_block_lw_bc: number of columns doesn't fit evenly into blocks.")
    nblocks = (ncol_l*nexp_l)/blocksize

    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_lw_bc: can't find file " // trim(fileName))
    !
    ! Allocate on assigment
    !

    ! surface temperature and emissivity can be either 1D or 2D , check for dimensions
    if(nf90_inq_varid(ncid, "surface_emissivity", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "surface_emissivity")

    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "surface_emissivity")

    if (ndims == 2) then ! (ncol, nexp)
      temp2D = read_field(ncid, "surface_emissivity", ncol_l, nexp_l)
    else if (ndims == 1) then
      temp2D  = spread(read_field(ncid, "surface_emissivity",  ncol_l), dim=2, ncopies=nexp_l)
    end if

    surface_emissivity  = reshape(temp2D, shape = [blocksize, nblocks])
    deallocate(temp2D)

    if(nf90_inq_varid(ncid, "surface_temperature", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "surface_temperature")

    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "surface_temperature")

    if (ndims == 2) then ! (ncol, nexp)
      temp2D  = read_field(ncid, "surface_temperature", ncol_l, nexp_l)
    else if (ndims == 1) then
      temp2D  = spread(read_field(ncid, "surface_temperature",  ncol_l), dim=2, ncopies=nexp_l)
    end if

    surface_temperature = reshape(temp2D, shape = [blocksize, nblocks])

    ncid = nf90_close(ncid)

  end subroutine read_and_block_lw_bc
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Create a pair of string arrays - one containing the chemical name of each gas, used by the k-distribution, and
  !   one containing the name as contained in the RFMIP input files - depending on the forcing scenario
  ! Forcing index (1 = all available greenhouse gases;
  !                2 = CO2, CH4, N2O, CFC11eq
  !                3 = CO2, CH4, N2O, CFC12eq, HFC-134eq
  !                All scenarios use 3D values of ozone, water vapor so those aren't listed here
  !
  subroutine determine_gas_names(concentrationFile, kdistFile, forcing_index, names_in_kdist, names_in_file)
    character(len=*),                             intent(in   ) :: concentrationFile, kdistFile
    integer,                                      intent(in   ) :: forcing_index
    character(len=32), dimension(:), allocatable, intent(inout) :: names_in_kdist, names_in_file
    ! ----------------
    integer :: num_gases, i
    character(len=32), dimension(11) :: &
      chem_name = ['co   ', &
                   'ch4  ', &
        				   'o2   ', &
        				   'n2o  ', &
        				   'n2   ', &
        				   'co2  ', &
        				   'CCl4 ', &
        				   'ch4  ', &
        				   'CH3Br', &
   			           'CH3Cl', &
                   'cfc22'], &
      conc_name = ['carbon_monoxide     ', &
                   'methane             ', &
                   'oxygen              ', &
          			   'nitrous_oxide       ', &
          			   'nitrogen            ', &
        				   'carbon_dioxide      ', &
        				   'carbon_tetrachloride', &
        				   'methane             ', &
        				   'methyl_bromide      ', &
        				   'methyl_chloride     ', &
                   'hcfc22              ']
    ! ----------------
    select case (forcing_index)
    case (1)
      call read_kdist_gas_names(kdistFile, names_in_kdist)
      allocate(names_in_file(size(names_in_kdist)))
      do i = 1, size(names_in_kdist)
        names_in_file(i) = trim(lower_case(names_in_kdist(i)))
        !
        ! Use a mapping between chemical formula and name if it exists
        !
        if(string_in_array(names_in_file(i), chem_name)) &
          names_in_file(i) = conc_name(string_loc_in_array(names_in_file(i), chem_name))
      end do
    case (2)
      num_gases = 9
      allocate(names_in_kdist(num_gases), names_in_file(num_gases))
      !
      ! Not part of the RFMIP specification, but oxygen is included because it's a major
      !    gas in some bands in the SW
      !
      names_in_kdist = ['no2  ','h2o  ', 'o3   ', 'co2  ', 'ch4  ', 'n2o  ', 'o2   ', 'cfc12', 'cfc11']
      names_in_file =  ['no2           ', &
                        'water_vapor   ', &
                        'ozone         ', &            
                        'carbon_dioxide', &
                        'methane       ', &
                        'nitrous_oxide ', &
                        'oxygen        ', &
                        'cfc12         ', &
                        'cfc11         ']
    case (3)
      num_gases = 6
      allocate(names_in_kdist(num_gases), names_in_file(num_gases))
      !
      ! Not part of the RFMIP specification, but oxygen is included because it's a major
      !    gas in some bands in the SW
      !
      names_in_kdist = ['co2    ', 'ch4    ', 'n2o    ', 'o2     ', 'cfc12  ', &
                        'hfc134a']
      names_in_file =  ['carbon_dioxide', &
                        'methane       ', &
                        'nitrous_oxide ', &
                        'oxygen        ', &
                        'cfc12eq       ', &
                        'hfc134aeq     ']
    case (4)
      num_gases = 9
      allocate(names_in_kdist(num_gases), names_in_file(num_gases))
      !
      ! CKDMIP gases only
      !                 
      names_in_kdist = ["h2o  ","co2  ", "o3   ", "n2o  ","ch4  ","o2   ","n2   ", "cfc11", "cfc12"]
      names_in_file =  ['water_vapor   ', &
                        'carbon_dioxide', &
                        'ozone         ', &
                        'nitrous_oxide ', &
                        'methane       ', &
                        'oxygen        ', &
                        'nitrogen      ', &
                        'cfc11         ', &
                        'cfc12         ']
    case default
      call stop_on_err("determine_gas_names: unknown value of forcing_index")
    end select

  end subroutine determine_gas_names
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Read the names of the gases known to the k-distribution
  !
  !
  subroutine read_kdist_gas_names(fileName, kdist_gas_names)
    character(len=*),          intent(in   ) :: fileName
    character(len=32), dimension(:), allocatable, &
                               intent(  out) :: kdist_gas_names
    ! ---------------------------
    integer :: ncid, varid
    character(len=9), parameter :: varName = "gas_names"
    ! ---------------------------
    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_kdist_gas_names: can't open file " // trim(fileName))

    allocate(kdist_gas_names(get_dim_size(ncid, 'absorber')))

    if(nf90_inq_varid(ncid, trim(varName), varid) /= NF90_NOERR) &
      call stop_on_err("read_kdist_gas_names: can't find variable " // trim(varName))
    if(nf90_get_var(ncid, varid, kdist_gas_names)  /= NF90_NOERR) &
      call stop_on_err("read_kdist_gas_names: can't read variable " // trim(varName))

    ncid = nf90_close(ncid)
  end subroutine read_kdist_gas_names
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Read and reshape gas concentrations. RRTMGP requires gas concentrations to be supplied via a class
  !   (ty_gas_concs). Gas concentrations are set via a call to gas_concs%set_vmr(name, values)
  !   where `name` is nominally the chemical formula for the gas in question and `values` may be
  !   a scalar, a 1-d profile assumed to apply to all columns, or an array of dimension (ncol, nlay).
  ! This routine outputs a vector nblocks long of these types so each element of the array can be passed to
  !   the rrtmgp gas optics calculation in turn.
  !
  ! This routine exploits RFMIP conventions: only water vapor and ozone vary by column within
  !   each experiment.
  ! Fields in the RFMIP file have a trailing _GM (global mean); some fields use a chemical formula and other
  !   a descriptive name, so a map is provided between these.
  !
  subroutine read_and_block_gases_ty(fileName, blocksize, gas_names, names_in_file, gas_conc_array)
    character(len=*),           intent(in   ) :: fileName
    integer,                    intent(in   ) :: blocksize
    character(len=*),  dimension(:), &
                                intent(inout   ) :: gas_names ! Names used by the k-distribution/gas concentration type
    character(len=*),  dimension(:), &
                                intent(in   ) :: names_in_file ! Corresponding names in the RFMIP file
    type(ty_gas_concs), dimension(:), allocatable, &
                                intent(  out) :: gas_conc_array

    ! ---------------------------
    integer :: ncid, varid, ndims, dimsize1, dimsize2
    integer :: nblocks
    integer :: b, g, ind, i
    integer, dimension(nf90_max_var_dims) :: dimids
    integer,  dimension(:,:),   allocatable :: exp_num
    real(wp), dimension(:),     allocatable :: gas_conc_temp_1d
    real(wp), dimension(:,:),   allocatable :: gas_conc_temp_2d, gas_conc_temp_2d_2
    real(wp), dimension(:,:,:), allocatable :: gas_conc_temp_3d, gas_conc_temp_3d_2 
    real(wp) :: scaling_factor ! explicit variable: bug in intel compilers, sometimes the scaling factor wasn't read on the fly otherwise
    character(len=32) :: varName
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) &
      call stop_on_err("read_and_block_lw_bc: Haven't read problem size yet.")
    if(mod(ncol_l*nexp_l, blocksize) /= 0 ) &
      call stop_on_err("read_and_block_lw_bc: number of columns doesn't fit evenly into blocks.")
    nblocks = (ncol_l*nexp_l)/blocksize
    allocate(gas_conc_array(nblocks))
    !
    ! gas_names contains 'no2' which isn't available in the RFMIP files. We should remove it
    !   here but that's kinda hard, so we set its concentration to 0 below.
    !  gas_names = gas_names(1:size(gas_names)-1)
    !call strip(gas_names,"no2")
    ! gas_names = REPLACE(gas_names,"no2","") 
    !  print *, gas_names
    do b = 1, nblocks
      call stop_on_err(gas_conc_array(b)%init(gas_names))
    end do

    ! print *, "gas names:", gas_names
    ! print *, gas_conc_array(b)%get_gas_names()

    !
    ! Which gases are known to the k-distribution and available in the files?
    !
    ! Experiment index for each colum
    allocate(exp_num(blocksize,nblocks))
    exp_num = reshape(spread([(b, b = 1, nexp_l)], 1, ncopies = ncol_l), shape = [blocksize, nblocks], order=[1,2])

    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(fileName))

    ! !
    ! ! Water vapor and ozone depend on col, lay, exp: look just like other fields
    ! !
    ! allocate(gas_conc_temp_3d(nlay_l,blocksize,nblocks))
    ! gas_conc_temp_3d = reshape(read_field(ncid, "water_vapor", nlay_l, ncol_l, nexp_l), &
    !                            shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "water_vapor")
    ! do b = 1, nblocks
    !   call stop_on_err(gas_conc_array(b)%set_vmr('h2o', gas_conc_temp_3d(:,:,b)))
    ! end do

    ! gas_conc_temp_3d = reshape(read_field(ncid, "ozone", nlay_l, ncol_l, nexp_l), &
    !                            shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "ozone")
    ! do b = 1, nblocks
    !   call stop_on_err(gas_conc_array(b)%set_vmr('o3', gas_conc_temp_3d(:,:,b)))
    !   !                                         nlay, ncol, nexp -> nlay, blocksize, nblock -> blocksize, nlay
    ! end do
    !

    ! EDIT: Water vapor and ozone can now be either 2D (nlay, ncol) or 3D (nlay, ncol, nexp), check for dimensions
    if(nf90_inq_varid(ncid, "water_vapor", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "water_vapor")
    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "water_vapor")

    if (ndims == 3) then ! (nlay, ncol, nexp)
      allocate(gas_conc_temp_3d(nlay_l,blocksize,nblocks))
      gas_conc_temp_3d = reshape(       read_field(ncid, "water_vapor", nlay_l,  ncol_l, nexp_l), &
                    shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "water_vapor")
    else if (ndims == 2) then ! (nlay, ncol)
      gas_conc_temp_3d = reshape(spread(read_field(ncid, "water_vapor", nlay_l,   ncol_l), dim = 3, ncopies = nexp_l), &
                     shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "water_vapor")
    else 
      call stop_on_err("water vapour needs to be either 2D (lay, col) or 3D (lay, col, exp)")
    end if 

    do b = 1, nblocks
      call stop_on_err(gas_conc_array(b)%set_vmr('h2o', gas_conc_temp_3d(:,:,b)))
    end do
    deallocate(gas_conc_temp_3d)

    if(nf90_inq_varid(ncid, "ozone", varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // "ozone")
    if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // "ozone")

    if (ndims == 3) then ! (nlay, ncol, nexp)
      allocate(gas_conc_temp_3d(nlay_l,blocksize,nblocks))
      gas_conc_temp_3d = reshape(       read_field(ncid, "ozone", nlay_l,  ncol_l, nexp_l), &
                    shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "ozone")
    else if (ndims == 2) then ! (nlay, ncol)
      gas_conc_temp_3d = reshape(spread(read_field(ncid, "ozone", nlay_l,   ncol_l), dim = 3, ncopies = nexp_l), &
                     shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "ozone")
    else 
      call stop_on_err("ozone needs to be either 2D (lay, col) or 3D (lay, col, exp)")
    end if 

    do b = 1, nblocks
      call stop_on_err(gas_conc_array(b)%set_vmr('o3', gas_conc_temp_3d(:,:,b)))
    end do
    deallocate(gas_conc_temp_3d)

    !
    ! EDIT: other gases are NOT necessarily a function of experiment only, check using if statement
    !
    do g = 1, size(gas_names)
      !
      ! Skip 3D fields above, also NO2 since RFMIP doesn't have this
      !
      varName = trim(names_in_file(g)) // "_GM"

      if(string_in_array(gas_names(g), ['h2o', 'o3 ', 'no2'])) cycle

      if(nf90_inq_varid(ncid, varName, varid) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't find variable " // varName)
      if(nf90_inquire_variable(ncid, varid, ndims = ndims) /= NF90_NOERR) &
      call stop_on_err("get_var_size: can't get information for variable " // varName)

      ! Read the values as a function of experiment

      if (ndims == 3) then ! this is a 3d field nlay*ncol*nexp
        scaling_factor = read_scaling(ncid, varName)
        gas_conc_temp_3d = reshape(read_field(ncid, varName, nlay_l, ncol_l, nexp_l), &
        shape = [nlay_l, blocksize, nblocks]) * scaling_factor
        
        do b = 1, nblocks
          call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), gas_conc_temp_3d(:,:,b)))                                
        end do

      else if (ndims == 2) then  ! this is a 2d field, EITHER ncol*nexp, nlay*ncol, or nlay*nexp...it got a little complicated

        scaling_factor = read_scaling(ncid, varName)

        if(nf90_inquire_variable(ncid, varid, ndims = ndims, dimids = dimids) /= NF90_NOERR) &
        call stop_on_err("get_var_size: can't get information for variable " // varName)

        if(nf90_inquire_dimension(ncid, dimids(1), len = dimsize1) /= NF90_NOERR) &
        call stop_on_err("get_var_size: can't get dim length for variable " // varName)

        if(nf90_inquire_dimension(ncid, dimids(2), len = dimsize2) /= NF90_NOERR) &
        call stop_on_err("get_var_size: can't get dim length for variable " // varName)

        if (dimsize1 == ncol_l .and. dimsize2 == nexp_l) then    ! (ncol, exp)

          gas_conc_temp_2d = read_field(ncid, varName, ncol_l, nexp_l) * scaling_factor
          gas_conc_temp_2d_2 = reshape(gas_conc_temp_2d, shape = [blocksize, nblocks])

          do b = 1, nblocks
            call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), &
                                    transpose(spread(gas_conc_temp_2d_2(:,b), dim=2, ncopies=nlay_l)) ))
          end do
          deallocate(gas_conc_temp_2d, gas_conc_temp_2d_2)

        else if (dimsize1 == nlay_l .and. dimsize2 == ncol_l) then   ! (nlay, ncol)
          gas_conc_temp_3d = reshape(spread(read_field(ncid, varName, nlay_l,   ncol_l), dim = 3, ncopies = nexp_l), &
          shape = [nlay_l, blocksize, nblocks]) * scaling_factor

          do b = 1, nblocks
            call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), gas_conc_temp_3d(:,:,b)))
          end do
          deallocate(gas_conc_temp_3d)

        else if  (dimsize1 == nlay_l .and. dimsize2 == nexp_l) then   ! (nlay, nexp)
          
          gas_conc_temp_2d = read_field(ncid, varName, nlay_l, nexp_l) * scaling_factor
          allocate(gas_conc_temp_3d(nlay_l, ncol_l, nexp_l))
          do i = 1, ncol_l
            gas_conc_temp_3d(:,i,:) = gas_conc_temp_2d 
          end do

          gas_conc_temp_3d_2 = reshape(gas_conc_temp_3d, shape = [nlay_l, blocksize, nblocks]) 

          do b = 1, nblocks
            call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), gas_conc_temp_3d_2(:,:,b)))                                
          end do
          deallocate(gas_conc_temp_2d, gas_conc_temp_3d, gas_conc_temp_3d_2)
          
        else 
           call stop_on_err("confused about dimensions of variable, can't write to blocks " // varName)
        end if

      else  ! ndims = 1
        scaling_factor = read_scaling(ncid, varName) 
        gas_conc_temp_1d = read_field(ncid, varName, nexp_l)
        gas_conc_temp_1d = scaling_factor * gas_conc_temp_1d 
        ! print *, "shape(gas_temp_1d):", shape(gas_conc_temp_1d)

        do b = 1, nblocks
          ! Does every value in this block belong to the same experiment?
          if(all(exp_num(1,b) == exp_num(2:,b))) then
            ! Provide a scalar value
            call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), gas_conc_temp_1d(exp_num(1,b))))
          else
            ! Create 2D field, blocksize x nlay, with scalar values from each experiment
            call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), &
            transpose(spread(gas_conc_temp_1d(exp_num(:,b)), dim=2, ncopies = nlay_l))))
          end if
        end do
        ! 
        ! NO2 is the one gas known to the k-distribution that isn't provided by RFMIP
        !   It would be better to remove it from
        !

      end if 
 
    end do

    print *, "setting no2 to zero" 
    do b = 1, nblocks
      call stop_on_err(gas_conc_array(b)%set_vmr('no2', 0._wp))
    end do

    if (allocated(gas_conc_temp_3d)) deallocate(gas_conc_temp_3d)
    if (allocated(gas_conc_temp_2d)) deallocate(gas_conc_temp_2d)
    if (allocated(gas_conc_temp_1d)) deallocate(gas_conc_temp_1d)

    ncid = nf90_close(ncid)

  end subroutine read_and_block_gases_ty

  !--------------------------------------------------------------------------------------------------------------------
  function read_scaling(ncid, varName)
    integer,          intent(in) :: ncid
    character(len=*), intent(in) :: varName
    real(wp)                     :: read_scaling

    integer           :: varid
    character(len=16) :: charUnits

    if(nf90_inq_varid(ncid, trim(varName), varid) /= NF90_NOERR) &
      call stop_on_err("read_scaling: can't find variable " // trim(varName))
    if(nf90_get_att(ncid, varid, "units", charUnits)  /= NF90_NOERR) &
      call stop_on_err("read_scaling: can't read attribute 'units' from variable " // trim(varName))
    read(charUnits, *) read_scaling
    return

  end function read_scaling
  !--------------------------------------------------------------------------------------------------------------------
  !
  ! Reshape values (nominally fluxes) from RTE order (nlev, ncol, nblocks)
  !   to RFMIP order (nlev, ncol, nexp),
  subroutine unblock(values, values_unblocked)
    real(wp), dimension(:,:,:),  & ! [nlay/+1, blocksize,nblocks]
                                intent(in   ) :: values
    real(wp), dimension(:,:,:),  & ! [nlay+1, ncol, nexp]
                                intent(out  ) :: values_unblocked
    ! ---------------------------
    integer :: ncid
    integer :: b, blocksize, nlev, nblocks
    real(wp), dimension(:,:), allocatable :: temp2d
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("unblock: Haven't read problem size yet.")
    nlev      = size(values,1)
    blocksize = size(values,2)
    nblocks   = size(values,3)
    if(nlev /= nlay_l+1)                   call stop_on_err('unblock: array values has the wrong number of levels')
    if(blocksize*nblocks /= ncol_l*nexp_l) call stop_on_err('unblock: array values has the wrong number of blocks/size')

    allocate(temp2D(nlev, ncol_l*nexp_l))
    do b = 1, nblocks
      temp2D(1:nlev, ((b-1)*blocksize+1):(b*blocksize)) = values(1:nlev,1:blocksize,b)
    end do
    
    values_unblocked = reshape(temp2d, shape = [nlev, ncol_l, nexp_l])

    deallocate(temp2d)
  end subroutine unblock
  !
  !
  ! Reshape values (nominally fluxes) from RTE order (ncol, nblocks)
  !   to RFMIP order (ncol, nexp), then write them to a user-specified variable
  !   in an existing netCDF file.
  subroutine unblock_and_write_2D(fileName, varName, values)
    character(len=*),           intent(in   ) :: fileName, varName
    real(wp), dimension(:,:),  & ! [blocksize, nblocks]
                                intent(in   ) :: values
    ! ---------------------------
    integer :: ncid
    integer :: b, blocksize, nlev, nblocks
    real(wp), dimension(:), allocatable :: temp1d
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("unblock_and_write 2D: Haven't read problem size yet.")
    blocksize = size(values,1)
    nblocks   = size(values,2)
    if(blocksize*nblocks /= ncol_l*nexp_l) call stop_on_err('unblock_and_write 2D: array values has the wrong number of blocks/size')

    allocate(temp1d(ncol_l*nexp_l))
    do b = 1, nblocks
      temp1d(((b-1)*blocksize+1):(b*blocksize)) = values(1:blocksize,b)
    end do
    !
    ! Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    !
    if(nf90_open(trim(fileName), NF90_WRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("unblock_and_write: can't find file " // trim(fileName))
    call stop_on_err(write_field(ncid, varName,  &
                                 reshape(temp1d, shape = [ncol_l, nexp_l])))

    ncid = nf90_close(ncid)
    deallocate(temp1d)
  end subroutine unblock_and_write_2D

  !
  ! Reshape values (nominally fluxes) from RTE order (nlev, ncol, nblocks)
  !   to RFMIP order (nlev, ncol, nexp), then write them to a user-specified variable
  !   in an existing netCDF file.
  !
  subroutine unblock_and_write_3D(fileName, varName, values)
    character(len=*),           intent(in   ) :: fileName, varName
    real(wp), dimension(:,:,:),  & ! [nlay/+1, blocksize, nblocks]
                                intent(in   ) :: values
    ! ---------------------------
    integer :: ncid
    integer :: b, blocksize, nlev, nblocks
    real(wp), dimension(:,:), allocatable :: temp2d
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("unblock_and_write 3D: Haven't read problem size yet.")
    nlev      = size(values,1)
    blocksize = size(values,2)
    nblocks   = size(values,3)
    ! if(nlev /= nlay_l+1)                   call stop_on_err('unblock_and_write: array values has the wrong number of levels')
    if(blocksize*nblocks /= ncol_l*nexp_l) call stop_on_err('unblock_and_write 3D: array values has the wrong number of blocks/size')

    allocate(temp2D(nlev, ncol_l*nexp_l))
    do b = 1, nblocks
      temp2D(1:nlev, ((b-1)*blocksize+1):(b*blocksize)) = values(1:nlev,1:blocksize,b)
    end do
    !
    ! Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    !
    if(nf90_open(trim(fileName), NF90_WRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("unblock_and_write: can't find file " // trim(fileName))
    call stop_on_err(write_field(ncid, varName,  &
                                 reshape(temp2d, shape = [nlev, ncol_l, nexp_l])))

    ncid = nf90_close(ncid)
    deallocate(temp2d)
  end subroutine unblock_and_write_3D


  subroutine unblock_and_write_4D_dp(fileName, varName, values)
    character(len=*),           intent(in   ) :: fileName, varName
    real(dp), dimension(:,:,:,:),  & !   (ngas, nlay/+1, block_size, nblocks) or (ngpt,...)
                                intent(in   ) :: values
    ! ---------------------------
    integer :: ncid
    integer :: b, blocksize, nlev, nblocks, nfirst, ibnd
    real(wp), dimension(:,:,:), allocatable :: temp3D
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("unblock_and_write 4D dp: Haven't read problem size yet.")
    nfirst      = size(values,1)
    nlev      = size(values,2)
    blocksize = size(values,3)
    nblocks   = size(values,4)
    !if(nlev /= nlay_l)                   call stop_on_err('unblock_and_write: array values has the wrong number of levels')
    if(blocksize*nblocks /= ncol_l*nexp_l) call stop_on_err('unblock_and_write 4D dp: array values has the wrong number of blocks/size')

    allocate(temp3D(nfirst, nlev, ncol_l*nexp_l))

    do b = 1, nblocks
       temp3D(:, :, ((b-1)*blocksize+1):(b*blocksize)) = values(:,:,1:blocksize,b)
    end do
    !
    ! Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    !

    if(nf90_open(trim(fileName), NF90_WRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("unblock_and_write: can't find file " // trim(fileName))
    call stop_on_err(write_field(ncid, varName,  &
                                 reshape(temp3D, shape = [nfirst, nlev, ncol_l, nexp_l])))

    ncid = nf90_close(ncid)
    deallocate(temp3D)
  end subroutine unblock_and_write_4D_dp

  subroutine unblock_and_write_4D_sp(fileName, varName, values)
    character(len=*),           intent(in   ) :: fileName, varName
    real(sp), dimension(:,:,:,:),  & !   (ngas, nlay/+1, block_size, nblocks) or (ngpt,...)
                                intent(in   ) :: values
    ! ---------------------------
    integer :: ncid
    integer :: b, blocksize, nlev, nblocks, nfirst, ibnd
    real(sp), dimension(:,:,:), allocatable :: temp3D
    ! ---------------------------
    if(any([ncol_l, nlay_l, nexp_l]  == 0)) call stop_on_err("unblock_and_write 4D sp: Haven't read problem size yet.")
    nfirst    = size(values,1)
    nlev      = size(values,2)
    blocksize = size(values,3)
    nblocks   = size(values,4)
    !if(nlev /= nlay_l)                   call stop_on_err('unblock_and_write: array values has the wrong number of levels')
    if(blocksize*nblocks /= ncol_l*nexp_l) call stop_on_err('unblock_and_write 4D sp: array values has the wrong number of blocks/size')

    allocate(temp3D(nfirst, nlev, ncol_l*nexp_l))

    do b = 1, nblocks
       temp3D(:, :, ((b-1)*blocksize+1):(b*blocksize)) = values(:,:,1:blocksize,b)
    end do
    !
    ! Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    !

    if(nf90_open(trim(fileName), NF90_WRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("unblock_and_write: can't find file " // trim(fileName))
    call stop_on_err(write_field(ncid, varName,  &
                                 reshape(temp3D, shape = [nfirst, nlev, ncol_l, nexp_l])))

    ncid = nf90_close(ncid)
    deallocate(temp3D)
  end subroutine unblock_and_write_4D_sp


  function write_4D_sp(ncid, varName, var) result(err_msg)
    integer,                    intent(in) :: ncid
    character(len=*),           intent(in) :: varName
    real(sp), dimension(:,:,:,:), intent(in) :: var
    character(len=128)                     :: err_msg

    integer :: varid

    err_msg = ""
    if(nf90_inq_varid(ncid, trim(varName), varid) /= NF90_NOERR) then
      err_msg = "write_field: can't find variable " // trim(varName)
      return
    end if
    if(nf90_put_var(ncid, varid, var)  /= NF90_NOERR) &
      err_msg = "write_field: can't write variable " // trim(varName)

  end function write_4d_sp
  !----------------------------

  !--------------------------------------------------------------------------------------------------------------------
  subroutine stop_on_err(msg)
    !
    ! Print error message and stop
    !
    use iso_fortran_env, only : error_unit
    character(len=*), intent(in) :: msg
    if(len_trim(msg) > 0) then
      write(error_unit,*) trim(msg)
      stop
    end if
  end subroutine
end module mo_rfmip_io
