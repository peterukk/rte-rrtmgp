subroutine stop_on_err(error_msg)
  use iso_fortran_env, only : error_unit
  character(len=*), intent(in) :: error_msg

  if(error_msg /= "") then
    write (error_unit,*) trim(error_msg)
    write (error_unit,*) "rte_rrtmgp_clouds stopping"
    error stop 1
  end if
end subroutine stop_on_err

subroutine vmr_2d_to_1d(gas_concs, gas_concs_garand, name, nlay, ncol)
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_rte_kind,           only: wp

  type(ty_gas_concs), intent(in)    :: gas_concs_garand
  type(ty_gas_concs), intent(inout) :: gas_concs
  character(len=*),   intent(in)    :: name
  integer,            intent(in)    :: nlay, ncol

  real(wp) :: tmp(nlay, ncol), tmp_col(nlay)

  !$acc data create(tmp, tmp_col)
  !!$omp target data map(alloc:tmp, tmp_col)
  call stop_on_err(gas_concs_garand%get_vmr(name, tmp))
  !$acc kernels
  !!$omp target
  tmp_col(:) = tmp(:, 1)
  !$acc end kernels
  !!$omp end target

  call stop_on_err(gas_concs%set_vmr       (name, tmp_col))
  !$acc end data
  !!$omp end target data
end subroutine vmr_2d_to_1d
! ----------------------------------------------------------------------------------
program rte_rrtmgp_clouds
  use mo_rte_kind,           only: wp, i8
  use mo_optical_props,      only: ty_optical_props, &
                                   ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str
  use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  use mo_cloud_optics,       only: ty_cloud_optics
  use mo_gas_concentrations, only: ty_gas_concs
  use mo_source_functions,   only: ty_source_func_lw
  use mo_fluxes,             only: ty_fluxes_broadband, ty_fluxes_flexible
  use mo_rte_lw,             only: rte_lw
  use mo_rte_sw,             only: rte_sw
  use mo_load_coefficients,  only: load_and_init
  use mo_load_cloud_coefficients, &
                             only: load_cld_lutcoeff, load_cld_padecoeff
  use mo_garand_atmos_io,    only: read_atmos, write_lw_fluxes, write_sw_fluxes
#ifdef USE_TIMING
  !
  ! Timing library
  !
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr_file, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead, gptlsetutr
#endif
  implicit none

#ifdef USE_PAPI  
#include "f90papi.h"
#endif  
  ! ----------------------------------------------------------------------------------
  ! Variables
  ! ----------------------------------------------------------------------------------
  ! Arrays: dimensions (col, lay)
  real(wp), dimension(:,:),   allocatable :: p_lay, t_lay, p_lev
  real(wp), dimension(:,:),   allocatable :: col_dry
  real(wp), dimension(:,:),   allocatable :: temp_array

  !
  ! Longwave only
  !
  real(wp), dimension(:,:),   allocatable :: t_lev
  real(wp), dimension(:),     allocatable :: t_sfc
  real(wp), dimension(:,:),   allocatable :: emis_sfc ! First dimension is band
  !
  ! Shortwave only
  !
  real(wp), dimension(:),     allocatable :: mu0
  real(wp), dimension(:,:),   allocatable :: sfc_alb_dir, sfc_alb_dif ! First dimension is band
  !
  ! Source functions
  !
  !   Longwave
  type(ty_source_func_lw), save               :: lw_sources
  !   Shortwave
  real(wp), dimension(:,:), allocatable, save :: toa_flux
  !
  ! Clouds
  !
  real(wp), allocatable, dimension(:,:) :: lwp, iwp, rel, rei
  logical,  allocatable, dimension(:,:) :: cloud_mask
  !
  ! Output variables
  !
  real(wp), dimension(:,:), target, &
                            allocatable :: flux_up, flux_dn, flux_dir
  !
  ! Derived types from the RTE and RRTMGP libraries
  !
  type(ty_gas_optics_rrtmgp) :: k_dist
  type(ty_cloud_optics)      :: cloud_optics
  type(ty_gas_concs)         :: gas_concs, gas_concs_garand, gas_concs_1col
  class(ty_optical_props_arry), &
                 allocatable :: atmos, clouds
  ! type(ty_fluxes_broadband)  :: fluxes
  type(ty_fluxes_flexible)  :: fluxes

  !
  ! Inputs to RRTMGP
  !
  logical :: top_at_1, is_sw, is_lw

  integer  :: ncol, nlay, nbnd, ngpt
  integer  :: icol, ilay, ibnd, iloop, igas
  real(wp) :: rel_val, rei_val

  character(len=8) :: char_input
  integer  :: nUserArgs=0, nloops
  logical :: use_luts = .true., write_fluxes = .true.
  integer, parameter :: ngas = 8
  character(len=3), dimension(ngas) &
                     :: gas_names = ['h2o', 'co2', 'o3 ', 'n2o', 'co ', 'ch4', 'o2 ', 'n2 ']

  character(len=256) :: input_file, k_dist_file, cloud_optics_file, timing_file
  !
  ! Timing variables
  !
  integer(kind=i8)              :: start, finish, start_all, finish_all, clock_rate, ret
  real(wp)                      :: avg
  integer(kind=i8), allocatable :: elapsed(:)
#ifdef USE_TIMING
  !
  ! Initialize timers
  !
  ret = gptlsetoption (gptlpercent, 1)        ! Turn on "% of" print
  ret = gptlsetoption (gptloverhead, 0)       ! Turn off overhead estimate
#ifdef USE_PAPI  
  ret = GPTLsetoption (PAPI_SP_OPS, 1);
#endif  
  ret =  gptlinitialize()
#endif
  ! NAR OpenMP CPU directives in compatible with OpenMP GPU directives
  !!$omp threadprivate( lw_sources, toa_flux, flux_up, flux_dn, flux_dir )
  ! ----------------------------------------------------------------------------------
  ! Code
  ! ----------------------------------------------------------------------------------
  !
  ! Parse command line for any file names, block size
  !
  ! rrtmgp_clouds rrtmgp-clouds.nc $RRTMGP_ROOT/rrtmgp/data/rrtmgp-data-lw-g256-2018-12-04.nc $RRTMGP_ROOT/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-lw.nc  128 1
  ! rrtmgp_clouds rrtmgp-clouds.nc $RRTMGP_ROOT/rrtmgp/data/rrtmgp-data-sw-g224-2018-12-04.nc $RRTMGP_ROOT/extensions/cloud_optics/rrtmgp-cloud-optics-coeffs-sw.nc  128 1
  nUserArgs = command_argument_count()
  nloops = 1
  if (nUserArgs <  4) call stop_on_err("Need to supply input_file k_distribution_file ncol.")
  if (nUserArgs >= 1) call get_command_argument(1,input_file)
  if (nUserArgs >= 2) call get_command_argument(2,k_dist_file)
  if (nUserArgs >= 3) call get_command_argument(3,cloud_optics_file)
  if (nUserArgs >= 4) then
    call get_command_argument(4, char_input)
    read(char_input, '(i8)') ncol
    if(ncol <= 0) call stop_on_err("Specify positive ncol.")
  end if
  if (nUserArgs >= 5) then
    call get_command_argument(5, char_input)
    read(char_input, '(i8)') nloops
    if(nloops <= 0) call stop_on_err("Specify positive nloops.")
  end if
  if (nUserArgs >  6) print *, "Ignoring command line arguments beyond the first five..."
  if(trim(input_file) == '-h' .or. trim(input_file) == "--help") then
    call stop_on_err("rrtmgp_clouds input_file absorption_coefficients_file cloud_optics_file ncol")
  end if
  !
  ! Read temperature, pressure, gas concentrations.
  !   Arrays are allocated as they are read
  !
  call read_atmos(input_file,                 &
                  p_lay, t_lay, p_lev, t_lev, &
                  gas_concs_garand, col_dry)
  deallocate(col_dry)
  nlay = size(p_lay, 1)
  ! For clouds we'll use the first column, repeated over and over
  call stop_on_err(gas_concs%init(gas_names))
  do igas = 1, ngas
    call vmr_2d_to_1d(gas_concs, gas_concs_garand, gas_names(igas), size(p_lay, 1), size(p_lay, 2))
  end do

  !  If we trusted in Fortran allocate-on-assign we could skip the temp_array here
  allocate(temp_array(nlay, ncol))
  temp_array = spread(p_lay(:,1), dim = 2, ncopies=ncol)
  call move_alloc(temp_array, p_lay)
  allocate(temp_array(nlay, ncol))
  temp_array = spread(t_lay(:,1), dim = 2, ncopies=ncol)
  call move_alloc(temp_array, t_lay)
  allocate(temp_array(nlay+1, ncol))
  temp_array = spread(p_lev(:,1), dim = 2, ncopies=ncol)
  call move_alloc(temp_array, p_lev)
  allocate(temp_array(nlay+1, ncol))
  temp_array = spread(t_lev(:,1), dim = 2, ncopies=ncol)
  call move_alloc(temp_array, t_lev)
  ! This puts pressure and temperature arrays on the GPU
  !$acc enter data copyin(p_lay, p_lev, t_lay, t_lev)
  !!$omp target enter data map(to:p_lay, p_lev, t_lay, t_lev)
  ! ----------------------------------------------------------------------------
  ! load data into classes
  call load_and_init(k_dist, k_dist_file, gas_concs)
  is_sw = k_dist%source_is_external()
  is_lw = .not. is_sw
  !
  ! Should also try with Pade calculations
  !  call load_cld_padecoeff(cloud_optics, cloud_optics_file)
  !
  if(use_luts) then
    call load_cld_lutcoeff (cloud_optics, cloud_optics_file)
  else
    call load_cld_padecoeff(cloud_optics, cloud_optics_file)
  end if
  call stop_on_err(cloud_optics%set_ice_roughness(2))
  ! ----------------------------------------------------------------------------
  !
  ! Problem sizes
  !
  nbnd = k_dist%get_nband()
  ngpt = k_dist%get_ngpt()
  top_at_1 = p_lay(1, 1) < p_lay(nlay,1)

  ! ----------------------------------------------------------------------------
  ! LW calculations neglect scattering; SW calculations use the 2-stream approximation
  !   Here we choose the right variant of optical_props.
  !
  if(is_sw) then
    allocate(ty_optical_props_2str::atmos)
    allocate(ty_optical_props_2str::clouds)
  else
    allocate(ty_optical_props_1scl::atmos)
    allocate(ty_optical_props_1scl::clouds)
  end if
  ! Clouds optical props are defined by band
  call stop_on_err(clouds%init(k_dist%get_band_lims_wavenumber()))
  !
  ! Allocate arrays for the optical properties themselves.
  !
  select type(atmos)
    class is (ty_optical_props_1scl)
      !! $acc enter data copyin(atmos)
      call stop_on_err(atmos%alloc_1scl(ncol, nlay, k_dist))
      !!$acc enter data copyin(atmos) create(atmos%tau)
      !!$omp target enter data map(alloc:atmos%tau)
    class is (ty_optical_props_2str)
      call stop_on_err(atmos%alloc_2str( ncol, nlay, k_dist))
      !!$acc enter data copyin(atmos) create(atmos%tau, atmos%ssa, atmos%g)
      !!$omp target enter data map(alloc:atmos%tau, atmos%ssa, atmos%g)
    class default
      call stop_on_err("rte_rrtmgp_clouds: Don't recognize the kind of optical properties ")
  end select
  select type(clouds)
    class is (ty_optical_props_1scl)
      call stop_on_err(clouds%alloc_1scl(ncol, nlay))
      !!$acc enter data copyin(clouds) create(clouds%tau)
      !!$omp target enter data map(alloc:clouds%tau)
    class is (ty_optical_props_2str)
      call stop_on_err(clouds%alloc_2str(ncol, nlay))
      !!$acc enter data copyin(clouds) create(clouds%tau, clouds%ssa, clouds%g)
      !!$omp target enter data map(alloc:clouds%tau, clouds%ssa, clouds%g)
    class default
      call stop_on_err("rte_rrtmgp_clouds: Don't recognize the kind of optical properties ")
  end select
  ! ----------------------------------------------------------------------------
  !  Boundary conditions depending on whether the k-distribution being supplied
  !   is LW or SW
  if(is_sw) then
    ! toa_flux is threadprivate
    !!$omp parallel
    allocate(toa_flux(ngpt, ncol))
    !!$omp end parallel
    !
    allocate(sfc_alb_dir(ngpt, ncol), sfc_alb_dif(ngpt, ncol), mu0(ncol))
    !!$acc enter data create(sfc_alb_dir, sfc_alb_dif, mu0)
    !!$omp target enter data map(alloc:sfc_alb_dir, sfc_alb_dif, mu0)
    ! Ocean-ish values for no particular reason
    !$acc kernels
    !!$omp target
    sfc_alb_dir = 0.06_wp
    sfc_alb_dif = 0.06_wp
    mu0 = .86_wp
    !$acc end kernels
    !!$omp end target
  else
    ! lw_sorces is threadprivate
    !!$omp parallel
    call stop_on_err(lw_sources%alloc(ncol, nlay, k_dist))
    !!$omp end parallel

    allocate(t_sfc(ncol), emis_sfc(nbnd, ncol))
    !$acc enter data create(t_sfc, emis_sfc)
    !!$omp target enter data map(alloc:t_sfc, emis_sfc)
    ! Surface temperature
    !$acc kernels
    !!$omp target
    t_sfc = t_lev(1, merge(nlay+1, 1, top_at_1))
    emis_sfc = 0.98_wp
    !$acc end kernels
    !!$omp end target
  end if
  ! ----------------------------------------------------------------------------
  !
  ! Fluxes
  !
  !!$omp parallel
  allocate(flux_up(nlay+1,ncol), flux_dn(nlay+1,ncol))
  !!$omp end parallel

  !$acc enter data create(flux_up, flux_dn)
  !!$omp target enter data map(alloc:flux_up, flux_dn)
  if(is_sw) then
    allocate(flux_dir(nlay+1,ncol))
    !$acc enter data create(flux_dir)
    !!$omp target enter data map(alloc:flux_dir)
  end if
  !
  ! Clouds
  !
  allocate(lwp(nlay,ncol), iwp(nlay,ncol), &
           rel(nlay,ncol), rei(nlay,ncol), cloud_mask(nlay,ncol))
  !$acc enter data create(cloud_mask, lwp, iwp, rel, rei)
  !!$omp target enter data map(alloc:cloud_mask, lwp, iwp, rel, rei)

  ! Restrict clouds to troposphere (> 100 hPa = 100*100 Pa)
  !   and not very close to the ground (< 900 hPa), and
  !   put them in 2/3 of the columns since that's roughly the
  !   total cloudiness of earth
  rel_val = 0.5 * (cloud_optics%get_min_radius_liq() + cloud_optics%get_max_radius_liq())
  rei_val = 0.5 * (cloud_optics%get_min_radius_ice() + cloud_optics%get_max_radius_ice())
  !$acc parallel loop collapse(2) copyin(t_lay) copyout(lwp, iwp, rel, rei)
  !!$omp target teams distribute parallel do simd collapse(2) map(to:t_lay) map(from:lwp, iwp, rel, rei)
  do icol=1,ncol
    do ilay=1,nlay
      cloud_mask(ilay,icol) = p_lay(ilay,icol) > 100._wp * 100._wp .and. &
                              p_lay(ilay,icol) < 900._wp * 100._wp .and. &
                              mod(icol, 3) /= 0
      !
      ! Ice and liquid will overlap in a few layers
      !
      lwp(ilay,icol) = merge(10._wp,  0._wp, cloud_mask(ilay,icol) .and. t_lay(ilay,icol) > 263._wp)
      iwp(ilay,icol) = merge(10._wp,  0._wp, cloud_mask(ilay,icol) .and. t_lay(ilay,icol) < 273._wp)
      rel(ilay,icol) = merge(rel_val, 0._wp, lwp(ilay,icol) > 0._wp)
      rei(ilay,icol) = merge(rei_val, 0._wp, iwp(ilay,icol) > 0._wp)
    end do
  end do
  print *, "min max lwp", minval(lwp), maxval(lwp)
  print *, "min max iwp", minval(iwp), maxval(iwp)
  !$acc exit data delete(cloud_mask)
  !!$omp target exit data map(release:cloud_mask)
  ! ----------------------------------------------------------------------------
  !
  ! Multiple iterations for big problem sizes, and to help identify data movement
  !   For CPUs we can introduce OpenMP threading over loop iterations
  !
  allocate(elapsed(nloops))
  !
#ifdef USE_TIMING
  ret =  gptlstart('cloudy_sky_total')
#endif
  call system_clock(start_all)
  !
  !!$omp parallel do firstprivate(fluxes)
  do iloop = 1, nloops
    call system_clock(start)
#ifdef USE_TIMING
    ret =  gptlstart('cloud_optics')
#endif
    call stop_on_err(                                      &
      cloud_optics%cloud_optics(lwp, iwp, rel, rei, clouds))
#ifdef USE_TIMING
    ret =  gptlstop('cloud_optics')
#endif
    !
    ! Solvers
    !
    fluxes%flux_up => flux_up(:,:)
    fluxes%flux_dn => flux_dn(:,:)
    if(is_lw) then
      ! !$acc enter data create(lw_sources, lw_sources%lay_source, lw_sources%lev_source_inc, lw_sources%lev_source_dec, lw_sources%sfc_source)
      ! !!$omp target enter data map(alloc:lw_sources%lay_source, lw_sources%lev_source_inc, lw_sources%lev_source_dec, lw_sources%sfc_source)
#ifdef USE_TIMING
    ret =  gptlstart('gas_optics_lw')
#endif
      call stop_on_err(k_dist%gas_optics(p_lay, p_lev, &
                                         t_lay, t_sfc, &
                                         gas_concs,    &
                                         atmos,        &
                                         lw_sources,   &
                                         tlev = t_lev))
#ifdef USE_TIMING
    ret =  gptlstop('gas_optics_lw')
    ret =  gptlstart('clouds_increment')
#endif
      call stop_on_err(clouds%increment(atmos))
#ifdef USE_TIMING
    ret =  gptlstop('clouds_increment')
    ret =  gptlstart('rte_lw')
#endif
      call stop_on_err(rte_lw(atmos, top_at_1, &
                              lw_sources,      &
                              emis_sfc,        &
                              fluxes))
#ifdef USE_TIMING
    ret =  gptlstop('rte_lw')
#endif
      ! !$acc exit data delete(lw_sources%lay_source, lw_sources%lev_source_inc, lw_sources%lev_source_dec, lw_sources%sfc_source, lw_sources)
      ! !!$omp target exit data map(release:lw_sources%lay_source, lw_sources%lev_source_inc, lw_sources%lev_source_dec, lw_sources%sfc_source)
    else
      !$acc enter data create(toa_flux)
      !!$omp target enter data map(alloc:toa_flux)
      fluxes%flux_dn_dir => flux_dir(:,:)
#ifdef USE_TIMING
    ret =  gptlstart('gas_optics_sw')
#endif
      call stop_on_err(k_dist%gas_optics(p_lay, p_lev, &
                                         t_lay,        &
                                         gas_concs,    &
                                         atmos,        &
                                         toa_flux))
      print *, "mean tau after gas optics", mean3(atmos%tau)                                        
#ifdef USE_TIMING
    ret =  gptlstop('gas_optics_sw')
    ret =  gptlstart('clouds_deltascale_increment')
#endif       
      call stop_on_err(clouds%delta_scale())
      call stop_on_err(clouds%increment(atmos))
      print *, "mean tau after cloud optics contribution", mean3(atmos%tau)
#ifdef USE_TIMING
    ret =  gptlstop('clouds_deltascale_increment')
    ret =  gptlstart('rte_sw')
#endif
      call stop_on_err(rte_sw(atmos, top_at_1, &
                              mu0,   toa_flux, &
                              sfc_alb_dir, sfc_alb_dif, &
                              fluxes))
#ifdef USE_TIMING
    ret =  gptlstop('rte_sw')
#endif
      !$acc exit data delete(toa_flux)
      !!$omp target exit data map(release:toa_flux)
    end if
    !print *, "******************************************************************"
    call system_clock(finish, clock_rate)
    elapsed(iloop) = finish - start
  end do

if(is_lw) then
  timing_file  = "timing.cloudy_lw"
else
  timing_file  = "timing.cloudy_sw"
end if
#ifdef USE_TIMING
  ret =  gptlstop('cloudy_sky_total')
  ret = gptlpr_file(trim(timing_file))
  ret = gptlfinalize()
#endif
  !
  call system_clock(finish_all, clock_rate)
  !
  !$acc exit data delete(lwp, iwp, rel, rei)
  !!$omp target exit data map(release:lwp, iwp, rel, rei)
  !$acc exit data delete(p_lay, p_lev, t_lay, t_lev)
  !!$omp target exit data map(release:p_lay, p_lev, t_lay, t_lev)

#if defined(_OPENACC) || defined(_OPENMP)
  avg = sum( elapsed(merge(2,1,nloops>1):) ) / real(merge(nloops-1,nloops,nloops>1))

  print *, "Execution times - min(s)        :", minval(elapsed) / real(clock_rate)
  print *, "                - avg(s)        :", avg / real(clock_rate)
  print *, "                - per column(ms):", avg / real(ncol) / (1.0e-3*clock_rate)
#else
  print *, "Execution times - total(s)      :", (finish_all-start_all) / real(clock_rate)
  print *, "                - per column(ms):", (finish_all-start_all) / real(ncol*nloops) / (1.0e-3*clock_rate)
#endif

  if(is_lw) then
    print *, "mean LW flux dn", mean2(flux_dn), "mean LW flux up", mean2(flux_up)
    !  mean LW flux dn   144.144470     mean LW flux up   269.762390    
    !$acc exit data copyout(flux_up, flux_dn)
    !!$omp target exit data map(from:flux_up, flux_dn)
    if(write_fluxes) call write_lw_fluxes(input_file, transpose(flux_up), transpose(flux_dn))
    !$acc exit data delete(t_sfc, emis_sfc)
    !!$omp target exit data map(release:t_sfc, emis_sfc)
  else
    print *, "mean SW flux dn", mean2(flux_dn), "mean SW flux up", mean2(flux_up)
    !  mean SW flux dn   946.975098     mean SW flux up   325.290985 
    !$acc exit data copyout(flux_up, flux_dn, flux_dir)
    !!$omp target exit data map(from:flux_up, flux_dn, flux_dir)
    if(write_fluxes) call write_sw_fluxes(input_file, transpose(flux_up), transpose(flux_dn), transpose(flux_dir))
    !$acc exit data delete(sfc_alb_dir, sfc_alb_dif, mu0)
    !!$omp target exit data map(release:sfc_alb_dir, sfc_alb_dif, mu0)
  end if
  !$acc enter data create(lwp, iwp, rel, rei)
  !!$omp target enter data map(alloc:lwp, iwp, rel, rei)
  contains

  function mean2(x) result(mean)
    real(wp), dimension(:,:), intent(in) :: x
    real(wp) :: mean
    
    mean = sum(sum(x, dim=1),dim=1) / size(x)
  
  end function mean2

  function mean3(x3) result(mean)
    implicit none 
    real(wp), dimension(:,:,:), intent(in) :: x3
    real(wp) :: mean
    
    mean = sum(sum(sum(x3, dim=1),dim=1),dim=1) / (size(x3))
  end function mean3

end program rte_rrtmgp_clouds
