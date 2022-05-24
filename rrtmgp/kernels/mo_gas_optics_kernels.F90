! This code is part of
! RRTM for GCM Applications - Parallel (RRTMGP)
!
! Eli Mlawer and Robert Pincus
! Andre Wehe and Jennifer Delamere
! email:  rrtmgp@aer.com
!
! Copyright 2015,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
!
! Description: Numeric calculations for gas optics. Absorption and Rayleigh optical depths,
!   source functions.

module mo_gas_optics_kernels
  use mo_rte_kind,          only : wp, wl, sp, dp
  use mod_network_rrtmgp,   only: rrtmgp_network_type, output_sgemm_tau, output_sgemm_pfrac, output_sgemm_lw
  use omp_lib
  use, intrinsic :: ISO_C_BINDING
#ifdef USE_TIMING
  ! Timing library
  use gptl,                  only: gptlstart, gptlstop
#endif

  implicit none
  public
  interface predict_nn_lw_blas
    module procedure predict_nn_lw_blas_sp, predict_nn_lw_blas_mp
  end interface predict_nn_lw_blas

  interface predict_nn_sw_blas
    module procedure predict_nn_sw_blas_sp, predict_nn_sw_blas_mp
  end interface predict_nn_sw_blas

#ifdef USE_TIMING
  integer :: ret, i
#endif

contains

  ! --------------------------------------------------------------------------------------
  ! Compute interpolation coefficients
  ! for calculations of major optical depths, minor optical depths, Rayleigh,
  ! and Planck fractions
  subroutine interpolation( &
                ncol,nlay,ngas,nflav,neta, npres, ntemp, &
                flavor,                                  &
                press_ref_log, temp_ref,press_ref_log_delta,    &
                temp_ref_min,temp_ref_delta,press_ref_trop_log, &
                vmr_ref,                                        &
                play,tlay,col_gas,                              &
                jtemp,fmajor,fminor,col_mix,tropo,jeta,jpress) bind(C, name="interpolation")
    ! input dimensions
    integer,                            intent(in) :: ncol, nlay
    integer,                            intent(in) :: ngas,nflav,neta,npres,ntemp
    integer,     dimension(2,nflav),    intent(in) :: flavor
    real(wp),    dimension(npres),      intent(in) :: press_ref_log
    real(wp),    dimension(ntemp),      intent(in) :: temp_ref
    real(wp),                           intent(in) :: press_ref_log_delta, &
                                                      temp_ref_min, temp_ref_delta, &
                                                      press_ref_trop_log
    real(wp),    dimension(2,0:ngas,ntemp), intent(in) :: vmr_ref

    ! inputs from profile or parent function
    real(wp),    dimension(nlay, ncol),        intent(in) :: play, tlay
    real(wp),    dimension(nlay, ncol,0:ngas), intent(in) :: col_gas

    ! outputs
    integer,     dimension(nlay, ncol),             intent(out) :: jtemp, jpress
    logical(wl), dimension(nlay, ncol),             intent(out) :: tropo
    integer,     dimension(2,    nflav,nlay, ncol), intent(out) :: jeta
    real(wp),    dimension(2,    nflav,nlay, ncol), intent(out) :: col_mix
    real(wp),    dimension(2,2,2,nflav,nlay, ncol), intent(out) :: fmajor
    real(wp),    dimension(2,2,  nflav,nlay, ncol), intent(out) :: fminor
    ! -----------------
    ! local
    real(wp),    dimension(nlay, ncol)      :: play_log
    real(wp),    dimension(nlay, ncol)      :: ftemp, fpress ! interpolation fraction for temperature, pressure
    real(wp) :: locpress ! needed to find location in pressure grid
    real(wp) :: ratio_eta_half ! ratio of vmrs of major species that defines eta=0.5
                               ! for given flavor and reference temperature level
    real(wp) :: eta, feta      ! binary_species_parameter, interpolation variable for eta
    real(wp) :: loceta         ! needed to find location in eta grid
    real(wp) :: ftemp_term
    ! -----------------
    ! local indexes
    integer :: ilay, icol, iflav, igases(2), itropo, itemp

    do icol = 1, ncol
      do ilay = 1, nlay
        ! index and factor for temperature interpolation
        jtemp(ilay,icol) = int((tlay(ilay,icol) - (temp_ref_min - temp_ref_delta)) / temp_ref_delta)
        jtemp(ilay,icol) = min(ntemp - 1, max(1, jtemp(ilay,icol))) ! limit the index range
        ftemp(ilay,icol) = (tlay(ilay,icol) - temp_ref(jtemp(ilay,icol))) / temp_ref_delta

        ! index and factor for pressure interpolation
        play_log(ilay,icol) = log(play(ilay,icol))
        locpress = 1._wp + (play_log(ilay,icol) - press_ref_log(1)) / press_ref_log_delta
        jpress(ilay,icol) = min(npres-1, max(1, int(locpress)))
        fpress(ilay,icol) = locpress - float(jpress(ilay,icol))

        ! determine if in lower or upper part of atmosphere
        tropo(ilay,icol) = play_log(ilay,icol) > press_ref_trop_log
      end do
    end do


    do icol = 1, ncol
      do ilay = 1, nlay
        ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
        itropo = merge(1,2,tropo(ilay,icol))
        ! loop over implemented combinations of major species
        do iflav = 1, nflav
          igases(:) = flavor(:,iflav)
          do itemp = 1, 2
            ! compute interpolation fractions needed for lower, then upper reference temperature level
            ! compute binary species parameter (eta) for flavor and temperature and
            !  associated interpolation index and factors
            ratio_eta_half = vmr_ref(itropo,igases(1),(jtemp(ilay,icol)+itemp-1)) / &
                             vmr_ref(itropo,igases(2),(jtemp(ilay,icol)+itemp-1))
            col_mix(itemp,iflav,ilay,icol) = col_gas(ilay,icol,igases(1)) + ratio_eta_half * col_gas(ilay,icol,igases(2))
            eta = merge(col_gas(ilay,icol,igases(1)) / col_mix(itemp,iflav,ilay,icol), 0.5_wp, &
                        col_mix(itemp,iflav,ilay,icol) > 2._wp * tiny(col_mix))
            loceta = eta * float(neta-1)
            jeta(itemp,iflav,ilay,icol) = min(int(loceta)+1, neta-1)
            feta = mod(loceta, 1.0_wp)
            ! compute interpolation fractions needed for minor species
            ! ftemp_term = (1._wp-ftemp(ilay,icol)) for itemp = 1, ftemp(ilay,icol) for itemp=2
            ftemp_term = (real(2-itemp, wp) + real(2*itemp-3, wp) * ftemp(ilay,icol))
            fminor(1,itemp,iflav,ilay,icol) = (1._wp-feta) * ftemp_term
            fminor(2,itemp,iflav,ilay,icol) =        feta  * ftemp_term
            ! compute interpolation fractions needed for major species
            fmajor(1,1,itemp,iflav,ilay,icol) = (1._wp-fpress(ilay,icol)) * fminor(1,itemp,iflav,ilay,icol)
            fmajor(2,1,itemp,iflav,ilay,icol) = (1._wp-fpress(ilay,icol)) * fminor(2,itemp,iflav,ilay,icol)
            fmajor(1,2,itemp,iflav,ilay,icol) =        fpress(ilay,icol)  * fminor(1,itemp,iflav,ilay,icol)
            fmajor(2,2,itemp,iflav,ilay,icol) =        fpress(ilay,icol)  * fminor(2,itemp,iflav,ilay,icol)
          end do ! reference temperatures
        end do ! iflav
      end do ! ilay,icol
    end do

  end subroutine interpolation
  ! --------------------------------------------------------------------------------------
  !
  ! Compute minor and major species opitcal depth from pre-computed interpolation coefficients
  !   (jeta,jtemp,jpress)
  !
  subroutine compute_tau_absorption(                &
                ncol,nlay,nbnd,ngpt,                &  ! dimensions
                ngas,nflav,neta,npres,ntemp,        &
                nminorlower, nminorklower,          & ! number of minor contributors, total num absorption coeffs
                nminorupper, nminorkupper,          &
                idx_h2o,                            &
                gpoint_flavor,                      &
                band_lims_gpt,                      &
                kmajor,                             &
                kminor_lower,                       &
                kminor_upper,                       &
                minor_limits_gpt_lower,             &
                minor_limits_gpt_upper,             &
                minor_scales_with_density_lower,    &
                minor_scales_with_density_upper,    &
                scale_by_complement_lower,          &
                scale_by_complement_upper,          &
                idx_minor_lower,                    &
                idx_minor_upper,                    &
                idx_minor_scaling_lower,            &
                idx_minor_scaling_upper,            &
                kminor_start_lower,                 &
                kminor_start_upper,                 &
                tropo,                              &
                col_mix,fmajor,fminor,              &
                play,tlay,col_gas,                  &
                jeta,jtemp,jpress,                  &
                tau) bind(C, name="compute_tau_absorption")
    ! ---------------------
    ! input dimensions
    integer,                                intent(in) :: ncol,nlay,nbnd,ngpt
    integer,                                intent(in) :: ngas,nflav,neta,npres,ntemp
    integer,                                intent(in) :: nminorlower, nminorklower,nminorupper, nminorkupper
    integer,                                intent(in) :: idx_h2o
    ! ---------------------
    ! inputs from object
    integer,     dimension(2,ngpt),                  intent(in) :: gpoint_flavor
    integer,     dimension(2,nbnd),                  intent(in) :: band_lims_gpt
    real(wp),    dimension(ngpt,neta,npres+1,ntemp), intent(in) :: kmajor
    real(wp),    dimension(nminorklower,neta,ntemp), intent(in) :: kminor_lower
    real(wp),    dimension(nminorkupper,neta,ntemp), intent(in) :: kminor_upper
    integer,     dimension(2,nminorlower),           intent(in) :: minor_limits_gpt_lower
    integer,     dimension(2,nminorupper),           intent(in) :: minor_limits_gpt_upper
    logical(wl), dimension(  nminorlower),           intent(in) :: minor_scales_with_density_lower
    logical(wl), dimension(  nminorupper),           intent(in) :: minor_scales_with_density_upper
    logical(wl), dimension(  nminorlower),           intent(in) :: scale_by_complement_lower
    logical(wl), dimension(  nminorupper),           intent(in) :: scale_by_complement_upper
    integer,     dimension(  nminorlower),           intent(in) :: idx_minor_lower
    integer,     dimension(  nminorupper),           intent(in) :: idx_minor_upper
    integer,     dimension(  nminorlower),           intent(in) :: idx_minor_scaling_lower
    integer,     dimension(  nminorupper),           intent(in) :: idx_minor_scaling_upper
    integer,     dimension(  nminorlower),           intent(in) :: kminor_start_lower
    integer,     dimension(  nminorupper),           intent(in) :: kminor_start_upper
    logical(wl), dimension(nlay, ncol),              intent(in) :: tropo
    ! ---------------------
    ! inputs from profile or parent function
    real(wp), dimension(2,    nflav,nlay, ncol       ), intent(in) :: col_mix
    real(wp), dimension(2,2,2,nflav,nlay, ncol       ), intent(in) :: fmajor
    real(wp), dimension(2,2,  nflav,nlay, ncol       ), intent(in) :: fminor
    real(wp), dimension(            nlay, ncol       ), intent(in) :: play, tlay      ! pressure and temperature
    real(wp), dimension(            nlay, ncol,0:ngas), intent(in) :: col_gas
    integer,  dimension(2,    nflav,nlay, ncol       ), intent(in) :: jeta
    integer,  dimension(            nlay, ncol       ), intent(in) :: jtemp
    integer,  dimension(            nlay, ncol       ), intent(in) :: jpress
    ! ---------------------
    ! output - optical depth
    real(wp), dimension(ngpt,nlay,ncol), intent(out) :: tau
    ! ---------------------
    ! Local variables
    !
    logical                    :: top_at_1
    integer, dimension(ncol,2) :: itropo_lower, itropo_upper
    ! ----------------------------------------------------------------

    ! ---------------------
    ! Layer limits of upper, lower atmospheres
    ! ---------------------
    top_at_1 = play(1,1) < play(nlay, 1)

    if(top_at_1) then
      itropo_lower(:, 1) = minloc(play, dim=1, mask=tropo)
      itropo_lower(:, 2) = nlay
      itropo_upper(:, 1) = 1
      itropo_upper(:, 2) = maxloc(play, dim=1, mask=(.not. tropo))
    else
      itropo_lower(:, 1) = 1
      itropo_lower(:, 2) = minloc(play, dim=1, mask= tropo)
      itropo_upper(:, 1) = maxloc(play, dim=1, mask=(.not. tropo))
      itropo_upper(:, 2) = nlay
    end if
    ! ---------------------
    ! Major Species
    ! ---------------------
    call gas_optical_depths_major(   &
          ncol,nlay,nbnd,ngpt,       & ! dimensions
          nflav,neta,npres,ntemp,    &
          gpoint_flavor,             &
          band_lims_gpt,             &
          kmajor,                    &
          col_mix,fmajor,            &
          jeta,tropo,jtemp,jpress,   &
          tau)

    ! ---------------------
    ! Minor Species - lower
    ! ---------------------
    call gas_optical_depths_minor(     &
           ncol,nlay,ngpt,             & ! dimensions
           ngas,nflav,ntemp,neta,      &
           nminorlower,nminorklower,   &
           idx_h2o,                    &
           gpoint_flavor(1,:),         &
           kminor_lower,               &
           minor_limits_gpt_lower,     &
           minor_scales_with_density_lower, &
           scale_by_complement_lower,  &
           idx_minor_lower,            &
           idx_minor_scaling_lower,    &
           kminor_start_lower,         &
           play, tlay,                 &
           col_gas,fminor,jeta,        &
           itropo_lower,jtemp,         &
           tau)

    ! ---------------------
    ! Minor Species - upper
    ! ---------------------
    call gas_optical_depths_minor(     &
           ncol,nlay,ngpt,             & ! dimensions
           ngas,nflav,ntemp,neta,      &
           nminorupper,nminorkupper,   &
           idx_h2o,                    &
           gpoint_flavor(2,:),         &
           kminor_upper,               &
           minor_limits_gpt_upper,     &
           minor_scales_with_density_upper, &
           scale_by_complement_upper,  &
           idx_minor_upper,            &
           idx_minor_scaling_upper,    &
           kminor_start_upper,         &
           play, tlay,                 &
           col_gas,fminor,jeta,        &
           itropo_upper,jtemp,         &
           tau)

  end subroutine compute_tau_absorption

  ! --------------------------------------------------------------------------------------
  !
  ! compute minor species optical depths
  !
  subroutine gas_optical_depths_major(ncol,nlay,nbnd,ngpt,&
                                      nflav,neta,npres,ntemp,      & ! dimensions
                                      gpoint_flavor, band_lims_gpt,   & ! inputs from object
                                      kmajor,                         &
                                      col_mix,fmajor,                 &
                                      jeta,tropo,jtemp,jpress,        & ! local input
                                      tau) bind(C, name="gas_optical_depths_major")
    ! input dimensions
    integer, intent(in) :: ncol, nlay, nbnd, ngpt, nflav,neta,npres,ntemp  ! dimensions

    ! inputs from object
    integer,  dimension(2,ngpt),  intent(in) :: gpoint_flavor
    integer,  dimension(2,nbnd),  intent(in) :: band_lims_gpt ! start and end g-point for each band
    real(wp), dimension(ngpt,neta,npres+1,ntemp), intent(in) :: kmajor

    ! inputs from profile or parent function
    real(wp),    dimension(2,    nflav,nlay, ncol), intent(in) :: col_mix
    real(wp),    dimension(2,2,2,nflav,nlay, ncol), intent(in) :: fmajor
    integer,     dimension(2,    nflav,nlay, ncol), intent(in) :: jeta
    logical(wl), dimension(nlay, ncol), intent(in) :: tropo
    integer,     dimension(nlay, ncol), intent(in) :: jtemp, jpress

    ! outputs
    real(wp), dimension(ngpt,nlay,ncol), intent(out) :: tau
    ! -----------------
    ! local variables
    ! real(wp) :: tau_major(ngpt) ! major species optical depth
    ! local index
    integer :: icol, ilay, iflav, ibnd, igpt, itropo
    integer :: gptS, gptE

    ! -----------------

    do icol = 1, ncol
      do ilay = 1, nlay
        ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
        itropo = merge(1,2,tropo(ilay,icol))
        ! optical depth calculation for major species
        do ibnd = 1, nbnd
          gptS = band_lims_gpt(1, ibnd)
          gptE = band_lims_gpt(2, ibnd)
          iflav = gpoint_flavor(itropo, gptS) !eta interpolation depends on band's flavor
          tau(gptS:gptE,ilay,icol)  = &
            ! interpolation in temperature, pressure, and eta
            interpolate3D_byflav(col_mix(:,iflav,ilay,icol),                                     &
                                 fmajor(:,:,:,iflav,ilay,icol), kmajor,                          &
                                 band_lims_gpt(1, ibnd), band_lims_gpt(2, ibnd),                 &
                                 jeta(:,iflav,ilay,icol), jtemp(ilay,icol),jpress(ilay,icol)+itropo)
          ! tau(gptS:gptE,ilay,icol) = tau(gptS:gptE,ilay,icol) + tau_major(gptS:gptE)
        end do ! igpt
      end do
    end do ! ilay

  end subroutine gas_optical_depths_major

  ! ----------------------------------------------------------
  !
  ! compute minor species optical depths
  !
  subroutine gas_optical_depths_minor(ncol,nlay,ngpt,        &
                                      ngas,nflav,ntemp,neta, &
                                      nminor,nminork,        &
                                      idx_h2o,               &
                                      gpt_flv,               &
                                      kminor,                &
                                      minor_limits_gpt,      &
                                      minor_scales_with_density,    &
                                      scale_by_complement,   &
                                      idx_minor, idx_minor_scaling, &
                                      kminor_start,          &
                                      play, tlay,            &
                                      col_gas,fminor,jeta,   &
                                      layer_limits,jtemp,    &
                                      tau) bind(C, name="gas_optical_depths_minor")
    integer,                                     intent(in   ) :: ncol,nlay,ngpt
    integer,                                     intent(in   ) :: ngas,nflav
    integer,                                     intent(in   ) :: ntemp,neta,nminor,nminork
    integer,                                     intent(in   ) :: idx_h2o
    integer,     dimension(ngpt),                intent(in   ) :: gpt_flv
    real(wp),    dimension(nminork,neta,ntemp),  intent(in   ) :: kminor
    integer,     dimension(2,nminor),            intent(in   ) :: minor_limits_gpt
    logical(wl), dimension(  nminor),            intent(in   ) :: minor_scales_with_density
    logical(wl), dimension(  nminor),            intent(in   ) :: scale_by_complement
    integer,     dimension(  nminor),            intent(in   ) :: kminor_start
    integer,     dimension(  nminor),            intent(in   ) :: idx_minor, idx_minor_scaling
    real(wp),    dimension(nlay, ncol),          intent(in   ) :: play, tlay
    real(wp),    dimension(nlay, ncol,0:ngas),   intent(in   ) :: col_gas
    real(wp),    dimension(2,2,nflav,nlay, ncol),intent(in   ) :: fminor
    integer,     dimension(2,  nflav,nlay, ncol),intent(in   ) :: jeta
    integer,     dimension(ncol, 2),             intent(in   ) :: layer_limits
    integer,     dimension(nlay, ncol),          intent(in   ) :: jtemp
    real(wp),    dimension(ngpt,nlay,ncol),      intent(inout) :: tau
    ! -----------------
    ! local variables
    real(wp), parameter :: PaTohPa = 0.01
    real(wp) :: vmr_fact, dry_fact             ! conversion from column abundance to dry vol. mixing ratio;
    real(wp) :: scaling, kminor_loc            ! minor species absorption coefficient, optical depth
    integer  :: icol, ilay, iflav, igpt, imnr
    integer  :: gptS, gptE
    real(wp), dimension(ngpt) :: tau_minor
    ! -----------------
    !
    ! Guard against layer limits being 0 -- that means don't do anything i.e. there are no
    !   layers with pressures in the upper or lower atmosphere respectively
    ! First check skips the routine entirely if all columns are out of bounds...
    !
    if(any(layer_limits(:,1) > 0)) then
      do imnr = 1, size(scale_by_complement,dim=1) ! loop over minor absorbers in each band
        do icol = 1, ncol
          !
          ! This check skips individual columns with no pressures in range
          !
          if(layer_limits(icol,1) > 0) then
            do ilay = layer_limits(icol,1), layer_limits(icol,2)
              !
              ! Scaling of minor gas absortion coefficient begins with column amount of minor gas
              !
              scaling = col_gas(ilay,icol,idx_minor(imnr))
              !
              ! Density scaling (e.g. for h2o continuum, collision-induced absorption)
              !
              if (minor_scales_with_density(imnr)) then
                !
                ! NOTE: P needed in hPa to properly handle density scaling.
                !
                scaling = scaling * (PaTohPa*play(ilay,icol)/tlay(ilay,icol))

                if(idx_minor_scaling(imnr) > 0) then  ! there is a second gas that affects this gas's absorption
                  vmr_fact = 1._wp / col_gas(ilay,icol,0)
                  dry_fact = 1._wp / (1._wp + col_gas(ilay,icol,idx_h2o) * vmr_fact)
                  ! scale by density of special gas
                  if (scale_by_complement(imnr)) then ! scale by densities of all gases but the special one
                    scaling = scaling * (1._wp - col_gas(ilay,icol,idx_minor_scaling(imnr)) * vmr_fact * dry_fact)

                  else
                    ! When using single precision, the scaling variable previously became infinity at this stage for some minor gases, due to
                    !(very big number)*         (very big number) . Fixed by computing last term first (vmr_fact is very small)
                    scaling = scaling *          (col_gas(ilay,icol,idx_minor_scaling(imnr)) * vmr_fact * dry_fact)

                  endif

                endif
              endif
              !
              ! Interpolation of absorption coefficient and calculation of optical depth
              !
              ! Which gpoint range does this minor gas affect?
              gptS = minor_limits_gpt(1,imnr)
              gptE = minor_limits_gpt(2,imnr)
              iflav = gpt_flv(gptS)
              tau_minor(gptS:gptE) = scaling *                   &
                                      interpolate2D_byflav(fminor(:,:,iflav,ilay,icol), &
                                                           kminor, &
                                                           kminor_start(imnr), kminor_start(imnr)+(gptE-gptS), &
                                                           jeta(:,iflav,ilay,icol), jtemp(ilay,icol))
              tau(gptS:gptE,ilay,icol) = tau(gptS:gptE,ilay,icol) + tau_minor(gptS:gptE)
            enddo
          end if
        enddo
      enddo
    end if
  end subroutine gas_optical_depths_minor
  

  ! ----------------------------------------------------------
  !
  ! compute Rayleigh scattering optical depths
  !
  subroutine compute_tau_rayleigh(ncol,nlay,nbnd,ngpt,         &
                                  ngas,nflav,neta,npres,ntemp, &
                                  gpoint_flavor,band_lims_gpt, &
                                  krayl,                       &
                                  idx_h2o, col_dry,col_gas,    &
                                  fminor,jeta,tropo,jtemp,     &
                                  tau_rayleigh) bind(C, name="compute_tau_rayleigh")
    integer,                                     intent(in ) :: ncol,nlay,nbnd,ngpt
    integer,                                     intent(in ) :: ngas,nflav,neta,npres,ntemp
    integer,     dimension(2,ngpt),              intent(in ) :: gpoint_flavor
    integer,     dimension(2,nbnd),              intent(in ) :: band_lims_gpt ! start and end g-point for each band
    real(wp),    dimension(ngpt,neta,ntemp,2),   intent(in ) :: krayl
    integer,                                     intent(in ) :: idx_h2o
    real(wp),    dimension(nlay, ncol),           intent(in ) :: col_dry
    real(wp),    dimension(nlay, ncol,0:ngas),    intent(in ) :: col_gas
    real(wp),    dimension(2,2,nflav,nlay, ncol), intent(in ) :: fminor
    integer,     dimension(2,  nflav,nlay, ncol), intent(in ) :: jeta
    logical(wl), dimension(nlay, ncol),           intent(in ) :: tropo
    integer,     dimension(nlay, ncol),           intent(in ) :: jtemp
    ! outputs
    real(wp),    dimension(ngpt,nlay,ncol),      intent(out) :: tau_rayleigh
    ! -----------------
    ! local variables
    real(wp) :: k(ngpt) ! rayleigh scattering coefficient
    integer  :: icol, ilay, iflav, ibnd, igpt, gptS, gptE
    integer  :: itropo
    ! -----------------
    do icol = 1, ncol
      do ilay = 1, nlay
        itropo = merge(1,2,tropo(ilay,icol)) ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
        do ibnd = 1, nbnd
          gptS = band_lims_gpt(1, ibnd)
          gptE = band_lims_gpt(2, ibnd)
          iflav = gpoint_flavor(itropo, gptS) !eta interpolation depends on band's flavor
          k(gptS:gptE) = interpolate2D_byflav(fminor(:,:,iflav,ilay,icol), &
                                              krayl(:,:,:,itropo),      &
                                              gptS, gptE, jeta(:,iflav,ilay,icol), jtemp(ilay,icol))
          tau_rayleigh(gptS:gptE,ilay,icol) = k(gptS:gptE) * &
                                              (col_gas(ilay,icol,idx_h2o)+col_dry(ilay,icol))
        end do
      end do
    end do
  end subroutine compute_tau_rayleigh

  ! ----------------------------------------------------------
  subroutine compute_Planck_source(             &
                    ncol, nlay, nbnd, ngpt,                &
                    nflav, neta, npres, ntemp, nPlanckTemp,&
                    tlay, tlev, tsfc, sfc_lay,             &
                    fmajor, jeta, tropo, jtemp, jpress,    &
                    gpoint_bands, band_lims_gpt,           &
                    temp_ref_min, totplnk_delta, pfracin, totplnk, gpoint_flavor, &
                    sfc_source, sfc_source_Jac, lay_source, lev_source, pfrac_inout)
    integer,                                    intent(in) :: ncol, nlay, nbnd, ngpt
    integer,                                    intent(in) :: nflav, neta, npres, ntemp, nPlanckTemp
    real(wp),    dimension(nlay, ncol  ),       intent(in) :: tlay
    real(wp),    dimension(nlay+1, ncol),       intent(in) :: tlev
    real(wp),    dimension(ncol       ),        intent(in) :: tsfc
    integer,                                    intent(in) :: sfc_lay
    ! Interpolation variables
    real(wp),    dimension(2,2,2,nflav,nlay, ncol), intent(in) :: fmajor
    integer,     dimension(2,    nflav,nlay, ncol), intent(in) :: jeta
    logical(wl), dimension(            nlay, ncol), intent(in) :: tropo
    integer,     dimension(            nlay, ncol), intent(in) :: jtemp, jpress
    ! Table-specific
    integer, dimension(ngpt),                     intent(in) :: gpoint_bands ! start and end g-point for each band
    integer, dimension(2, nbnd),                  intent(in) :: band_lims_gpt ! start and end g-point for each band
    real(wp),                                     intent(in) :: temp_ref_min, totplnk_delta
    real(wp), dimension(ngpt,neta,npres+1,ntemp), intent(in) :: pfracin
    real(wp), dimension(nPlanckTemp,nbnd),        intent(in) :: totplnk
    integer,  dimension(2,ngpt),                  intent(in) :: gpoint_flavor
    ! Outputs
    real(wp), dimension(ngpt,     ncol),          intent(out) :: sfc_source
    real(wp), dimension(ngpt,     ncol),          intent(out) :: sfc_source_Jac
    real(wp), dimension(ngpt,nlay,ncol),          intent(out) :: lay_source
    real(wp), dimension(ngpt,nlay+1,ncol),        intent(out) :: lev_source
    real(wp), dimension(ngpt,nlay,ncol), optional,target,&
                                                  intent(inout) :: pfrac_inout

    ! -----------------
    ! local                                ! Planck functions per band
    real(wp), dimension(nbnd       )    :: planck_function_sfc
    real(wp), dimension(nbnd       )    :: planck_function_sfc_Jac
    real(wp), dimension(nbnd        )   :: planck_function_lev, planck_function_lay
    real(wp), dimension(ngpt,nlay   ), target  :: pfrac_local ! Planck fraction per g-point
    real(wp), dimension(:,:), pointer, contiguous :: pfrac
    integer  :: ilay, icol, igpt, ibnd, itropo, iflav
    integer  :: gptS, gptE
    real(wp), dimension(2), parameter :: one          = [1._wp, 1._wp]
    real(wp), parameter               :: delta_Tsurf  = 1.0_wp

    ! -----------------    
    do icol = 1, ncol
      if(present(pfrac_inout)) then 
        pfrac => pfrac_inout(:,:,icol)
      else 
        pfrac => pfrac_local
      end if
      do ilay = 1, nlay
        ! Planck function by band for layers and levels
        planck_function_lev(:) = interpolate1D(tlev(ilay,icol),  temp_ref_min, totplnk_delta, totplnk)
        planck_function_lay(:) = interpolate1D(tlay(ilay,icol),  temp_ref_min, totplnk_delta, totplnk)

        ! Calculation of fraction of band's Planck irradiance associated with each g-point
        ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
        itropo = merge(1,2,tropo(ilay,icol))
        do ibnd = 1, nbnd
          gptS = band_lims_gpt(1, ibnd)
          gptE = band_lims_gpt(2, ibnd)
          iflav = gpoint_flavor(itropo, gptS) !eta interpolation depends on band's flavor
          pfrac(gptS:gptE,ilay) = &
            ! interpolation in temperature, pressure, and eta
            interpolate3D_byflav(one, fmajor(:,:,:,iflav,ilay,icol), pfracin, &
                          band_lims_gpt(1, ibnd), band_lims_gpt(2, ibnd),                 &
                          jeta(:,iflav,ilay,icol), jtemp(ilay,icol),jpress(ilay,icol)+itropo)

          ! Compute source irradiance for g-point, equals band irradiance x fraction for g-point
          lev_source(gptS:gptE, ilay, icol) = pfrac(gptS:gptE,ilay) * planck_function_lev(ibnd)
          lay_source(gptS:gptE, ilay, icol) = pfrac(gptS:gptE,ilay) * planck_function_lay(ibnd)
        end do ! band
      end do ! ilay

      ! Planck function by band for the surface and nlay+1
      planck_function_sfc(:)          = interpolate1D(tsfc(icol),               temp_ref_min, totplnk_delta, totplnk)
      planck_function_sfc_Jac(:)      = interpolate1D(tsfc(icol) + delta_Tsurf, temp_ref_min, totplnk_delta, totplnk)
      planck_function_lev(:)          = interpolate1D(tlev(nlay+1,icol),        temp_ref_min, totplnk_delta, totplnk)

      do ibnd = 1, nbnd
        gptS = band_lims_gpt(1, ibnd)
        gptE = band_lims_gpt(2, ibnd)

        ! Source irradiance for nlay+1
        lev_source(gptS:gptE,nlay+1,icol) = pfrac(gptS:gptE,nlay) * planck_function_lev(ibnd)

        ! Surface source irradiance
        sfc_source    (gptS:gptE, icol) = pfrac(gptS:gptE,sfc_lay) * planck_function_sfc(ibnd)
        sfc_source_Jac(gptS:gptE, icol) = pfrac(gptS:gptE,sfc_lay) * &
                              (planck_function_sfc_Jac(ibnd) - planck_function_sfc(ibnd))
      end do

    end do ! icol

  end subroutine compute_Planck_source

  ! ----------------------------------------------------------
  ! Like compute_Planck_source, but Planck fraction has already been computed by a neural network
  pure subroutine compute_Planck_source_nn(             &
                    ncol, nlay, nbnd, ngpt, nPlanckTemp, &
                    tlay, tlev, tsfc, sfc_lay,             &
                    gpt_bands, band_lims_gpt,           &
                    temp_ref_min, totplnk_delta, totplnk, &
                    sfc_source, sfc_source_Jac, pfrac, lev_source)
    integer,                                      intent(in)    :: ncol, nlay, nbnd, ngpt, nPlanckTemp
    real(wp), dimension(nlay,  ncol),             intent(in)    :: tlay
    real(wp), dimension(nlay+1,ncol),             intent(in)    :: tlev
    real(wp), dimension(ncol       ),             intent(in)    :: tsfc
    integer,                                      intent(in)    :: sfc_lay
    integer,  dimension(ngpt),                    intent(in)    :: gpt_bands    ! the band number for each g-point
    integer,  dimension(2, nbnd),                 intent(in)    :: band_lims_gpt ! start and end g-point for each band
    real(wp),                                     intent(in)    :: temp_ref_min, totplnk_delta
    real(wp), dimension(nPlanckTemp,nbnd),        intent(in)    :: totplnk
    ! outputs
    real(wp), dimension(ngpt,       ncol),        intent(out)   :: sfc_source
    real(wp), dimension(ngpt,       ncol),        intent(out)   :: sfc_source_Jac
    real(wp), dimension(ngpt,nlay,  ncol),        intent(inout) :: pfrac ! Planck fraction, which
                                                                ! becomes lay_source on output (trick to save memory)
    real(wp), dimension(ngpt,nlay+1,ncol),        intent(out)   :: lev_source
    ! -----------------
    ! local
    real(wp), dimension(nbnd )  :: planck_function_sfc
    real(wp), dimension(nbnd )  :: planck_function_sfc_Jac
    real(wp), dimension(nbnd )  :: planck_function_lev, planck_function_lay
    real(wp), parameter         :: delta_Tsurf  = 1.0_wp
    integer                     :: ilay, icol, igpt, ibnd, gptS, gptE

    ! -----------------    
    do icol = 1, ncol

      ! Source for surface and nlay+1
      planck_function_sfc(:)          = interpolate1D(tsfc(icol),               temp_ref_min, totplnk_delta, totplnk)
      planck_function_sfc_Jac(:)      = interpolate1D(tsfc(icol) + delta_Tsurf, temp_ref_min, totplnk_delta, totplnk)
      planck_function_lev(:)          = interpolate1D(tlev(nlay+1,icol),        temp_ref_min, totplnk_delta, totplnk)

      do ibnd = 1, nbnd
        gptS = band_lims_gpt(1, ibnd)
        gptE = band_lims_gpt(2, ibnd)

        ! Source for last level
        lev_source(gptS:gptE,nlay+1,icol) = pfrac(gptS:gptE,nlay,icol) * planck_function_lev(ibnd)

        ! Surface source
        sfc_source    (gptS:gptE, icol) = pfrac(gptS:gptE,sfc_lay,icol) * planck_function_sfc(ibnd)
        sfc_source_Jac(gptS:gptE, icol) = pfrac(gptS:gptE,sfc_lay,icol) * &
                              (planck_function_sfc_Jac(ibnd) - planck_function_sfc(ibnd))
      end do

      ! Source for levels and layers
      do ilay = 1, nlay
        planck_function_lev(:) = interpolate1D(tlev(ilay,icol),  temp_ref_min, totplnk_delta, totplnk)
        planck_function_lay(:) = interpolate1D(tlay(ilay,icol),  temp_ref_min, totplnk_delta, totplnk)

        do ibnd = 1, nbnd
          gptS = band_lims_gpt(1, ibnd)
          gptE = band_lims_gpt(2, ibnd)
          !$OMP SIMD
          do igpt = gptS, gptE
            lev_source(igpt, ilay, icol) = pfrac(igpt,ilay,icol) * planck_function_lev(ibnd)
            pfrac     (igpt, ilay, icol) = pfrac(igpt,ilay,icol) * planck_function_lay(ibnd)
            ! NOTE: "pfrac" is now actually lay_source
          end do
        end do ! band
      end do ! lay
    end do ! col

  end subroutine compute_Planck_source_nn

  ! --------------------------------------------------------------------------------------
  !
  ! LW neural network kernel using matrix-matrix GEMM computations, used if working precision is
  !  set as single precision (avoids temporary output array, which is faster)
  !
  subroutine predict_nn_lw_blas_sp(               &
                    ncol, nlay, ngpt, ninputs,    & 
                    nn_inputs, col_dry_wk,        &
                    neural_nets,                  &
                    tau, pfrac)
    ! inputs
    integer,                            intent(in)  :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                        intent(in)  :: nn_inputs     
    real(sp), dimension(nlay,ncol), target,  &     
                                        intent(in)  :: col_dry_wk     
    !^needs to be assumed-shape (and explicit shaped in parent code) for OpenMP?                                                           
    ! The neural network models
    type(rrtmgp_network_type), dimension(:), &
                                        intent(in)  :: neural_nets

    ! outputs
    real(sp), dimension(ngpt,nlay,ncol), target, &
                                        intent(out) :: pfrac, tau
    ! local
    real(sp), dimension(:,:), contiguous, pointer   :: input, output
    real(sp), dimension(:),   contiguous, pointer   :: input_coldry   
    real(sp), dimension(2*ngpt,nlay,ncol), target   :: outp_both
    integer                                         :: ilay, icol, igpt, nobs
    real(sp), dimension(ngpt)                       :: ymeans,ystd ! standard-scaling coefficients

    !  PREDICT PLANCK FRACTIONS
    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])
    
    call C_F_POINTER (C_LOC(col_dry_wk), input_coldry, [nobs])

    if (size(neural_nets)==2) then

#ifdef USE_TIMING
    ret =  gptlstart('compute_tau')
#endif
    call C_F_POINTER (C_LOC(tau), output, [ngpt,nobs])

    call neural_nets(1) % output_sgemm_tau(ninputs, ngpt, nobs, input, &
                          input_coldry, output)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau')
#endif
#ifdef USE_TIMING
    ret =  gptlstart('compute_pfrac')
#endif
    call C_F_POINTER (C_LOC(pfrac), output, [ngpt,nobs])

    call neural_nets(2) % output_sgemm_pfrac(ninputs, ngpt, nobs, input, output)
#ifdef USE_TIMING
    ret =  gptlstop('compute_pfrac')
#endif
  
    else 
      call C_F_POINTER (C_LOC(outp_both), output, [2*ngpt,nobs])

      call neural_nets(1) % output_sgemm_lw(ninputs, 2*ngpt, nobs, input, output)

#ifdef USE_TIMING
    ret =  gptlstart('write')
#endif   
      ystd = neural_nets(1)%coeffs_output_std(1:ngpt)
      ymeans = neural_nets(1)%coeffs_output_mean(1:ngpt)

      do icol = 1, ncol
        do ilay = 1, nlay
          !$OMP SIMD
          do igpt= 1, ngpt 
          ! Postprocess absorption output: reverse standard scaling and square root scaling
            tau(igpt,ilay,icol) = (ystd(igpt) * outp_both(igpt,ilay,icol) + ymeans(igpt))**8
            ! Optical depth from cross-sections
            tau(igpt,ilay,icol) = tau(igpt,ilay,icol)*col_dry_wk(ilay,icol)
            ! Planck fraction: reverse square root scaling
            pfrac(igpt,ilay,icol) = outp_both(igpt+ngpt,ilay,icol)*outp_both(igpt+ngpt,ilay,icol)
          end do 
        end do
      end do

#ifdef USE_TIMING
    ret =  gptlstop('write')
#endif   
    end if

  end subroutine predict_nn_lw_blas_sp  

  ! --------------------------------------------------------------------------------------
  !
  ! LW neural network kernel using matrix-matrix GEMM computations, used if working precision 
  ! is set as double precision (does computations in single precision but has to use temporary output array)
  !
  subroutine predict_nn_lw_blas_mp(                  &
                    ncol, nlay, ngpt, ninputs,       & 
                    nn_inputs, col_dry_wk,                   &
                    neural_nets,                  &
                    tau, pfrac)
    ! inputs
    integer,                              intent(in)    :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                          intent(in)    :: nn_inputs 
    real(dp), dimension(nlay,ncol),       intent(in)    :: col_dry_wk
    ! The neural network models
    type(rrtmgp_network_type), dimension(:),     intent(in)    :: neural_nets

    real(dp), dimension(ngpt,nlay,ncol),  intent(out)   :: pfrac, tau
    ! local
    real(sp), dimension(:,:), contiguous, pointer :: input,  output
    real(sp), dimension(:), contiguous,   pointer :: input_coldry          
    real(sp), dimension(nlay,ncol),       target  :: col_dry_wk_sp                                          
    real(sp), dimension(ngpt,nlay*ncol)           :: tmp_output
    real(sp), dimension(2*ngpt,nlay,ncol), target :: outp_both
    real(sp), dimension(ngpt)                     :: ymeans,ystd ! standard-scaling coefficients
    integer                                       :: ilay, icol, nobs

    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])

    col_dry_wk_sp = real(col_dry_wk, sp)
    call C_F_POINTER (C_LOC(col_dry_wk_sp), input_coldry, [nobs])

    if (size(neural_nets)==2) then

#ifdef USE_TIMING
      ret =  gptlstart('compute_tau')
#endif
      call neural_nets(1) % output_sgemm_tau(ninputs, ngpt, nobs, input, &
                          input_coldry, tmp_output)
      tau = reshape(tmp_output,(/ngpt,nlay,ncol/))

#ifdef USE_TIMING
      ret =  gptlstop('compute_tau')
      ret =  gptlstart('compute_pfrac')
#endif
      call neural_nets(2) % output_sgemm_pfrac(ninputs, ngpt, nobs, input, tmp_output)                      
#ifdef USE_TIMING
      ret =  gptlstart('output_reshape')
#endif
      pfrac = reshape(tmp_output,(/ngpt,nlay,ncol/))  
#ifdef USE_TIMING
      ret =  gptlstop('output_reshape')
#endif
#ifdef USE_TIMING
      ret =  gptlstop('compute_pfrac')
#endif

    else 
      call C_F_POINTER (C_LOC(outp_both), output, [2*ngpt,nobs])

      call neural_nets(1) % output_sgemm_lw(ninputs, 2*ngpt, nobs, input, output)

#ifdef USE_TIMING
    ret =  gptlstart('write')
#endif   
      ystd = neural_nets(1)%coeffs_output_std(1:ngpt)
      ymeans = neural_nets(1)%coeffs_output_mean(1:ngpt)

      do icol = 1, ncol
        do ilay = 1, nlay
          ! Postprocess absorption output: reverse standard scaling and square root scaling
          tau(1:ngpt,ilay,icol) = (ystd(1:ngpt) * outp_both(1:ngpt,ilay,icol) + ymeans(1:ngpt))**8
          ! Optical depth from cross-sections
          tau(1:ngpt,ilay,icol) = tau(1:ngpt,ilay,icol)*col_dry_wk_sp(ilay,icol)

          pfrac(1:ngpt,ilay,icol) = outp_both(ngpt+1:2*ngpt,ilay,icol)*outp_both(ngpt+1:2*ngpt,ilay,icol)
        end do
      end do

#ifdef USE_TIMING
    ret =  gptlstop('write')
#endif   

    end if
  end subroutine predict_nn_lw_blas_mp

  ! --------------------------------------------------------------------------------------
  !
  ! SW neural network kernel using matrix-matrix GEMM computations, used if working precision 
  ! is set as single precision
  !
  subroutine predict_nn_sw_blas_sp(               &
                    ncol, nlay, ngpt, ninputs,       & 
                    nn_inputs, col_dry_wk,                    &
                    neural_nets,                  &
                    tau, ssa)
    ! inputs
    integer,                            intent(in)    :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                        intent(in)    :: nn_inputs 
    real(sp), dimension(nlay,ncol), target, &     
                                          intent(in)    :: col_dry_wk                                    
    ! The neural network models
    type(rrtmgp_network_type), dimension(2),   intent(in)    :: neural_nets
    ! outputs
    real(wp), dimension(ngpt,nlay,ncol), target, &
                                        intent(out) :: tau 
    real(wp), dimension(ngpt,nlay,ncol), target, &
                           optional,    intent(out) :: ssa                                     
    ! ^ wp = sp but using wp here for consistency with combine_2_str_opt                                                                  
    ! local
    real(sp), dimension(:,:), contiguous, pointer     :: input, output, output_ray
    real(sp), dimension(:),   contiguous, pointer     :: input_coldry   
    integer                                           :: nobs

    
    ! PREDICT PLANCK FRACTIONS
    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])

    call C_F_POINTER (C_LOC(col_dry_wk), input_coldry, [nobs])

#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_abs')
#endif
    call C_F_POINTER (C_LOC(tau), output, [ngpt,nobs])

    call neural_nets(1) % output_sgemm_tau(ninputs, ngpt, nobs, input, &
                          input_coldry, output)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_abs')
#endif

    if (present(ssa)) then

#define INLINE_COMBINE
#ifdef INLINE_COMBINE
! ----- merge combine_2str with NN kernel? yes --------
#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_ray_and_combine')
#endif

      ! output = tau_abs, tau_tot call
      ! output_ray = tau_ray, ssa after call
      call C_F_POINTER (C_LOC(ssa), output_ray, [ngpt,nobs])

      call neural_nets(2) % output_sgemm_tau(ninputs, ngpt, nobs, input, input_coldry, output_ray, output)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_ray_and_combine')
#endif

#else
! ----- merge combine_2str with NN kernel? no --------

#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_ray')
#endif
      call C_F_POINTER (C_LOC(ssa), output, [ngpt,nobs])

      call neural_nets(2) % output_sgemm_tau(ninputs, ngpt, nobs, input, input_coldry, output)
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_ray')
    ret =  gptlstart('combine_taus_compute_ssa')
#endif
      ! Now compute tau_tot = tau_ray + tau_abs and ssa = tau_ray / tau_tot
      ! inputs: tau_abs (called tau) and tau_ray (called ssa)
      ! first argument becomes tau_tot and second becomes ssa
      call combine_2str_opt(ncol, nlay, ngpt, tau, ssa) 
#ifdef USE_TIMING
    ret =  gptlstop('combine_taus_compute_ssa')
#endif
#endif
! ----- merge combine_2str with NN kernel? --------

    end if
  end subroutine predict_nn_sw_blas_sp

  ! --------------------------------------------------------------------------------------
  !
  ! SW neural network kernel using matrix-matrix GEMM computations, used if working precision 
  ! is set as double precision (does computations in single precision but has to use temporary output array)
  !
  subroutine predict_nn_sw_blas_mp(               &
                    ncol, nlay, ngpt, ninputs,       & 
                    nn_inputs, col_dry_wk,                    &
                    neural_nets,                  &
                    tau, ssa)
    ! inputs
    integer,                            intent(in)    :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                        intent(in)    :: nn_inputs 
    real(dp), dimension(nlay,ncol),     intent(in)    :: col_dry_wk                                    
    ! The neural network models
    type(rrtmgp_network_type), dimension(2),   intent(in)    :: neural_nets

    ! outputs
    real(wp), dimension(ngpt,nlay,ncol), target, &
                                        intent(out) :: tau 
    real(wp), dimension(ngpt,nlay,ncol), target, &
                           optional,    intent(out) :: ssa                                     
    ! ^ wp = dp but using wp here for consistency with combine_2_str_opt                                                                
    ! local
    real(sp), dimension(nlay,ncol),       target      :: col_dry_wk_sp                                     
    real(sp), dimension(:,:), contiguous, pointer     :: input
    real(sp), dimension(:),   contiguous, pointer     :: input_coldry   
    real(sp), dimension(ngpt,nlay*ncol)               :: tmp_output
    integer                                           :: nobs

    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])

    col_dry_wk_sp = real(col_dry_wk, sp)
    call C_F_POINTER (C_LOC(col_dry_wk_sp), input_coldry, [nobs])

#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_abs')
#endif
    call neural_nets(1) % output_sgemm_tau(ninputs, ngpt, nobs, input, input_coldry, tmp_output)
    
    tau = reshape(tmp_output,(/ngpt,nlay,ncol/))
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_abs')
#endif

    if (present(ssa)) then
#ifdef USE_TIMING
    ret =  gptlstart('compute_tau_ray')
#endif
      call neural_nets(2) % output_sgemm_tau(ninputs, ngpt, nobs, input, input_coldry, tmp_output)

      ssa = reshape(tmp_output,(/ngpt,nlay,ncol/))
#ifdef USE_TIMING
    ret =  gptlstop('compute_tau_ray')
#endif
      ! Now compute tau_tot = tau_ray + tau_abs and ssa = tau_ray / tau_tot
      ! inputs: tau_abs (called tau) and tau_ray (called ssa)
      ! first argument becomes tau_tot and second becomes ssa
      call combine_2str_opt(ncol, nlay, ngpt, tau, ssa) 
    end if

  end subroutine predict_nn_sw_blas_mp

  ! ----------------------------------------------------------
  !
  ! One dimensional interpolation -- return all values along second table dimension
  !
  pure function interpolate1D(val, offset, delta, table) result(res)
    ! input
    real(wp), intent(in) :: val,    & ! axis value at which to evaluate table
                            offset, & ! minimum of table axis
                            delta     ! step size of table axis
    real(wp), dimension(:,:), contiguous, &
              intent(in) :: table ! dimensions (axis, values)
    ! output
    real(wp), dimension(size(table,dim=2)) :: res

    ! local
    real(wp) :: val0 ! fraction index adjusted by offset and delta
    integer :: index ! index term
    real(wp) :: frac ! fractional term
    ! -------------------------------------
    val0 = (val - offset) / delta
    frac = val0 - int(val0) ! get fractional part
    index = min(size(table,dim=1)-1, max(1, int(val0)+1)) ! limit the index range
    res(:) = table(index,:) + frac * (table(index+1,:) - table(index,:))
  end function interpolate1D


  pure function interpolate1D_nocheck(val, offset, delta, table) result(res)
    ! input
    real(wp), intent(in) :: val,    & ! axis value at which to evaluate table
                            offset, & ! minimum of table axis
                            delta     ! step size of table axis
    real(wp), dimension(:,:), contiguous, &
              intent(in) :: table ! dimensions (axis, values)
    ! output
    real(wp), dimension(size(table,dim=2)) :: res

    ! local
    real(wp) :: val0 ! fraction index adjusted by offset and delta
    integer :: index ! the first table index after which val occurs
    real(wp) :: frac ! fractional term
    ! -------------------------------------
    val0 = (val - offset) / delta
    index = int(val0) + 1 ! incremented here, Fortran because indexing starts from 1
    frac = val0 - (index-1) ! get fractional part
    res(:) = table(index,:) + frac * (table(index+1,:) - table(index,:))
  end function interpolate1D_nocheck


  ! ----------------------------------------------------------------------------------------
  !   This function returns a single value from a subset (in gpoint) of the k table
  !
  pure function interpolate2D(fminor, k, igpt, jeta, jtemp) result(res)
    real(wp), dimension(2,2), intent(in) :: fminor ! interpolation fractions for minor species
                                       ! index(1) : reference eta level (temperature dependent)
                                       ! index(2) : reference temperature level
    real(wp), dimension(:,:,:), contiguous, intent(in) :: k ! (g-point, eta, temp)
    integer,                    intent(in) :: igpt, jtemp ! interpolation index for temperature
    integer, dimension(2),      intent(in) :: jeta ! interpolation index for binary species parameter (eta)
    real(wp)                             :: res ! the result

    res =  &
      fminor(1,1) * k(igpt, jeta(1)  , jtemp  ) + &
      fminor(2,1) * k(igpt, jeta(1)+1, jtemp  ) + &
      fminor(1,2) * k(igpt, jeta(2)  , jtemp+1) + &
      fminor(2,2) * k(igpt, jeta(2)+1, jtemp+1)
  end function interpolate2D
  ! ----------------------------------------------------------
  !   This function returns a range of values from a subset (in gpoint) of the k table
  !
  pure function interpolate2D_byflav(fminor, k, gptS, gptE, jeta, jtemp) result(res)
    real(wp), dimension(2,2), intent(in) :: fminor ! interpolation fractions for minor species
                                       ! index(1) : reference eta level (temperature dependent)
                                       ! index(2) : reference temperature level
    real(wp), dimension(:,:,:), intent(in) :: k ! (g-point, eta, temp)
    integer,                    intent(in) :: gptS, gptE, jtemp ! interpolation index for temperature
    integer, dimension(2),      intent(in) :: jeta ! interpolation index for binary species parameter (eta)
    real(wp), dimension(gptE-gptS+1)       :: res ! the result

    ! Local variable
    integer :: igpt
    ! each code block is for a different reference temperature
    do igpt = 1, gptE-gptS+1
      res(igpt) = fminor(1,1) * k(gptS+igpt-1, jeta(1)  , jtemp  ) + &
                  fminor(2,1) * k(gptS+igpt-1, jeta(1)+1, jtemp  ) + &
                  fminor(1,2) * k(gptS+igpt-1, jeta(2)  , jtemp+1) + &
                  fminor(2,2) * k(gptS+igpt-1, jeta(2)+1, jtemp+1)
    end do
  end function interpolate2D_byflav
  ! ----------------------------------------------------------
  ! interpolation in temperature, pressure, and eta
  pure function interpolate3D(scaling, fmajor, k, igpt, jeta, jtemp, jpress) result(res)
    real(wp), dimension(2),     intent(in) :: scaling
    real(wp), dimension(2,2,2), intent(in) :: fmajor ! interpolation fractions for major species
                                                     ! index(1) : reference eta level (temperature dependent)
                                                     ! index(2) : reference pressure level
                                                     ! index(3) : reference temperature level
    real(wp), dimension(:,:,:,:),intent(in) :: k ! (gpt, eta,temp,press)
    integer,                     intent(in) :: igpt
    integer, dimension(2),       intent(in) :: jeta ! interpolation index for binary species parameter (eta)
    integer,                     intent(in) :: jtemp ! interpolation index for temperature
    integer,                     intent(in) :: jpress ! interpolation index for pressure
    real(wp)                                :: res ! the result
    ! each code block is for a different reference temperature
    res =  &
      scaling(1) * &
      ( fmajor(1,1,1) * k(igpt, jeta(1)  , jpress-1, jtemp  ) + &
        fmajor(2,1,1) * k(igpt, jeta(1)+1, jpress-1, jtemp  ) + &
        fmajor(1,2,1) * k(igpt, jeta(1)  , jpress  , jtemp  ) + &
        fmajor(2,2,1) * k(igpt, jeta(1)+1, jpress  , jtemp  ) ) + &
      scaling(2) * &
      ( fmajor(1,1,2) * k(igpt, jeta(2)  , jpress-1, jtemp+1) + &
        fmajor(2,1,2) * k(igpt, jeta(2)+1, jpress-1, jtemp+1) + &
        fmajor(1,2,2) * k(igpt, jeta(2)  , jpress  , jtemp+1) + &
        fmajor(2,2,2) * k(igpt, jeta(2)+1, jpress  , jtemp+1) )
  end function interpolate3D
  ! ----------------------------------------------------------
  pure function interpolate3D_byflav(scaling, fmajor, k, gptS, gptE, jeta, jtemp, jpress) result(res)
    real(wp), dimension(2),     intent(in) :: scaling
    real(wp), dimension(2,2,2), intent(in) :: fmajor ! interpolation fractions for major species
                                                     ! index(1) : reference eta level (temperature dependent)
                                                     ! index(2) : reference pressure level
                                                     ! index(3) : reference temperature level
    real(wp), dimension(:,:,:,:),intent(in) :: k ! (gpt, eta,temp,press)
    integer,                     intent(in) :: gptS, gptE
    integer, dimension(2),       intent(in) :: jeta ! interpolation index for binary species parameter (eta)
    integer,                     intent(in) :: jtemp ! interpolation index for temperature
    integer,                     intent(in) :: jpress ! interpolation index for pressure
    real(wp), dimension(gptE-gptS+1)        :: res ! the result

    ! Local variable
    integer :: igpt
    ! each code block is for a different reference temperature
    do igpt = 1, gptE-gptS+1
      res(igpt) =  &
        scaling(1) * &
        ( fmajor(1,1,1) * k(gptS+igpt-1, jeta(1)  , jpress-1, jtemp  ) + &
          fmajor(2,1,1) * k(gptS+igpt-1, jeta(1)+1, jpress-1, jtemp  ) + &
          fmajor(1,2,1) * k(gptS+igpt-1, jeta(1)  , jpress  , jtemp  ) + &
          fmajor(2,2,1) * k(gptS+igpt-1, jeta(1)+1, jpress  , jtemp  ) ) + &
        scaling(2) * &
        ( fmajor(1,1,2) * k(gptS+igpt-1, jeta(2)  , jpress-1, jtemp+1) + &
          fmajor(2,1,2) * k(gptS+igpt-1, jeta(2)+1, jpress-1, jtemp+1) + &
          fmajor(1,2,2) * k(gptS+igpt-1, jeta(2)  , jpress  , jtemp+1) + &
          fmajor(2,2,2) * k(gptS+igpt-1, jeta(2)+1, jpress  , jtemp+1) )
    end do
  end function interpolate3D_byflav
  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, g
  ! No reorder needed
  !
  pure subroutine combine_2str(ncol, nlay, ngpt, tau_rayleigh, tau, ssa, g) &
      bind(C, name="combine_2str")
    integer,                                intent(in) :: ncol, nlay, ngpt
    real(wp), dimension(ngpt, nlay, ncol),  intent(in   ) :: tau_rayleigh
    real(wp), dimension(ngpt, nlay, ncol),  intent(inout) :: tau, ssa, g ! inout because components are allocated
    ! -----------------------
    integer  :: icol, ilay, igpt
    ! -----------------------

    g(:,:,:) = 0._wp

    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
           tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_rayleigh(igpt,ilay,icol)
           !g  (igpt,ilay,icol) = 0._wp

           ! This conditional is not needed, tau is always >> tiny()?
          !  if(tau(igpt,ilay,icol) > 2._wp * tiny( tau(igpt,ilay,icol))) then
          !    ssa(igpt,ilay,icol) = tau_rayleigh(igpt,ilay,icol) / tau(igpt,ilay,icol)
          !  else
          !    ssa(igpt,ilay,icol) = 0._wp
          !  end if
           ssa(igpt,ilay,icol) = tau_rayleigh(igpt,ilay,icol) / tau(igpt,ilay,icol)
           
           ! FIX for bug when using GFortran compilers with --fast-math, ssa can become slightly larger than 1
           ssa(igpt,ilay,icol) = min(ssa(igpt,ilay,icol), 1.0_wp)
        end do
      end do
    end do
  end subroutine combine_2str

  !
  ! Combine absorption and Rayleigh optical depths for total tau, ssa
  !
  pure subroutine combine_2str_opt(ncol, nlay, ngpt, tau, tau_ray) &
      bind(C, name="combine_2str_opt")
    integer,                                intent(in)    :: ncol, nlay, ngpt
    real(wp), dimension(ngpt, nlay, ncol),  intent(inout) :: tau     ! tau_abs on input, tau_tot on output
    real(wp), dimension(ngpt, nlay, ncol),  intent(inout) :: tau_ray ! tau_ray on input, ssa on output
    ! -----------------------
    integer  :: icol, ilay, igpt
    ! -----------------------
    associate (ssa => tau_ray)

    do icol = 1, ncol
      do ilay = 1, nlay
        ! do igpt = 1, ngpt
        !   tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_ray(igpt,ilay,icol) 

        !   if(tau(igpt,ilay,icol) > 2._wp * tiny( tau(igpt,ilay,icol))) then
        !     ssa(igpt,ilay,icol) = tau_ray(igpt,ilay,icol) / tau(igpt,ilay,icol)
        !     ! ! FIX for bug when using GFortran compilers with --fast-math, ssa can become slightly larger than 1
        !     !   tau_ray(igpt,ilay,icol) = min(tau_ray(igpt,ilay,icol), 1.0_wp)
        !   else
        !       ssa(igpt,ilay,icol) = 0._wp
        !   end if
        ! end do

        ! Is the check for very small tau needed? Near-zero tau not present in RFMIP data nor checked for in ecRAD
        do igpt = 1, ngpt
          tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_ray(igpt,ilay,icol) 
          ssa(igpt,ilay,icol) = tau_ray(igpt,ilay,icol) / tau(igpt,ilay,icol)
        end do
      end do
    end do

    end associate
  end subroutine combine_2str_opt
  ! ----------------------------------------------------------
  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, p
  !   using Rayleigh scattering phase function
  ! No reorder needed
  !
  pure subroutine combine_nstr(ncol, nlay, ngpt, nmom, tau_rayleigh, tau, ssa, p) &
      bind(C, name="combine_nstr")
    integer, intent(in) :: ncol, nlay, ngpt, nmom
    real(wp), dimension(ngpt,nlay,ncol), intent(in ) :: tau_rayleigh
    real(wp), dimension(ngpt, nlay, ncol), intent(inout) :: tau, ssa
    real(wp), dimension(ngpt, nlay, ncol,nmom), &
                                         intent(inout) :: p
    ! -----------------------
    integer :: icol, ilay, igpt, imom
    ! -----------------------
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! tau_tot = tau_abs + tau_rayleigh
          tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_rayleigh(igpt,ilay,icol)
          if(tau(igpt,ilay,icol) > 2._wp * tiny( tau(igpt,ilay,icol))) then
            ssa(igpt,ilay,icol) = tau_rayleigh(igpt,ilay,icol) / tau(igpt,ilay,icol)
          else
            ssa(igpt,ilay,icol) = 0._wp
          end if
          do imom = 1, nmom
            p(imom,igpt,ilay,icol) = 0.0_wp
          end do
          if(nmom >= 2) p(2,igpt,ilay,icol) = 0.1_wp
        end do
      end do
    end do
  end subroutine combine_nstr
  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, g
  !
  pure subroutine combine_and_reorder_2str(ncol, nlay, ngpt, tau_abs, tau_rayleigh, tau, ssa, g) &
      bind(C, name="combine_and_reorder_2str")
    integer,                             intent(in) :: ncol, nlay, ngpt
    real(wp), dimension(ngpt,nlay,ncol), intent(in   ) :: tau_abs, tau_rayleigh
    real(wp), dimension(ncol,nlay,ngpt), intent(inout) :: tau, ssa, g ! inout because components are allocated
    ! -----------------------
    integer  :: icol, ilay, igpt
    real(wp) :: t
    ! -----------------------
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
           t = tau_abs(igpt,ilay,icol) + tau_rayleigh(igpt,ilay,icol)
           tau(icol,ilay,igpt) = t
           g  (icol,ilay,igpt) = 0._wp
           if(t > 2._wp * tiny(t)) then
             ssa(icol,ilay,igpt) = tau_rayleigh(igpt,ilay,icol) / t
           else
             ssa(icol,ilay,igpt) = 0._wp
           end if
        end do
      end do
    end do
  end subroutine combine_and_reorder_2str
  ! ----------------------------------------------------------
  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, p
  !   using Rayleigh scattering phase function
  !
  pure subroutine combine_and_reorder_nstr(ncol, nlay, ngpt, nmom, tau_abs, tau_rayleigh, tau, ssa, p) &
      bind(C, name="combine_and_reorder_nstr")
    integer, intent(in) :: ncol, nlay, ngpt, nmom
    real(wp), dimension(ngpt,nlay,ncol), intent(in ) :: tau_abs, tau_rayleigh
    real(wp), dimension(ncol,nlay,ngpt), intent(inout) :: tau, ssa
    real(wp), dimension(ncol,nlay,ngpt,nmom), &
                                         intent(inout) :: p
    ! -----------------------
    integer :: icol, ilay, igpt, imom
    real(wp) :: t
    ! -----------------------
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          t = tau_abs(igpt,ilay,icol) + tau_rayleigh(igpt,ilay,icol)
          tau(icol,ilay,igpt) = t
          if(t > 2._wp * tiny(t)) then
            ssa(icol,ilay,igpt) = tau_rayleigh(igpt,ilay,icol) / t
          else
            ssa(icol,ilay,igpt) = 0._wp
          end if
          do imom = 1, nmom
            p(imom,icol,ilay,igpt) = 0.0_wp
          end do
          if(nmom >= 2) p(2,icol,ilay,igpt) = 0.1_wp
        end do
      end do
    end do
  end subroutine combine_and_reorder_nstr
end module mo_gas_optics_kernels

