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
  use mo_rte_kind,      only: wp, wl, sp, dp
  use mod_network,      only: network_type, output_sgemm_pfrac_acc, output_sgemm_tau_acc
  use, intrinsic :: ISO_C_BINDING
  
  implicit none

  interface predict_nn_lw_blas
    module procedure predict_nn_lw_blas_sp, predict_nn_lw_blas_mp
  end interface predict_nn_lw_blas


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
                jtemp,fmajor,fminor,col_mix,tropo,jeta,jpress,play_log) bind(C, name="interpolation")
    ! input dimensions
    integer,                            intent(in) :: nlay,ncol
    integer,                            intent(in) :: ngas,nflav,neta,npres,ntemp
    integer,     dimension(2,nflav),    intent(in) :: flavor
    real(wp),    dimension(npres),      intent(in) :: press_ref_log
    real(wp),    dimension(ntemp),      intent(in) :: temp_ref
    real(wp),                           intent(in) :: press_ref_log_delta, &
                                                      temp_ref_min, temp_ref_delta, &
                                                      press_ref_trop_log
    real(wp),    dimension(2,0:ngas,ntemp), intent(in) :: vmr_ref

    ! inputs from profile or parent function
    real(wp),    dimension(nlay,ncol),        intent(in) :: play, tlay
    real(wp),    dimension(nlay,ncol,0:ngas), intent(in) :: col_gas

    ! outputs
    integer,     dimension(nlay,ncol),            intent(out) :: jtemp, jpress
    logical(wl), dimension(nlay,ncol),            intent(out) :: tropo
    integer,     dimension(2,    nflav,nlay,ncol), intent(out) :: jeta
    real(wp),    dimension(2,    nflav,nlay,ncol), intent(out) :: col_mix
    real(wp),    dimension(2,2,2,nflav,nlay,ncol), intent(out) :: fmajor
    real(wp),    dimension(2,2,  nflav,nlay,ncol), intent(out) :: fminor
    real(wp),    dimension(nlay, ncol),            intent(out) :: play_log
    ! -----------------
    ! local
    real(wp), dimension(nlay,ncol) :: ftemp, fpress ! interpolation fraction for temperature, pressure
    real(wp) :: locpress ! needed to find location in pressure grid
    real(wp) :: ratio_eta_half ! ratio of vmrs of major species that defines eta=0.5
                               ! for given flavor and reference temperature level
    real(wp) :: eta, feta      ! binary_species_parameter, interpolation variable for eta
    real(wp) :: loceta         ! needed to find location in eta grid
    real(wp) :: ftemp_term
    ! -----------------
    ! local indexes
    integer :: ilay, icol, iflav, igases(2), itropo, itemp

    !$acc data present(jtemp, jpress, jeta, tropo, fmajor)

    !$acc enter data copyin(flavor,press_ref_log,temp_ref,vmr_ref)
    !$acc enter data create(col_mix,fminor)
    !$acc enter data create(ftemp,fpress)

    !$acc parallel loop gang vector collapse(2)
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

    ! loop over implemented combinations of major species
    ! PGI BUG WORKAROUND: if present(vmr_ref) isn't there, OpenACC runtime
    ! thinks it isn't present.
    !$acc parallel loop gang vector collapse(4) private(igases) present(vmr_ref)
    do icol = 1, ncol
      do ilay = 1, nlay
        ! loop over implemented combinations of major species
        do iflav = 1, nflav
          do itemp = 1, 2
            igases(:) = flavor(:,iflav)
            ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
            itropo = merge(1,2,tropo(ilay,icol))
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

    !$acc exit data delete(flavor,press_ref_log,temp_ref,vmr_ref)

    ! copyout deletes data from device?
    ! !$acc exit data copyout(jtemp,jpress,tropo,jeta,col_mix,fmajor,fminor)

    !$acc exit data delete(ftemp,fpress)

    !$acc end data

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
    logical(wl), dimension(nlay,ncol),               intent(in) :: tropo
    ! ---------------------
    ! inputs from profile or parent function
    real(wp), dimension(2,    nflav,nlay,ncol       ), intent(in) :: col_mix
    real(wp), dimension(2,2,2,nflav,nlay,ncol       ), intent(in) :: fmajor
    real(wp), dimension(2,2,  nflav,nlay,ncol       ), intent(in) :: fminor
    real(wp), dimension(            nlay,ncol       ), intent(in) :: play, tlay      ! pressure and temperature
    real(wp), dimension(            nlay,ncol,0:ngas), intent(in) :: col_gas
    integer,  dimension(2,    nflav,nlay,ncol       ), intent(in) :: jeta
    integer,  dimension(            nlay,ncol       ), intent(in) :: jtemp
    integer,  dimension(            nlay,ncol       ), intent(in) :: jpress
    ! ---------------------
    ! output - optical depth
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau
    
    ! ---------------------
    ! Local variables
    !
    logical(wl)                :: top_at_1
    integer, dimension(ncol,2) :: itropo_lower, itropo_upper
    integer                    :: icol, idx_tropo

    ! ----------------------------------------------------------------

    !$acc enter data create(itropo_lower, itropo_upper)

    !$acc data present(play, tlay, tropo, gpoint_flavor, jeta, jtemp, col_gas, fminor, tau)

    ! ---------------------
    ! Layer limits of upper, lower atmospheres
    ! ---------------------
    top_at_1 = play(1,1) < play(nlay, 1)
    if(top_at_1) then
      !$acc parallel loop
      do icol = 1,ncol
        itropo_lower(icol,2) = nlay
        itropo_lower(icol,1) = minloc(play(:,icol), dim=1, mask=tropo(:,icol))
        itropo_upper(icol,1) = 1
        itropo_upper(icol,2) = maxloc(play(:,icol), dim=1, mask=(.not. tropo(:,icol)))
      end do
    else
      !$acc parallel loop
      do icol = 1,ncol
        itropo_lower(icol,1) = 1
        itropo_lower(icol,2) = minloc(play(:,icol), dim=1, mask=tropo(:,icol))
        itropo_upper(icol,2) = nlay
        itropo_upper(icol,1) = maxloc(play(:,icol), dim=1, mask=(.not.tropo(:,icol)))
      end do
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
    idx_tropo = 1
    call gas_optical_depths_minor(     &
           ncol,nlay,ngpt,             & ! dimensions
           ngas,nflav,ntemp,neta,      &
           nminorlower,nminorklower,   &
           idx_h2o,idx_tropo,          &
           gpoint_flavor,              &
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
    idx_tropo = 2
    call gas_optical_depths_minor(     &
           ncol,nlay,ngpt,             & ! dimensions
           ngas,nflav,ntemp,neta,      &
           nminorupper,nminorkupper,   &
           idx_h2o,idx_tropo,          &
           gpoint_flavor,              &
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

    !$acc end data 

    !$acc exit data delete(itropo_lower,itropo_upper)
    !$acc exit data delete(gpoint_flavor, col_gas, fminor)
    !$acc exit data copyout(tau)

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
    real(wp), dimension(ngpt,nlay,ncol), intent(inout) :: tau
    ! -----------------
    ! local variables
    real(wp) :: tau_major ! major species optical depth
    ! local index
    integer :: icol, ilay, iflav, ibnd, igpt, itropo
    integer :: gptS, gptE
    ! -----------------

    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
          itropo = merge(1,2,tropo(ilay,icol))
          iflav = gpoint_flavor(itropo, igpt) !eta interpolation depends on band's flavor
          tau_major = &
            ! interpolation in temperature, pressure, and eta
            interpolate3D(col_mix(:,iflav,ilay,icol),                   &
                                 fmajor(:,:,:,iflav,ilay,icol), kmajor, &
                                 igpt, jeta(:,iflav,ilay,icol), jtemp(ilay,icol),jpress(ilay,icol)+itropo)
          tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_major
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
                                      idx_h2o,idx_tropo,     &
                                      gpt_flv,               &
                                      kminor,                &
                                      minor_limits_gpt,      &
                                      minor_scales_with_density,    &
                                      scale_by_complement,   &
                                      idx_minor, idx_minor_scaling, &
                                      kminor_start,        &
                                      play, tlay,          &
                                      col_gas,fminor,jeta, &
                                      layer_limits,jtemp,  &
                                      tau) bind(C, name="gas_optical_depths_minor")
    integer,                                     intent(in   ) :: ncol,nlay,ngpt
    integer,                                     intent(in   ) :: ngas,nflav
    integer,                                     intent(in   ) :: ntemp,neta,nminor,nminork
    integer,                                     intent(in   ) :: idx_h2o, idx_tropo
    integer,     dimension(2, ngpt),             intent(in   ) :: gpt_flv
    real(wp),    dimension(nminork,neta,ntemp),  intent(in   ) :: kminor
    integer,     dimension(2,nminor),            intent(in   ) :: minor_limits_gpt
    logical(wl), dimension(  nminor),            intent(in   ) :: minor_scales_with_density
    logical(wl), dimension(  nminor),            intent(in   ) :: scale_by_complement
    integer,     dimension(  nminor),            intent(in   ) :: kminor_start
    integer,     dimension(  nminor),            intent(in   ) :: idx_minor, idx_minor_scaling
    real(wp),    dimension(nlay, ncol),          intent(in   ) :: play, tlay
    real(wp),    dimension(nlay, ncol,0:ngas),   intent(in   ) :: col_gas
    real(wp),    dimension(2,2,nflav,nlay,ncol), intent(in   ) :: fminor
    integer,     dimension(2,  nflav,nlay,ncol), intent(in   ) :: jeta
    integer,     dimension(ncol, 2),             intent(in   ) :: layer_limits
    integer,     dimension(nlay,ncol),           intent(in   ) :: jtemp
    real(wp),    dimension(ngpt,nlay,ncol),      intent(inout) :: tau
    ! -----------------
    ! local variables
    real(wp), parameter :: PaTohPa = 0.01
    real(wp) :: vmr_fact, dry_fact             ! conversion from column abundance to dry vol. mixing ratio;
    real(wp) :: scaling, kminor_loc, tau_minor ! minor species absorption coefficient, optical depth
    integer  :: icol, ilay, iflav, igpt, imnr
    integer  :: gptS, gptE
    integer  :: minor_start, minor_loc, extent

    real(wp) :: myplay, mytlay, mycol_gas_h2o, mycol_gas_imnr, mycol_gas_0
    real(wp) :: myfminor(2,2)
    integer  :: myjtemp, myjeta(2), max_gpt_diff, igpt0
    ! -----------------

    extent = size(scale_by_complement,dim=1)

    ! Find the largest number of g-points per band
    max_gpt_diff = maxval( minor_limits_gpt(2,:) - minor_limits_gpt(1,:) )

    !$acc parallel loop gang vector collapse(3)
    do icol = 1, ncol
      do ilay = 1 , nlay
        do igpt0 = 0, max_gpt_diff
          !
          ! This check skips individual columns with no pressures in range
          !
          if ( layer_limits(icol,1) <= 0 .or. ilay < layer_limits(icol,1) .or. ilay > layer_limits(icol,2) ) cycle

          myplay  = play (ilay, icol)
          mytlay  = tlay (ilay, icol)
          myjtemp = jtemp(ilay, icol)
          mycol_gas_h2o = col_gas(ilay, icol,idx_h2o)
          mycol_gas_0   = col_gas(ilay, icol,0)

          do imnr = 1, extent

            scaling = col_gas(ilay, icol,idx_minor(imnr))
            if (minor_scales_with_density(imnr)) then
              !
              ! NOTE: P needed in hPa to properly handle density scaling.
              !
              scaling = scaling * (PaTohPa * myplay/mytlay)

              if(idx_minor_scaling(imnr) > 0) then  ! there is a second gas that affects this gas's absorption
                mycol_gas_imnr = col_gas(ilay, icol,idx_minor_scaling(imnr))
                vmr_fact = 1._wp / mycol_gas_0
                dry_fact = 1._wp / (1._wp + mycol_gas_h2o * vmr_fact)
                ! scale by density of special gas
                if (scale_by_complement(imnr)) then ! scale by densities of all gases but the special one
                  scaling = scaling * (1._wp - mycol_gas_imnr * vmr_fact * dry_fact)
                else
                  scaling = scaling *          (mycol_gas_imnr * vmr_fact * dry_fact)
                endif
              endif
            endif

            scaling = scaling * (1._wp - col_gas(icol,ilay,idx_minor_scaling(imnr)) * vmr_fact * dry_fact)

            !
            ! Interpolation of absorption coefficient and calculation of optical depth
            !
            ! Which gpoint range does this minor gas affect?
            gptS = minor_limits_gpt(1,imnr)
            gptE = minor_limits_gpt(2,imnr)

            ! Find the actual g-point to work on
            igpt = igpt0 + gptS

            ! Proceed only if the g-point is within the correct range
            if (igpt <= gptE) then
              ! What is the starting point in the stored array of minor absorption coefficients?
              minor_start = kminor_start(imnr)

              tau_minor = 0._wp
              iflav = gpt_flv(idx_tropo,igpt) ! eta interpolation depends on flavor
              minor_loc = minor_start + (igpt - gptS) ! add offset to starting point
              kminor_loc = interpolate2D(fminor(:,:,iflav,ilay, icol), kminor, minor_loc, &
                                          jeta(:,iflav,ilay, icol), myjtemp)
                                                 
              tau_minor = kminor_loc * scaling
              !$acc atomic update
              tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_minor
            endif

          enddo

        enddo
      enddo
    enddo


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
    real(wp),    dimension(nlay,ncol),           intent(in ) :: col_dry
    real(wp),    dimension(nlay,ncol,0:ngas),    intent(in ) :: col_gas
    real(wp),    dimension(2,2,nflav,nlay,ncol), intent(in ) :: fminor
    integer,     dimension(2,  nflav,nlay,ncol), intent(in ) :: jeta
    logical(wl), dimension(nlay,ncol),           intent(in ) :: tropo
    integer,     dimension(nlay,ncol),           intent(in ) :: jtemp
    ! outputs
    real(wp),    dimension(ngpt,nlay,ncol),      intent(out) :: tau_rayleigh
    ! -----------------
    ! local variables
    real(wp) :: k ! rayleigh scattering coefficient
    integer  :: icol, ilay, iflav, igpt
    integer  :: itropo
    ! -----------------
    !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          itropo = merge(1,2,tropo(ilay, icol)) ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
          iflav = gpoint_flavor(itropo, igpt)
          k = interpolate2D(fminor(:,:,iflav,ilay, icol), &
                            krayl(:,:,:,itropo),      &
                            igpt, jeta(:,iflav,ilay, icol), jtemp(ilay, icol))
          tau_rayleigh(igpt,ilay,icol) =  k * (col_gas(ilay, icol,idx_h2o)+col_dry(ilay, icol))
        end do
      end do
    end do
  end subroutine compute_tau_rayleigh

  ! ----------------------------------------------------------
  ! Calculation of fraction of band's Planck irradiance associated with each g-point
! ----------------------------------------------------------
  subroutine compute_source_bybnd_pfrac_bygpt(             &
                    ncol, nlay, nbnd, ngpt,                &
                    nflav, neta, npres, ntemp, nPlanckTemp,&
                    tlay, tlev, tsfc,             &
                    fmajor, jeta, tropo, jtemp, jpress,    &
                    gpoint_bands, band_lims_gpt,           &
                    temp_ref_min, totplnk_delta, pfracin, totplnk, gpoint_flavor, &
                    sfc_source_bnd, sfc_source_bnd_Jac,    &
                    lay_source_bnd, lev_source_bnd, pfrac) bind(C, name="compute_source_bybnd_pfrac_bygpt")
    integer,                                    intent(in) :: ncol, nlay, nbnd, ngpt
    integer,                                    intent(in) :: nflav, neta, npres, ntemp, nPlanckTemp
    real(wp),    dimension(nlay, ncol  ),        intent(in) :: tlay
    real(wp),    dimension(nlay+1, ncol),        intent(in) :: tlev
    real(wp),    dimension(ncol       ),        intent(in) :: tsfc
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

    real(wp), dimension(nbnd,     ncol),          intent(inout) :: sfc_source_bnd
    real(wp), dimension(nbnd,     ncol),          intent(inout) :: sfc_source_bnd_Jac
    real(wp), dimension(nbnd,nlay,ncol),          intent(inout) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),        intent(inout) :: lev_source_bnd
    real(wp), dimension(ngpt,nlay,ncol),          intent(inout) :: pfrac
    ! -----------------
    ! local
    integer  :: ilay, icol, igpt, ibnd, itropo, iflav
    integer  :: gptS, gptE
    real(wp), dimension(2), parameter :: one = [1._wp, 1._wp]
    real(wp), parameter               :: delta_Tsurf = 1.0_wp

    ! -----------------    

    ! Calculation of fraction of band's Planck irradiance associated with each g-point
    
    !$acc data present(sfc_source_bnd,sfc_source_bnd_Jac,lev_source_bnd,lay_source_bnd,tsfc,tlay,tlev,temp_ref_min,totplnk_delta,jpress,jtemp,jeta,fmajor,tropo,totplnk,pfrac)
    
    !$acc parallel loop 
    do icol = 1, ncol
      !
      ! Planck function by band for the surface
      !
      call interpolate1D(tsfc(icol),   temp_ref_min, totplnk_delta, totplnk, sfc_source_bnd(:,icol))
      call interpolate1D(tsfc(icol) + delta_Tsurf,   temp_ref_min, totplnk_delta, totplnk, sfc_source_bnd_Jac(:,icol))
      call interpolate1D(tlev(1,icol), temp_ref_min, totplnk_delta, totplnk, lev_source_bnd(:,1, icol))
    end do

    ! explicit loop unrolling
    !$acc parallel loop collapse(2)
    do icol = 1, ncol, 2
      do ilay = 1, nlay
        call interpolate1D(tlev(ilay+1,icol),  temp_ref_min, totplnk_delta, totplnk, lev_source_bnd(:,ilay+1,icol))
        call interpolate1D(tlay(ilay,icol),    temp_ref_min, totplnk_delta, totplnk, lay_source_bnd(:,ilay,icol))

        if (icol < ncol) then
          call interpolate1D(tlev(ilay+1,icol+1),  temp_ref_min, totplnk_delta, totplnk, lev_source_bnd(:,ilay+1,icol+1))
          call interpolate1D(tlay(ilay,icol+1),    temp_ref_min, totplnk_delta, totplnk, lay_source_bnd(:,ilay,icol+1))
        end if
      end do
    end do

   !$acc parallel loop collapse(3)
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
        ! Calculation of fraction of band's Planck irradiance associated with each g-point
        ! itropo = 1 lower atmosphere; itropo = 2 upper atmosphere
          itropo = merge(1,2,tropo(ilay,icol))
          iflav = gpoint_flavor(itropo, igpt) !eta interpolation depends on band's flavor
          pfrac(igpt,ilay,icol) = &
            ! interpolation in temperature, pressure, and eta
            interpolate3D(one, fmajor(:,:,:,iflav,ilay,icol), pfracin, &
                          igpt, jeta(:,iflav,ilay,icol), jtemp(ilay,icol),jpress(ilay,icol)+itropo)              
        end do ! g-point
      end do ! ilay
    end do ! icol

    !$acc end data

  end subroutine compute_source_bybnd_pfrac_bygpt

  subroutine compute_source_bybnd(                    &
                    ncol, nlay, nbnd,                 &
                    ntemp, nPlanckTemp,                   &
                    tlay, tlev, tsfc,            &
                    temp_ref_min, totplnk_delta, totplnk, &
                    sfc_source_bnd, sfc_source_bnd_Jac,   &
                    lay_source_bnd, lev_source_bnd) bind(C, name="compute_Planck_bybnd")
    integer,                                    intent(in) :: ncol, nlay, nbnd
    integer,                                    intent(in) :: ntemp, nPlanckTemp
    real(wp),    dimension(nlay,ncol  ),        intent(in) :: tlay
    real(wp),    dimension(nlay+1,ncol),        intent(in) :: tlev
    real(wp),    dimension(ncol       ),        intent(in) :: tsfc

    real(wp),                                     intent(in) :: temp_ref_min, totplnk_delta
    real(wp), dimension(nPlanckTemp,nbnd),        intent(in) :: totplnk

    real(wp), dimension(nbnd,     ncol),          intent(out) :: sfc_source_bnd
    real(wp), dimension(nbnd,     ncol),          intent(out) :: sfc_source_bnd_Jac
    real(wp), dimension(nbnd,nlay,ncol),          intent(out) :: lay_source_bnd
    real(wp), dimension(nbnd,nlay+1,ncol),        intent(out) :: lev_source_bnd
    ! -----------------
    ! local
    integer  :: ilay, icol, ibnd
    real(wp), parameter                             :: delta_Tsurf = 1.0_wp

    !$acc data present(sfc_source_bnd,lev_source_bnd,lay_source_bnd,tsfc,tlay,tlev,temp_ref_min,totplnk_delta,totplnk)
    
    !$acc parallel loop
    do icol = 1, ncol
      !
      ! Planck function by band for the surface
      ! Compute surface source irradiance for g-point, equals band irradiance x fraction for g-point
      !
      call interpolate1D(tsfc(icol),   temp_ref_min, totplnk_delta, totplnk, sfc_source_bnd(:,icol))
      call interpolate1D(tsfc(icol) + delta_Tsurf,   temp_ref_min, totplnk_delta, totplnk, sfc_source_bnd_Jac(:,icol))
      call interpolate1D(tlev(1,icol), temp_ref_min, totplnk_delta, totplnk, lev_source_bnd(:,1, icol))
    end do ! icol

    !$acc parallel loop collapse(2)
    do icol = 1, ncol
      do ilay = 1, nlay
        call interpolate1D(tlay(ilay,icol),    temp_ref_min, totplnk_delta, totplnk, lay_source_bnd(:,ilay,icol))
        call interpolate1D(tlev(ilay+1,icol),  temp_ref_min, totplnk_delta, totplnk, lev_source_bnd(:,ilay+1,icol))
      end do ! ilay
    end do ! icol

    !$acc end data

  end subroutine compute_source_bybnd

  ! --------------------------------------------------------------------------------------
  !
  ! neural network kernel using matrix-matrix GEMM computations, used if working precision is set as double precision
  ! (does computations in single precision but has to use temporary output array)
  !
  subroutine predict_nn_lw_blas_mp(                  &
                    ncol, nlay, ngpt, ninputs,       & 
                    nn_inputs,                    &
                    neural_nets,                  &
                    tau, pfrac)
    ! inputs
    integer,                              intent(in)    :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                          intent(in)    :: nn_inputs 
    ! The neural network models
    type(network_type), dimension(2),     intent(in)    :: neural_nets

    real(dp), dimension(ngpt,nlay,ncol),  intent(inout) :: pfrac, tau
    ! local
    real(sp), dimension(:,:), contiguous, &
                              pointer   :: input
    real(sp), dimension(ngpt,nlay*ncol) :: tmp_output
    integer                             :: nobs

    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])
    
! #ifdef USE_TIMING
!     ret =  gptlstart('tmp_init')
! #endif
!     tmp_output = 0.0_sp
! #ifdef USE_TIMING
!     ret =  gptlstop('tmp_init')
! #endif
! #ifdef USE_TIMING
!     ret =  gptlstart('compute_pfrac')
! #endif
!     call neural_nets(1) % nn_kernel_m(ninputs, ngpt, nobs, input, tmp_output)
!     pfrac = reshape(tmp_output,(/ngpt,nlay,ncol/))

! #ifdef USE_TIMING
!     ret =  gptlstop('compute_pfrac')
!     ret =  gptlstart('compute_tau')
! #endif
!     call neural_nets(2) % nn_kernel_m(ninputs, ngpt, nobs, input, tmp_output)
! #ifdef USE_TIMING
!     ret =  gptlstart('output_reshape')
! #endif
!     tau = reshape(tmp_output,(/ngpt,nlay,ncol/))  
! #ifdef USE_TIMING
!     ret =  gptlstop('output_reshape')
! #endif
! #ifdef USE_TIMING
!     ret =  gptlstop('compute_tau')
! #endif

  end subroutine predict_nn_lw_blas_mp

  ! --------------------------------------------------------------------------------------
  !
  ! neural network kernel using matrix-matrix GEMM computations, used if working precision is set as single precision
  ! (avoids temporary output array, which is faster)
  !

    subroutine predict_nn_lw_blas_sp(               &
                    ncol, nlay, ngpt, ninputs,       & 
                    nn_inputs,                    &
                    neural_nets,                  &
                    tau, pfrac)
    ! inputs
    integer,                            intent(in)    :: ncol, nlay, ngpt, ninputs
    real(sp), dimension(ninputs,nlay,ncol), target, &     
                                        intent(in)    :: nn_inputs 
    ! The neural network models
    type(network_type), dimension(2),   intent(in)    :: neural_nets

    ! outputs
    real(sp), dimension(ngpt,nlay,ncol), target, &
                                        intent(out) :: pfrac, tau
    ! local
    real(sp), dimension(:,:), contiguous, pointer     :: input, output
    integer                                           :: ilay, icol, nobs

    
    ! PREDICT PLANCK FRACTIONS
    nobs = nlay*ncol
    call C_F_POINTER (C_LOC(nn_inputs), input, [ninputs,nobs])

    call C_F_POINTER (C_LOC(tau), output, [ngpt,nobs])

    call neural_nets(2) % output_sgemm_tau_acc(ninputs, ngpt, nobs, input, output)

    call C_F_POINTER (C_LOC(pfrac), output, [ngpt,nobs])

    call neural_nets(1) %  output_sgemm_pfrac_acc(ninputs,ngpt,nobs,input, output)

  end subroutine predict_nn_lw_blas_sp

  ! subroutine predict_nn_lw_blas_sp(               &
  !                   ncol, nlay, ngpt, ninputs,       & 
  !                   nn_inputs,                    &
  !                   neural_nets,                  &
  !                   tau, pfrac)
  !   ! inputs
  !   integer,                            intent(in)    :: ncol, nlay, ngpt, ninputs
  !   real(sp), dimension(ninputs,nlay,ncol), target, &     
  !                                       intent(in)    :: nn_inputs 
  !   ! The neural network models
  !   type(network_type), dimension(2),   intent(in)    :: neural_nets

  !   ! outputs
  !   real(sp), dimension(ngpt,nlay,ncol), target, &
  !                                       intent(out) :: pfrac, tau
  !   ! local
  !   real(sp), dimension(ngpt,nlay*ncol) :: tmp_output
  !   real(sp), dimension(ninputs,nlay*ncol) :: tmp_input
                      
  !   integer     :: ilay, icol, nobs, igpt, iobs, igas

  !   ! PREDICT PLANCK FRACTIONS
  !   nobs = nlay*ncol

  !   !$acc enter data create(tmp_input, tmp_output)

  !   !$acc parallel loop collapse(3)
  !   do icol = 1, ncol
  !     do ilay = 1, nlay
  !         do igas = 1, ninputs
  !           iobs = icol*ilay
  !           tmp_input(igas,iobs) = nn_inputs(igas, ilay, icol)
  !         end do 
  !     end do 
  !   end do

  !   call neural_nets(2) % output_sgemm_tau_acc(ninputs, ngpt, nobs, tmp_input, tmp_output)
  
  !   !$acc parallel loop collapse(3)
  !   do icol = 1, ncol
  !     do ilay = 1, nlay
  !         do igpt = 1, ngpt
  !           iobs = icol*ilay
  !           tau(igpt,ilay,icol) = tmp_output(igpt, iobs)
  !         end do 
  !     end do 
  !   end do

  !   call neural_nets(1) %  output_sgemm_pfrac_acc(ninputs,ngpt,nobs,tmp_input, tmp_output)

  !   !$acc parallel loop collapse(3)
  !   do icol = 1, ncol
  !     do ilay = 1, nlay
  !         do igpt = 1, ngpt
  !           iobs = icol*ilay
  !           pfrac(igpt,ilay,icol) = tmp_output(igpt, iobs)
  !         end do 
  !     end do 
  !   end do

  !   !$acc exit data delete(tmp_input, tmp_output)

  ! end subroutine predict_nn_lw_blas_sp

  ! ----------------------------------------------------------
  !
  ! One dimensional interpolation -- return all values along second table dimension
  !
  subroutine interpolate1D(val, offset, delta, table, res)
  !$acc routine seq
    ! input
    real(wp), intent(in) :: val,    & ! axis value at which to evaluate table
                            offset, & ! minimum of table axis
                            delta     ! step size of table axis
    real(wp), dimension(:,:), contiguous, &
              intent(in) :: table ! dimensions (axis, values)
    ! output
    real(wp), intent(out), dimension(size(table,dim=2)) :: res

    ! local
    real(wp) :: val0 ! fraction index adjusted by offset and delta
    integer :: index ! index term
    real(wp) :: frac ! fractional term
    ! -------------------------------------
    val0 = (val - offset) / delta
    frac = val0 - int(val0) ! get fractional part
    index = min(size(table,dim=1)-1, max(1, int(val0)+1)) ! limit the index range
    res(:) = table(index,:) + frac * (table(index+1,:) - table(index,:))
  end subroutine interpolate1D
  ! ------------
  !   This function returns a single value from a subset (in gpoint) of the k table
  !
  function interpolate2D(fminor, k, igpt, jeta, jtemp) result(res)
  !$acc routine seq
    real(wp), dimension(2,2), intent(in) :: fminor ! interpolation fractions for minor species
                                       ! index(1) : reference eta level (temperature dependent)
                                       ! index(2) : reference temperature level
    real(wp), dimension(:,:,:), intent(in) :: k ! (g-point, eta, temp)
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
  ! interpolation in temperature, pressure, and eta
  function interpolate3D(scaling, fmajor, k, igpt, jeta, jtemp, jpress) result(res)
  !$acc routine seq
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

  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, g
  !
  pure subroutine combine_2str(ncol, nlay, ngpt, tau_rayleigh, tau, ssa, g) &
      bind(C, name="combine_2str")
    integer,                                intent(in) :: ncol, nlay, ngpt
    real(wp), dimension(ngpt,nlay, ncol),    intent(in   ) :: tau_rayleigh
    real(wp), dimension(ngpt, nlay, ncol),  intent(inout) :: tau, ssa, g ! inout because components are allocated
    ! -----------------------
    integer  :: icol, ilay, igpt
    real(wp) :: t
    ! -----------------------
    do icol = 1, ncol
      do ilay = 1, nlay
        do igpt = 1, ngpt
          ! tau_tot = tau_abs + tau_rayleigh
           tau(igpt,ilay,icol) = tau(igpt,ilay,icol) + tau_rayleigh(igpt,ilay,icol)
           g  (igpt,ilay,icol) = 0._wp
           if(tau(igpt,ilay,icol) > 2._wp * tiny( tau(igpt,ilay,icol))) then
            ! ssa = tau_rayleigh / tau_tot
             ssa(igpt,ilay,icol) = tau_rayleigh(igpt,ilay,icol) / tau(igpt,ilay,icol)
           else
             ssa(igpt,ilay,icol) = 0._wp
           end if
        end do
      end do
    end do
  end subroutine combine_2str
  ! ----------------------------------------------------------
  !
  ! Combine absoprtion and Rayleigh optical depths for total tau, ssa, p
  !   using Rayleigh scattering phase function
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
    real(wp) :: t
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
  
end module mo_gas_optics_kernels
