module mod_network_rrtmgp

    ! Changelog:
    !        P. Ukkonen, 24.1.2022: Create new network_rrtmgp type which extends mod_network
    !                               All RRTMGP-specific things are put here
    ! 
     
  use mo_rte_kind, only: sp
  use mod_network
  use mo_simple_netcdf,      only: get_dim_size, read_field, read_char_vec
  use netcdf
#ifdef USE_TIMING
  ! Timing library
  use gptl,                  only: gptlstart, gptlstop
#endif

#ifdef USE_OPENACC
  use cublas 
  use openacc
#define sgemm cublassgemm
#endif

  implicit none
 
#ifdef USE_TIMING
  integer  :: ret, i
#endif

  private

  public :: rrtmgp_network_type, output_sgemm_pfrac, output_sgemm_tau, output_sgemm_lw


  type, extends(network_type) :: rrtmgp_network_type

  ! ! ---------------------------------------------------------------------------------------
  ! ! -------------------  Coefficients for scaling inputs and outputs ----------------------
  ! ! ---------------------------------------------------------------------------------------

  real(sp),       dimension(:), allocatable :: coeffs_input_min, coeffs_input_max
  real(sp),       dimension(:), allocatable :: coeffs_output_mean, coeffs_output_std
  character(32),  dimension(:), allocatable :: input_names 
  ! ---------------------------------------------------------------------------------------


  contains


    procedure, public, pass(self) :: load_netcdf
    ! Inference kernels using BLAS and custom post-processing
    procedure, public, pass(self) :: output_sgemm_pfrac, output_sgemm_tau, output_sgemm_lw  

  end type rrtmgp_network_type


contains

  subroutine load_netcdf(self, filename)
    ! Loads the network from file.
    class(rrtmgp_network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer :: ncid, n, num_layers, nx, dim1, dim2, varid
    integer, allocatable :: dims(:)
    character(len=20) :: varname_weight, varname_bias
    character(len=5) :: charN
    ! real(sp), dimension(:,:), allocatable :: tmpvar
    ! real(sp), dimension(:), allocatable :: tmpbias
    character(len=32), dimension(:), allocatable :: string_array

    if(nf90_open(trim(filename), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("mod_network_rrtmgp:load_netcdf: can't find file " // trim(filename))

    num_layers = get_dim_size(ncid, 'nn_layers')
    ! print *, "num nn layers", num_layers

    allocate(dims(num_layers+1))

    nx = get_dim_size(ncid, 'nn_dim_input')
    dims(1) = nx
    dims(2:num_layers+1) = read_field(ncid, "nn_dimsize",  num_layers)

    call self % init(dims)
      
    !!$acc enter data copyin(self) 
    !!$acc enter data copyin(self % dims)  
    !!$acc enter data copyin(self % layers)
    do n = 1, num_layers
      dim1 = dims(n)
      dim2 = dims(n+1)
      ! create string containing the variable name
      write(charN, '(I5)') n
      varname_weight = 'nn_weights_'// trim(adjustl(charN)) // ''
      varname_bias   = 'nn_bias_'// trim(adjustl(charN)) // ''
      
      self % layers(n) % w =  transpose(read_field(ncid, varname_weight,  dim2, dim1))
      self % layers(n) % b =  read_field(ncid, varname_bias,  dim2)
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )  
      !$acc enter data copyin(self % layers(n)%b, self % layers(n)%w_transposed) async   
    end do

    ! Load and set layer activation functions
    string_array = read_char_vec(ncid, 'nn_activation_char', num_layers)
    do n = 1, num_layers
      call self % layers(n) % set_activation(adjustl(string_array(n)))
    end do    
    deallocate(string_array)

    ! Load and set input names
    string_array = read_char_vec(ncid, 'nn_inputs_char', nx)
    self % input_names = string_array

    ! Load and set scaling coefficients
    self % coeffs_input_min =  read_field(ncid, 'nn_input_coeffs_min',  nx)
    self % coeffs_input_max =  read_field(ncid, 'nn_input_coeffs_max',  nx)

    if(nf90_inq_varid(ncid, 'nn_output_coeffs_mean', varid) == NF90_NOERR) then
      self % coeffs_output_mean = read_field(ncid, 'nn_output_coeffs_mean',  dim2)
    end if
    if(nf90_inq_varid(ncid, 'nn_output_coeffs_std', varid) == NF90_NOERR) then
      self % coeffs_output_std = read_field(ncid, 'nn_output_coeffs_std',  dim2)
    end if
  end subroutine load_netcdf


  subroutine output_sgemm_tau(self, nx, ngpt, nbatch, x, coldry, output, output2)
    ! Like output_sgemm_flat but for computing OPTICAL DEPTH, inlining the post-processing
    ! Additional inputs: number of dry air molecules (coldry) and the mean and standard deviation
    ! used for normalization (ymeans, ysigma)
    use, intrinsic :: iso_c_binding
    class(rrtmgp_network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ngpt, nbatch
    real(sp), dimension(nx, nbatch),  intent(in)  :: x      ! (features, nbatch)
    real(sp), dimension(nbatch),      intent(in)  :: coldry ! number of dry air molecules
    real(sp), dimension(ngpt, nbatch),intent(out) :: output ! optical depth (or ssa if output2 present)
    real(sp), dimension(ngpt, nbatch), optional, intent(out) :: output2 ! absorption od=> total od

    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
                                          target  :: a1, a2  
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer                       :: n, j, neurons, nlayers, i
    real(sp), dimension(ngpt)                     :: ymeans,ystd ! standard-scaling coefficients

    if (.not. allocated(self%coeffs_output_mean)) &
      stop "output_sgemm_tau: NN output scaling coefficients missing"
    ymeans = self%coeffs_output_mean
    ystd   = self%coeffs_output_std
    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc data create(a1, a2) copyin(ymeans,ystd)

    associate(layers=>self%layers)
      
      ! Assign pointers to layer weights, biases and input-output arrays
      wt      => layers(1) % w_transposed
      a       => a1
      a_next  => a2
      b       => layers(1) % b
#ifdef USE_TIMING
    ret =  gptlstart('first_sgemm')
#endif  
      ! 1. Multiply inputs with the weights of the first layer
      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data
#ifdef USE_TIMING
    ret =  gptlstop('first_sgemm')
#endif  
      ! 2. Add biases and activation
     call layers(1) % bias_and_activation(a, b)
      
      ! 3. Repeat steps 1-2 until final layer reached
      do n = 2, nlayers-1

        wt => layers(n) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers, the previous output is the next input
        if(mod(n,2) .EQ. 0) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      wt => layers(n) % w_transposed
      b  => layers(n) % b
#ifdef USE_TIMING
    ret =  gptlstart('last_sgemm')
#endif  
      !$acc host_data use_device(wt, a, output)
      call sgemm("N","N", ngpt, nbatch, neurons, 1.0, wt, ngpt, a, neurons, 0.0, output, ngpt)
      !$acc end host_data
#ifdef USE_TIMING
    ret =  gptlstop('last_sgemm')
#endif  
      
      !$acc parallel loop collapse(2) default(present)
      do j = 1, nbatch
        do i = 1, ngpt
          ! Add bias to obtain model output (linear layer, no activation) 
          output(i, j) = output(i, j) + b(i)
          ! Postprocess 1: reverse standard scaling and square root scaling
          ! output(i, j) = (ysigma*output(i, j) + ymeans(i))**8
          output(i, j) = (ystd(i)*output(i, j) + ymeans(i))**8

          ! Postprocess 2: scale with number of dry air molecules to obtain optical depth
          output(i, j) = output(i, j) * coldry(j)

          ! One-line solution
          ! output(i, j) = ((ystd(i)* (output(i, j) + b(i)) + ymeans(i))**8) * coldry(j)

          if (present(output2)) then 
            ! output = tau_rayleigh => ssa
            ! output2 = tau_abs => tau_tot
            output2(i,j) = output2(i,j) + output(i,j)
            output(i,j) =  output(i,j) / output2(i,j)
          end if
        end do
      end do

    end associate
    !$acc end data
                                              
  end subroutine

  subroutine output_sgemm_pfrac(self, nx, ny, nbatch, x, output)
    ! Like output_sgemm_tau but for predicting Planck fraction, which has different post-processing
    class(rrtmgp_network_type),              intent(in), target  :: self ! a neural network model
    integer, intent(in)                           :: nx, ny, nbatch
    real(sp), dimension(nx, nbatch), intent(in)  :: x            ! Model input
    real(sp), dimension(ny, nbatch), intent(out) :: output       ! Model output
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
                                          target  :: a1, a2       ! Temporary output/input between layers, of shape (neurons, nbatch)
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next    ! The input to a layer is the output of the previous layer. To avoid memory
                                                                  ! movement, we can use pointers and just switch them around after each layer
    real(sp), dimension(:,:), contiguous, pointer :: wt           ! Weights
    real(sp), dimension(:),   contiguous, pointer :: b            ! BIases
    integer :: n, j, neurons, nlayers, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc data create(a1, a2)
    associate(layers=>self%layers)    ! so it's easier to read

      ! FIRST HIDDEN LAYER (input layer)
      wt => layers(1) % w_transposed  ! Set the weights to the weights of the first layer
      a  => a1                        
      b  => layers(1) % b            

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)  ! uses GPU version if USE_OPENACC=1
      !$acc end host_data

      call layers(1) % bias_and_activation(a, b)

      ! INTERMEDIATE LAYERS
      a_next => a2

      do n = 2, nlayers-1

        wt => layers(n) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers
        if(mod(n,2) .EQ. 0) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      wt => layers(n) % w_transposed
      b  => layers(n) % b
      !$acc host_data use_device(wt, a, output)
      call sgemm("N","N", ny, nbatch, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      !!$acc parallel loop gang default(present)
      ! do j = 1, nbatch
      !   !$acc loop vector
      !   do i = 1, ny
      !     output(i, j) = output(i, j ) + b(i)
      !     output(i, j) = max(0.0_sp, output(i, j)) !RELU activation
      !     output(i, j) = output(i, j)*output(i, j)
      !   end do
      ! end do
      call layers(n) % bias_and_activation(output, b)
      !$acc kernels
      output = output*output
      !$acc end kernels
      end associate

    !$acc end data

  end subroutine

  subroutine output_sgemm_lw(self, nx, ngpt, nbatch, x, output)
    ! Like output_sgemm_flat but for computing OPTICAL DEPTH, inlining the post-processing
    ! Additional inputs: number of dry air molecules (coldry) and the mean and standard deviation
    ! used for normalization (ymeans, ysigma)
    use, intrinsic :: iso_c_binding
    class(rrtmgp_network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ngpt, nbatch
    real(sp), dimension(nx, nbatch),  intent(in)  :: x      ! (features, nbatch)
    real(sp), dimension(ngpt, nbatch),intent(out) :: output !
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
                                          target  :: a1, a2  
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer                       :: n, j, neurons, nlayers, i
    real(sp), dimension(ngpt)                     :: ymeans,ystd ! standard-scaling coefficients

    if (.not. allocated(self%coeffs_output_mean)) &
      stop "output_sgemm_tau: NN output scaling coefficients missing"
    ymeans = self%coeffs_output_mean
    ystd   = self%coeffs_output_std
    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc data create(a1, a2) copyin(ymeans,ystd)
    associate(layers=>self%layers)
      
      ! Assign pointers to layer weights, biases and input-output arrays
      wt      => layers(1) % w_transposed
      a       => a1
      a_next  => a2
      b       => layers(1) % b
#ifdef USE_TIMING
    ret =  gptlstart('first_sgemm')
#endif  
      ! 1. Multiply inputs with the weights of the first layer
      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data
#ifdef USE_TIMING
    ret =  gptlstop('first_sgemm')
#endif  
      ! 2. Add biases and activation
     call layers(1) % bias_and_activation(a, b)
      
      ! 3. Repeat steps 1-2 until final layer reached
      do n = 2, nlayers-1

        wt => layers(n) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers, the previous output is the next input
        if(mod(n,2) .EQ. 0) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      wt => layers(n) % w_transposed
      b  => layers(n) % b
#ifdef USE_TIMING
    ret =  gptlstart('last_sgemm')
#endif   
      !$acc host_data use_device(wt, a, output)
      call sgemm("N","N", ngpt, nbatch, neurons, 1.0, wt, ngpt, a, neurons, 0.0, output, ngpt)
      !$acc end host_data
#ifdef USE_TIMING
    ret =  gptlstop('last_sgemm')
#endif 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, nbatch
        do i = 1, ngpt
          ! Add bias to obtain model output (linear layer, no activation) 
          output(i, j) = output(i, j) + b(i)
        end do
      end do

    end associate
    !$acc end data
                                              
  end subroutine

  ! subroutine output_sgemm_tau_sgemmbatched(self, nx, ny, nbatch, x, coldry, ymeans, ysigma, output)
  !   use, intrinsic :: iso_c_binding
  !   use cudafor
  !   use cublas
  !   integer, parameter :: blocksize = 128
  !   class(rrtmgp_network_type),              intent(in), target  :: self
  !   integer, intent(in)                           :: nx, ny, nbatch
  !   real(sp), dimension(nx, nbatch), intent(in)  :: x      ! (features, nbatch)
  !   real(sp), dimension(nbatch),     intent(in)  :: coldry 
  !   real(sp), dimension(ny),          intent(in)  :: ymeans
  !   real(sp),                         intent(in)  :: ysigma
  !   real(sp), dimension(ny, nbatch), intent(out) :: output ! (outputs, nbatch) 
  !   real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
  !                                         target  :: a1, a2  
  !   real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
  !   real(sp), dimension(:,:), contiguous, pointer :: wt
  !   real(sp), dimension(:),   contiguous, pointer :: b
  !   integer      :: n, j, neurons, nlayers, i,nb, stat
  !   real(sp), dimension(:,:,:), contiguous, pointer :: output_b, a_b
  !   real(sp), allocatable :: wt_b(:,:)
  !   type(c_devptr), dimension(nbatch/blocksize) :: devptr_A, devptr_B, devptr_C
  !   type(cublasHandle) :: handle
  !   real(sp) :: alpha, beta

  !   neurons = size(self % layers(1) % w_transposed, 1)
  !   nlayers = size(self % layers)

  !   wt_b = self % layers(nlayers) % w_transposed

  !   !$acc enter data create(a1, a2) copyin(ymeans)
  !   associate(layers=>self%layers)
      
  !     wt => layers(1) % w_transposed
  !     a  => a1
  !     b  => layers(2) % b

  !     !$acc host_data use_device(wt, x, a)
  !     call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
  !     !$acc end host_data

  !     !$acc parallel loop gang vector collapse(2) default(present)
  !     do j = 1, nbatch
  !       do i = 1, neurons
  !         a(i, j) = a(i, j ) + b(i)
  !         call activation_softsign(a(i, j))
  !       end do
  !     end do

  !     ! INTERMEDIATE LAYERS
  !     a_next => a2
  !     do n = 3, nlayers-1

  !       wt => layers(n-1) % w_transposed
  !       b => layers(n) % b

  !       !$acc host_data use_device(wt, a, a_next)
  !       call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
  !       !$acc end host_data

  !       !$acc parallel loop gang vector collapse(2) default(present)
  !       do j = 1, nbatch
  !         do i = 1 , neurons 
  !           a_next(i, j) = a_next(i, j ) + b(i)
  !           call activation_softsign(a_next(i, j))
  !         end do
  !       end do 

  !       ! Swap pointers, the previous output is the next input
  !       if(mod(n,2) .EQ. 1) then
  !         a       => a2
  !         a_next  => a1  
  !       else
  !         a       => a1
  !         a_next  => a2
  !       end if

  !     end do

  !     wt => layers(n-1) % w_transposed
  !     b  => layers(n) % b


    
  !     nb = nbatch/blocksize
  !     call C_F_POINTER (C_LOC(output), output_b, [ny,blocksize,nb])
  !     call C_F_POINTER (C_LOC(a2), a_b, [neurons,blocksize,nb])
  !     !call C_F_POINTER (C_LOC(layers(n-1) % w_transposed), wt_b, [ny,neurons,nb])

  !     stat = cublasCreate(handle)

  !     !$acc data create(devptr_A, devptr_B, devptr_C) copyin(wt_b)


  !     !$acc host_data use_device(wt_b, a_b, output_b)
  !     ! Set device pointers to device arrays
  !     do i = 1, nb
  !       devptr_A(i) = c_devloc(wt_b(1,1))
  !       devptr_B(i) = c_devloc(a_b(1,1,i))
  !       devptr_C(i) = c_devloc(output_b(1,1,i))
  !     enddo
  !     !$acc end host_data

  !     alpha = 1.0
  !     beta = 0.0

  !     !$acc update device(devptr_A, devptr_B, devptr_C)

  !     stat = cudaDeviceSynchronize()
            

  !     !$acc host_data use_device(devptr_A, devptr_B, devptr_C)
  !   ! batched DGEMM: C = alpha*A*B + beta*C
  !     stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
  !           ny, blocksize, neurons, &
  !           alpha,         &
  !           devptr_A, ny, &
  !           devptr_B, neurons, &
  !           beta,          &
  !           devptr_C, ny, &
  !           nb)
  !     !$acc end host_data
  !     !$acc end data

  !     ! !$acc host_data use_device(wt, a, output)
  !     ! call sgemm("N","N", ny, nbatch, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
  !     ! !$acc end host_data

  !     n = nlayers

  !     !$acc parallel loop gang vector collapse(2) default(present)
  !     do j = 1, nbatch
  !       !$OMP SIMD
  !       do i = 1, ny
  !         ! Compute outputs and scale them to obtain molecular absorption 
  !         ! output(i, j) = (ysigma*(output(i, j) + b(i)) + ymeans_lw_tau(i))**8

  !         ! Scale with number of dry air molecules to obtain optical depth
  !         ! output(i, j) =  output(i, j) * coldry(j)

  !         ! One-line solution
  !         output(i, j) = ((ysigma* (output(i, j) + b(i)) + ymeans(i))**8) * coldry(j)

  !       end do
  !     end do

  !   end associate
  !   !$acc exit data detach(a,a_next) delete(a1, a2, ymeans)

                                              
  ! end subroutine output_sgemm_tau_sgemmbatched

end module mod_network_rrtmgp
