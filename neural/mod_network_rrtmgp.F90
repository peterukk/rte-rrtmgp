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
  ! List of character for case conversion
character(len=26), parameter :: LOWER_CASE_CHARS = 'abcdefghijklmnopqrstuvwxyz'
character(len=26), parameter :: UPPER_CASE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  private

  public :: rrtmgp_network_type, output_sgemm_pfrac, output_sgemm_tau
  
  ! ! ---------------------------------------------------------------------------------------
  ! ! -------------------  Coefficients for scaling inputs and outputs ----------------------
  ! ! ---------------------------------------------------------------------------------------
  ! ! integer, parameter :: numinputs_all = 19
  ! ! ! Strings specifying all the RRTMGP gases which can be used by neural networks, these are matched to available gases
  ! character(32), dimension(19) :: nn_rrtmgp_inputs_all = [character(len=32)  :: 'tlay', 'play', 'h2o',   'o3', 'co2',  &
  ! 'n2o', 'ch4', 'cfc11', 'cfc12', 'co', 'ccl4', 'cfc22', 'hfc143a', 'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4', 'no2'] 

  type, extends(network_type), public :: rrtmgp_network_type

  ! ---------------------------------------------------------------------------------------
  ! -------------------  Coefficients for scaling inputs and outputs ----------------------
  ! ---------------------------------------------------------------------------------------
  ! integer, parameter :: numinputs_all = 19
  ! ! Strings specifying all the RRTMGP gases which can be used by neural networks, these are matched to available gases
  character(32), dimension(19) :: nn_rrtmgp_inputs_all = [character(len=32)  :: 'tlay', 'play', 'h2o',   'o3', 'co2',  &
  'n2o', 'ch4', 'cfc11', 'cfc12', 'co', 'ccl4', 'cfc22', 'hfc143a', 'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4', 'no2'] 
  real(sp), dimension(:), allocatable :: nn_coeffs_input_min, nn_coeffs_input_max
  character(32), dimension(:), allocatable :: nn_rrtmgp_inputs 
  ! ! Scaling coefficients for neural network inputs, used to scale values to between 0 and 1 (min-max-scaling)
  ! ! Furthermore, some inputs are power or log-scaled. The f irst two inputs are pressure and temperature, followed by gases.
  ! ! The inputs are:                                             tlay,   log(play), h2o**(1/4),o3**(1/4),co2, ... 
  ! real(sp), dimension(ninputs_all)   :: nn_input_minvals =  (/ 1.60E2, 5.15E-3,   1.01E-2,   4.36E-3,  1.41E-4, 0.00E0, 2.55E-8, 0.00E0, 0.00E0, &
  ! 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0 /)
  ! real(sp), dimension(ninputs_all)   :: nn_input_maxvals =  (/ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, 2.3000003E-3, 5.8135214E-7, & 
  ! 3.6000001E-6, 2.0000002E-9, 5.3385213E-10, 1.3127458E-6, 1.0316801E-10, 2.3845328E-10, &
  !  7.7914392E-10, 9.8880004E-10, 3.1067642E-11, 1.3642075E-11, 4.2330001E-10, 1.6702625E-10, 3.0E-10 /)
  ! ---------------------------------------------------------------------------------------

  real(sp), dimension(3) :: testnum = (/ 1.60E2, 5.15E-3,   1.01E-2   /)


  contains


    procedure, public, pass(self) :: load_netcdf
    procedure, public, pass(self) :: output_sgemm_pfrac, output_sgemm_tau  ! Inference kernels using BLAS and custom post-processing

  end type rrtmgp_network_type


contains

  subroutine load_netcdf(self, filename)
    ! Loads the network from file.
    class(rrtmgp_network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer :: ncid, n, num_layers, nx, dim1, dim2
    integer, allocatable :: dims(:)
    character(len=20) :: activation_type, varname_weight, varname_bias
    character(len=5) :: charN
    real(sp), dimension(:,:), allocatable :: tmpvar
    real(sp), dimension(:), allocatable :: tmpbias
    character(len=32), dimension(:), allocatable :: activation_strings

    if(nf90_open(trim(filename), NF90_NOWRITE, ncid) /= NF90_NOERR) &
      call stop_on_err("mod_network_rrtmgp:load_netcdf: can't find file " // trim(filename))

    num_layers = get_dim_size(ncid, 'nn_layers')
    ! print *, "num nn layers", num_layers

    allocate(dims(num_layers+1))

    nx = get_dim_size(ncid, 'nn_dim_input')
    dims(1) = nx
    dims(2:num_layers+1) = read_field(ncid, "nn_dimsize",  num_layers)

    call self % init(dims)
      
    !$acc enter data copyin(self) 
    !$acc enter data copyin(self % dims)  
    !$acc enter data copyin(self % layers)
    do n = 1, num_layers
      dim1 = dims(n)
      dim2 = dims(n+1)
      ! create string containing the variable name
      write(charN, '(I5)') n
      varname_weight = 'nn_weights_'// trim(adjustl(charN)) // ''
      varname_bias   = 'nn_bias_'// trim(adjustl(charN)) // ''
      
      self % layers(n) % w =  transpose(read_field(ncid, varname_weight,  dim2, dim1))
      self % layers(n) % b =  read_field(ncid, varname_bias,  dim2)
      !$acc enter data copyin(self % layers(n) % w, self % layers(n) % b) async
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )  
      !$acc enter data copyin(self % layers(n) % w_transposed) async   

      ! print *, "set weight to", self % layers(n) % w
      ! print *, "set bias to", self % layers(n) % w

    end do

    activation_strings = read_char_vec(ncid, 'nn_activation_char', num_layers)

    do n = 1, num_layers
      call self % layers(n) % set_activation(adjustl(activation_strings(n)))
    end do    
  end subroutine load_netcdf

  pure function string_in_array(string, array)
    character(len=*),               intent(in) :: string
    character(len=*), dimension(:), intent(in) :: array
    logical                                    :: string_in_array

    integer :: i
    character(len=len_trim(string)) :: lc_string

    string_in_array = .false.
    lc_string = lower_case(trim(string))
    do i = 1, size(array)
      if(lc_string == lower_case(trim(array(i)))) then
        string_in_array = .true.
        exit
      end if
    end do
  end function string_in_array
  pure function lower_case( input_string ) result( output_string )
    character(len=*), intent(in) :: input_string
    character(len=len(input_string)) :: output_string
    integer :: i, n

    ! Copy input string
    output_string = input_string

    ! Convert case character by character
    do i = 1, len(output_string)
      n = index(UPPER_CASE_CHARS, output_string(i:i))
      if ( n /= 0 ) output_string(i:i) = LOWER_CASE_CHARS(n:n)
    end do
  end function
  subroutine output_sgemm_tau(self, nx, ngpt, nbatch, x, coldry, ymeans, ysigma, output)
    ! Like output_sgemm_flat but for computing OPTICAL DEPTH, inlining the post-processing
    ! Additional inputs: number of dry air molecules (coldry) and the mean and standard deviation
    ! used for normalization (ymeans, ysigma)
    use, intrinsic :: iso_c_binding
    class(rrtmgp_network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ngpt, nbatch
    real(sp), dimension(nx, nbatch),  intent(in)  :: x      ! (features, nbatch)
    real(sp), dimension(nbatch),      intent(in)  :: coldry ! number of dry air molecules
    real(sp), dimension(ngpt),        intent(in)  :: ymeans ! standard-scaling coefficient 
    real(sp),                         intent(in)  :: ysigma ! standard-scaling coefficient
    real(sp), dimension(ngpt, nbatch),intent(out) :: output ! absorption cross-section g-point vectors
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
                                          target  :: a1, a2  
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer                       :: n, j, neurons, nlayers, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc enter data create(a1, a2) copyin(ymeans)
    associate(layers=>self%layers)
      
      ! Assign pointers to layer weights, biases and input-output arrays
      wt      => layers(1) % w_transposed
      a       => a1
      a_next  => a2
      b       => layers(1) % b
      
      ! 1. Multiply inputs with the weights of the first layer
      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

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

      !$acc host_data use_device(wt, a, output)
      call sgemm("N","N", ngpt, nbatch, neurons, 1.0, wt, ngpt, a, neurons, 0.0, output, ngpt)
      !$acc end host_data

      !$acc parallel loop gang default(present)
      do j = 1, nbatch
        !$OMP SIMD
        !$acc loop vector
        do i = 1, ngpt
          ! Add bias to obtain model output (linear layer, no activation) 
          output(i, j) = output(i, j) + b(i)
          ! Postprocess 1: reverse standard scaling and square root scaling
          output(i, j) = (ysigma*output(i, j) + ymeans(i))**8
          ! Postprocess 2: scale with number of dry air molecules to obtain optical depth
          output(i, j) = output(i, j) * coldry(j)

          ! One-line solution
          ! output(i, j) = ((ysigma* (output(i, j) + b(i)) + ymeans(i))**8) * coldry(j)
        end do
      end do

    end associate
    !$acc exit data detach(a,a_next) delete(a1, a2, ymeans)
                                              
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

    !$acc enter data create(a1, a2)
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

      !$acc parallel loop gang default(present)
      do j = 1, nbatch
        !$acc loop vector
        do i = 1, ny
          output(i, j) = output(i, j ) + b(i)
          output(i, j) = max(0.0_sp, output(i, j)) !RELU activation
          output(i, j) = output(i, j)*output(i, j)
        end do
      end do
      ! call layers(n) % bias_and_activation(output, b)
      ! output = output*output

      end associate

    !$acc exit data detach(a,a_next) delete(a1, a2)

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
