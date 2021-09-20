module mod_network

  use mo_rte_kind, only: sp
  use mod_layer, only: layer_type
   use mod_activation, only : softsign_mat_b
#ifdef USE_TIMING
  ! Timing library
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif

#ifdef USE_OPENACC
  use cublas 
  use openacc
#define sgemm cublassgemm
#endif

  implicit none
 
#ifdef USE_TIMING
  integer, private :: ret, i
#endif

  public! :: network_type

  type :: network_type

    type(layer_type), allocatable :: layers(:)
    integer,          allocatable :: dims(:)
  contains

    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: output_opt, output_opt_flatmodel       ! Vector input, matrix-vector product
    procedure, public, pass(self) :: output_sgemm_pfrac, output_sgemm_tau   ! Matrix-matrix using BLAS
    procedure, public, pass(self) :: output_sgemm_flat, output_sgemm_flat_byrows
    procedure, public, pass(self) :: save
    procedure, public, pass(self) :: set_activation

  end type network_type

  interface network_type
    module procedure net_constructor
  endinterface network_type

contains

  type(network_type) function net_constructor(dims, activation) result(net)
    ! network class constructor. Size of input array dims indicates the total
    ! number of layers (input + hidden + output), and the value of its elements
    ! corresponds the size of each layer.
    integer, intent(in) :: dims(:)
    character(len=*), intent(in), optional :: activation
    call net % init(dims)
    if (present(activation)) then
      call net % set_activation(activation)
    else
      call net % set_activation('sigmoid')
    end if
    ! call net % sync(1)
  end function net_constructor


  subroutine init(self, dims)
    ! Allocates and initializes the layers with given dimensions dims.
    class(network_type), intent(in out) :: self
    integer, intent(in) :: dims(:)
    integer :: n
    allocate(self%dims(size(dims)))
    self % dims = dims
    if (.not. allocated(self % layers)) allocate(self % layers(size(dims)))
    do n = 1, size(dims) - 1
      self % layers(n) = layer_type(dims(n), dims(n+1))
    end do
    self % layers(n) = layer_type(dims(n), 1)
    self % layers(1) % b = 0.0_sp
    self % layers(size(dims)) % w = 0.0_sp
    self % layers(size(dims)) % w_transposed = 0.0_sp
  end subroutine init

  subroutine load(self, filename)
    ! Loads the network from file.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    character(len=20) :: activation_type

    integer :: fileunit, n, num_layers
    integer, allocatable :: dims(:)
    
    open(newunit=fileunit, file=filename, status='old', action='read')
    read(fileunit, fmt=*) num_layers
    allocate(dims(num_layers))
    read(fileunit, fmt=*) dims
    call self % init(dims)
   !$acc enter data copyin(self) 
   !$acc enter data copyin(self % dims)  
   !$acc enter data copyin(self % layers)
    do n = 2, size(self % dims)
      read(fileunit, fmt=*) self % layers(n) % b
      !$acc enter data copyin(self % layers(n) % b) async
    end do
    !$acc wait
    
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) self % layers(n) % w
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )   
     !$acc enter data copyin(self % layers(n) % w_transposed) async   
    end do
    
    call self % layers(1) % set_activation('linear')
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) activation_type
      call self % layers(n+1) % set_activation(activation_type)
    end do    

    close(fileunit)
    !$acc wait
  end subroutine load


  pure subroutine output_opt(self, x, output)
    class(network_type),    intent(in)  :: self
    real(sp), dimension(:), intent(in)  :: x
    real(sp), dimension(:), intent(out) :: output
    ! Local variables
    real(sp), allocatable   :: a(:)
    integer,  dimension(2)  :: matsize
    integer                 :: n

    associate(layers => self % layers)
      matsize = shape(layers(1) % w_transposed)
      a = matvecmul(layers(1) % w_transposed, x, matsize(1), matsize(2)) + layers(2) % b
      ! sigmoid activation: using an "inout" subroutine to avoid array copy 
      call layers(2) % activation(a)
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        matsize = shape(layers(n-1) % w_transposed)
        a = matvecmul(layers(n-1) % w_transposed, a, matsize(1), matsize(2)) + layers(n) % b
        call layers(n) % activation(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      matsize = shape(layers(n-1) % w_transposed)
      output = (matvecmul(layers(n-1) % w_transposed, a, matsize(1), matsize(2)) + layers(n) % b)
      call layers(n) % activation(output)
    end associate
    
  end subroutine

  pure subroutine linear(x)
    real(sp), intent(inout) :: x(:)
    x = x
  end subroutine

  pure subroutine output_opt_flatmodel(self, x, output)
    ! Use forward propagation to compute the output of the network.
    ! For computational efficiency, following changes are implemented:
    ! 1) Outputs are allocated outside of function, 
    ! 2) use of explicit-shape intermediate array that assumes the number of neurons are the same for all hidden layers,
    ! 3) activation functions are replaced with a subroutine that modifies the arguments (sigmoid), activation from final layer removed (linear activation=redundant 1:1 copy)
    ! 4) matmul replaced by custom function which is faster than matmul for matrix-vector multiplication
    ! 5) weights have been pre-transposed in the load routine.
    ! This procedure is much faster than the original when using gfortran -O3 -march=native or ifort -O3.
    ! For lower optimization levels the custom function (4) may be SLOWER
    class(network_type),    intent(in)  :: self
    real(sp), dimension(:), intent(in)  :: x
    real(sp), dimension(:), intent(out) :: output
    ! Local variables
    ! The signal/tensor passing through the network
    real(sp), dimension(size(self % layers(1) % w_transposed,1))  :: a 
    integer :: n, neurons

    neurons = size(self % layers(1) % w_transposed, 1)

    associate(layers => self % layers)
      a = matvecmul(layers(1) % w_transposed, x, neurons, size(x)) + layers(2) % b
      call layers(2) % activation(a)
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        a = matvecmul(layers(n-1) % w_transposed, a, neurons, neurons) + layers(n) % b
        call layers(n) % activation(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      output = (matvecmul(layers(n-1) % w_transposed, a, size(output), neurons) + layers(n) % b)
      call layers(n) % activation(output)
    end associate
  end subroutine

  subroutine output_sgemm_flat(self, nx, ny, nbatch, x, output)
    ! Optimized batched inference function for a multi-output (regression) neural network using BLAS/cuBLAS
    ! Takes a 2D input array where the columns are the input vectors, outer dimension
    ! is the number of samples (nbatch, block size) which could be ncol*nlay
    ! Always in single-precision (sgemm) because more precision is redundant (nets trained in sp)
    ! this version assumes a "flat" network structure with regard to hidden neurons (nneur1=nneur2)    
    !
    !                             Layer Weights (T)     Layer Inputs          Layer Outputs
    ! Input (first hidden) layer    (nneur1 x nx)     * (nx x nbatch )      = (nneur1 x nbatch) 
    ! Other hidden layers           (nneur2 x nneur1) * (nneur1 x nbatch )  = (nneur2 x nbatch) 
    ! output layer                  (ny x nneur2)     * (nneur2 x nbatch )  = (ny x nbatch)  
    ! in GEMM terms:                  A (m x k)       *  B (k x N )         = C (m x N) 
    ! T = weights have been pre-transposed
    !
    class(network_type),              intent(in), target  :: self ! a neural network model
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

    ! print *, "neurons", neurons, "nbatch", nbatch, "nx", nx, "ny", ny

    !$acc enter data create(a1, a2)
    associate(layers=>self%layers)    ! so it's easier to read

      ! FIRST LAYER
      wt => layers(1) % w_transposed 
      a  => a1                        
      b  => layers(2) % b            

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)  ! uses GPU version if USE_OPENACC=1
      !$acc end host_data

      call layers(2) % bias_and_activation(a, b)

      ! INTERMEDIATE LAYERS
      a_next => a2

      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers
        if(mod(n,2) .EQ. 1) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do
      
      ! Output layer
      wt => layers(n-1) % w_transposed
      b  => layers(n) % b

      !$acc host_data use_device(wt, a, output)
      call sgemm("N","N", ny, nbatch, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      call layers(n) % bias_and_activation(output, b)

      end associate
    !$acc exit data detach(a,a_next) delete(a1, a2)

  end subroutine

  subroutine output_sgemm_tau(self, nx, ngpt, nbatch, x, coldry, ymeans, ysigma, output)
    ! Like output_sgemm_flat but for computing OPTICAL DEPTH, inlining the post-processing
    ! Additional inputs: number of dry air molecules (coldry) and the mean and standard deviation
    ! used for normalization (ymeans, ysigma)
    use, intrinsic :: iso_c_binding
    class(network_type),              intent(in), target  :: self
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
      b       => layers(2) % b
      
      ! 1. Multiply inputs with the weights of the first layer

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

      ! 2. Add biases and activation
     call layers(2) % bias_and_activation(a, b)
      
      ! 3. Repeat steps 1-2 until final layer reached
      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers, the previous output is the next input
        if(mod(n,2) .EQ. 1) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      wt => layers(n-1) % w_transposed
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
    class(network_type),              intent(in), target  :: self ! a neural network model
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

      ! FIRST LAYER
      wt => layers(1) % w_transposed  ! Set the weights to the weights of the first layer
      a  => a1                        
      b  => layers(2) % b            

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)  ! uses GPU version if USE_OPENACC=1
      !$acc end host_data

      call layers(2) % bias_and_activation(a, b)

      ! INTERMEDIATE LAYERS
      a_next => a2

      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        call layers(n) % bias_and_activation(a_next, b)

        ! Swap pointers
        if(mod(n,2) .EQ. 1) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      wt => layers(n-1) % w_transposed
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

  subroutine output_sgemm_flat_byrows(self, nx, ny, nbatch, x, output)
    ! Generic inference function using BLAS/cuBLAS for batched prediction (many samples at a time)
    ! In this version, the dimensions of inputs and outputs are reversed so that the individual
    ! NN input/output vectors are strided across the outer dimension
    ! This may be faster for very small networks
    !                               layer input           weights             layer output
    ! input (first hidden) layer    (nbatch x nx)     * (nx x nneur1)       = (nbatch x nneur1)
    ! other hidden layers           (nbatch x nneur1) * (nneur1 x nneur2)   = (nbatch * nneur2)  
    ! output layer                  (nbatch x nneur2) * (nneur2 x ny )      = (nbatch x ny)
    ! in GEMM terms:                  A (m x k)       *  B (k x N )         = C (m x N) 
    ! T = transposed weights
    class(network_type),              intent(in), target  :: self ! a neural network model
    integer, intent(in)                           :: nx, ny, nbatch
    real(sp), dimension(nbatch, nx), intent(in)   :: x            ! Model input
    real(sp), dimension(nbatch, ny), intent(out)  :: output       ! Model output
    real(sp), dimension(nbatch, size(self % layers(1) % w_transposed, 1)), &
                                      target      :: a1, a2       ! Temporary output/input between layers
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next    ! The input to a layer is the output of the previous layer. To avoid memory
                                                                  ! movement, we can use pointers and just switch them around after each layer
    real(sp), dimension(:,:), contiguous, pointer :: w            ! Weights
    real(sp), dimension(:),   contiguous, pointer :: b            ! BIases
    integer :: n, j, neurons, nlayers, i

    neurons = size(self % layers(1) % w, 2)
    nlayers = size(self % layers)

    !$acc enter data create(a1, a2)
    associate(layers=>self%layers)    ! so it's easier to read

      ! FIRST LAYER
      w   => layers(1) % w  ! Set the weights to the weights of the first layer
      a   => a1                        
      b   => layers(2) % b            
#ifdef USE_TIMING
  ret =  gptlstart('first_sgemm')
#endif
      !$acc host_data use_device(w, x, a)
      call sgemm('N','N', nbatch, neurons, nx, 1.0, x, nbatch, w, nx, 0.0, a, nbatch)
      !$acc end host_data
#ifdef USE_TIMING
  ret =  gptlstop('first_sgemm')
#endif
#ifdef USE_TIMING
    ret =  gptlstart('bias_and_activ1')
#endif
      call layers(2) % bias_and_activation(a, b)
#ifdef USE_TIMING
    ret =  gptlstop('bias_and_activ1')
#endif
      ! INTERMEDIATE LAYERS
      a_next => a2
      
      do n = 3, nlayers-1

        w => layers(n-1) % w
        b => layers(n) % b
#ifdef USE_TIMING
  ret =  gptlstart('middle_sgemm')
#endif
        !$acc host_data use_device(w, a, a_next)
        call sgemm("N","N", nbatch, neurons, neurons, 1.0, a, nbatch, w, neurons, 0.0, a_next, nbatch)
        !$acc end host_data
#ifdef USE_TIMING
  ret =  gptlstop('middle_sgemm')
#endif
#ifdef USE_TIMING
    ret =  gptlstart('bias_and_activ_mid')
#endif
        call layers(n) % bias_and_activation(a_next, b)
#ifdef USE_TIMING
    ret =  gptlstop('bias_and_activ_mid')
#endif
        ! Swap pointers
        if(mod(n,2) .EQ. 1) then
          a       => a2
          a_next  => a1  
        else
          a       => a1
          a_next  => a2
        end if

      end do

      w => layers(n-1) % w
      b  => layers(n) % b
#ifdef USE_TIMING
  ret =  gptlstart('last_sgemm')
#endif
      !$acc host_data use_device(w, a, output)
      call sgemm("N","N", nbatch, ny, neurons, 1.0, a, nbatch, w, neurons, 0.0, output, nbatch)
      !$acc end host_data
#ifdef USE_TIMING
  ret =  gptlstop('last_sgemm')
#endif
#ifdef USE_TIMING
    ret =  gptlstart('bias_and_activ_fin')
#endif
      call layers(n) % bias_and_activation(output, b)
#ifdef USE_TIMING
    ret =  gptlstop('bias_and_activ_fin')
#endif
      end associate
    !$acc exit data detach(a,a_next) delete(a1, a2)

  end subroutine


elemental subroutine reluu(x) 
!$acc routine seq
  !! REctified Linear Unit (RELU) activation subroutine.
  real(sp), intent(inout) :: x
  x = max(0.0_sp, x)
end subroutine reluu

  
elemental subroutine activation_softsign(x)
!$acc routine seq
  real(sp), intent(inout) :: x
  x = x / (abs(x) + 1)
end subroutine



pure function matvecmul(matA,vecB,nrow,ncol)
    implicit none
    integer, intent(in) :: nrow,ncol
    real(sp), intent(in) :: matA(nrow,ncol)
    real(sp), intent(in) :: vecB(ncol)
    integer :: i,j
    real(sp) :: matvecmul(nrow)

    ! each (row,here ncol) element in b (length e.g. 256) is obtained by :
    ! loop through the different elements in vecB (length 50), multiply by the corresponding
    ! column (still 50) element in matA for that particular row (outer loop), add the 50 values together

    matvecmul = 0.0_sp
    do j=1,ncol !   y            X          alpha
        matvecmul = matvecmul + matA(:,j) * vecB(j) !length 256. this is for one column, and then the columns need to be added together ( b = b + ..)
    enddo

end function matvecmul

subroutine gemm_matvecmul(m, k, n, matA, matB, matC)
    implicit none
    integer, intent(in) :: m,k,n
    real(sp), intent(in) :: matA(m,k), matB(k,n)
    real(sp), intent(out) :: matC(m,n)
    integer :: i,j, jj

    ! m = 4
    ! k = 8

    do jj = 1,n ! n > 1000

      matC(:,jj) = 0.0_sp
      do j=1,k 
        matC(:,jj) = matC(:,jj) + matA(:,j) * matB(j,jj) 
        ! call saxpy(m,matB(j,jj), matA(:,j), 1, matC(:,jj), 1)
      enddo


    end do

end subroutine gemm_matvecmul

pure subroutine forward_pass(nrow,ncol,nbatch, matA,x,b, a)
!                           nx,  nneur,nbatch,weights,x,b,a
  implicit none
  integer, intent(in)   :: nrow  ! Nneur
  integer, intent(in)   :: ncol  ! Nx
  integer, intent(in)   :: nbatch
  real(sp),intent(in)  :: matA(nrow,ncol) ! weights (Nneur x Nx) = layers(1) % w_transposed
  real(sp), intent(in)  :: x(ncol,nbatch) ! (nx, nbatch)
  real(sp), intent(in)  :: b(nrow)
  real(sp), intent(out) :: a(nrow,nbatch) ! (nneur, nbatch)
  real(sp):: scal
  integer :: i,j,k

  !$acc parallel loop gang worker default(present) 
  do k  = 1, nbatch
    !$acc loop vector
    do i = 1, nrow
      scal = 0.0_sp
      do j=1,ncol 
        scal = scal + matA(i,j) * x(j,k) 
      end do 
      a(i,k) = scal + b(i)
      call activation_softsign(a(i, k))
    end do  
  end do

end subroutine forward_pass

  subroutine save(self, filename)
    ! Saves the network to a file.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer :: fileunit, n
    open(newunit=fileunit, file=filename)
    write(fileunit, fmt=*) size(self % dims)
    write(fileunit, fmt=*) self % dims
    do n = 2, size(self % dims)
      write(fileunit, fmt=*) self % layers(n) % b
    end do
    do n = 1, size(self % dims) - 1
      write(fileunit, fmt=*) self % layers(n) % w
    end do
    close(fileunit)
  end subroutine save

  pure subroutine set_activation(self, activation)
    ! A thin wrapper around layer % set_activation().
    ! This method can be used to set an activation function
    ! for all layers at once.
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: activation
    integer :: n
    do concurrent (n = 1:size(self % layers))
      call self % layers(n) % set_activation(activation)
    end do
  end subroutine set_activation


  ! subroutine output_sgemm_tau_sgemmbatched(self, nx, ny, nbatch, x, coldry, ymeans, ysigma, output)
  !   use, intrinsic :: iso_c_binding
  !   use cudafor
  !   use cublas
  !   integer, parameter :: blocksize = 128
  !   class(network_type),              intent(in), target  :: self
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

end module mod_network
