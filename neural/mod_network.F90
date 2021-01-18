module mod_network

  use mo_rte_kind, only: sp
  use mod_layer, only: layer_type
#ifdef USE_TIMING
  ! Timing library
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif
#ifdef USE_OPENACC
  use cublas 
  use openacc
#endif

  implicit none
 
#ifdef USE_TIMING
  integer, private :: ret, i
#endif
  !private
  public! :: network_type

  type :: network_type

    type(layer_type), allocatable :: layers(:)
    integer, allocatable :: dims(:)
  contains

    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: output_opt, output_opt_flatmodel       ! Vector input, matrix-vector product
    procedure, public, pass(self) :: output_matmul_flatmodel                ! Matrix input, matrix-matrix product
    procedure, public, pass(self) :: output_sgemm_pfrac, output_sgemm_tau   ! Matrix-matrix but using BLAS
    procedure, public, pass(self) :: save
    procedure, public, pass(self) :: set_activation
    !procedure, public, pass(self) :: sync

  end type network_type

#ifdef USE_OPENACC
#define sgemm cublassgemm
#endif

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
      !$acc enter data copyin(self % layers(n) % b)
    end do
    
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) self % layers(n) % w
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )   
     !$acc enter data copyin(self % layers(n) % w_transposed)    
    end do
    
    call self % layers(1) % set_activation('linear')
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) activation_type
      call self % layers(n+1) % set_activation(activation_type)
    end do    

    close(fileunit)
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

  subroutine output_matmul_flatmodel(self, nx, ny, nsample, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network
    class(network_type),    intent(in)          :: self
    integer,                intent(in)          :: nx,ny,nsample
    real(sp), dimension(nx,  nsample), &
                            intent(in)          :: x      ! (features, nsample)
    real(sp), dimension(ny, nsample), &
                            intent(out)         :: output ! (outputs, nsample)
    ! Local variables
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample)  :: a
    integer :: n, neurons, isample

    neurons = size(self % layers(1) % w_transposed, 1)

    associate(layers => self % layers)
      a = matmul(layers(1) % w_transposed, x) + spread( layers(2) % b, 2, nsample )
      call layers(2) % activation_m(a)

      do n = 3, size(layers)-1
        a = matmul(layers(n-1) % w_transposed, a) + spread( layers(n) % b, 2, nsample )
        call layers(n) % activation_m(a)
      end do

      output = matmul(layers(n-1) % w_transposed, a)  + spread( layers(n) % b, 2, nsample )
      call layers(n) % activation_m(output)
    end associate
  end subroutine

  subroutine output_sgemm_tau(self, nx, ny, nsample, x, coldry, ymeans, ysigma, output)
    ! Optimized inference function for optical depth, using BLAS/cuBLAS and includes post-processing of outputs.
    ! This routine takes a 2D input data array for batched predictions with a feed-forward network.
    ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
    ! always in single-precision (sgemm) because why waste precision on NNs which don't need it 
    !
    !                                   Layer Weights            Layer Inputs                Layer Outputs
    ! First layer :                      (Nneurons x Nx)       * (Nx x Nsample )          = (Nneurons x Nsample) 
    ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons x Nsample )    = (Nneurons x Nsample) 
    ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons x Nsample )    = (Ngpoints x Nsample)  
    ! in GEMM terms:                         A                 *         B                = C
    !                                     (m x k)              *      (k * N )            = (m  * N)  
    use, intrinsic :: iso_c_binding
    class(network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ny, nsample
    real(sp), dimension(nx, nsample), intent(in)  :: x      ! (features, nsample)
    real(sp), dimension(nsample),     intent(in)  :: coldry 
    real(sp), dimension(ny),          intent(in)  :: ymeans
    real(sp),                         intent(in)  :: ysigma
    real(sp), dimension(ny, nsample), intent(out) :: output ! (outputs, nsample) 
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample), &
                                          target  :: a1, a2  
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer                       :: n, isample, neurons, nlayers, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc data create(a1, a2) copyin(ymeans) present(x, output, coldry)
    associate(layers=>self%layers)
      
      wt => layers(1) % w_transposed
      a  => a1
      b  => layers(2) % b

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nsample, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

      !$acc parallel loop gang vector collapse(2) present(a, b)
      do isample = 1, nsample
        do i = 1, neurons
          a(i, isample) = a(i, isample ) + b(i)
          call softsignn(a(i, isample))
        end do
      end do

      ! INTERMEDIATE LAYERS
      a_next => a2
      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        !$acc parallel loop gang vector collapse(2) present(a_next, b)
        do isample = 1, nsample
          do i = 1 , neurons 
            a_next(i, isample) = a_next(i, isample ) + b(i)
            call softsignn(a_next(i, isample))
          end do
        end do 

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
      call sgemm("N","N", ny, nsample, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      n = nlayers

      !$acc parallel loop collapse(2) present(b)
      do isample = 1, nsample
        !DIR$ SIMD
        !DIR$ VECTOR ALIGNED
        do i = 1, ny
          ! Compute outputs and scale them to obtain molecular absorption 
          ! output(i, isample) = (ysigma_lw_tau*(output(i, isample) + b(i)) + ymeans_lw_tau(i))**8

          ! Scale with number of dry air molecules to obtain optical depth
          ! output(i, isample) =  output(i, isample) * coldry(isample)

          ! One-line solution
          output(i, isample) = ((ysigma* (output(i, isample) + layers(n) % b(i)) + ymeans(i))**8) * coldry(isample)
        end do
      end do

    end associate
    !$acc end data

                                              
  end subroutine

  subroutine output_sgemm_pfrac(self, nx, ny, nsample, x, output)
    ! Optimized inference function for Planck fraction, using BLAS/cuBLAS for batched prediction (many samples at a time)
    ! Includes post-processing of outputs. The inputs have already been pre-processed
    class(network_type),              intent(in), target  :: self ! a neural network model
    integer, intent(in)                           :: nx, ny, nsample
    real(sp), dimension(nx, nsample), intent(in)  :: x            ! Model input
    real(sp), dimension(ny, nsample), intent(out) :: output       ! Model output
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample), &
                                          target  :: a1, a2       ! Temporary output/input between layers, of shape (neurons, nsample)
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next    ! The input to a layer is the output of the previous layer. To avoid memory
                                                                  ! movement, we can use pointers and just switch them around after each layer
    real(sp), dimension(:,:), contiguous, pointer :: wt           ! Weights
    real(sp), dimension(:),   contiguous, pointer :: b            ! BIases
    integer :: n, isample, neurons, nlayers, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)

    !$acc data create(a1, a2) present(x, output)

    associate(layers=>self%layers)    ! so it's easier to read

      ! FIRST LAYER
      
      wt => layers(1) % w_transposed  ! Set the weights to the weights of the first layer
      a  => a1                        
      b  => layers(2) % b            

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nsample, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)  ! uses GPU version if USE_OPENACC=1
      !$acc end host_data

      !$acc parallel loop gang vector collapse(2) present(a, b)
      do isample = 1, nsample
        do i = 1, neurons
          a(i, isample) = a(i, isample ) + b(i)  ! 1. add bias
          call softsignn(a(i, isample))          ! 2. use activation function, here the softsign
        end do
      end do

      ! INTERMEDIATE LAYERS
      a_next => a2

      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        !$acc parallel loop gang vector collapse(2) present(a_next, b)
        do isample = 1, nsample
          do i = 1 , neurons 
            a_next(i, isample) = a_next(i, isample ) + b(i)
            call softsignn(a_next(i, isample))
          end do
        end do 

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
      call sgemm("N","N", ny, nsample, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      n = nlayers

      !$acc parallel loop gang vector collapse(2) present(b)
      do isample = 1, nsample
        do i = 1, ny
          output(i, isample) = output(i, isample ) + b(i)
          output(i, isample) = max(0.0_sp, output(i, isample))
          output(i, isample) = output(i, isample)*output(i, isample)
        end do
      end do

      end associate
      !$acc end data 

  end subroutine


elemental subroutine reluu(x) 
!$acc routine seq
  !! REctified Linear Unit (RELU) activation subroutine.
  real(sp), intent(inout) :: x
  x = max(0.0_sp, x)
end subroutine reluu

  
elemental subroutine softsignn(x)
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
    do j=1,ncol !  50
        matvecmul = matvecmul + matA(:,j) * vecB(j) !length 256. this is for one column, and then the columns need to be added together ( b = b + ..)
    enddo

end function matvecmul

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

  ! subroutine sync(self, image)
  !   ! Broadcasts network weights and biases from
  !   ! specified image to all others.
  !   class(network_type), intent(in out) :: self
  !   integer, intent(in) :: image
  !   integer :: n
  !   if (num_images() == 1) return
  !   layers: do n = 1, size(self % dims)
  !     call co_broadcast(self % layers(n) % b, image)
  !     call co_broadcast(self % layers(n) % w, image)
  !   end do layers
  ! end subroutine sync

end module mod_network
