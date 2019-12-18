module mod_network

  use mo_rte_kind, only: wp
  use mod_layer, only: layer_type
  
  implicit none
  real(wp), dimension(256) :: output_scaler_means = (/ 0.67_wp, 0.78_wp, 0.84_wp, 0.9_wp, &
  0.96_wp, 1.04_wp, 1.15_wp, 1.3_wp, 1.53_wp, 1.74_wp, 1.82_wp, &
  1.92_wp, 2.03_wp, 2.18_wp, 2.4_wp, 2.6_wp, 0.45_wp, 0.5_wp, 0.56_wp, 0.63_wp, 0.68_wp, 0.74_wp, &
  0.83_wp, 0.94_wp, 1.15_wp, 1.34_wp, 1.43_wp, 1.54_wp, 1.69_wp, 1.89_wp, 2.19_wp, 2.46_wp, 0.41_wp, &
  0.43_wp, 0.47_wp, 0.52_wp, 0.57_wp, 0.63_wp, 0.7_wp, 0.79_wp, 0.93_wp, 1.07_wp, 1.13_wp, 1.2_wp,&
  1.28_wp, 1.41_wp, 1.53_wp, 1.59_wp, 0.76_wp, 0.85_wp, 0.9_wp, 0.95_wp, 1.01_wp, 1.09_wp, 1.2_wp,&
  1.36_wp, 1.62_wp, 1.85_wp, 1.97_wp, 2.09_wp, 2.22_wp, 2.36_wp, 2.47_wp, 2.53_wp, 0.36_wp, 0.39_wp,&
  0.43_wp, 0.49_wp, 0.56_wp, 0.62_wp, 0.69_wp, 0.77_wp, 0.91_wp, 1.04_wp, 1.1_wp, 1.16_wp, 1.24_wp,&
  1.35_wp, 1.49_wp, 1.64_wp, 0.33_wp, 0.34_wp, 0.35_wp, 0.34_wp, 0.34_wp, 0.34_wp, 0.35_wp, 0.36_wp,&
  0.4_wp, 0.43_wp, 0.45_wp, 0.46_wp, 0.47_wp, 0.49_wp, 0.5_wp, 0.5_wp, 0.38_wp, 0.42_wp, 0.46_wp,&
  0.49_wp, 0.52_wp, 0.55_wp, 0.58_wp, 0.63_wp, 0.7_wp, 0.77_wp, 0.79_wp, 0.81_wp, 0.85_wp, 0.88_wp,&
  0.93_wp, 0.95_wp, 0.37_wp, 0.38_wp, 0.39_wp, 0.4_wp, 0.41_wp, 0.43_wp, 0.46_wp, 0.5_wp, 0.58_wp,&
  0.65_wp, 0.67_wp, 0.7_wp, 0.74_wp, 0.8_wp, 0.86_wp, 0.88_wp, 0.38_wp, 0.42_wp, 0.46_wp, 0.5_wp,&
  0.55_wp, 0.59_wp, 0.65_wp, 0.74_wp, 0.88_wp, 1.01_wp, 1.07_wp, 1.14_wp, 1.21_wp, 1.31_wp, 1.44_wp,&
  1.53_wp, 0.52_wp, 0.56_wp, 0.59_wp, 0.62_wp, 0.67_wp, 0.74_wp, 0.82_wp, 0.95_wp, 1.13_wp, 1.27_wp,&
  1.33_wp, 1.4_wp, 1.49_wp, 1.59_wp, 1.71_wp, 1.77_wp, 0.59_wp, 0.65_wp, 0.69_wp, 0.73_wp, 0.78_wp,&
  0.84_wp, 0.94_wp, 1.06_wp, 1.24_wp, 1.39_wp, 1.46_wp, 1.54_wp, 1.64_wp, 1.74_wp, 1.85_wp, 1.93_wp,&
  0.28_wp, 0.32_wp, 0.36_wp, 0.39_wp, 0.42_wp, 0.46_wp, 0.51_wp, 0.58_wp, 0.7_wp, 0.81_wp, 0.85_wp,&
  0.91_wp, 0.97_wp, 1.03_wp, 1.11_wp, 1.18_wp, 0.35_wp, 0.4_wp, 0.44_wp, 0.48_wp, 0.53_wp, 0.57_wp,&
  0.62_wp, 0.69_wp, 0.77_wp, 0.82_wp, 0.83_wp, 0.83_wp, 0.81_wp, 0.83_wp, 0.88_wp, 0.91_wp, 0.69_wp,&
  0.82_wp, 0.96_wp, 1.12_wp, 1.24_wp, 1.34_wp, 1.45_wp, 1.64_wp, 1.97_wp, 2.26_wp, 2.37_wp, 2.5_wp,&
  2.67_wp, 2.89_wp, 3.04_wp, 3.1_wp, 0.22_wp, 0.25_wp, 0.27_wp, 0.28_wp, 0.3_wp, 0.32_wp, 0.34_wp,&
  0.35_wp, 0.38_wp, 0.39_wp, 0.4_wp, 0.41_wp, 0.42_wp, 0.44_wp, 0.46_wp, 0.49_wp, 0.28_wp, 0.32_wp,&
  0.36_wp, 0.39_wp, 0.43_wp, 0.47_wp, 0.52_wp, 0.6_wp, 0.72_wp, 0.83_wp, 0.87_wp, 0.93_wp, 1._wp,&
  1.08_wp, 1.14_wp, 1.19_wp /)
  real(wp) :: sigma = 0.7591194_wp
  !private
  public! :: network_type

  type :: network_type

    type(layer_type), allocatable :: layers(:)
    integer, allocatable :: dims(:)
    procedure(kernel_interface),      pointer :: nn_kernel
    procedure(kernel_interface_m),    pointer :: nn_kernel_m

  contains

    procedure, public :: change_kernel
    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: output_opt, output_opt_flatmodel       ! Vector input, matrix-vector product
    procedure, public, pass(self) :: output_matmul_flatmodel                ! Matrix input, matrix-matrix product
    procedure, public, pass(self) :: output_sgemm, output_sgemm_flatmodel   ! Matrix input, matrix-matrix product using BLAS
    procedure, public, pass(self) :: output_sgemm_flatmodel_tau
    procedure, public, pass(self) :: save
    procedure, public, pass(self) :: set_activation
    !procedure, public, pass(self) :: sync


  end type network_type

  interface network_type
    module procedure net_constructor
  endinterface network_type

  abstract interface
    subroutine kernel_interface(self, x, output)
      import network_type, wp
      class(network_type),    intent(in) :: self
      real(wp), dimension(:), intent(in)  :: x
      real(wp), dimension(:), intent(out) :: output
    end subroutine

    subroutine kernel_interface_m(self, x, output)
      import network_type, wp
      class(network_type),    intent(in) :: self
      real(wp), dimension(:,:), intent(in)  :: x
      real(wp), dimension(:,:), intent(out) :: output
    end subroutine
  end interface

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
    self % dims = dims
    if (.not. allocated(self % layers)) allocate(self % layers(size(dims)))
    do n = 1, size(dims) - 1
      self % layers(n) = layer_type(dims(n), dims(n+1))
    end do
    self % layers(n) = layer_type(dims(n), 1)
    self % layers(1) % b = 0.0_wp
    self % layers(size(dims)) % w = 0.0_wp
    self % layers(size(dims)) % w_transposed = 0.0_wp
    self % nn_kernel   => output_opt
    self % nn_kernel_m => output_sgemm ! Would be more consistent to use output_matmul but this one is generally very slow..
  end subroutine init

  subroutine change_kernel(self, nn_kernel, nn_kernel_m)
    class(network_type), intent(inout) :: self
    procedure(kernel_interface)   :: nn_kernel
    procedure(kernel_interface_m) :: nn_kernel_m
    self % nn_kernel    => nn_kernel
    self % nn_kernel_m  => nn_kernel_m
  end subroutine change_kernel

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

    do n = 2, size(self % dims)
      read(fileunit, fmt=*) self % layers(n) % b
    end do
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) self % layers(n) % w
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )       
    end do

    call self % layers(1) % set_activation('linear')
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) activation_type
      call self % layers(n+1) % set_activation(activation_type)
    end do    

    close(fileunit)
  end subroutine load

  ! function output(self, x) result(a)
  !   ! Use forward propagation to compute the output of the network.
  !   class(network_type), intent(in) :: self
  !   real(wp), intent(in) :: x(:)
  !   real(wp), allocatable :: a(:)
  !   integer :: n
  !   associate(layers => self % layers)
  !     a = self % layers(2) % activation(matmul(transpose(layers(1) % w), x) + layers(2) % b)
  !     do n = 3, size(layers)
  !       a = self % layers(n) % activation(matmul(transpose(layers(n-1) % w), a) + layers(n) % b)
  !     end do
  !   end associate
  ! end function output

  subroutine output_opt(self, x, output)
    class(network_type),    intent(in)  :: self
    !integer, dimension(:)   intent(in)  :: neurons
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    real(wp), allocatable   :: a(:)
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
    real(wp), intent(inout) :: x(:)
    x = x
  end subroutine

  subroutine output_opt_flatmodel(self, x, output)
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
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    ! The signal/tensor passing through the network
    real(wp), dimension(size(self % layers(1) % w_transposed,1))        :: a 
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

  subroutine output_matmul_flatmodel(self, x, output)
    ! Use forward propagation to compute the output of the network.
    ! In this version the inputs are a matrix/2D array, not 1D array (matrix-vector operations become matrix-matrix)
    class(network_type),      intent(in) :: self
    real(wp), dimension(:,:), intent(in)  :: x ! (features, num_sample)
    real(wp), dimension(:,:), intent(out) :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), size(x,2))  :: a
    integer :: n, neurons, isample, num_sample

    num_sample = size(x,2)
    neurons = size(self % layers(1) % w_transposed, 1)

    associate(layers => self % layers)
      a = matmul(layers(1) % w_transposed, x) ! + transpose(spread( layers(2) % b, 1, num_sample ))
      do isample = 1, num_sample
        a(:,isample) = a(:,isample ) + layers(2) % b
      end do
      call layers(2) % activation_m(a)

      do n = 3, size(layers)-1
        a = matmul(layers(n-1) % w_transposed, a) !+ transpose(spread( layers(n) % b, 1, num_sample ))
        do isample = 1, num_sample
          a(:,isample) = a(:,isample ) + layers(n) % b
        end do 
        call layers(n) % activation_m(a)
      end do

      output = matmul(layers(n-1) % w_transposed, a) ! + transpose(spread( layers(n) % b, 1, num_sample ))
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
      end do
      call layers(n) % activation_m(output)
    end associate
  end subroutine


  subroutine output_sgemm(self, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),      intent(in)    :: self
    real(wp), dimension(:,:), intent(in)    :: x      ! (features, num_sample)
    real(wp), dimension(:,:), intent(out)   :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), allocatable   :: a(:,:), a_next(:,:)
    real(wp)                :: alpha, beta
    integer,  dimension(2)  :: matsize
    integer                 :: n, num_sample, isample, neurons

    alpha = 1.0_wp
    beta = 0.0_wp
    output = 0.0_wp

    num_sample = size(x,2)

    associate(layers => self % layers)
      matsize = shape(layers(1) % w_transposed)
      allocate(a(matsize(1),num_sample))
      ! Multiply weights of this layer with input
      call sgemm("N","N",matsize(1), num_sample, matsize(2), alpha, layers(1) % w_transposed, matsize(1), x, matsize(2), beta, a, matsize(1))
      ! Add biases
      do isample = 1, num_sample
        a(:,isample) = a(:,isample ) + layers(2) % b
      end do
      ! a = a + transpose(spread( layers(2) % b, 1, num_sample ))
      ! Activation (2D array)
      call layers(2) % activation_m(a)
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        matsize = shape(layers(n-1) % w_transposed)
        allocate(a_next(matsize(1),num_sample))
        ! Multiply weights of this layer with input
        call sgemm("N","N",matsize(1),num_sample,matsize(2),alpha,layers(n-1) % w_transposed,matsize(1),a,matsize(2),beta,a_next,matsize(1))
        deallocate(a)
        ! Add biases
        do isample = 1, num_sample
          a_next(:,isample) = a_next(:,isample ) + layers(n) % b
        end do 
        ! a_next = a_next + transpose(spread( layers(n) % b, 1, num_sample ))
        ! Activation
        call layers(n) % activation_m(a_next)
        a = a_next
        deallocate(a_next)
      end do

      matsize = shape(layers(n-1) % w_transposed)
      call sgemm("N","N",matsize(1), num_sample, matsize(2), alpha, layers(n-1) % w_transposed, matsize(1), a, matsize(2), beta, output, matsize(1))
      ! Add biases
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
      end do
      call layers(n) % activation_m(output)
      !  output = output + transpose(spread( layers(n) % b, 1, num_sample ))
    end associate
  end subroutine

  
  subroutine output_sgemm_flatmodel(self, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),      intent(in)    :: self
    real(wp), dimension(:,:), intent(in)    :: x      ! (features, num_sample)
    real(wp), dimension(:,:), intent(out)   :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), size(x,2))  :: a,a_next
    real(wp)                :: alpha, beta
    integer                 :: n, num_sample, isample, neurons, num_inputs, ngpt

    a       = 0.0_wp
    alpha   = 1.0_wp
    beta    = 0.0_wp
    output  = 0.0_wp

    neurons = size(self % layers(1) % w_transposed, 1)
    num_inputs  = size(x,1)
    num_sample  = size(x,2)
    ngpt        = size(output,1)

    associate(layers => self % layers)

      call sgemm("N","N",neurons, num_sample, num_inputs, alpha, layers(1) % w_transposed, neurons, x, num_inputs, beta, a, neurons)
      do isample = 1, num_sample
        a(:,isample) = a(:,isample ) + layers(2) % b
      end do
      call layers(2) % activation_m(a)

      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        ! call sgemm("N","N",neurons,num_sample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a, neurons)
        call sgemm("N","N",neurons,num_sample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a_next, neurons)
        do isample = 1, num_sample
          a_next(:,isample) = a_next(:,isample ) + layers(n) % b
        end do 
        call layers(n) % activation_m(a_next)
        a = a_next
      end do

      call sgemm("N","N",ngpt, num_sample, neurons, alpha, layers(n-1) % w_transposed, ngpt, a, neurons, beta, output, ngpt)
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
      end do
      call layers(n) % activation_m(output)

      end associate
  end subroutine

  subroutine output_sgemm_flatmodel_tau(self, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),      intent(in)    :: self
    real(wp), dimension(:,:), intent(in)    :: x      ! (features, num_sample)
    real(wp), dimension(:,:), intent(out)   :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), size(x,2))  :: a,a_next
    real(wp)                :: alpha, beta
    integer                 :: n, num_sample, isample, neurons, num_inputs, ngpt

    a       = 0.0_wp
    alpha   = 1.0_wp
    beta    = 0.0_wp
    output  = 0.0_wp

    neurons = size(self % layers(1) % w_transposed, 1)
    num_inputs  = size(x,1)
    num_sample  = size(x,2)
    ngpt        = size(output,1)

    associate(layers => self % layers)

      call sgemm("N","N",neurons, num_sample, num_inputs, alpha, layers(1) % w_transposed, neurons, x, num_inputs, beta, a, neurons)
      do isample = 1, num_sample
        a(:,isample) = a(:,isample ) + layers(2) % b
      end do
      call layers(2) % activation_m(a)

      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        ! call sgemm("N","N",neurons,num_sample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a, neurons)
        call sgemm("N","N",neurons,num_sample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a_next, neurons)
        do isample = 1, num_sample
          a_next(:,isample) = a_next(:,isample ) + layers(n) % b
        end do 
        call layers(n) % activation_m(a_next)
        a = a_next
      end do

      call sgemm("N","N",ngpt, num_sample, neurons, alpha, layers(n-1) % w_transposed, ngpt, a, neurons, beta, output, ngpt)
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
        output(:,isample) = sigma*output(:,isample) + output_scaler_means
      end do
      call layers(n) % activation_m(output)

      
      end associate
  end subroutine

  function matvecmul(matA,vecB,nrow,ncol)
        implicit none
        integer, intent(in) :: nrow,ncol
        real(wp), intent(in) :: matA(nrow,ncol)
        real(wp), intent(in) :: vecB(ncol)
        integer :: i,j
        real(wp) :: matvecmul(nrow)

        ! each (row,here ncol) element in b (length e.g. 256) is obtained by :
        ! loop through the different elements in vecB (length 50), multiply by the corresponding
        ! column (still 50) element in matA for that particular row (outer loop), add the 50 values together

        matvecmul = 0.0_wp
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
