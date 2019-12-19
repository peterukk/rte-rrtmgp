module mod_network

  use mo_rte_kind, only: wp
  use mod_layer, only: layer_type
  
  implicit none
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
    procedure, public, pass(self) :: output_sgemm_flatmodel_standardscaling
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

    subroutine kernel_interface_m(self, num_inputs,num_outputs,num_sample, x, output)
    import network_type, wp
    class(network_type),      intent(in)    :: self
    integer, intent(in) :: num_inputs,num_outputs,num_sample
    real(wp), dimension(num_inputs, num_sample), intent(in)      :: x      ! (features, num_sample)
    real(wp), dimension(num_outputs,num_sample), intent(out)     :: output ! (outputs, num_sample)
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

  subroutine output_matmul_flatmodel(self, num_inputs, num_outputs, num_sample, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),    intent(in)          :: self
    integer,                intent(in)          :: num_inputs,num_outputs,num_sample
    real(wp), dimension(num_inputs,  num_sample), &
                            intent(in)          :: x      ! (features, num_sample)
    real(wp), dimension(num_outputs, num_sample), &
                            intent(out)         :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), num_sample)  :: a
    integer :: n, neurons, isample

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


  subroutine output_sgemm(self, num_inputs, num_outputs, num_sample, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),    intent(in)          :: self
    integer,                intent(in)          :: num_inputs,num_outputs,num_sample
    real(wp), dimension(num_inputs, num_sample), &
                            intent(in)          :: x      ! (features, num_sample)
    real(wp), dimension(num_outputs,num_sample), &
                            intent(out)         :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), allocatable   :: a(:,:), a_next(:,:)
    real(wp)                :: alpha, beta
    integer,  dimension(2)  :: matsize
    integer                 :: n, isample, neurons

    alpha = 1.0_wp
    beta = 0.0_wp
    output = 0.0_wp

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

  
  subroutine output_sgemm_flatmodel(self, num_inputs,num_outputs,num_sample, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! sgemm = single-precision (wp = sp)
    class(network_type),      intent(in)    :: self
    integer, intent(in) :: num_inputs,num_outputs,num_sample
    real(wp), dimension(num_inputs, num_sample), intent(in)      :: x      ! (features, num_sample)
    real(wp), dimension(num_outputs,num_sample), intent(out)     :: output ! (outputs, num_sample)
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), size(x,2))  :: a,a_next
    real(wp)                :: alpha, beta
    integer                 :: n, isample, neurons 

    a       = 0.0_wp
    alpha   = 1.0_wp
    beta    = 0.0_wp
    output  = 0.0_wp

    neurons = size(self % layers(1) % w_transposed, 1)

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

      call sgemm("N","N",num_outputs, num_sample, neurons, alpha, layers(n-1) % w_transposed, num_outputs, a, neurons, beta, output, num_outputs)
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
      end do
      call layers(n) % activation_m(output)

      end associate
  end subroutine

  subroutine output_sgemm_flatmodel_standardscaling(self, num_inputs,num_outputs,num_sample, x, output, output_means, output_sigma)
    ! Like the previous procedure but including post-processing of outputs. 
    ! Use of optional argument would be neater, unfortunately there is a performance loss at least on intel compilers. 
    class(network_type),      intent(in)    :: self
    integer, intent(in) :: num_inputs,num_outputs,num_sample
    real(wp), dimension(num_inputs, num_sample), intent(in)     :: x      ! (features, num_sample)
    real(wp), dimension(num_outputs,num_sample), intent(out)    :: output ! (outputs, num_sample)
    ! Optional
    real(wp), dimension(num_outputs),           intent(in)      :: output_means
    real(wp) ,                                  intent(in)      :: output_sigma 
    ! Local variables
    real(wp), dimension(size(self % layers(1) % w_transposed, 1), size(x,2))  :: a,a_next
    real(wp)                :: alpha, beta
    integer                 :: n, isample, neurons 

    a       = 0.0_wp
    alpha   = 1.0_wp
    beta    = 0.0_wp
    output  = 0.0_wp

    neurons = size(self % layers(1) % w_transposed, 1)
    !num_inputs  = size(x,1)
    !num_sample  = size(x,2)
    !num_outputs        = size(output,1)

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

      call sgemm("N","N",num_outputs, num_sample, neurons, alpha, layers(n-1) % w_transposed, num_outputs, a, neurons, beta, output, num_outputs)
      do isample = 1, num_sample
        output(:,isample) = output(:,isample ) + layers(n) % b
        output(:,isample) = output_sigma*output(:,isample) + output_means
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
