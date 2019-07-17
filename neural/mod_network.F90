module mod_network

  use mo_rte_kind, only: wp
  use mod_layer, only: layer_type
  
  implicit none

  private
  public :: network_type

  type :: network_type

    type(layer_type), allocatable :: layers(:)
    integer, allocatable :: dims(:)

  contains

    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: output
    procedure, public, pass(self) :: output_flatmodel_sig_opt
    procedure, public, pass(self) :: output_flatmodel_sig_opt2
    procedure, public, pass(self) :: output_flatmodel_sgemv
    procedure, public, pass(self) :: output_flatmodel_dgemv
    procedure, public, pass(self) :: save
    procedure, public, pass(self) :: set_activation
    !procedure, public, pass(self) :: sync


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
    self % dims = dims
    if (.not. allocated(self % layers)) allocate(self % layers(size(dims)))
    do n = 1, size(dims) - 1
      self % layers(n) = layer_type(dims(n), dims(n+1))
    end do
    self % layers(n) = layer_type(dims(n), 1)
    self % layers(1) % b = 0
    self % layers(size(dims)) % w = 0
    self % layers(size(dims)) % w_transposed = 0
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

    do n = 2, size(self % dims)
      read(fileunit, fmt=*) self % layers(n) % b
    end do
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) self % layers(n) % w
      self % layers(n) % w_transposed = transpose(self % layers(n) % w )
      
      !print*, shape(transpose(self % layers(n) % w ))
       
    end do

    call self % layers(1) % set_activation('linear')
    do n = 1, size(self % dims) - 1
      read(fileunit, fmt=*) activation_type
      print*, n
      print*, activation_type
      call self % layers(n+1) % set_activation(activation_type)
    end do

    close(fileunit)
  end subroutine load

  function output(self, x) result(a)
    ! Use forward propagation to compute the output of the network.
    class(network_type), intent(in) :: self
    real(wp), intent(in) :: x(:)
    real(wp), allocatable :: a(:)
    integer :: n
    associate(layers => self % layers)
      !print *, "shape:", shape( transpose(layers(1) % w))
      a = self % layers(2) % activation(matmul(transpose(layers(1) % w), x) + layers(2) % b)
      !print *, "a30:", a(30)

      do n = 3, size(layers)
        a = self % layers(n) % activation(matmul(transpose(layers(n-1) % w), a) + layers(n) % b)
      end do
    end associate
  end function output

  subroutine output_flatmodel_sig_opt(self, x, neurons, output)
    ! Use forward propagation to compute the output of the network.
    ! Output allocated outside of function, explicit-shape intermediate array,(number of neurons same in each hidden layer)
    ! activation functions are replaced with a subroutine that modifies the arguments (sigmoid)
    class(network_type),    intent(in) :: self
    integer,            intent(in)  :: neurons
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    real(wp), dimension(neurons)        :: a
    integer :: n

    associate(layers => self % layers)
      a = matmul(transpose(layers(1) % w), x) + layers(2) % b
      ! sigmoid activation: using an "inout" subroutine to avoid array copy
      call sigmoid_subroutine(a)
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        a = matmul(transpose(layers(n-1) % w), a) + layers(n) % b
        call sigmoid_subroutine(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      output = (matmul(transpose(layers(n-1) % w), a) + layers(n) % b)
    end associate
  end subroutine

  subroutine output_flatmodel_sig_opt2(self, x, neurons, output)
    ! Use forward propagation to compute the output of the network.
    ! Output allocated outside of function, explicit-shape intermediate array,(number of neurons same in each hidden layer)
    ! activation functions are replaced with a subroutine that modifies the arguments (sigmoid),
    ! matmul replaced by custom function for matrix-vector multiplication.
    ! ONLY faster with gfortran -O3 -march=native, or ifort -O3, otherwise the custom function is slower
    class(network_type),    intent(in) :: self
    integer,            intent(in)  :: neurons
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    real(wp), dimension(neurons)        :: a

    integer :: n
    !print *, "hiho"
    associate(layers => self % layers)
      a = matvecmul(layers(1) % w_transposed,x,neurons,size(x)) + layers(2) % b
      ! sigmoid activation: using an "inout" subroutine to avoid array copy 
      call sigmoid_subroutine(a)
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        a = matvecmul(layers(n-1) % w_transposed, a, neurons, neurons) + layers(n) % b
        call sigmoid_subroutine(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      output = (matvecmul(layers(n-1) % w_transposed, a, size(output), neurons) + layers(n) % b)
    end associate
  end subroutine

  subroutine output_flatmodel_sgemv(self, x, neurons, output)
    ! Use forward propagation to compute the output of the network.
    ! Output allocated outside of function, explicit-shape intermediate array,(number of neurons same in each hidden layer)
    ! activation functions are replaced with a subroutine that modifies the arguments (sigmoid),
    ! matmul replaced by custom function for matrix-vector multiplication
    class(network_type),    intent(in) :: self
    integer,            intent(in)  :: neurons
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    real(wp), dimension(neurons)        :: a !, c
    integer :: n, incx, incy
    real(wp) :: alpha,beta
    alpha = 1.0
    beta = 0.0
    incx = 1
    incy = 1

    associate(layers => self % layers)
      call dgemv("N",neurons,size(x),alpha,layers(1) % w_transposed,neurons,x,incx,beta,a,incy)
      a = a + layers(2) % b
      call sigmoid_subroutine(a) ! sigmoid activation: using an "inout" subroutine to avoid array copy 
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        ! to avoid having to allocate another output array c (of size neurons), don't use sgemv here
        ! For deep neural networks with more than 2-3 hidden layers, it's probably worth using sgemv
        a = matvecmul(layers(n-1) % w_transposed, a, neurons, neurons)
        ! call sgemv("N",neurons,neurons,alpha,layers(n-1) % w_transposed,neurons,a,incx,beta,c,incy)'
        a = a + layers(n) % b
        ! a = c + layers(n) % b
        call sigmoid_subroutine(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      call dgemv("N",size(output),neurons,alpha,layers(n-1) % w_transposed,size(output),a,incx,beta,output,incy)
      output = output + layers(n) % b
    end associate
  end subroutine

  subroutine output_flatmodel_dgemv(self, x, neurons, output)
    ! Use forward propagation to compute the output of the network.
    ! Output allocated outside of function, explicit-shape intermediate array,(number of neurons same in each hidden layer)
    ! activation functions are replaced with a subroutine that modifies the arguments (sigmoid),
    ! matmul replaced by custom function for matrix-vector multiplication
    class(network_type),    intent(in) :: self
    integer,            intent(in)  :: neurons
    real(wp), dimension(:), intent(in)  :: x
    real(wp), dimension(:), intent(out) :: output
    ! Local variables
    real(wp), dimension(neurons)        :: a !, c
    integer :: n, incx, incy
    real(wp) :: alpha,beta
    alpha = 1.0
    beta = 0.0
    incx = 1
    incy = 1

    associate(layers => self % layers)
      call dgemv("N",neurons,size(x),alpha,layers(1) % w_transposed,neurons,x,incx,beta,a,incy)
      a = a + layers(2) % b
      call sigmoid_subroutine(a) ! sigmoid activation: using an "inout" subroutine to avoid array copy 
      ! INTERMEDIATE LAYERS
      do n = 3, size(layers)-1
        ! to avoid having to allocate another output array c (of size neurons), don't use sgemv here
        ! For deep neural networks with more than 2-3 hidden layers, it's probably worth using sgemv
        a = matvecmul(layers(n-1) % w_transposed, a, neurons, neurons)
        ! call sgemv("N",neurons,neurons,alpha,layers(n-1) % w_transposed,neurons,a,incx,beta,c,incy)'
        a = a + layers(n) % b
        ! a = c + layers(n) % b
        call sigmoid_subroutine(a)
      end do
      ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
      call dgemv("N",size(output),neurons,alpha,layers(n-1) % w_transposed,size(output),a,incx,beta,output,incy)
      output = output + layers(n) % b
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

        matvecmul = 0.0
        do j=1,ncol !  50
            matvecmul = matvecmul + matA(:,j) * vecB(j) !length 256. this is for one column, and then the columns need to be added together ( b = b + ..)
        enddo

  end function matvecmul

  subroutine sigmoid_subroutine(x)
    real(wp), dimension(:), intent(inout) :: x
    x = 1 / (1 + exp(-x))
    ! x = x / (1 + abs(x)  ! WOULD BE SLIGHTLY FASTER (10-15%)
  end subroutine



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
