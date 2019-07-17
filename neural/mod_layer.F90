module mod_layer

  ! Defines the layer type and its methods.

  use mod_activation
  use mo_rte_kind, only: wp
  use mod_random, only: randn

  implicit none

  private
  public :: layer_type

  type :: layer_type
    real(wp), allocatable :: a(:) ! activations
    real(wp), allocatable :: b(:) ! biases
    real(wp), allocatable :: w(:,:) ! weights
    real(wp), allocatable :: w_transposed(:,:) ! weights
    real(wp), allocatable :: z(:) ! arg. to activation function
    procedure(activation_function), pointer, nopass :: activation => null()
    procedure(activation_function), pointer, nopass :: activation_prime => null()
  contains
    procedure, public, pass(self) :: set_activation
  end type layer_type

  interface layer_type
    module procedure constructor
  end interface layer_type

contains

  type(layer_type) function constructor(this_size, next_size) result(layer)
    ! Layer class constructor. this_size is the number of neurons in the layer.
    ! next_size is the number of neurons in the next layer, used to allocate
    ! the weights.
    integer, intent(in) :: this_size, next_size
    allocate(layer % a(this_size))
    allocate(layer % z(this_size))
    layer % a = 0
    layer % z = 0
    layer % w = randn(this_size, next_size) / this_size
    layer % w_transposed = transpose(layer % w)
    layer % b = randn(this_size)
  end function constructor

  pure subroutine set_activation(self, activation)
    ! Sets the activation function. Input string must match one of
    ! provided activation functions, otherwise it defaults to sigmoid.
    ! If activation not present, defaults to sigmoid.
    class(layer_type), intent(in out) :: self
    character(len=*), intent(in) :: activation
    select case(trim(activation))
      case('gaussian')
        self % activation => gaussian
        self % activation_prime => gaussian_prime
      case('relu')
        self % activation => relu
        self % activation_prime => relu_prime
      case('sigmoid')
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
      case('step')
        self % activation => step
        self % activation_prime => step_prime
      case('tanh')
        self % activation => tanhf
        self % activation_prime => tanh_prime
      case('linear')
        self % activation => linear
        self % activation_prime => linear_prime
      case default
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
    end select
  end subroutine set_activation

end module mod_layer
