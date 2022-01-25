module mod_layer

  ! Defines the layer type and its methods.

  ! Changelog: 
  !   P.Ukkonen, 23.1.2022 : Change the code so that input layer is not included in layers (see mod_network)


  use mod_activation
  use mo_rte_kind, only: sp
  use mod_random, only: randn

  implicit none

  private
  public :: layer_type

  type :: layer_type
    real(sp), allocatable :: b(:) ! biases
    real(sp), allocatable :: w(:,:) ! weights
    real(sp), allocatable :: w_transposed(:,:) ! weights
    !!  !$acc policy<copylayer> copyin(b, w, w_transposed)
    procedure(activation_interface),        pointer, nopass :: activation !=> null()
    procedure(bias_and_activation_interface),pointer, nopass :: bias_and_activation !=> null()

  contains
  
    procedure, public, pass(self) :: set_activation
  end type layer_type

  ! interface activation
  !   procedure activation_func
  !   procedure activation_func_bias
  ! end interface activation

  interface layer_type
    module procedure constructor
  end interface layer_type

contains

  ! type(layer_type) function constructor(this_size, next_size) result(layer)
  !   ! Layer class constructor. this_size is the number of neurons in the layer.
  !   ! next_size is the number of neurons in the next layer, used to allocate
  !   ! the weights.
  !   integer, intent(in) :: this_size, next_size
  !   allocate(layer % w(this_size,next_size))
  !   allocate(layer % w_transposed(next_size,this_size))
  !   allocate(layer % b(this_size))
  !   layer % b = randn(this_size)
  ! end function constructor

  type(layer_type) function constructor(last_size, this_size) result(layer)
    ! Layer class constructor. this_size is the number of neurons in the layer.
    ! next_size is the number of neurons in the next layer, used to allocate
    ! the weights.
    integer, intent(in) :: last_size, this_size
    allocate(layer % w(last_size, this_size))
    allocate(layer % w_transposed(this_size, last_size))
    allocate(layer % b(this_size))
    ! layer % b = randn(this_size)
  end function constructor

  subroutine set_activation(self, activation)
    ! Sets the activation function. Input string must match one of
    ! provided activation functions, otherwise it defaults to sigmoid.
    ! If activation not present, defaults to sigmoid.
    class(layer_type), intent(in out) :: self
    character(len=*), intent(in) :: activation
    select case(trim(activation))
      case('gaussian')
        self % activation => gaussian
        ! self % bias_and_activation => gaussian_b
      case('relu')
        self % activation             => relu
        self % bias_and_activation    => relu_mat_b
      case('sigmoid')
        self % activation             => sigmoid
        self % bias_and_activation    => sigmoid_mat_b
      case('hard_sigmoid')
        self % activation             => hard_sigmoid
        self % bias_and_activation    => hard_sigmoid_mat_b
      case('softsign')
        self % activation             => softsign
        self % bias_and_activation    => softsign_mat_b
      case('tanh')
        self % activation => tanhf
      case('linear')
        self % activation             => linear
        self % bias_and_activation    => linear_mat_b
      case default
        print *, "failed to read activation function, setting to linear"
        self % activation             => linear
        self % bias_and_activation    => linear_mat_b
    end select
    ! if (activation(1:8)=='gaussian') then
    !     self % activation => gaussian
    !     ! self % bias_and_activation => gaussian_b
    ! else if (activation(1:4)=='relu') then
    !     self % activation             => relu
    !     self % bias_and_activation    => relu_mat_b
    ! else if (activation(1:7)=='sigmoid') then
    !     self % activation             => sigmoid
    !     self % bias_and_activation    => sigmoid_mat_b
    ! else if (activation(1:12)=='hard_sigmoid') then
    !     self % activation             => hard_sigmoid
    !     self % bias_and_activation    => hard_sigmoid_mat_b
    ! else if (activation(1:8)=='softsign') then
    !     self % activation             => softsign
    !     self % bias_and_activation    => softsign_mat_b
    ! else if (activation(1:4)=='tanh') then
    !     self % activation => tanhf
    ! else if (activation(1:6)=='linear') then
    !     self % activation             => linear
    !     self % bias_and_activation    => linear_mat_b
    ! else
    !     print *, "failed to read activation function, setting to linear"
    !     self % activation             => linear
    !     self % bias_and_activation    => linear_mat_b
    ! end if
  end subroutine set_activation

end module mod_layer
