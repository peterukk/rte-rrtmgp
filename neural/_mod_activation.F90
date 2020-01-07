module mod_activation

  ! A collection of activation subroutines and their derivatives.

  use mo_rte_kind, only: wp

  implicit none

  public

  abstract interface
    pure subroutine activation_vec(x)
      import :: wp
      real(wp), intent(inout) :: x(:)
    end subroutine activation_vec

    pure subroutine activation_mat(x)
      import :: wp
      real(wp), intent(inout) :: x(:,:)
    end subroutine activation_mat
  end interface

contains

  pure subroutine gaussian(x) 
    ! Gaussian activation subroutine.
    real(wp), intent(inout) :: x(:)
    x = exp(-x**2)
  end subroutine gaussian

  pure subroutine gaussian_m(x) 
    real(wp), intent(inout) :: x(:,:)
    x = exp(-x**2)
  end subroutine gaussian_m


  elemental subroutine relu(x) 
    !! REctified Linear Unit (RELU) activation subroutine.
    real(wp), intent(inout) :: x(:)
    x = max(0.0_wp, x)
  end subroutine relu

  pure subroutine relu_m(x) 
    real(wp), intent(inout) :: x(:,:)
    x = max(0.0_wp, x)
  end subroutine relu_m


  pure subroutine sigmoid(x) 
    ! Sigmoid activation subroutine.
    real(wp), intent(inout) :: x(:)
    x = 1 / (1 + exp(-x))
  end subroutine sigmoid

  pure subroutine sigmoid_m(x) 
    real(wp), intent(inout) :: x(:,:)
    x = 1 / (1 + exp(-x))
  end subroutine sigmoid_m


  pure subroutine tanhf(x) 
    ! Tangent hyperbolic activation subroutine.
    ! Same as the intrinsic tanh, but must be
    ! defined here so that we can use procedure
    ! pointer with it.
    real(wp), intent(inout) :: x(:)
    x = tanh(x)
  end subroutine tanhf

  pure subroutine tanhf_m(x) 
    real(wp), intent(inout) :: x(:,:)
    x = tanh(x)
  end subroutine tanhf_m


  pure subroutine softsign(x)
    real(wp), intent(inout) :: x(:)
    x = x / (abs(x) + 1)
  end subroutine

  pure subroutine softsign_m(x)
    real(wp), intent(inout) :: x(:,:)
    x = x / (abs(x) + 1)
  end subroutine

  pure subroutine linear(x)
    real(wp), intent(inout) :: x(:)
    x = x
  end subroutine

  pure subroutine linear_m(x)
    real(wp), intent(inout) :: x(:,:)
    x = x
  end subroutine

end module mod_activation
