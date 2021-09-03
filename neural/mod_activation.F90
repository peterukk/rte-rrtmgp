module mod_activation

  ! A collection of activation subroutines and their derivatives.

  use mo_rte_kind, only: sp

  implicit none
  
  public

  abstract interface 

    pure subroutine activation_interface(x)
      import :: sp
      real(sp), intent(inout) :: x(:)
    end subroutine activation_interface

    pure subroutine bias_and_activation_interface(x, b)
      ! This one works on batched 2D data, and adds bias before activation
      import :: sp
      real(sp), intent(inout) :: x(:,:)
      real(sp), intent(in)    :: b(:)
    end subroutine bias_and_activation_interface

  end interface

contains

  pure subroutine gaussian(x) 
    ! Gaussian activation subroutine.
    real(sp), intent(inout) :: x(:)
    x = exp(-x**2)
  end subroutine gaussian

  pure subroutine tanhf(x) 
    ! Tangent hyperbolic activation subroutine.
    ! Same as the intrinsic tanh, but must be
    ! defined here so that we can use procedure
    ! pointer with it.
    real(sp), intent(inout) :: x(:)
    x = tanh(x)
  end subroutine tanhf

  pure subroutine relu(x) 
    !! REctified Linear Unit (RELU) activation subroutine.
    real(sp), intent(inout) :: x(:)
    x = max(0.0_sp, x)
  end subroutine relu

  pure subroutine relu_mat_b(x,b)
    real(sp), intent(inout) :: x(:,:) 
    real(sp), intent(in)    :: b(:) !
    integer :: i,j
    if (size(x,dim=1) == size(b)) then
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(i)
          x(i,j) = max(0.0_sp, x(i,j) )
        end do
      end do
    else 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(j)
          x(i,j) = max(0.0_sp, x(i,j) )
        end do
      end do
    end if
  end subroutine

  pure subroutine sigmoid(x) 
    ! Sigmoid activation subroutine.
    real(sp), intent(inout) :: x(:)
    x = 1 / (1 + exp(-x))
  end subroutine sigmoid

  pure subroutine sigmoid_mat_b(x,b)
    real(sp), intent(inout) :: x(:,:) 
    real(sp), intent(in)    :: b(:) !
    integer :: i,j
    if (size(x,dim=1) == size(b)) then
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(i)
          x(i,j) = 1 / (1 + exp(-x(i,j)))
        end do
      end do
    else 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(j)
          x(i,j) = 1 / (1 + exp(-x(i,j)))
        end do
      end do
    end if
  end subroutine

  pure subroutine softsign(x)
    real(sp), intent(inout) :: x(:)
    x = x / (abs(x) + 1)
  end subroutine

  pure subroutine softsign_mat_b(x,b)
    real(sp), intent(inout) :: x(:,:) 
    real(sp), intent(in)    :: b(:) !
    integer :: i,j
    if (size(x,dim=1) == size(b)) then
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(i)
          x(i,j) = x(i,j) / (abs(x(i,j)) + 1)
        end do
      end do
    else 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(j)
          x(i,j) = x(i,j) / (abs(x(i,j)) + 1)
        end do
      end do      
    end if
  end subroutine

  pure subroutine hard_sigmoid(x)
    real(sp), intent(inout) :: x(:)
    x = max(0.0_sp, min(1.0_sp, 0.2_sp*x + 0.5_sp))
  end subroutine

  pure subroutine hard_sigmoid_mat_b(x,b)
    real(sp), intent(inout) :: x(:,:) 
    real(sp), intent(in)    :: b(:) !
    integer :: i,j
    if (size(x,dim=1) == size(b)) then
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(i)
          x(i,j) = max(0.0_sp, min(1.0_sp, 0.2_sp*x(i,j) + 0.5_sp))
        end do
      end do
    else 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(j)
          x(i,j) = max(0.0_sp, min(1.0_sp, 0.2_sp*x(i,j) + 0.5_sp))
        end do
      end do
    end if
  end subroutine

  pure subroutine linear(x)
    real(sp), intent(inout) :: x(:)
    x = x
  end subroutine

  pure subroutine linear_mat_b(x,b)
    real(sp), intent(inout) :: x(:,:) 
    real(sp), intent(in)    :: b(:)
    integer :: i,j
    if (size(x,dim=1) == size(b)) then
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2)
        do i = 1, size(x,dim=1)
          x(i,j) = x(i,j) + b(i)
        end do
      end do
    else 
      !$acc parallel loop collapse(2) default(present)
      do j = 1, size(x,dim=2), 2
        !$OMP SIMD
        do i = 1, size(x,dim=1)
          x(i,j)   = x(i,j)   + b(j)
          x(i,j+1) = x(i,j+1) + b(j+1)
        end do
      end do
    end if
  end subroutine
  
end module mod_activation
