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


  subroutine output_sgemm_tau(self, nx, ngpt, nbatch, x, coldry, ymeans, ysigma, output)
    ! Optimized inference function for optical depth, using BLAS/cuBLAS and includes post-processing of outputs.
    ! This routine takes a 2D input data array for batched predictions with a feed-forward network.
    ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
    ! always in single-precision (sgemm) because why not
    !
    !                                   Layer Weights            Layer Inputs                Layer Outputs
    ! First layer :                      (Nneurons x Nx)       * (Nx x nbatch )          = (Nneurons x nbatch) 
    ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons x nbatch )    = (Nneurons x nbatch) 
    ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons x nbatch )    = (Ngpoints x nbatch)  
    ! in GEMM terms:                         A                 *         B                = C
    !                                     (m x k)              *      (k * N )            = (m  * N)  
    use, intrinsic :: iso_c_binding
    class(network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ngpt, nbatch
    real(sp), dimension(nx, nbatch), intent(in)  :: x      ! (features, nbatch)
    real(sp), dimension(nbatch),     intent(in)  :: coldry 
    real(sp), dimension(ngpt),          intent(in)  :: ymeans
    real(sp),                         intent(in)  :: ysigma
    real(sp), dimension(ngpt, nbatch), intent(out) :: output ! (outputs, nbatch) 
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
      
      ! Assign pointers to layer weights, input-output arrays and biases
      wt      => layers(1) % w_transposed
      a       => a1
      a_next  => a2
      b       => layers(2) % b
      
      ! 1. Multiply inputs with weights of first layer

      !$acc host_data use_device(wt, x, a)
      call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

      ! 2. Add biases and activation

      !$acc parallel loop collapse(2) default(present)
      do j = 1, nbatch
        do i = 1, neurons
          a(i, j) = a(i, j ) + b(i)
          call activation_softsign(a(i, j))
        end do
      end do

      ! 3. Repeat steps 1-2 until final layer reached
      
      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        !$acc parallel loop collapse(2) default(present)
        do j = 1, nbatch
          do i = 1 , neurons 
            a_next(i, j) = a_next(i, j ) + b(i)
            call activation_softsign(a_next(i, j))
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
      call sgemm("N","N", ngpt, nbatch, neurons, 1.0, wt, ngpt, a, neurons, 0.0, output, ngpt)
      !$acc end host_data

      n = nlayers

      !$acc parallel loop gang default(present)
      do j = 1, nbatch
        !$OMP SIMD
        !$acc loop vector
        do i = 1, ngpt
          ! Compute outputs and scale them to obtain molecular absorption 
          ! output(i, j) = (ysigma*(output(i, j) + b(i)) + ymeans_lw_tau(i))**8

          ! Scale with number of dry air molecules to obtain optical depth
          ! output(i, j) =  output(i, j) * coldry(j)

          ! One-line solution
          output(i, j) = ((ysigma* (output(i, j) + b(i)) + ymeans(i))**8) * coldry(j)

        end do
      end do

    end associate
    !$acc exit data detach(a,a_next) delete(a1, a2, ymeans)

                                              
  end subroutine

!   subroutine output_sgemm_tau(self, nx, ngpt, nbatch, x, coldry, ymeans, ysigma, output)
!     ! Optimized inference function for optical depth, using BLAS/cuBLAS and includes post-processing of outputs.
!     ! This routine takes a 2D input data array for batched predictions with a feed-forward network.
!     ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
!     ! always in single-precision (sgemm) because why not
!     !
!     !                                   Layer Weights            Layer Inputs                Layer Outputs
!     ! First layer :                      (Nneurons x Nx)       * (Nx x nbatch )          = (Nneurons x nbatch) 
!     ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons x nbatch )    = (Nneurons x nbatch) 
!     ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons x nbatch )    = (Ngpoints x nbatch)  
!     ! in GEMM terms:                         A                 *         B                = C
!     !                                     (m x k)              *      (k * N )            = (m  * N)  
!     use, intrinsic :: iso_c_binding
!     class(network_type),              intent(in), target  :: self
!     integer, intent(in)                           :: nx, ngpt, nbatch
!     real(sp), dimension(nx, nbatch), intent(in)  :: x      ! (features, nbatch)
!     real(sp), dimension(nbatch),     intent(in)  :: coldry 
!     real(sp), dimension(ngpt),          intent(in)  :: ymeans
!     real(sp),                         intent(in)  :: ysigma
!     real(sp), dimension(ngpt, nbatch), intent(out) :: output ! (outputs, nbatch) 
!     real(sp), dimension(size(self % layers(1) % w_transposed, 1), nbatch), &
!                                           target  :: a1, a2  
!     real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
!     real(sp), dimension(:,:), contiguous, pointer :: wt
!     real(sp), dimension(:),   contiguous, pointer :: b
!     integer                       :: n, j, neurons, nlayers, i, ng, nneur
!     real(sp), allocatable :: output2(:,:), wt2(:,:), inp2(:,:), output3(:,:), wt3(:,:), inp3(:,:)

!     neurons = size(self % layers(1) % w_transposed, 1)
!     nlayers = size(self % layers)

!     !$acc enter data create(a1, a2) copyin(ymeans)
!     associate(layers=>self%layers)
      
!       ! Assign pointers to layer weights, input-output arrays and biases
!       wt      => layers(1) % w_transposed
!       a       => a1
!       a_next  => a2
!       b       => layers(2) % b
      
!       ! 1. Multiply inputs with weights of first layer

!       !$acc host_data use_device(wt, x, a)
!       call sgemm('N','N', neurons, nbatch, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
!       !$acc end host_data

!       ! 2. Add biases and activation

!       !$acc parallel loop collapse(2) default(present)
!       do j = 1, nbatch
!         do i = 1, neurons
!           a(i, j) = a(i, j ) + b(i)
!           call activation_softsign(a(i, j))
!         end do
!       end do

!       ! 3. Repeat steps 1-2 until final layer reached
      
!       do n = 3, nlayers-1

!         wt => layers(n-1) % w_transposed
!         b => layers(n) % b

!         !$acc host_data use_device(wt, a, a_next)
!         call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
!         !$acc end host_data

!         !$acc parallel loop collapse(2) default(present)
!         do j = 1, nbatch
!           do i = 1 , neurons 
!             a_next(i, j) = a_next(i, j ) + b(i)
!             call activation_softsign(a_next(i, j))
!           end do
!         end do 

!         ! Swap pointers, the previous output is the next input
!         if(mod(n,2) .EQ. 1) then
!           a       => a2
!           a_next  => a1  
!         else
!           a       => a1
!           a_next  => a2
!         end if

!       end do

!       wt => layers(n-1) % w_transposed
!       b  => layers(n) % b

! #ifdef USE_TIMING
!     ret =  gptlstart('(256,58) X (58,nobs)')
! #endif

!       !$acc host_data use_device(wt, a, output)
!       call sgemm("N","N", ngpt, nbatch, neurons, 1.0, wt, ngpt, a, neurons, 0.0, output, ngpt)
!       !$acc end host_data

! #ifdef USE_TIMING
!     ret =  gptlstop('(256,58) X (58,nobs)')
! #endif

!        !(Ngpoints x Nneurons) * (Nneurons x nbatch )    = (Ngpoints x nbatch)  

!       ng = 672
!       nneur = 58
    
!       allocate(wt2(ng,nneur), inp2(nneur,nbatch), output2(ng,nbatch))
!       !$acc enter data create(output2, wt2,inp2)
!       !$acc kernels
!       wt2 = 0.5_sp
!       inp2 = 0.5_sp
!       !$acc end kernels
! #ifdef USE_TIMING
!     ret =  gptlstart('(672,58) X (58,nobs)')
! #endif
!       !$acc host_data use_device(wt2, inp2, output2)
!       call sgemm("N","N", ng, nbatch, nneur, 1.0, wt2, ng, inp2, nneur, 0.0, output2, ng)
!       !$acc end host_data
! #ifdef USE_TIMING
!     ret =  gptlstop('(672,58) X (58,nobs)')
! #endif
!       !$acc exit data delete(output2,wt2,inp2)
!       deallocate(output2,wt2,inp2)

!       ng = 3
!       nneur = 3
!       allocate(wt2(ng,nneur), inp2(nneur,nbatch*ngpt), output2(ng,nbatch*ngpt))
!       !$acc enter data create(output2, wt2,inp2)
!       !$acc kernels
!       wt2 = 0.5_sp
!       inp2 = 0.5_sp
!       !$acc end kernels
! #ifdef USE_TIMING
!     ret =  gptlstart('(3,6) X (6,nobs*ngpt)')
! #endif
!       !$acc host_data use_device(wt2, inp2, output2)
!       call sgemm("N","N", ng, nbatch*ngpt, nneur, 1.0, wt2, ng, inp2, nneur, 0.0, output2, ng)
!       !$acc end host_data
! #ifdef USE_TIMING
!     ret =  gptlstop('(3,6) X (6,nobs*ngpt)')
! #endif
!       !$acc exit data delete(output2,wt2,inp2)


!     ! RTE-NN1 + RRTMGP		rte			  ref		        3*224 --> (3*1)	bb flux                   nlay*ncol
!     ! RTE-NN2 + RRTMGP		rte			  ref		        3*224 --> (3*224) gpt flux	              nlay*ncol
!     ! RTE-NN3 + RRTMGP		rte			  ref		        3*1   --> (3*1)	gpt flux	                nlay*ncol*ngpt	  
!     ! RTE-NN4 + RRTMGP		rte			  ref		        3*1   --> (4*1) reftrans scalars	        nlay*ncol*ngpt

!       n = nlayers

!       !$acc parallel loop gang default(present)
!       do j = 1, nbatch
!         !$OMP SIMD
!         !$acc loop vector
!         do i = 1, ngpt
!           ! Compute outputs and scale them to obtain molecular absorption 
!           ! output(i, j) = (ysigma*(output(i, j) + b(i)) + ymeans_lw_tau(i))**8

!           ! Scale with number of dry air molecules to obtain optical depth
!           ! output(i, j) =  output(i, j) * coldry(j)

!           ! One-line solution
!           output(i, j) = ((ysigma* (output(i, j) + b(i)) + ymeans(i))**8) * coldry(j)

!         end do
!       end do

!     end associate
!     !$acc exit data detach(a,a_next) delete(a1, a2, ymeans)

                                              
!   end subroutine

  

  ! subroutine output_sgemm_tau(self, nx, ny, nbatch, x, coldry, ymeans, ysigma, output)
  !   ! Optimized inference function for optical depth, using BLAS/cuBLAS and includes post-processing of outputs.
  !   ! This routine takes a 2D input data array for batched predictions with a feed-forward network.
  !   ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
  !   ! always in single-precision (sgemm) because why waste precision on NNs which don't need it 
  !   !
  !   !                                   Layer Weights            Layer Inputs                Layer Outputs
  !   ! First layer :                      (Nneurons x Nx)       * (Nx x nbatch )          = (Nneurons x nbatch) 
  !   ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons x nbatch )    = (Nneurons x nbatch) 
  !   ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons x nbatch )    = (Ngpoints x nbatch)  
  !   ! in GEMM terms:                         A                 *         B                = C
  !   !                                     (m x k)              *      (k * N )            = (m  * N)  
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

                                              
  ! end subroutine

  subroutine output_sgemm_pfrac(self, nx, ny, nbatch, x, output)
    ! Optimized inference function for Planck fraction, using BLAS/cuBLAS for batched prediction (many samples at a time)
    ! Includes post-processing of outputs. The inputs have already been pre-processed
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

      !$acc parallel loop collapse(2) default(present)
      do j = 1, nbatch
        do i = 1, neurons
          a(i, j) = a(i, j ) + b(i)  ! 1. add bias
          call activation_softsign(a(i, j))          ! 2. use activation function, here the softsign
        end do
      end do

      ! INTERMEDIATE LAYERS
      a_next => a2

      do n = 3, nlayers-1

        wt => layers(n-1) % w_transposed
        b  => layers(n) % b

        !$acc host_data use_device(wt, a, a_next)
        call sgemm("N","N", neurons, nbatch, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        !$acc parallel loop collapse(2) default(present)
        do j = 1, nbatch
          do i = 1 , neurons 
            a_next(i, j) = a_next(i, j ) + b(i)
            call activation_softsign(a_next(i, j))
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
      call sgemm("N","N", ny, nbatch, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      n = nlayers

      !$acc parallel loop gang default(present)
      do j = 1, nbatch
        !$acc loop vector
        do i = 1, ny
          output(i, j) = output(i, j ) + b(i)
          output(i, j) = max(0.0_sp, output(i, j)) !RELU activation
          output(i, j) = output(i, j)*output(i, j)
        end do
      end do

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
    do j=1,ncol !  50
        matvecmul = matvecmul + matA(:,j) * vecB(j) !length 256. this is for one column, and then the columns need to be added together ( b = b + ..)
    enddo

end function matvecmul

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


end module mod_network
