module mod_network

    use mo_rte_kind, only: sp
    use mod_layer, only: layer_type
  #ifdef USE_TIMING
    ! Timing library
    use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                     gptlpercent, gptloverhead
  #endif
  
    implicit none
    real(sp), parameter :: tau_sigma = 0.7591194_sp
    real(sp), parameter, dimension(256) :: tau_gpt_means = (/ 0.67_sp, 0.78_sp, 0.84_sp, 0.9_sp, &
    0.96_sp, 1.04_sp, 1.15_sp, 1.3_sp, 1.53_sp, 1.74_sp, 1.82_sp, &
    1.92_sp, 2.03_sp, 2.18_sp, 2.4_sp, 2.6_sp, 0.45_sp, 0.5_sp, 0.56_sp, 0.63_sp, 0.68_sp, 0.74_sp, &
    0.83_sp, 0.94_sp, 1.15_sp, 1.34_sp, 1.43_sp, 1.54_sp, 1.69_sp, 1.89_sp, 2.19_sp, 2.46_sp, 0.41_sp, &
    0.43_sp, 0.47_sp, 0.52_sp, 0.57_sp, 0.63_sp, 0.7_sp, 0.79_sp, 0.93_sp, 1.07_sp, 1.13_sp, 1.2_sp,&
    1.28_sp, 1.41_sp, 1.53_sp, 1.59_sp, 0.76_sp, 0.85_sp, 0.9_sp, 0.95_sp, 1.01_sp, 1.09_sp, 1.2_sp,&
    1.36_sp, 1.62_sp, 1.85_sp, 1.97_sp, 2.09_sp, 2.22_sp, 2.36_sp, 2.47_sp, 2.53_sp, 0.36_sp, 0.39_sp,&
    0.43_sp, 0.49_sp, 0.56_sp, 0.62_sp, 0.69_sp, 0.77_sp, 0.91_sp, 1.04_sp, 1.1_sp, 1.16_sp, 1.24_sp,&
    1.35_sp, 1.49_sp, 1.64_sp, 0.33_sp, 0.34_sp, 0.35_sp, 0.34_sp, 0.34_sp, 0.34_sp, 0.35_sp, 0.36_sp,&
    0.4_sp, 0.43_sp, 0.45_sp, 0.46_sp, 0.47_sp, 0.49_sp, 0.5_sp, 0.5_sp, 0.38_sp, 0.42_sp, 0.46_sp,&
    0.49_sp, 0.52_sp, 0.55_sp, 0.58_sp, 0.63_sp, 0.7_sp, 0.77_sp, 0.79_sp, 0.81_sp, 0.85_sp, 0.88_sp,&
    0.93_sp, 0.95_sp, 0.37_sp, 0.38_sp, 0.39_sp, 0.4_sp, 0.41_sp, 0.43_sp, 0.46_sp, 0.5_sp, 0.58_sp,&
    0.65_sp, 0.67_sp, 0.7_sp, 0.74_sp, 0.8_sp, 0.86_sp, 0.88_sp, 0.38_sp, 0.42_sp, 0.46_sp, 0.5_sp,&
    0.55_sp, 0.59_sp, 0.65_sp, 0.74_sp, 0.88_sp, 1.01_sp, 1.07_sp, 1.14_sp, 1.21_sp, 1.31_sp, 1.44_sp,&
    1.53_sp, 0.52_sp, 0.56_sp, 0.59_sp, 0.62_sp, 0.67_sp, 0.74_sp, 0.82_sp, 0.95_sp, 1.13_sp, 1.27_sp,&
    1.33_sp, 1.4_sp, 1.49_sp, 1.59_sp, 1.71_sp, 1.77_sp, 0.59_sp, 0.65_sp, 0.69_sp, 0.73_sp, 0.78_sp,&
    0.84_sp, 0.94_sp, 1.06_sp, 1.24_sp, 1.39_sp, 1.46_sp, 1.54_sp, 1.64_sp, 1.74_sp, 1.85_sp, 1.93_sp,&
    0.28_sp, 0.32_sp, 0.36_sp, 0.39_sp, 0.42_sp, 0.46_sp, 0.51_sp, 0.58_sp, 0.7_sp, 0.81_sp, 0.85_sp,&
    0.91_sp, 0.97_sp, 1.03_sp, 1.11_sp, 1.18_sp, 0.35_sp, 0.4_sp, 0.44_sp, 0.48_sp, 0.53_sp, 0.57_sp,&
    0.62_sp, 0.69_sp, 0.77_sp, 0.82_sp, 0.83_sp, 0.83_sp, 0.81_sp, 0.83_sp, 0.88_sp, 0.91_sp, 0.69_sp,&
    0.82_sp, 0.96_sp, 1.12_sp, 1.24_sp, 1.34_sp, 1.45_sp, 1.64_sp, 1.97_sp, 2.26_sp, 2.37_sp, 2.5_sp,&
    2.67_sp, 2.89_sp, 3.04_sp, 3.1_sp, 0.22_sp, 0.25_sp, 0.27_sp, 0.28_sp, 0.3_sp, 0.32_sp, 0.34_sp,&
    0.35_sp, 0.38_sp, 0.39_sp, 0.4_sp, 0.41_sp, 0.42_sp, 0.44_sp, 0.46_sp, 0.49_sp, 0.28_sp, 0.32_sp,&
    0.36_sp, 0.39_sp, 0.43_sp, 0.47_sp, 0.52_sp, 0.6_sp, 0.72_sp, 0.83_sp, 0.87_sp, 0.93_sp, 1._sp,&
    1.08_sp, 1.14_sp, 1.19_sp /)
  
  #ifdef USE_TIMING
    integer, private :: ret, i
  #endif
    !private
    public! :: network_type
  
    type :: network_type
  
      type(layer_type), allocatable :: layers(:)
      integer, allocatable :: dims(:)
      !!  !$acc policy<copynetwork> copyin(layers, dims)
      procedure(kernel_interface),      pointer :: nn_kernel
      procedure(kernel_interface_m),    pointer :: nn_kernel_m
  
    contains
  
      procedure, public :: change_kernel
      procedure, public, pass(self) :: init
      procedure, public, pass(self) :: load
      procedure, public, pass(self) :: output_opt, output_opt_flatmodel       ! Vector input, matrix-vector product
      procedure, public, pass(self) :: output_matmul_flatmodel                ! Matrix input, matrix-matrix product
  #ifdef USE_OPENACC
      procedure, public, pass(self) :: output_sgemm_pfrac_flat_acc, output_sgemm_tau_flat_acc
  #else
      procedure, public, pass(self) :: output_sgemm   ! Matrix input, matrix-matrix product using BLAS
      procedure, public, pass(self) :: output_sgemm_pfrac, output_sgemm_tau, output_sgemv_flatmodel
  #endif
      procedure, public, pass(self) :: save
      procedure, public, pass(self) :: set_activation
      !procedure, public, pass(self) :: sync
  
  
    end type network_type
  
    interface network_type
      module procedure net_constructor
    endinterface network_type
  
    abstract interface
      pure subroutine kernel_interface(self, x, output)
        import network_type, sp
        class(network_type),    intent(in)    :: self
        real(sp), dimension(:), intent(in)    :: x
        real(sp), dimension(:), intent(out) :: output
      end subroutine
  
      subroutine kernel_interface_m(self, nx, ny, nsample, x, output)
      import network_type, sp
      class(network_type),              intent(in)      :: self
      integer,                          intent(in)      :: nx,ny,nsample
      real(sp), dimension(nx, nsample), intent(in)      :: x      ! (features, nsample)
      real(sp), dimension(ny, nsample), intent(out)   :: output ! (outputs, nsample)
      end subroutine
    end interface
  
  contains
  
    type(network_type) function net_constructor(dims) result(net)
      ! network class constructor. Size of input array dims indicates the total
      ! number of layers (input + hidden + output), and the value of its elements
      ! corresponds the size of each layer.
      integer, intent(in) :: dims(:)
      ! character(len=*), intent(in), optional :: activation
      call net % init(dims)
      call net % set_activation('sigmoid')
      ! end if
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
  
      self % nn_kernel   => output_opt
  #ifndef USE_OPENACC
      self % nn_kernel_m => output_sgemm ! Would be more consistent to use output_matmul but this one is generally very slow..
  #endif
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
     ! !$acc enter data copyin(self) 
     ! !$acc enter data copyin(self % dims)  
     ! !$acc enter data copyin(self % layers)
      do n = 2, size(self % dims)
        read(fileunit, fmt=*) self % layers(n) % b
        ! !$acc enter data copyin(self % layers(n) % b)
      end do
      
      do n = 1, size(self % dims) - 1
        read(fileunit, fmt=*) self % layers(n) % w
        self % layers(n) % w_transposed = transpose(self % layers(n) % w )   
       ! !$acc enter data copyin(self % layers(n) % w_transposed)    
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
      real(sp), dimension(size(self % layers(1) % w_transposed,1))        :: a 
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
  
  #ifdef USE_OPENACC
  
    subroutine output_sgemm_tau_flat_acc(self, nx, ny, nsample, x, output)
      ! Inference function for tau, using cuBLAS and includes post-processing of outputs.
      !
      !                                   Layer Weights            Layer Inputs                Layer Outputs
      ! First layer :                      (Nneurons x Nx)       * (Nx * Nsample )           = (Nneurons * Nsample) 
      ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons * Nsample )     = (Nneurons * Nsample) 
      ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons * Nsample )     = (Ngpoints  * Nsample)  
      use cublas 
      use openacc
      use, intrinsic :: iso_c_binding
      class(network_type),              intent(in), target  :: self
      integer, intent(in)                           :: nx, ny, nsample
      real(sp), dimension(nx, nsample), intent(in)  :: x      ! (features, nsample)
      real(sp), dimension(ny, nsample), intent(out) :: output ! (outputs, nsample) 
      real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample), &
                                            target  :: a1, a2  
      real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
      !real(sp), dimension(:,:), contiguous, pointer :: wt
      real(sp), dimension(:),   allocatable         :: b, b_last
      real(sp), dimension(:,:), allocatable          :: wt, wt_first, wt_last, ax
      integer,  dimension(:),   allocatable         :: layersizes
      integer                       :: n, isample, neurons, nlayers, layersize, i, i1, i2, i3, i4
  
      neurons = size(self % layers(1) % w_transposed, 1)
      nlayers = size(self % layers)
      allocate(layersizes(nlayers))
      do i = 1, nlayers
        layersizes(i) = size(self%layers(i) % b)
      end do
      allocate(b(neurons))
      allocate(b_last(ny))
      allocate(wt_first(neurons, nx))
      allocate(wt(neurons,neurons))
      allocate(wt_last(ny,neurons))
      allocate(ax(neurons,nsample))
  
      !$acc enter data create(a1, a2, output, ax)    
      !$acc enter data copyin(nlayers, layersizes, neurons, nsample, nx, ny)
  
      !$acc data present(layersizes, x, output, a1, a2)
      associate(layers=>self%layers)
        
        !wt => layers(1) % w_transposed
        wt_first = layers(1) % w_transposed
        a  => a1
        !b => layers(2) % b
        b = layers(2) % b
  
        !$acc enter data attach(a)
        !$acc data present(a) copyin(b, wt_first)
        !$acc host_data use_device(wt_first, x, ax)
        call cublassgemm('N','N', neurons, nsample, nx, 1.0, wt_first, neurons, x, nx, 0.0, ax, neurons)
        !$acc end host_data
  
        !! !$acc data present(a) copyin(b)
        !$acc parallel loop gang vector collapse(2)
        do isample = 1, nsample
          do i = 1, layersizes(2)  
            a(i, isample) = a(i, isample ) + b(i)
            call softsignn(a(i, isample))
          end do
        end do
        !$acc end data
  
        print *, "HEJJJ"
  
        ! ! INTERMEDIATE LAYERS
        ! a_next => a2
        ! !$acc enter data attach(a_next)
        ! do n = 3, nlayers-1
  
        !   ! wt => layers(n-1) % w_transposed
        !   wt = layers(n-1) % w_transposed
        !   b = layers(n) % b
  
        !   !$acc data present(a,a_next) copyin(b, wt)
        !   print *, "loop n:", n, "a(1,1)", a(1,1), "a_next(1,1)", a_next(1,1)
        !   !$acc host_data use_device(wt, a, a_next)
        !   call cublassgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !   !$acc end host_data
  
        !   print *, "HEJJJ2"
  
  
        !   !b => layers(n) % b
      
  
        !   ! 
        !   !$acc parallel loop gang vector collapse(2) 
        !   do isample = 1, nsample
        !     do i = 1 , layersizes(n)  
        !       a_next(i, isample) = a_next(i, isample ) + b(i)
        !       call softsignn(a_next(i, isample))
        !     end do
        !   end do 
        !   !$acc end data
  
        !   ! Swap pointers
        !   if(mod(n,2) .EQ. 1) then
        !     a       => a2
        !     a_next  => a1  
        !   else
        !     a       => a1
        !     a_next  => a2
        !   end if
        !   !$acc enter data attach(a,a_next)
        !   print *, "HEJJJ2.1"
  
        ! end do
  
        ! ! wt => layers(n-1) % w_transposed
        ! n = nlayers
  
        ! wt_last = layers(n-1) % w_transposed
        ! b_last = layers(n) % b
  
        ! !$acc data copyin(b_last, wt_last) 
        ! !$acc host_data use_device(wt, a, output)
        ! call cublassgemm("N","N", ny, nsample, neurons, 1.0, wt_last, ny, a, neurons, 0.0, output, ny)
        ! !$acc end host_data
  
        ! print *, "HEJJJ3"
  
        ! !n = nlayers
        ! !b => layers(n) % b
    
        
        ! !$acc parallel loop gang vector collapse(2)
        ! do isample = 1, nsample
        !   do i = 1, layersizes(n)
        !     output(i, isample) = (tau_sigma*(output(i, isample)+b_last(i)) + tau_gpt_means(i))**8
        !   end do
        ! end do
        ! !$acc end data
  
      end associate
      !$acc end data 
                                             
      !$acc exit data delete(nlayers, layersizes, neurons, nsample, nx, ny, b, b_last, wt, wt_first, wt_last)
      !$acc exit data delete(a1, a2, a, a_next)
      
    end subroutine
  
    subroutine output_sgemm_pfrac_flat_acc(self, nx, ny, nsample, x, output)
      ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
      ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
      ! sgemm = single-precision (sp = sp)
      use cublas 
      use openacc
      use, intrinsic :: iso_c_binding
      class(network_type),              intent(in), target  :: self
      integer, intent(in)                           :: nx, ny, nsample
      real(sp), dimension(nx, nsample), intent(in)  :: x      ! (features, nsample)
      real(sp), dimension(ny, nsample), intent(out) :: output ! (outputs, nsample) 
      real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample), &
                                            target  :: a1, a2  
      real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
      !real(sp), dimension(:,:), contiguous, pointer :: wt
      real(sp), dimension(:),   allocatable         :: b, b_last
      real(sp), dimension(:,:), allocatable          :: wt, wt_first, wt_last
      integer,  dimension(:),   allocatable         :: layersizes
      integer                       :: n, isample, neurons, nlayers, layersize, i
  
      neurons = size(self % layers(1) % w_transposed, 1)
      nlayers = size(self % layers)
      allocate(layersizes(nlayers))
      do i = 1, nlayers
        layersizes(i) = size(self%layers(i) % b)
      end do
      allocate(b(neurons))
      allocate(b_last(ny))
      allocate(wt_first(neurons, nx))
      allocate(wt(neurons,neurons))
      allocate(wt_last(ny,neurons))
  
      !$acc enter data create(a1, a2, output)    
      !$acc enter data copyin(nlayers, layersizes, neurons, nsample, nx, ny)
  
      !$acc data present(layersizes, x, output, a1, a2)
      associate(layers=>self%layers)
        
        !wt => layers(1) % w_transposed
        wt_first = layers(1) % w_transposed
        a  => a1
        !b => layers(2) % b
        b = layers(2) % b
  
        !$acc enter data attach(a)
        !$acc data present(a) copyin(b, wt_first)
        !$acc host_data use_device(wt_first, x, a)
        call cublassgemm('N','N', neurons, nsample, nx, 1.0, wt_first, neurons, x, nx, 0.0, a, neurons)
        !$acc end host_data
  
        !! !$acc data present(a) copyin(b)
        !$acc parallel loop gang vector collapse(2)
        do isample = 1, nsample
          do i = 1, layersizes(2)  
            a(i, isample) = a(i, isample ) + b(i)
            call softsignn(a(i, isample))
          end do
        end do
        !$acc end data
  
        ! INTERMEDIATE LAYERS
        a_next => a2
        do n = 3, nlayers-1
  
          ! wt => layers(n-1) % w_transposed
          wt = layers(n-1) % w_transposed
          b = layers(n) % b
  
          !$acc enter data attach(a_next)
          !$acc data present(a_next) copyin(b, wt)
          !$acc host_data use_device(wt, a, a_next)
          call cublassgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
          !$acc end host_data
  
          !b => layers(n) % b
      
  
          ! 
          !$acc parallel loop gang vector collapse(2) 
          do isample = 1, nsample
            do i = 1 , layersizes(n)  
              a_next(i, isample) = a_next(i, isample ) + b(i)
              call softsignn(a_next(i, isample))
            end do
          end do 
          !$acc end data
  
          ! Swap pointers
          if(mod(n,2) .EQ. 1) then
            a       => a2
            a_next  => a1  
          else
            a       => a1
            a_next  => a2
          end if
  
        end do
  
        ! wt => layers(n-1) % w_transposed
        n = nlayers
  
        wt_last = layers(n-1) % w_transposed
        b_last = layers(n) % b
  
        !$acc data present(output) copyin(b_last, wt_last) 
        !$acc host_data use_device(wt, a, output)
        call cublassgemm("N","N", ny, nsample, neurons, 1.0, wt_last, ny, a, neurons, 0.0, output, ny)
        !$acc end host_data
  
        !n = nlayers
        !b => layers(n) % b
    
        !$acc parallel loop gang vector collapse(2)
        do isample = 1, nsample
          do i = 1, layersizes(n)
            output(i, isample) = output(i, isample ) + b_last(i)
            output(i, isample) = max(0.0_sp, output(i, isample))
            output(i, isample) = output(i, isample)*output(i, isample)
          end do
        end do
        !$acc end parallel
        !$acc end data
  
  
        end associate
        !$acc end data 
  
      !$acc exit data delete(nlayers, layersizes, neurons, nsample, nx, ny, b, b_last, wt, wt_first, wt_last)
      !$acc exit data delete(a1, a2, a, a_next)
  
    end subroutine
  
  #else
  
    subroutine output_sgemv_flatmodel(self, x, output)
      class(network_type),    intent(in)  :: self
      real(sp), dimension(:), intent(in)  :: x
      real(sp), dimension(:), intent(out) :: output
      ! Local variables
      ! The signal/tensor passing through the network
      real(sp), dimension(size(self % layers(1) % w_transposed,1))        :: a, c
      integer :: n, incx, incy, neurons
      real(sp) :: alpha,beta
      alpha = 1.0_sp
      beta = 0.0_sp
      incx = 1
      incy = 1
  
      neurons = size(self % layers(1) % w_transposed, 1)
  
      associate(layers => self % layers)
        call sgemv("N", neurons, size(x), alpha, layers(1) % w_transposed, neurons, x, incx, beta, a, incy)
        a = a + layers(2) % b
        call layers(2) % activation(a)
        ! INTERMEDIATE LAYERS
        do n = 3, size(layers)-1
          ! to avoid having to allocate another output array c (of size neurons), don't use sgemv here
          ! For deep neural networks with more than 2-3 hidden layers, it's probably worth using sgemv
          !a = matvecmul(layers(n-1) % w_transposed, a, neurons, neurons)
          call sgemv("N",neurons,neurons,alpha,layers(n-1) % w_transposed,neurons,a,incx,beta,c,incy)
          ! a = a + layers(n) % b
          a = c + layers(n) % b
          call layers(n) % activation(a)
        end do
        ! LAST LAYER (LINEAR ACTIVATION = do nothing, just add biases)
        call sgemv("N",size(output), neurons, alpha, layers(n-1) % w_transposed, size(output), a, incx, beta, output, incy)
        output = output + layers(n) % b
        call layers(n) % activation(output)
      end associate
    end subroutine
  
    subroutine output_sgemm(self, nx, ny, nsample, x, output)
      ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
      ! Using BLAS for the matrix-matrix computations
      ! sgemm = single-precision (sp)
      class(network_type),    intent(in)          :: self
      integer,                intent(in)          :: nx,ny,nsample
      real(sp), dimension(nx, nsample), &
                              intent(in)          :: x      ! (features, nsample)
      real(sp), dimension(ny, nsample), &
                              intent(out)       :: output ! (outputs, nsample)
      ! Local variables
      real(sp), allocatable   :: a(:,:), a_next(:,:)
      real(sp)                :: alpha, beta
      integer,  dimension(2)  :: matsize
      integer                 :: n, isample, neurons
  
      alpha = 1.0_sp
      beta = 0.0_sp
      output = 0.0_sp
  
      associate(layers => self % layers)
        matsize = shape(layers(1) % w_transposed)
        allocate(a(matsize(1),nsample))
        ! Multiply weights with the inputs (matrix-matrix dot-product) using BLAS 
        call sgemm("N","N",matsize(1), nsample, matsize(2), alpha, layers(1) % w_transposed, matsize(1), x, matsize(2), beta, a, matsize(1))
  
        do isample = 1, nsample
          a(:,isample) = a(:,isample ) + layers(2) % b  ! Add biases of first layer
          call layers(2) % activation(a(:,isample))     ! Use activation function of first layer
        end do
  
        ! INTERMEDIATE LAYERS
        do n = 3, size(layers)-1
          matsize = shape(layers(n-1) % w_transposed)
          allocate(a_next(matsize(1),nsample))
          call sgemm("N","N",matsize(1),nsample,matsize(2),alpha,layers(n-1) % w_transposed,matsize(1),a,matsize(2),beta,a_next,matsize(1))
          deallocate(a)
          do isample = 1, nsample
            a_next(:,isample) = a_next(:,isample ) + layers(n) % b  ! Add biases
            call layers(n) % activation(a_next(:,isample))          ! Activation 
          end do 
          a = a_next
          deallocate(a_next)
        end do
  
        matsize = shape(layers(n-1) % w_transposed)
        call sgemm("N","N",matsize(1), nsample, matsize(2), alpha, layers(n-1) % w_transposed, matsize(1), a, matsize(2), beta, output, matsize(1))
        do isample = 1, nsample
            output(:,isample) = output(:,isample ) + layers(n) % b ! Add biases
            call layers(n) % activation(output(:,isample))         ! Activation of the final layer
        end do
      end associate
    end subroutine
  
  subroutine output_sgemm_pfrac(self, nx, ny, nsample, x, output)
      ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
      ! Assuming "flat model" i.e. all hidden layers have the same number of neurons
      ! sgemm = single-precision (sp = sp)
      class(network_type),      intent(in)    :: self
      integer, intent(in)                     :: nx, ny, nsample
      real(sp), dimension(nx, nsample), &
                                intent(in)    :: x      ! (features, nsample)
      real(sp), dimension(ny, nsample), &
                                intent(out)   :: output ! (outputs, nsample)
      ! Local variables
      real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample) &
                                              :: a, a_next
  
      real(sp)                                :: alpha, beta
      integer                                 :: n, isample, i, neurons, layersize
  
      alpha   = 1.0_sp
      beta    = 0.0_sp
  
      neurons = size(self % layers(1) % w_transposed, 1)
  
      associate(layers => self % layers)
  #ifdef USE_TIMING
        ret =  gptlstart('sgemm_pfrac')
  #endif
        ! First layer: multiply input matrix with weights in the first layer
        call sgemm("N","N", neurons, nsample, nx, alpha, layers(1) % w_transposed, neurons, x, nx, beta, a, neurons)
  #ifdef USE_TIMING
        ret =  gptlstop('sgemm_pfrac')
  #endif
        ! Add biases and use activation function
        layersize = size(layers(2) % b)
        !dir$ vector aligned
        do concurrent (isample = 1 : nsample, i = 1 : layersize)
          ! do concurrent (i = 1 : layersize)    
            a(i, isample) = a(i, isample ) + layers(2) % b(i)
            call softsignn(a(i, isample))
          ! end do
        end do
  
        ! Intermediate layers: in each layer, multiply the signal matrix a with weights in that layer, add biases, and activation
        do n = 3, size(layers)-1
  #ifdef USE_TIMING
          ret =  gptlstart('sgemm_pfrac')
  #endif
          call sgemm("N","N", neurons,nsample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a_next, neurons)
  #ifdef USE_TIMING
          ret =  gptlstop('sgemm_pfrac')
  #endif
          layersize = size(layers(n) % b)
          !dir$ vector aligned
          do concurrent (isample = 1 : nsample, i = 1 : layersize)
            ! do concurrent (i = 1 : layersize)  
              a_next(i, isample) = a_next(i, isample ) + layers(n) % b(i)
              call softsignn(a_next(i, isample))
            ! end do
          end do 
          a = a_next
        end do
  #ifdef USE_TIMING
        ret =  gptlstart('sgemm_pfrac')
  #endif
        call sgemm("N","N",ny, nsample, neurons, alpha, layers(n-1) % w_transposed, ny, a, neurons, beta, output, ny)
  #ifdef USE_TIMING
        ret =  gptlstop('sgemm_pfrac')
  #endif
        layersize = size(layers(n) % b)
  
        !dir$ vector aligned
        do concurrent (isample = 1 : nsample, i = 1 : layersize)  
          ! do concurrent (i = 1 : layersize)  
            output(i, isample) = output(i, isample ) + layers(n) % b(i)
            call reluu(output(i, isample))
            output(i, isample) = output(i, isample)**2
          ! end do
        end do
  
        ! print *, "max,min PFRAC", maxval(output), minval(output)
  
        end associate
    end subroutine
  
  subroutine output_sgemm_tau(self, nx, ny, nsample, x, output)
      ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
      ! Assuming "flat model" i.e. all hidden layers have the same number of neurons
      ! sgemm = single-precision (sp = sp)
      ! This procedure for predicting optical depths includes post-processing of outputs. 
  
      class(network_type),              intent(in)  :: self
      integer, intent(in)                           :: nx, ny, nsample
      real(sp), dimension(nx, nsample), intent(in)  :: x        ! (features, nsample)
      real(sp), dimension(ny, nsample), intent(out) :: output ! (outputs, nsample)
  
      ! LOCAL VARIABLES
      ! The "signal" i.e. output of hidden layers, assumed to not change shape
      real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample) &
                                                    :: a, a_next
      
      real(sp)                :: alpha, beta
      integer                 :: n, isample, neurons, layersize, i
  
      neurons = size(self % layers(1) % w_transposed, 1)
      alpha   = 1.0_sp
      beta    = 0.0_sp
  
      associate(layers => self % layers)
  #ifdef USE_TIMING
        ret =  gptlstart('sgemm_tau')
  #endif
        call sgemm("N","N",neurons, nsample, nx, alpha, layers(1) % w_transposed, neurons, x, nx, beta, a, neurons)
        ! First layer :  (nneur x nx) * (nx * nsamp )   = (nneur * nsamp)   
  
  #ifdef USE_TIMING
        ret = gptlstop('sgemm_tau')
        ret = gptlstart('add_signal_bias_and_activation')
  #endif
        layersize = size(layers(2) % b)
        
        !dir$ vector aligned
        do concurrent (isample = 1 : nsample, i = 1 : layersize)
          a(i, isample) = a(i, isample ) + layers(2) % b(i)
          call softsignn(a(i, isample))
        end do
  
  #ifdef USE_TIMING
        ret =  gptlstop('add_signal_bias_and_activation')
  #endif
        ! INTERMEDIATE LAYERS
        do n = 3, size(layers)-1
  #ifdef USE_TIMING
        ret =  gptlstart('sgemm_tau')
  #endif
          call sgemm("N","N",neurons,nsample,neurons,alpha,layers(n-1) % w_transposed, neurons, a, neurons, beta, a_next, neurons)
  #ifdef USE_TIMING
        ret =  gptlstop('sgemm_tau')
        ret =  gptlstart('add_signal_bias_and_activation')
  #endif
          layersize = size(layers(n) % b)
          !dir$ vector aligned
          do concurrent (isample = 1 : nsample, i = 1 : layersize)
            ! do concurrent (i = 1 : layersize)  
              a_next(i, isample) = a_next(i, isample ) + layers(n) % b(i)
              call softsignn(a_next(i, isample))
            ! end do
          end do 
  #ifdef USE_TIMING
        ret =  gptlstop('add_signal_bias_and_activation')
  #endif
        a = a_next
  
        end do
  #ifdef USE_TIMING
        ret =  gptlstart('sgemm_tau')
  #endif
        call sgemm("N","N",ny, nsample, neurons, alpha, layers(n-1) % w_transposed, ny, a, neurons, beta, output, ny)
  #ifdef USE_TIMING
        ret =  gptlstop('sgemm_tau')
        ret =  gptlstart('add_output_bias_and_scale')
  #endif
        layersize = size(layers(n) % b)
        !dir$ vector aligned
        do concurrent (isample = 1 : nsample, i = 1 : layersize)
          ! do concurrent (i = 1 : layersize)  
            output(i, isample) = output(i, isample ) + layers(n) % b(i)
            output(i, isample) = (tau_sigma*output(i, isample) + tau_gpt_means(i))**8
          ! end do
        end do
  #ifdef USE_TIMING
        ret =  gptlstop('add_output_bias_and_scale')
  #endif      
        end associate
    end subroutine
  
  #endif  
  ! if using openACC endif
  
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
  