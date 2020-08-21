module mod_network

  use mo_rte_kind, only: sp
  use mod_layer, only: layer_type
#ifdef USE_TIMING
  ! Timing library
  use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption, &
                                   gptlpercent, gptloverhead
#endif

  implicit none
  ! real(sp), parameter :: ysigma_lw_tau = 0.0008864219_sp
  ! real(sp), parameter, dimension(256) :: ymeans_lw_tau = (/ &
  ! 0.0007013_sp, 0.00081638_sp, 0.00088407_sp, 0.0009376_sp, 0.00100272_sp,  &
  ! 0.00108871_sp, 0.00119965_sp, 0.00136057_sp, 0.00162019_sp, 0.00185225_sp,&
  ! 0.00195079_sp, 0.00206555_sp, 0.00220559_sp, 0.00240607_sp, 0.00271468_sp,&
  ! 0.00301905_sp, 0.00046654_sp, 0.00051556_sp, 0.00058341_sp, 0.00065088_sp,&
  ! 0.00070464_sp, 0.00076876_sp, 0.00085896_sp, 0.00097935_sp, 0.00119935_sp,&
  ! 0.00141034_sp, 0.00151006_sp, 0.00163442_sp, 0.00180711_sp, 0.00204042_sp,&
  ! 0.00242276_sp, 0.00280669_sp, 0.00042887_sp, 0.00046485_sp, 0.00051584_sp,&
  ! 0.00058201_sp, 0.00065572_sp, 0.00072477_sp, 0.00080584_sp, 0.00091423_sp,&
  ! 0.00110042_sp, 0.00127067_sp, 0.00134649_sp, 0.00144005_sp, 0.00155864_sp,&
  ! 0.0017359_sp, 0.00192395_sp, 0.00202425_sp, 0.00089971_sp, 0.00101015_sp,&
  ! 0.0010735_sp, 0.00113382_sp, 0.00121697_sp, 0.00131415_sp, 0.0014548_sp,&
  ! 0.00166414_sp, 0.00201406_sp, 0.0023286_sp, 0.00248189_sp, 0.0026516_sp,&
  ! 0.0028395_sp, 0.00309552_sp, 0.00329059_sp, 0.00340846_sp, 0.00040419_sp,&
  ! 0.00044149_sp, 0.00049501_sp, 0.00057646_sp, 0.00065665_sp, 0.00072959_sp,&
  ! 0.0008178_sp, 0.00092567_sp, 0.00110577_sp, 0.00127773_sp, 0.00135464_sp,&
  ! 0.00144639_sp, 0.00155456_sp, 0.00172624_sp, 0.00195144_sp, 0.00219249_sp,&
  ! 0.00037311_sp, 0.00038832_sp, 0.00039876_sp, 0.00039357_sp, 0.00039073_sp,&
  ! 0.00039439_sp, 0.00040139_sp, 0.00041544_sp, 0.00045041_sp, 0.00048733_sp,&
  ! 0.00050021_sp, 0.000511_sp, 0.0005222_sp, 0.00053226_sp, 0.00053905_sp,&
  ! 0.00053686_sp, 0.00045033_sp, 0.00050789_sp, 0.00056031_sp, 0.0006055_sp,&
  ! 0.00064596_sp, 0.0006884_sp, 0.00073842_sp, 0.00080476_sp, 0.00091462_sp,&
  ! 0.00100837_sp, 0.00104174_sp, 0.00107803_sp, 0.00112003_sp, 0.00116588_sp,&
  ! 0.00121704_sp, 0.00124574_sp, 0.00043096_sp, 0.00044609_sp, 0.00045882_sp,&
  ! 0.00047328_sp, 0.00049173_sp, 0.00051617_sp, 0.00055129_sp, 0.00060892_sp,&
  ! 0.00070759_sp, 0.00078732_sp, 0.00081574_sp, 0.00084897_sp, 0.00089626_sp,&
  ! 0.00096228_sp, 0.00103707_sp, 0.00107244_sp, 0.00044382_sp, 0.00047923_sp,&
  ! 0.00052424_sp, 0.00056825_sp, 0.00061594_sp, 0.00066949_sp, 0.00073642_sp,&
  ! 0.00083537_sp, 0.00099988_sp, 0.00115576_sp, 0.00122428_sp, 0.00130514_sp,&
  ! 0.0013967_sp, 0.00150886_sp, 0.00164095_sp, 0.00173281_sp, 0.0005398_sp,&
  ! 0.00058346_sp, 0.00061954_sp, 0.00065103_sp, 0.00069784_sp, 0.00076837_sp,&
  ! 0.00086104_sp, 0.00099005_sp, 0.00119531_sp, 0.00135744_sp, 0.00142613_sp,&
  ! 0.00151644_sp, 0.00163775_sp, 0.0017866_sp, 0.00194151_sp, 0.00203961_sp,&
  ! 0.00062092_sp, 0.00068568_sp, 0.00072482_sp, 0.00076629_sp, 0.00081684_sp,&
  ! 0.00088733_sp, 0.00098261_sp, 0.00111583_sp, 0.00132001_sp, 0.00149657_sp,&
  ! 0.00157718_sp, 0.00167974_sp, 0.00181246_sp, 0.00196578_sp, 0.00211621_sp,&
  ! 0.00222501_sp, 0.00027775_sp, 0.00031307_sp, 0.00034595_sp, 0.00037792_sp,&
  ! 0.00040985_sp, 0.00044745_sp, 0.00049738_sp, 0.00056636_sp, 0.00068384_sp,&
  ! 0.00079073_sp, 0.00083684_sp, 0.00089007_sp, 0.00094933_sp, 0.00101663_sp,&
  ! 0.00109136_sp, 0.00115981_sp, 0.00041082_sp, 0.00047212_sp, 0.0005221_sp,&
  ! 0.00057267_sp, 0.00062689_sp, 0.00068311_sp, 0.00074801_sp, 0.00083137_sp,&
  ! 0.0009249_sp, 0.00097404_sp, 0.00098701_sp, 0.0009677_sp, 0.00091862_sp,&
  ! 0.00092232_sp, 0.00097919_sp, 0.00099403_sp, 0.00082178_sp, 0.00097702_sp,&
  ! 0.00115414_sp, 0.00134279_sp, 0.00149137_sp, 0.00160743_sp, 0.00175431_sp,&
  ! 0.00198898_sp, 0.00242579_sp, 0.00281907_sp, 0.00299135_sp, 0.00319233_sp,&
  ! 0.00345435_sp, 0.00384145_sp, 0.00410627_sp, 0.00420549_sp, 0.00022183_sp,&
  ! 0.00025186_sp, 0.00027004_sp, 0.0002846_sp, 0.0003047_sp, 0.00032399_sp,&
  ! 0.00034067_sp, 0.00035919_sp, 0.00038545_sp, 0.00040524_sp, 0.00041355_sp,&
  ! 0.00042393_sp, 0.00043629_sp, 0.00045243_sp, 0.0004797_sp, 0.00050912_sp,&
  ! 0.00030737_sp, 0.00035681_sp, 0.00039763_sp, 0.00043568_sp, 0.00047533_sp,&
  ! 0.00052387_sp, 0.00058468_sp, 0.00066867_sp, 0.00081455_sp, 0.00094435_sp,&
  ! 0.00099829_sp, 0.00107121_sp, 0.00117441_sp, 0.00127705_sp, 0.00135903_sp,0.00142707 /)

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
    ! First layer :                      (Nneurons x Nx)       * (Nx x Nsample )          = (Nneurons x Nsample) 
    ! Intermediate layers :              (Nneurons x Nneurons) * (Nneurons x Nsample )    = (Nneurons x Nsample) 
    ! Final layer:                       (Ngpoints x Nneurons) * (Nneurons x Nsample )    = (Ngpoints x Nsample)  
    ! in GEMM terms:                         A                 *         B                = C
    !                                     (m x k)              *      (k * N )            = (m  * N)  
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
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer,  dimension(:),   allocatable         :: layersizes
    integer                       :: n, isample, neurons, nlayers, layersize, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)
    allocate(layersizes(nlayers))
    do i = 1, nlayers
      layersizes(i) = size(self%layers(i) % b)
    end do

    !$acc enter data create(a1, a2)    
    !$acc enter data copyin(nlayers, layersizes, neurons, nsample, nx, ny)

    !$acc data present(layersizes, x, output, a1, a2)
    associate(layers=>self%layers)
      
      wt => layers(1) % w_transposed
      a  => a1
      b => layers(2) % b

      !$acc host_data use_device(wt, x, a)
      call cublassgemm('N','N', neurons, nsample, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

      !$acc data present(a, b)
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

        wt => layers(n-1) % w_transposed

        !$acc host_data use_device(wt, a, a_next)
        call cublassgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        b => layers(n) % b

        !$acc data present(a_next, b)
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

      wt => layers(n-1) % w_transposed

      !$acc host_data use_device(wt, a, output)
      call cublassgemm("N","N", ny, nsample, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      n = nlayers

      b => layers(n) % b

      !$acc parallel loop collapse(2) present(b)
      do isample = 1, nsample
        do i = 1, layersizes(n)
          ! Compute outputs and scale them to obtain molecular absorption 
          output(i, isample) = (ysigma_lw_tau*(output(i, isample)+b(i)) + ymeans_lw_tau(i))**8
          ! Scale with number of dry air molecules to obtain optical depth
        end do
      end do

    end associate
    !$acc end data 
                                           
    !$acc exit data delete(nlayers, layersizes, neurons, nsample, nx, ny)
    !$acc exit data delete(a1, a2, a, a_next)
    
  end subroutine

  subroutine output_sgemm_pfrac_flat_acc(self, nx, ny, nsample, x, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! Assuming "flat model" i.e. the hidden layers have the same number of neurons
    ! sgemm = single-precision (sp = sp)
    use cublas 
    use openacc
    class(network_type),              intent(in), target  :: self
    integer, intent(in)                           :: nx, ny, nsample
    real(sp), dimension(nx, nsample), intent(in)  :: x      ! (features, nsample)
    real(sp), dimension(ny, nsample), intent(out) :: output ! (outputs, nsample) 
    real(sp), dimension(size(self % layers(1) % w_transposed, 1), nsample), &
                                          target  :: a1, a2  
    real(sp), dimension(:,:), contiguous, pointer :: a, a_next  
    real(sp), dimension(:,:), contiguous, pointer :: wt
    real(sp), dimension(:),   contiguous, pointer :: b
    integer,  dimension(:),   allocatable         :: layersizes
    integer                       :: n, isample, neurons, nlayers, layersize, i

    neurons = size(self % layers(1) % w_transposed, 1)
    nlayers = size(self % layers)
    allocate(layersizes(nlayers))
    do i = 1, nlayers
      layersizes(i) = size(self%layers(i) % b)
    end do

    !$acc enter data create(a1, a2)                                              
    !$acc enter data copyin(nlayers, layersizes, neurons, nsample, nx, ny)

    !$acc data present(layersizes, x, output, a1, a2)
    associate(layers=>self%layers)
      
      wt => layers(1) % w_transposed
      a  => a1
      b => layers(2) % b

      !$acc host_data use_device(wt, x, a)
      call cublassgemm('N','N', neurons, nsample, nx, 1.0, wt, neurons, x, nx, 0.0, a, neurons)
      !$acc end host_data

      !$acc data present(a, b)
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

        wt => layers(n-1) % w_transposed

        !$acc host_data use_device(wt, a, a_next)
        call cublassgemm("N","N", neurons, nsample, neurons, 1.0, wt, neurons, a, neurons, 0.0, a_next, neurons)
        !$acc end host_data

        b => layers(n) % b

        !$acc data present(a_next, b)
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

      wt => layers(n-1) % w_transposed

      !$acc host_data use_device(wt, a, output)
      call cublassgemm("N","N", ny, nsample, neurons, 1.0, wt, ny, a, neurons, 0.0, output, ny)
      !$acc end host_data

      n = nlayers

      b => layers(n) % b

      !$acc parallel loop gang vector collapse(2) present(b)
      do isample = 1, nsample
        do i = 1, layersizes(n)
          output(i, isample) = output(i, isample ) + b(i)
          output(i, isample) = max(0.0_sp, output(i, isample))
          output(i, isample) = output(i, isample)*output(i, isample)
        end do
      end do

      end associate
      !$acc end data 

    !$acc exit data delete(nlayers, layersizes, neurons, nsample, nx, ny)
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
          a_next(i, isample) = a_next(i, isample ) + layers(n) % b(i)
          call softsignn(a_next(i, isample))
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
        output(i, isample) = output(i, isample ) + layers(n) % b(i)
        call reluu(output(i, isample))
        output(i, isample) = output(i, isample)*output(i, isample)
      end do

      end associate
  end subroutine

subroutine output_sgemm_tau(self, nx, ny, nsample, x, coldry, ymeans, ysigma, output)
    ! Use this routine for a 2D input data array to process all the samples simultaenously in a feed-forward network.
    ! Assuming "flat model" i.e. all hidden layers have the same number of neurons
    ! sgemm = single-precision (sp = sp)
    ! This procedure for predicting optical depths includes post-processing of outputs. 

    class(network_type),              intent(in)  :: self
    integer, intent(in)                           :: nx, ny, nsample
    real(sp), dimension(nx, nsample), intent(in)  :: x        ! (features, nsample)
    real(sp), dimension(nsample),     intent(in)  :: coldry      
    real(sp), dimension(ny),          intent(in)  :: ymeans
    real(sp),                         intent(in)  :: ysigma
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
            a_next(i, isample) = a_next(i, isample ) + layers(n) % b(i)
            call softsignn(a_next(i, isample))
        end do 
#ifdef USE_TIMING
      ret =  gptlstop('add_signal_bias_and_activation')
#endif
      a = a_next

      end do
#ifdef USE_TIMING
      ret =  gptlstart('sgemm_tau_lastlayer')
#endif
      call sgemm("N","N",ny, nsample, neurons, alpha, layers(n-1) % w_transposed, ny, a, neurons, beta, output, ny)
#ifdef USE_TIMING
      ret =  gptlstop('sgemm_tau_lastlayer')
      ret =  gptlstart('add_output_bias_and_scale')
#endif
      layersize = size(layers(n) % b)
      !dir$ vector aligned
      do concurrent (isample = 1 : nsample, i = 1 : layersize)
        !output(i, isample) = output(i, isample ) + layers(n) % b(i)
        !output(i, isample) = (ysigma_lw_tau*output(i, isample) + ymeans_lw_tau(i))**8
        !output(i, isample) = output(i, isample) *col_dry_wk(ilay,icol)

        ! In one line: add bias for get model output, standard-scale and power-scale to obtain molecular absorption, 
        ! and finally multiply with coldry to get optical depth
        output(i, isample) = ((ysigma* (output(i, isample) + layers(n) % b(i)) + ymeans(i))**8) *coldry(isample)
      end do
#ifdef USE_TIMING
      ret =  gptlstop('add_output_bias_and_scale')
#endif      
      end associate
  end subroutine

#endif  
! using openACC endif

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
