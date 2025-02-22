module mod_io

  use mo_rte_kind, only: wp

  implicit none

  private

  public :: read_binary_file

  interface read_binary_file
    module procedure :: read_binary_file_1d, read_binary_file_2d
  end interface read_binary_file

contains

  subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: dtype, nrec
    real(wp), allocatable, intent(in out) :: array(:)
    integer :: fileunit
    allocate(array(nrec))
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * nrec, status='old')
    read(fileunit, rec=1) array
    close(fileunit)
  end subroutine read_binary_file_1d

  subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
    character(len=*), intent(in) :: filename
    integer, intent(in) :: dtype, dsize, nrec
    real(wp), allocatable, intent(in out) :: array(:,:)
    integer :: fileunit, i
    allocate(array(dsize, nrec))
    open(newunit=fileunit, file=filename, access='direct',&
         action='read', recl=dtype * dsize, status='old')
    do i = 1, nrec
      read(fileunit, rec=i) array(:,i)
    end do
    close(fileunit)
  end subroutine read_binary_file_2d

end module mod_io
