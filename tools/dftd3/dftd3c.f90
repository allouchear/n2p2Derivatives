! Simple demonstration of how to use dftd3 as a library.
!
! Copyright (C) 2016, Bálint Aradi
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 1, or (at your option)
! any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! For the GNU General Public License, see <http://www.gnu.org/licenses/>
!

! Tests the dftd3 API by calculating the dispersion for a small DNA fragment.
!
subroutine dftd3c(instr, version, nAtoms, atnum, ccoords, edisp, cgrads)
  use dftd3_api
  use, intrinsic :: iso_c_binding
  implicit none
  character(kind=c_char), dimension(*), intent(IN) :: instr
  integer, parameter :: wp = kind(1.0d0)
  integer :: i,j,k
  integer :: nAtoms
  integer :: version
  integer :: atnum(nAtoms)
  real(wp) :: ccoords(3*nAtoms)
  real(wp) :: cgrads(3*nAtoms)

  real(wp) :: stress(3,3)
  real(wp), allocatable :: grads(:,:)
  real(wp), allocatable :: coords(:,:)
  character(len=100) :: func=" "
  integer :: l

  ! Same conversion factor as in dftd3
  ! Lattice vectors in Angstrom as found in dna.xyz/dna.poscar
  ! They must be converted to Bohr before passed to dftd3

  type(dftd3_input) :: input
  type(dftd3_calc) :: dftd3
  real(wp) :: edisp

  l=0
  do
      if (instr(l+1) == C_NULL_CHAR) exit
      func(l+1:l+1) = instr(l+1)
      l = l + 1
  end do

  allocate(grads(3,nAtoms))
  allocate(coords(3,nAtoms))
  k=1
  do i=1,3
  do j=1,nAtoms
     coords(i,j) = ccoords(k)
     k = k + 1
  enddo
  enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Initialize input
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! You can set input variables if you like, or just leave them on their
  ! defaults, which are the same as the dftd3 program uses.
  
  !! Threebody interactions (default: .false.)
  !input%threebody = .true.
  !
  !! Numerical gradients (default: .false.)
  !input%numgrad = .false.
  !
  !! Cutoffs (below you find the defaults)
  !input%cutoff = sqrt(9000.0_wp)
  !input%cutoff_cn = sqrt(1600.0_wp)

  ! Initialize dftd3
  call dftd3_init(dftd3, input)

  ! Choose functional. Alternatively you could set the parameters manually
  ! by the dftd3_set_params() function.
  !call dftd3_set_functional(dftd3, func='dftb3', version=version, tz=.false.)
  call dftd3_set_functional(dftd3, func=func, version=version, tz=.false.)

  ! Calculate dispersion and gradients for non-periodic case
  call dftd3_dispersion(dftd3, coords, atnum, edisp, grads)
  write(*, "(A)") "*** Dispersion for non-periodic case"
  write(*, "(A,ES20.12)") "Energy [au]:", edisp
  write(*, "(A)") "Gradients [au]:"
  write(*, "(3ES20.12)") grads
  write(*, *)
  
  k=1
  do i=1,3
  do j=1,nAtoms
     cgrads(k) = grads(i,j)
     k = k + 1
  enddo
  enddo
  deallocate(grads)
  deallocate(coords)

end subroutine dftd3c

subroutine dftd3cpbc(instr, version, nAtoms, atnum, ccoords, clatVecs,  edisp, cgrads, cstress)
  use dftd3_api
  use, intrinsic :: iso_c_binding
  implicit none
  character(kind=c_char), dimension(*), intent(IN) :: instr
  integer, parameter :: wp = kind(1.0d0)
  integer :: i,j,k
  integer :: nAtoms
  integer :: version
  integer :: atnum(nAtoms)
  real(wp) :: ccoords(3*nAtoms)
  real(wp) :: cgrads(3*nAtoms)
  real(wp) :: clatVecs(9)
  real(wp) :: cstress(9)

  real(wp) :: stress(3,3)
  real(wp) :: latVecs(3,3)
  real(wp), allocatable :: grads(:,:)
  real(wp), allocatable :: coords(:,:)
  character(len=100) :: func=" "
  integer :: l

  ! Same conversion factor as in dftd3
  ! Lattice vectors in Angstrom as found in dna.xyz/dna.poscar
  ! They must be converted to Bohr before passed to dftd3

  type(dftd3_input) :: input
  type(dftd3_calc) :: dftd3
  real(wp) :: edisp

  l=0
  do
      if (instr(l+1) == C_NULL_CHAR) exit
      func(l+1:l+1) = instr(l+1)
      l = l + 1
  end do

  allocate(grads(3,nAtoms))
  allocate(coords(3,nAtoms))
  k=1
  do i=1,3
  do j=1,nAtoms
     coords(i,j) = ccoords(k)
     k = k + 1
  enddo
  enddo

  k=1
  do i=1,3
  do j=1,3
     latVecs(i,j) = clatVecs(k)
     k = k + 1
  enddo
  enddo

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Initialize input
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! You can set input variables if you like, or just leave them on their
  ! defaults, which are the same as the dftd3 program uses.
  
  !! Threebody interactions (default: .false.)
  !input%threebody = .true.
  !
  !! Numerical gradients (default: .false.)
  !input%numgrad = .false.
  !
  !! Cutoffs (below you find the defaults)
  !input%cutoff = sqrt(9000.0_wp)
  !input%cutoff_cn = sqrt(1600.0_wp)

  ! Initialize dftd3
  call dftd3_init(dftd3, input)

  ! Choose functional. Alternatively you could set the parameters manually
  ! by the dftd3_set_params() function.
  call dftd3_set_functional(dftd3, func=func, version=version, tz=.false.)

  ! Calculate dispersion and gradients for periodic case
  call dftd3_pbc_dispersion(dftd3, coords, atnum, latVecs, edisp, grads, stress)
  write(*, "(A)") "*** Dispersion for periodic case"
  write(*, "(A,ES20.12)") "Energy [au]:", edisp
  write(*, "(A)") "Gradients [au]:"
  write(*, "(3ES20.12)") grads
  write(*, "(A)") "Stress [au]:"
  write(*, "(3ES20.12)") stress
  k=1
  do i=1,3
  do j=1,3
     cstress(k) = stress(i,j)
     k = k + 1
  enddo
  enddo
  k=1
  do i=1,3
  do j=1,nAtoms
     cgrads(k) = grads(i,j)
     k = k + 1
  enddo
  enddo
  deallocate(grads)
  deallocate(coords)

end subroutine dftd3cpbc
