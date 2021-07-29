#!/bin/make -f

###############################################################################
# EXTERNAL LIBRARY PATHS
###############################################################################
# Enter here paths to GSL or EIGEN if they are not in your standard include
# path. DO NOT completely remove the entry, leave at least "./".
# my office
PROJECT_GSL=./
#PROJECT_EIGEN=/sps/ilm/allouche/Softwares/eigen3
PROJECT_EIGEN=/usr/include/eigen3/

#lunxfrt
#PROJECT_GSL=/home/theochem/allouche/Softwares/gsl
#PROJECT_EIGEN=/home/theochem/allouche/Softwares/eigen3

###############################################################################
# COMPILERS AND FLAGS
###############################################################################
PROJECT_CC=g++
PROJECT_MPICC=mpic++
#PROJECT_CFLAGS=-O3 -march=native -std=c++11 -fopenmp  -L /home/theochem/allouche/Softwares/gsl/lib 
PROJECT_CFLAGS=-O3 -march=native -std=c++11 -fopenmp  
PROJECT_CFLAGS_MPI=-Wno-long-long
#PROJECT_DEBUG=-g -pedantic-errors -Wall -Wextra
#PROJECT_DEBUG=-pg 
#PROJECT_DEBUG=-pg --no-pie -fPIC
PROJECT_DEBUG=
PROJECT_TEST=--coverage -fno-default-inline -fno-inline -fno-inline-small-functions -fno-elide-constructors
PROJECT_AR=ar
PROJECT_ARFLAGS=-rcsv
PROJECT_CFLAGS_BLAS=
PROJECT_LDFLAGS_BLAS=-lblas

###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################
# with dipole derivatives
PROJECT_OPTIONS+= -DHIGH_DERIVATIVES

# with dipole derivatives
#PROJECT_OPTIONS+= -DDIPOLE_DERIVATIVES

# Do not use symmetry function groups.
#PROJECT_OPTIONS+= -DNOSFGROUPS

# Do not use cutoff function cache.
#PROJECT_OPTIONS+= -DNOCFCACHE

# Build with dummy Stopwatch class.
#PROJECT_OPTIONS+= -DNOTIME

# Disable check for low number of neighbors.
#PROJECT_OPTIONS+= -DNONEIGHCHECK

# Compile without MPI support.
#PROJECT_OPTIONS+= -DNOMPI

# Use BLAS together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_BLAS

# Disable all C++ asserts (also Eigen debugging).
#PROJECT_OPTIONS+= -DNDEBUG

# Use Intel MKL together with Eigen.
#PROJECT_OPTIONS+= -DEIGEN_USE_MKL_ALL

# Disable Eigen multi threading.
PROJECT_OPTIONS+= -DEIGEN_DONT_PARALLELIZE
