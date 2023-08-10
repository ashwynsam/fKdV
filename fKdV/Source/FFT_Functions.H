#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputeForwardFFT(const MultiFab&    mf,
                       MultiFab&          mf_dft_real,
                       MultiFab&          mf_dft_imag,
                       const Geometry&    geom,
                       long               npts);

void ComputeInverseFFT(MultiFab&        mf_2,
                       const MultiFab&  mf_dft_real,
                       const MultiFab&  mf_dft_imag,
                       GpuArray<int, 3> n_cell,
                       const Geometry&  geom);
