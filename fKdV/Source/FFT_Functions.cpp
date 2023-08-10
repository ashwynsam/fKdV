#include "FFT_Functions.H"

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

// Function accepts a multifab 'mf' and computes the FFT, storing it in mf_dft_real amd mf_dft_imag multifabs
void ComputeForwardFFT(const MultiFab&    mf,
		       MultiFab&          mf_dft_real,
		       MultiFab&          mf_dft_imag,
		       const Geometry&    geom,
		       long               npts)
{ 
    // **********************************
    // COPY INPUT MULTIFAB INTO A MULTIFAB WITH ONE BOX
    // **********************************

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(geom.Domain());
    DistributionMapping dm_onegrid(ba_onegrid);

    // storage for phi and the dft
    MultiFab mf_onegrid         (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // copy phi into phi_onegrid
    mf_onegrid.ParallelCopy(mf, 0, 0, 1);

    // **********************************
    // COMPUTE FFT
    // **********************************

#ifdef AMREX_USE_CUDA
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif

    // For scaling on forward FFTW
    Real sqrtnpts = std::sqrt(npts);

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;

    for (MFIter mfi(mf_onegrid); mfi.isValid(); ++mfi) {

      // grab a single box including ghost cell range
      Box realspace_bx = mfi.fabbox();

      // size of box including ghost cell range
      IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

      // this is the size of the box, except the 0th component is 'halved plus 1'
      IntVect spectral_bx_size = fft_size;
      spectral_bx_size[0] = fft_size[0]/2 + 1;

      // spectral box
      Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));

      spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                 The_Device_Arena()));
      spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory

      FFTplan fplan;

#ifdef AMREX_USE_CUDA

#if (AMREX_SPACEDIM == 1)
      cufftResult result = cufftPlan1d(&fplan, fft_size[0], CUFFT_D2Z);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan1d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 2)
      cufftResult result = cufftPlan2d(&fplan, fft_size[1], fft_size[0], CUFFT_D2Z);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan2d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 3)
      cufftResult result = cufftPlan3d(&fplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_D2Z);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan3d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#endif

#else // host

#if (AMREX_SPACEDIM == 1)
      fplan = fftw_plan_dft_r2c_1d(fft_size[0],
                   mf_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 2)
      fplan = fftw_plan_dft_r2c_2d(fft_size[1], fft_size[0],
                   mf_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#elif (AMREX_SPACEDIM == 3)
      fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                   mf_onegrid[mfi].dataPtr(),
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   FFTW_ESTIMATE);
#endif

#endif

      forward_plan.push_back(fplan);
    }

    ParallelDescriptor::Barrier();

    // ForwardTransform
    for (MFIter mfi(mf_onegrid); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
      cufftSetStream(forward_plan[i], Gpu::gpuStream());
      cufftResult result = cufftExecD2Z(forward_plan[i],
                    mf_onegrid[mfi].dataPtr(),
                    reinterpret_cast<FFTcomplex*>
                    (spectral_field[i]->dataPtr()));
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " forward transform using cufftExec failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#else
      fftw_execute(forward_plan[i]);
#endif
    }

    // copy data to a full-sized MultiFab
    // this involves copying the complex conjugate from the half-sized field
    // into the appropriate place in the full MultiFab
    for (MFIter mfi(mf_dft_real_onegrid); mfi.isValid(); ++mfi) {

      Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

      Array4<Real> const& realpart = mf_dft_real_onegrid.array(mfi);
      Array4<Real> const& imagpart = mf_dft_imag_onegrid.array(mfi);

      Box bx = mfi.fabbox();

      amrex:: ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
      /*
        Copying rules:

        For domains from (0,0,0) to (Nx-1,Ny-1,Nz-1)

        For any cells with i index >= Nx/2, these values are complex conjugates of the corresponding
        entry where (Nx-i,Ny-j,Nz-k) UNLESS that index is zero, in which case you use 0.

        e.g. for an 8^3 domain, any cell with i index

        Cell (6,2,3) is complex conjugate of (2,6,5)

        Cell (4,1,0) is complex conjugate of (4,7,0)  (note that the FFT is computed for 0 <= i <= Nx/2)
      */
          if (i <= bx.length(0)/2) {
          // copy value
              realpart(i,j,k) = spectral(i,j,k).real();
              imagpart(i,j,k) = spectral(i,j,k).imag();

	      realpart(i,j,k) /= std::sqrt(npts);
              imagpart(i,j,k) /= std::sqrt(npts);

          }
          else{
	      realpart(i,j,k) = 0.;
              imagpart(i,j,k) = 0.;
	  }	  
      });
    }
  
    // Copy the full multifabs back into the output multifabs
    mf_dft_real.ParallelCopy(mf_dft_real_onegrid, 0, 0, 1);
    mf_dft_imag.ParallelCopy(mf_dft_imag_onegrid, 0, 0, 1);

    // destroy fft plan
    for (int i = 0; i < forward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(forward_plan[i]);
#else
        fftw_destroy_plan(forward_plan[i]);
#endif
    }

// // Reconstruct FFT with reflections
//    for(amrex::MFIter mfi(mf_fft); mfi.isValid(); ++mfi){
//        const amrex::Box& bx = mfi.validbox();
//        //const amrex::Array4<amrex::Real>& mf_fft_array = mf_fft.array(mfi);
//        const amrex::Array4<amrex::Real>& mf_real_array = mf_real.array(mfi);
//        const amrex::Array4<amrex::Real>& mf_img_array = mf_img.array(mfi);
//
//        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//        {mf_real_array(n_cell - 1 - i, j , k) = mf_real_array(i + 1, j, k);
//         mf_img_array(n_cell - 1 - i, j, k) = -mf_img_array(i + 1, j , k);
//        });
//
//     }
//

}


// This function takes the real and imaginary parts of data from the frequency domain and performs an inverse FFT, storing the result in 'mf_2'
// The FFTW c2r function is called which accepts complex data in the frequency domain and returns real data in the normal cartesian plane
void ComputeInverseFFT(MultiFab&                        mf_2,
		       const MultiFab&                  mf_dft_real,
                       const MultiFab&                  mf_dft_imag,				   
		       GpuArray<int, 3>                 n_cell,
                       const Geometry&                  geom)
{

    // create a new BoxArray and DistributionMapping for a MultiFab with 1 grid
    BoxArray ba_onegrid(geom.Domain());
    DistributionMapping dm_onegrid(ba_onegrid);
    
    // Declare multifabs to store entire dataset in one grid.
    MultiFab mf_onegrid_2 (ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_real_onegrid(ba_onegrid, dm_onegrid, 1, 0);
    MultiFab mf_dft_imag_onegrid(ba_onegrid, dm_onegrid, 1, 0);

    // Copy distributed multifabs into one grid multifabs
    mf_dft_real_onegrid.ParallelCopy(mf_dft_real, 0, 0, 1);
    mf_dft_imag_onegrid.ParallelCopy(mf_dft_imag, 0, 0, 1);

#ifdef AMREX_USE_CUDA
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    // Copy the contents of the real and imaginary FFT Multifabs into 'spectral_field'
    for (MFIter mfi(mf_dft_real_onegrid); mfi.isValid(); ++mfi) {

      // grab a single box including ghost cell range
      Box realspace_bx = mfi.fabbox();

      // size of box including ghost cell range
      IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

      // this is the size of the box, except the 0th component is 'halved plus 1'
      IntVect spectral_bx_size = fft_size;
      spectral_bx_size[0] = fft_size[0]/2 + 1;

      // spectral box
      Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));

      spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                 The_Device_Arena()));
      spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory
      
        // Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();
        Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

        Array4<Real> const& realpart = mf_dft_real_onegrid.array(mfi);
        Array4<Real> const& imagpart = mf_dft_imag_onegrid.array(mfi);

        Box bx = mfi.fabbox();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {        
            if (i <= bx.length(0)/2) {
                GpuComplex<Real> copy(realpart(i,j,k),imagpart(i,j,k));
                spectral(i,j,k) = copy;
            }   
        });
    }

    // Compute the inverse FFT on spectral_field and store it in 'mf_onegrid_2'
    Vector<FFTplan> backward_plan;

    // Now that we have a spectral field full of the data from the DFT..
    // We perform the inverse DFT on spectral field and store it in mf_onegrid_2
    for (MFIter mfi(mf_onegrid_2); mfi.isValid(); ++mfi) {

       // grab a single box including ghost cell range
       Box realspace_bx = mfi.fabbox();

       // size of box including ghost cell range
       IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

       FFTplan bplan;

#ifdef AMREX_USE_CUDA

#if (AMREX_SPACEDIM == 1)
      cufftResult result = cufftPlan1d(&bplan, fft_size[0], CUFFT_Z2D);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan1d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 2)
      cufftResult result = cufftPlan2d(&bplan, fft_size[1], fft_size[0], CUFFT_Z2D);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan2d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#elif (AMREX_SPACEDIM == 3)
      cufftResult result = cufftPlan3d(&bplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_Z2D);
      if (result != CUFFT_SUCCESS) {
    AllPrint() << " cufftplan3d forward failed! Error: "
              << cufftErrorToString(result) << "\n";
      }
#endif

#else // host

#if (AMREX_SPACEDIM == 1)
      bplan = fftw_plan_dft_c2r_1d(fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);

#elif (AMREX_SPACEDIM == 2)
      bplan = fftw_plan_dft_c2r_2d(fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);

#elif (AMREX_SPACEDIM == 3)
      bplan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
                   reinterpret_cast<FFTcomplex*>
                   (spectral_field.back()->dataPtr()),
                   mf_onegrid_2[mfi].dataPtr(),
                   FFTW_ESTIMATE);
#endif

#endif

      backward_plan.push_back(bplan);// This adds an instance of bplan to the end of backward_plan
      }

    for (MFIter mfi(mf_onegrid_2); mfi.isValid(); ++mfi) {
      int i = mfi.LocalIndex();

#ifdef AMREX_USE_CUDA
      cufftSetStream(backward_plan[i], Gpu::gpuStream());
      cufftResult result = cufftExecZ2D(backward_plan[i],
                           reinterpret_cast<FFTcomplex*>
                           (spectral_field[i]->dataPtr()),
                           mf_onegrid_2[mfi].dataPtr());
       if (result != CUFFT_SUCCESS) {
         AllPrint() << " inverse transform using cufftExec failed! Error: "
         << cufftErrorToString(result) << "\n";
       }
#else
      fftw_execute(backward_plan[i]);
#endif

      // Standard scaling after fft and inverse fft using FFTW

#if (AMREX_SPACEDIM == 1)
      mf_onegrid_2[mfi] /= n_cell[0];
#elif (AMREX_SPACEDIM == 2)
      mf_onegrid_2[mfi] /= n_cell[0]*n_cell[1];
#elif (AMREX_SPACEDIM == 3)
      mf_onegrid_2[mfi] /= n_cell[0]*n_cell[1]*n_cell[2];
#endif

    }

    // copy contents of mf_onegrid_2 into mf
    mf_2.ParallelCopy(mf_onegrid_2, 0, 0, 1);

    // destroy ifft plan
    for (int i = 0; i < backward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(backward_plan[i]);
#else
        fftw_destroy_plan(backward_plan[i]);
#endif

    }

}
