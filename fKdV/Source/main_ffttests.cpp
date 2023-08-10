#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_MFParallelFor.H> //For the second newer method
#include <AMReX_PlotFileUtil.H> //For ploting the MultiFab
#include "myfunc.H"
#include "FFT_Functions.H"


int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {

        int ncomp = 1;
        int n_cell = 8;
        int max_grid_size = 2;
 	amrex::GpuArray<int, 3> n_cells = {n_cell, n_cell, n_cell}; // Number of cells in each dimension

	 // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
	amrex::IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
	amrex:: IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));
	amrex:: IntVect dom_hi_fft(AMREX_D_DECL(n_cell/2 - 2, n_cell/2 - 2, n_cell/2 - 2));
	
        amrex::Box domain(dom_lo, dom_hi);
        amrex::Box domain_fft(dom_lo, dom_hi_fft);

        amrex::BoxArray ba(domain);
	amrex::BoxArray ba_fft(domain_fft);

        // chop the single grid into many small boxes
        ba.maxSize(max_grid_size);
        ba_fft.maxSize(max_grid_size);
	

        // Distribution Mapping
        amrex::DistributionMapping dm(ba);
        amrex::DistributionMapping dm_fft(ba_fft);

        //Define MuliFab
        amrex::MultiFab mf(ba, dm, ncomp, 0);
        amrex::MultiFab mf_2(ba, dm, ncomp, 0);
        amrex::MultiFab mf_real(ba, dm, ncomp, 0);
        amrex::MultiFab mf_img(ba, dm, ncomp, 0);
        amrex::MultiFab mf_fft(ba_fft, dm_fft, ncomp, 0);
        amrex::MultiFab Plt(ba, dm, 3, 0);
	
	
	amrex::RealBox real_box({AMREX_D_DECL( 0., 0., 0.)},
                     {AMREX_D_DECL( 1., 1., 1.)});

        amrex::Geometry geom(domain, &real_box);

	// extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
       
        // define k
	amrex::Vector<amrex::Real> k_x(n_cell);

        // Generate array 'k' with values -N/2 to N/2 - 1 and apply fftshift
        for (int i = 0; i < n_cell; ++i) {
          k_x[i] = 2 * M_PI  * (i - n_cell / 2);
        }

        // FFTSHIFT on array 'k'
        std::rotate(k_x.begin(), k_x.begin() + n_cell / 2, k_x.end());


//	for (int i = 0; i < n_cell; ++i) {
//		amrex::Print() << k_x[i] << "\n";
//	}



	//Fill a MultiFab with Data
        for(amrex::MFIter mfi(mf); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& mf_array = mf.array(mfi);

	    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            { 
	    	Real x = (i+0.5) * dx[0];
		mf_array(i,j,k) = std::sin(x);
            });
         }

	amrex::Print() << "Print FFT here"  << "\n";

	//Compute forward fft
	ComputeForwardFFT(mf, mf_real, mf_img, geom, 1);

	//Here I will test to see if ifft(k * fft(U)) will give the full set even though fft is half zeros
	for(amrex::MFIter mfi(mf_real); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& mf_real_array = mf_real.array(mfi);
            const amrex::Array4<amrex::Real>& mf_img_array = mf_img.array(mfi);
		
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
    		mf_real_array(i,j,k) = k_x[i] * mf_real_array(i,j,k);
    		mf_img_array(i,j,k) = k_x[i] * mf_img_array(i,j,k);		
            });

         }


//	// Reconstruct FFT with reflections
//	for(amrex::MFIter mfi(mf_real); mfi.isValid(); ++mfi){
//            //const amrex::Box& bx = mfi.validbox();
//            const amrex::Array4<amrex::Real>& mf_real_array = mf_real.array(mfi);
//            const amrex::Array4<amrex::Real>& mf_img_array = mf_img.array(mfi);
//		
//            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//            {	
//	    	if (i < n_cell / 2 - 1){
//	        	mf_real_array(n_cell - 1 - i, j , k) = mf_real_array(i + 1, j, k);
//             		mf_img_array(n_cell - 1 - i, j, k) = -mf_img_array(i + 1, j , k);
//	        }    
//            });
//
//         }

	//Compute inverse fft
	ComputeInverseFFT(mf_2, mf_real, mf_img, n_cells, geom);

        //Plot MultiFab Data
	const std::string& pltfile = amrex::Concatenate("plt",0,5); //here 5 is the number of digits in the name of the file
        MultiFab::Copy(Plt, mf_real, 0, 0, 1, 0);
        MultiFab::Copy(Plt, mf_img , 0, 1, 1, 0);
        MultiFab::Copy(Plt, mf_2 , 0, 2, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt, {"Real","Img","ifft"}, geom, 0., 0);



    }
    amrex::Finalize();
}
