#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include "myfunc.H"
#include "FFT_Functions.H"
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_MFParallelFor.H> //For the second newer method

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

Real sech(Real x) {
    return 1.0 / std::cosh(x);
}

Real U_analy(Real x, Real t, Real A_coef, Real G_coef) {
    Real a1 = std::sqrt(A_coef/6.); 
    Real lmd = - (2. * A_coef - 3. * G_coef) / 6.;
    Real U = A_coef * std::pow(sech(a1 * (x + lmd * t)), 2.);
    return U;
}

Real forceterm(Real x, Real t, Real A_coef, Real G_coef) {
	Real a1 = std::sqrt(A_coef/6.);
	Real lmd = - (2. * A_coef - 3. * G_coef) / 6.;
	Real f = A_coef * G_coef * std::pow(sech(a1 * (x + lmd * t)), 2.);
	return f;
}

void complex_mult(Real a, Real b, Real c, Real d, Real& out_real, Real& out_img){
	out_real = a * c - b * d;
	out_img = a * d + b * c;
}

void main_main ()
{
    Real total_step_strt_time = ParallelDescriptor::second();

    // **********************************
    // SIMULATION PARAMETERS

    amrex::GpuArray<int, 3> n_cell; // Number of cells in each dimension

    int max_grid_size;

    int nsteps; // I need to make sure that this is an integer

    // how often to write a plotfile
    int plot_int;

    Real dt;

    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    // PDE parameters to solve the equation of the form u_t + a * uu_x + b * u_xxx = g * f
    Real gamma; //gamma
    Real alpha; //alpha
    Real beta; //beta

    // force function parameters
    Real A_coef;
    Real G_coef;
   

    // inputs parameters
    {
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        amrex::Vector<int> temp_int(AMREX_SPACEDIM);
        if (pp.queryarr("n_cell",temp_int)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                n_cell[i] = temp_int[i];
            }
        }
        pp.get("max_grid_size",max_grid_size);
        pp.get("nsteps",nsteps);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        pp.get("dt",dt);

	amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
        if (pp.queryarr("prob_lo",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_lo[i] = temp[i];
            }
        }
        if (pp.queryarr("prob_hi",temp)) {
            for (int i=0; i<AMREX_SPACEDIM; ++i) {
                prob_hi[i] = temp[i];
            }
        }
	pp.get("gamma", gamma);
	pp.get("alpha", alpha);
	pp.get("beta", beta);
	pp.get("A_coef", A_coef);
	pp.get("G_coef", G_coef);

    }
 

    // **********************************
    // SIMULATION SETUP

    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [-L,L] in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});
    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);
    //geom.define(domain, real_box, CoordSys::cartesian);
    // geom(domain, &real_box);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 0;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate 3 U multifabs; two will store the older states, the other the new.
    MultiFab U_nm1(ba, dm, Ncomp, Nghost); //U_n-1  
    MultiFab U_n(ba, dm, Ncomp, Nghost);   //U_n
    MultiFab U_np1(ba, dm, Ncomp, Nghost); //U_n+1
    MultiFab U_n_sq(ba, dm, Ncomp, Nghost); //U_n.^2
    MultiFab U_n_sq_real(ba, dm, Ncomp, Nghost);// The multi fabs with _real or _img are the multifabs where the fft data is stored
    MultiFab U_n_sq_img(ba, dm, Ncomp, Nghost);
    MultiFab U_nm1_real(ba, dm, Ncomp, Nghost);
    MultiFab U_nm1_img(ba, dm, Ncomp, Nghost);
    MultiFab U_np1_real(ba, dm, Ncomp, Nghost);
    MultiFab U_np1_img(ba, dm, Ncomp, Nghost);
    MultiFab F_t_real(ba, dm, Ncomp, Nghost);
    MultiFab F_t_img(ba, dm, Ncomp, Nghost);
    MultiFab U_exact(ba, dm, Ncomp, Nghost); //U_exact  
    MultiFab F_t(ba, dm, Ncomp, Nghost); //Force term 
    MultiFab Plt(ba, dm, 3, 0); //plotting of numerical solution, exact solution, and force term 

    // time = starting time in the simulation
    Real time = 0.0;

    // define k
    amrex::Vector<amrex::Real> k_x(n_cell[0]);

    // Generate array 'k' with values -N/2 to N/2 - 1 and apply fftshift
    for (int i = 0; i < n_cell[0]; ++i) {
        k_x[i] = (2 * M_PI) / (2 * prob_hi[0])  * (i - n_cell[0] / 2);
    }

    // FFTSHIFT on array 'k'
    std::rotate(k_x.begin(), k_x.begin() + n_cell[0] / 2, k_x.end());

    // **********************************
    // INITIALIZE DATA

    // some simplifications for function definiton (turn this into a seperate function in the future
    Real a1 = std::sqrt(A_coef/6.); // check sqrt syntax
    Real lmd = - (2. * A_coef - 3. * G_coef) / 6. ;


    // loop over box U_nm1 and U_n
    for (MFIter mfi(U_nm1); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& Unm1 = U_nm1.array(mfi);
	const Array4<Real>& Un = U_n.array(mfi);
	const Array4<Real>& Uexact = U_exact.array(mfi);
	const Array4<Real>& Ft = F_t.array(mfi);

        // set U_nm1 = A * sech^2(a1 * (x + lmd * t))
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real x = prob_lo[0]+ (i+0.5) * dx[0];
            Unm1(i,j,k) = A_coef * std::pow(sech(a1 * x),2.0) ; // t = 0
	    Un(i,j,k) = A_coef * std::pow(sech(a1 * (x + lmd * dt)),2.0); //t = dt
	    Uexact(i,j,k) = U_analy(x, dt, A_coef, G_coef); 
	    Ft(i,j,k) = forceterm(x, dt, A_coef, G_coef);
        });
    }
 

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,7); //here 5 is the number of digits in the name of the file
        MultiFab::Copy(Plt, U_n, 0, 0, 1, 0);
        MultiFab::Copy(Plt, U_exact , 0, 1, 1, 0);
        MultiFab::Copy(Plt, F_t, 0, 2, 1, 0);	
	WriteSingleLevelPlotfile(pltfile, Plt, {"Un","Uexact","Ft"}, geom, time, step);
    }
    
    //update time
    time = time + dt;

    //Time advancement 
    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        //U_nm1.FillBoundary(geom.periodicity());
	//U_n.FillBoundary(geom.periodicity()); 

	// Square U_n amd calculate Force term
	for(amrex::MFIter mfi(U_n); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& Un = U_n.array(mfi);
            const amrex::Array4<amrex::Real>& Un_sq = U_n_sq.array(mfi);
            const amrex::Array4<amrex::Real>& Ft = F_t.array(mfi);
	
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Un_sq(i,j,k) = std::pow(Un(i,j,k) , 2.) ;

		Real x = prob_lo[0]+ (i+0.5) * dx[0];
		Ft(i,j,k) = forceterm(x, time, A_coef, G_coef);
            });

        }

	// Compute FFTs
	ComputeForwardFFT(U_nm1, U_nm1_real, U_nm1_img, geom, 1);
	ComputeForwardFFT(U_n_sq, U_n_sq_real, U_n_sq_img, geom, 1);
	ComputeForwardFFT(F_t, F_t_real, F_t_img, geom, 1);
	
        // loop over boxes
        for ( MFIter mfi(U_nm1); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& Unm1_real = U_nm1_real.array(mfi);
	    const Array4<Real>& Unm1_img = U_nm1_img.array(mfi);
            const Array4<Real>& Unp1_real = U_np1_real.array(mfi);
            const Array4<Real>& Unp1_img = U_np1_img.array(mfi);
            const Array4<Real>& Un_sq_real = U_n_sq_real.array(mfi);
            const Array4<Real>& Un_sq_img = U_n_sq_img.array(mfi);
            const Array4<Real>& Ft_real = F_t_real.array(mfi);
            const Array4<Real>& Ft_img = F_t_img.array(mfi);
            const Array4<Real>& Uexact = U_exact.array(mfi);

            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
	    	//Simplifying coeffecients
	    	Real z = dt * beta * std::pow(k_x[i], 3.);
		Real C1_real = 1. / (1 + std::pow(z, 2.));
		Real C1_img = z / (1 + std::pow(z, 2.));
		Real C2_real = 1.;
		Real C2_img = z;
		Real C3_img = -dt * alpha * k_x[i];
		Real C4_img = 2 * gamma * dt * k_x[i];
		
		Real termUnm1_real, termUnm1_img, termUnsq_real, termUnsq_img, termFt_real, termFt_img, termSum_real, termSum_img, termFull_real, termFull_img;
		complex_mult(C2_real, C2_img, Unm1_real(i,j,k), Unm1_img(i,j,k), termUnm1_real, termUnm1_img);
		complex_mult(0., C3_img, Un_sq_real(i,j,k), Un_sq_img(i,j,k), termUnsq_real, termUnsq_img);
		complex_mult(0., C4_img, Ft_real(i,j,k), Ft_img(i,j,k), termFt_real, termFt_img);
		termSum_real = termUnm1_real + termUnsq_real + termFt_real;
		termSum_img = termUnm1_img + termUnsq_img + termFt_img;
		complex_mult(C1_real, C1_img, termSum_real, termSum_img, termFull_real, termFull_img);
		Unp1_real(i,j,k) = termFull_real;
		Unp1_img(i,j,k) = termFull_img;
    			    
            });
        }

	//Compte Inverse FFT
	ComputeInverseFFT(U_np1, U_np1_real, U_np1_img , n_cell, geom);

        // update time
        time = time + dt;

        // copy new solution into old solution
        MultiFab::Copy(U_nm1, U_n, 0, 0, 1, 0);
	MultiFab::Copy(U_n, U_np1, 0, 0, 1, 0);

        // Tell the I/O Processor to write out which step we're doing
        //amrex::Print() << "Advanced step " << step << "\n";
        
	if (plot_int > 0 && step % plot_int == 0) {
		
            // Write a plotfile of the current data (plot_int was defined in the inputs file)
            const std::string& pltfile = amrex::Concatenate("plt",step,7); //here 5 is the number of digits in the name of the file
            MultiFab::Copy(Plt, U_n, 0, 0, 1, 0);
            MultiFab::Copy(Plt, U_exact , 0, 1, 1, 0);
            MultiFab::Copy(Plt, F_t, 0, 2, 1, 0);
            WriteSingleLevelPlotfile(pltfile, Plt, {"Un","Uexact","Ft"}, geom, time, step);
        }
    }
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";
}
