#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include "myfunc.H"


using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

double sech(double x) {
    return 1.0 / std::cosh(x);
}

double U_analy(double x, double t, double A_coef, double G_coef) {
    double a1 = std::sqrt(A_coef/6.); 
    double lmd = - (2. * A_coef - 3. * G_coef) / 6.;
    double U = A_coef * std::pow(sech(a1 * (x + lmd * t)), 2.);
    return U;
}

double forceterm(double x, double t, double A_coef, double G_coef) {
	double a1 = std::sqrt(A_coef/6.);
	double lmd = - (2. * A_coef - 3. * G_coef) / 6.;
	double f = A_coef * G_coef * std::pow(sech(a1 * (x + lmd * t)), 2.);
	return f;
}

void main_main ()
{

    // **********************************
    // SIMULATION PARAMETERS

    // number of cells on each side of the domain
    amrex::GpuArray<int, 3> n_cell; // Number of cells in each dimension

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps; // I need to make sure that this is an integer

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;

    // domain dimensions
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

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
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

    // This defines the physical box, [-L,L] in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});
    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate 3 U multifabs; two will store the older states, the other the new.
    MultiFab U_nm1(ba, dm, Ncomp, Nghost); //U_n-1  
    MultiFab U_n(ba, dm, Ncomp, Nghost);   //U_n
    MultiFab U_np1(ba, dm, Ncomp, Nghost); //U_n+1
    MultiFab U_exact(ba, dm, Ncomp, Nghost); //U_exact  
    MultiFab F_t(ba, dm, Ncomp, Nghost); //Force term 
    MultiFab Plt(ba, dm, 3, 0); //plotting of numerical solution, exact solution, and force term 

    // time = starting time in the simulation
    Real time = 0.0;

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
        const std::string& pltfile = amrex::Concatenate("plt",step,5); //here 5 is the number of digits in the name of the file
        MultiFab::Copy(Plt, U_n, 0, 0, 1, 0);
        MultiFab::Copy(Plt, U_exact , 0, 1, 1, 0);
        MultiFab::Copy(Plt, F_t, 0, 2, 1, 0);	
	WriteSingleLevelPlotfile(pltfile, Plt, {"Un","Uexact","Ft"}, geom, time, step);
    }
    
    //update time
    time = time + dt;

    //Time advancement 
    for (int step = 2; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        U_nm1.FillBoundary(geom.periodicity());
	U_n.FillBoundary(geom.periodicity()); 

        // loop over boxes
        for ( MFIter mfi(U_nm1); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& Unm1 = U_nm1.array(mfi);
	    const Array4<Real>& Un = U_n.array(mfi);
            const Array4<Real>& Unp1 = U_np1.array(mfi);

            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            { Unp1(i,j,k) = 0.; // updatestep
            });
        }

        // update time
        time = time + dt;

        // copy new solution into old solution
        MultiFab::Copy(U_nm1, U_n, 0, 0, 1, 0);
	MultiFab::Copy(U_n, U_np1, 0, 0, 1, 0);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5); //here 5 is the number of digits in the name of the file
            MultiFab::Copy(Plt, U_nm1, 0, 0, 1, 0);
            MultiFab::Copy(Plt, U_n, 0, 1, 1, 0);
            WriteSingleLevelPlotfile(pltfile, Plt, {"Unm1","Un"}, geom, time, step);
        }
    }
}


