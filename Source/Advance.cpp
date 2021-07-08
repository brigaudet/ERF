#include <cmath>

#include "ERF.H"
#include "RK3.H"
#include "IndexDefines.H"
/* BJG */
#include "EOS.H"
#include "Constants.H"

using namespace amrex;

Real
ERF::advance(Real time, Real dt, int amr_iteration, int amr_ncycle)
{
  /** the main driver for a single level implementing the time advance.

         @param time the current simulation time
         @param dt the timestep to advance (e.g., go from time to time + dt)
         @param amr_iteration where we are in the current AMR subcycle.  Each
                         level will take a number of steps to reach the
                         final time of the coarser level below it.  This
                         counter starts at 1
         @param amr_ncycle  the number of subcycles at this level
  */

  /*  BJG  */
  //  amrex::Print() << "phys_bc.lo(0,1,2) in Advance:  " << phys_bc.lo(0) << " " << phys_bc.lo(1) << " " << phys_bc.lo(2) << std::endl;


  BL_PROFILE("ERF::advance()");

  int finest_level = parent->finestLevel();

  if (level < finest_level && do_reflux) {
    getFluxReg(level + 1).reset();
  }

//  Real dt_new = dt;

  BL_PROFILE("ERF::do_rk3_advance()");

  // Check that we are not asking to advance stuff we don't know to
  // if (src_list.size() > 0) amrex::Abort("Have not integrated other sources
  // into MOL advance yet");

  for (int i = 0; i < num_state_type; ++i) {
    bool skip = false;
    if (!skip) {
      state[i].allocOldData();
      state[i].swapTimeLevels(dt);
    }
  }

  MultiFab& S_old = get_old_data(State_Type);
  MultiFab& S_new = get_new_data(State_Type);

  MultiFab& U_old = get_old_data(X_Vel_Type);
  MultiFab& V_old = get_old_data(Y_Vel_Type);
  MultiFab& W_old = get_old_data(Z_Vel_Type);

  MultiFab& U_new = get_new_data(X_Vel_Type);
  MultiFab& V_new = get_new_data(Y_Vel_Type);
  MultiFab& W_new = get_new_data(Z_Vel_Type);

  /*  BJG  */

  int nx = geom.Domain().bigEnd(0);
  int ny = geom.Domain().bigEnd(1);
  int nz = geom.Domain().bigEnd(2);
  // amrex::Print() << "nx, ny, nz:  " << nx << " " << ny << " " << nz << std::endl;   

 int nvars = S_old.nComp(); 

 /*  end BJG */


  // Fill level 0 ghost cells (including at periodic boundaries)
  //TODO: Check if we should consider the number of ghost cells as a function of spatial order here itself
  S_old.FillBoundary(geom.periodicity());
  U_old.FillBoundary(geom.periodicity());
  V_old.FillBoundary(geom.periodicity());
  W_old.FillBoundary(geom.periodicity());

  const Real* dx = geom.CellSize();


  const auto dxarray = geom.CellSizeArray();


  /*  Below is test of tbx nodaltilebox  BJG */

 for ( MFIter mfi(U_old,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        
    const Box& tbx = mfi.nodaltilebox(0);

    const Array4<Real> & uarray = U_old.array(mfi);

    amrex::ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      /*  FOExtrap at low end of z direction */

      if( phys_bc.lo(2) == 2 ) {
	uarray(i,j,-1,0) = uarray(i,j,0,0);
      }


      /* SlipWall at low end of z direction
      Ghost cell value equals that at low end of domain    */

      if( phys_bc.lo(2) == 4 ) {
	uarray(i,j,-1,0) = uarray(i,j,0,0);
      }

      /* NoSlipWall at low end of z direction
      For now, we will do the below method of appying the condition on the z-face.
      Thus the average of ghost cell value and lower bound value should be zero, instead of ghost cell value being zero.
      Hopefully this will be OK   */

      if ( phys_bc.lo(2) == 5 ) {
	uarray(i,j,-1,0) = -uarray(i,j,0,0);
      }

      /*  FOExtrap at high end of z direction */

      if( phys_bc.hi(2) == 2 ) {
	uarray(i,j,nz+1,0) = uarray(i,j,nz,0);
      }

      /* Slip Wall at high end of z direction */

      if ( phys_bc.hi(2) == 4 ) {
	uarray(i,j,nz+1,0) = uarray(i,j,nz,0);
      }

      /* NoSlip Wall at high end of z direction */

      if ( phys_bc.hi(2) == 5 ) {
	uarray(i,j,nz+1,0) = -uarray(i,j,nz,0);
      }

    
    });

      }


  /*  Repeat for V_old BJG */

 for ( MFIter mfi(V_old,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        
    const Box& tby = mfi.nodaltilebox(1);

    const Array4<Real> & varray = V_old.array(mfi);

    amrex::ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {


      /*  FOExtrap at low end of z direction */

      if( phys_bc.lo(2) == 2 ) {
	varray(i,j,-1,0) = varray(i,j,0,0);
      }

      /* NoSlipWall at low end of z direction
      For now, we will do the below method of appying the condition on the z-face.
      Thus the average of ghost cell value and lower bound value should be zero, instead of ghost cell value being zero.
      Hopefully this will be OK   */

      if ( phys_bc.lo(2) == 5 ) {
	varray(i,j,-1,0) = -varray(i,j,0,0);
      }

      /*  FOExtrap at high end of z direction */

      if( phys_bc.hi(2) == 2 ) {
	varray(i,j,nz+1,0) = varray(i,j,nz,0);
      }


      /* Slip Wall at high end of z direction */

      if ( phys_bc.hi(2) == 4 ) {
	varray(i,j,nz+1,0) = varray(i,j,nz,0);
      }

      /* NoSlip Wall at high end of z direction */

      if ( phys_bc.hi(2) == 5 ) {
	varray(i,j,nz+1,0) = -varray(i,j,nz,0);
      }


    
    });
  }


  /*  Repeat for W_old BJG */


 for ( MFIter mfi(W_old,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        
    const Box& tbz = mfi.nodaltilebox(2);
    
    const auto lo = lbound(tbz);
    const auto hi = ubound(tbz);
    //    amrex::Print() << "lo0, lo2, hi0, hi2:  " << lo.x << " " << lo.z << " " << hi.x << " " << hi.z << std::endl;    
    
    const Array4<Real> & warray = W_old.array(mfi);

    amrex::ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      /*  FOExtrap at low end of z direction */

      if( phys_bc.lo(2) == 2 ) {
	warray(i,j,-1,0) = warray(i,j,0,0);
      }


      /* SlipWall at low end of z direction
      For w, directly set value at lowest index to zero.  
      Set value at ghost point to negative of second lowest index, to prevent
      forcing on w at the lower boundary from viscous diffusion   */   

      if ( phys_bc.lo(2) == 4 ) {
	warray(i,j,0,0) = 0.0;
        warray(i,j,-1,0) = -warray(i,j,1,0);
      }


      /* NoSlipWall at low end of z direction
	 Same as SlipWall for w */

      if ( phys_bc.lo(2) == 5 ) {
	warray(i,j,0,0) = 0.0;
        warray(i,j,-1,0) = -warray(i,j,1,0);
      }


      /*  In z direction, warray has one more index than scalars */


      /*  FOExtrap at high end of z direction */

      if( phys_bc.hi(2) == 2 ) {
	warray(i,j,nz+2,0) = warray(i,j,nz+1,0);
      }


      if ( phys_bc.hi(2) == 4 ) {
	warray(i,j,nz+1,0) = 0.0;
        warray(i,j,nz+2,0) = -warray(i,j,nz,0);
      }

      if ( phys_bc.hi(2) == 5 ) {
	warray(i,j,nz+1,0) = 0.0;
        warray(i,j,nz+2,0) = -warray(i,j,nz,0);
      }




    
    });
  }

		       /* For scalars, below is retained from Zhao's code, but converted to z direction.
                       Apply not just to scalar field (RhoScalar_comp), but to Density (Rho).
                       Assume zero gradient (homogeneous Neumann) on density. 
                       Treatment of RhoTheta is such so that, when converted to pressure, is consistent with
                       zero tendency on normal velocity (w)  */
                       /* TODO:  for inhomogeneous Neumann case dx should be dz. */

 Real gravity = use_gravity? CONST_GRAV: 0.0;
 // CONST_GRAV is positive, but grav is assumed to be negative in equations below BJG
    const    Array<Real,AMREX_SPACEDIM> grav{0.0, 0.0, -gravity};
    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};
    amrex::Print() << "grav_gpu[2]:  " << grav_gpu[2] << std::endl;

 for ( MFIter mfi(S_old,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        
    const Box& bx = mfi.tilebox();
    const Box& tbx = mfi.nodaltilebox(0);
    const Box& tby = mfi.nodaltilebox(1);
    const Box& tbz = mfi.nodaltilebox(2);

    const Array4<Real> & cu = S_old.array(mfi);
    //amrex::Print() << "Advance before b.c cu = " << cu(0,0,0,Scalar_comp) <<  "  " << cu(-1,0,0,Scalar_comp) << "  " << cu(nx,0,0,Scalar_comp) <<  "  " << cu(nx+1,0,0,Scalar_comp) << std::endl;

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {

      /*  FOExtrap at low end of z direction, apply to all conserved scalars */

      if( phys_bc.lo(2) == 2 ) {
       cu(i,j,-1,RhoScalar_comp) = cu(i,j,0,RhoScalar_comp);
       cu(i,j,-1,Rho_comp) =  cu(i,j,0,Rho_comp);
       cu(i,j,-1,RhoTheta_comp) = cu(i,j,0,RhoTheta_comp);
      }



      /*  NoSlipWall and SlipWall have the same scalar boundary conditions */

      if ( phys_bc.lo(2) == 4 ) {

      // Neumann
       cu(i,j,-1,RhoScalar_comp) = 0.0*(*dx) + cu(i,j,0,RhoScalar_comp);
       cu(i,j,-1,Rho_comp) = 0.0*(*dx) + cu(i,j,0,Rho_comp);
       Real rhotheta = cu(i,j,0,RhoTheta_comp);
       Real pressure = getPgivenRTh(rhotheta);
       Real pressurem1 =  ( -grav_gpu[2] * dxarray[2] / 2.0 ) * (cu(i,j,0,Rho_comp) + cu(i,j,-1,Rho_comp)) + pressure; 
       cu(i,j,-1,RhoTheta_comp) = getRThgivenP(pressurem1);
      }


      if ( phys_bc.lo(2) == 5 ) {

      // Neumann
       cu(i,j,-1,RhoScalar_comp) = 0.0*(*dx) + cu(i,j,0,RhoScalar_comp);
       cu(i,j,-1,Rho_comp) = 0.0*(*dx) + cu(i,j,0,Rho_comp);
       Real rhotheta = cu(i,j,0,RhoTheta_comp);
       Real pressure = getPgivenRTh(rhotheta);
       Real pressurem1 =  ( -grav_gpu[2] * dxarray[2] / 2.0 ) * (cu(i,j,0,Rho_comp) + cu(i,j,-1,Rho_comp)) + pressure; 
       cu(i,j,-1,RhoTheta_comp) = getRThgivenP(pressurem1);
      }


      /*  FOExtrap at high end of z direction, apply to all conserved scalars */

      if( phys_bc.lo(2) == 2 ) {
       cu(i,j,nz+1,RhoScalar_comp) = cu(i,j,nz,RhoScalar_comp);
       cu(i,j,nz+1,Rho_comp) =  cu(i,j,nz,Rho_comp);
       cu(i,j,nz+1,RhoTheta_comp) = cu(i,j,nz,RhoTheta_comp);
      }




      if ( phys_bc.hi(2) == 4 ) {

      // Neumann
       cu(i,j,nz+1,RhoScalar_comp) = 0.0*(*dx) + cu(i,j,nz,RhoScalar_comp);
       cu(i,j,nz+1,Rho_comp) = 0.0*(*dx) + cu(i,j,nz,Rho_comp);
       Real rhotheta = cu(i,j,nz,RhoTheta_comp);
       Real pressure = getPgivenRTh(rhotheta);
       Real pressurep1 =  -( -grav_gpu[2] * dxarray[2] / 2.0 ) * (cu(i,j,nz+1,Rho_comp) + cu(i,j,nz,Rho_comp)) + pressure; 
       cu(i,j,nz+1,RhoTheta_comp) = getRThgivenP(pressurep1);
       //    amrex::Print() << "scalar,rho,rhotheta:  " << cu(i,j,nz+1,RhoScalar_comp) << " " << cu(i,j,nz+1,Rho_comp) << " " << cu(i,j,nz+1,RhoTheta_comp) << std::endl;    
      }


      if ( phys_bc.hi(2) == 5 ) {

      // Neumann
       cu(i,j,nz+1,RhoScalar_comp) = 0.0*(*dx) + cu(i,j,nz,RhoScalar_comp);
       cu(i,j,nz+1,Rho_comp) = 0.0*(*dx) + cu(i,j,nz,Rho_comp);
       Real rhotheta = cu(i,j,nz,RhoTheta_comp);
       Real pressure = getPgivenRTh(rhotheta);
       Real pressurep1 =  -( -grav_gpu[2] * dxarray[2] / 2.0 ) * (cu(i,j,nz+1,Rho_comp) + cu(i,j,nz,Rho_comp)) + pressure; 
       cu(i,j,nz+1,RhoTheta_comp) = getRThgivenP(pressurep1);
       //    amrex::Print() << "scalar,rho,rhotheta:  " << cu(i,j,nz+1,RhoScalar_comp) << " " << cu(i,j,nz+1,Rho_comp) << " " << cu(i,j,nz+1,RhoTheta_comp) << std::endl;    
      }


    });

 }

		       /*  end BJG */








  const BoxArray&            ba = S_old.boxArray();
  const DistributionMapping& dm = S_old.DistributionMap();

  /*  int nvars = S_old.nComp();  */

  // Place-holder for source array -- for now just set to 0
  MultiFab source(ba,dm,nvars,1); 
  source.setVal(0.0);

  // Place-holder for eta array -- shear viscosity -- for now just set to 0
  MultiFab eta(ba,dm,1,1); 
  eta.setVal(0.0);

  // Place-holder for zeta array -- bulk viscosity -- for now just set to 0
  MultiFab zeta(ba,dm,1,1); 
  zeta.setVal(0.0);

  // Place-holder for kappa array -- thermal conducitivity --for now just set to 0
  MultiFab kappa(ba,dm,1,1); 
  kappa.setVal(0.0);

  // TODO: We won't need faceflux, edgeflux, and centflux when using the new code architecture. Remove them.
  // Fluxes (except momentum) at faces. This should comprise of advective as well as diffusive fluxes.
  // There are separate variables to handle the momentum at the faces
  std::array< MultiFab, AMREX_SPACEDIM > faceflux;
  //faceflux[0] is of size (ncells_x + 1, ncells_y    , ncells_z    )
  faceflux[0].define(convert(ba,IntVect(1,0,0)), dmap, nvars, 0);
  //faceflux[1] is of size (ncells_x    , ncells_y + 1, ncells_z    )
  faceflux[1].define(convert(ba,IntVect(0,1,0)), dmap, nvars, 0);
  //faceflux[2] is of size (ncells_x    , ncells_y    , ncells_z + 1)
  faceflux[2].define(convert(ba,IntVect(0,0,1)), dmap, nvars, 0);

  // Edge fluxes for {x, y, z}-momentum equations
  std::array< MultiFab, 2 > edgeflux_x; // v, w
  std::array< MultiFab, 2 > edgeflux_y; // u, w
  std::array< MultiFab, 2 > edgeflux_z; // u, v

  edgeflux_x[0].define(convert(ba,IntVect(1,1,0)), dmap, 1, 0); // v
  edgeflux_x[1].define(convert(ba,IntVect(1,0,1)), dmap, 1, 0); // w

  edgeflux_y[0].define(convert(ba,IntVect(1,1,0)), dmap, 1, 0); // u
  edgeflux_y[1].define(convert(ba,IntVect(0,1,1)), dmap, 1, 0); // w

  edgeflux_z[0].define(convert(ba,IntVect(1,0,1)), dmap, 1, 0); // u
  edgeflux_z[1].define(convert(ba,IntVect(0,1,1)), dmap, 1, 0); // v

  std::array< MultiFab, AMREX_SPACEDIM > cenflux;
  cenflux[0].define(ba,dmap,1,1); // 0-2: rhoU, rhoV, rhoW
  cenflux[1].define(ba,dmap,1,1);
  cenflux[2].define(ba,dmap,1,1);

  // TODO: Better make it a member of the ERF class. Need to deal with static stuff.
  SolverChoice solverChoice(use_advection, use_diffusion, use_smagorinsky, use_gravity, spatial_order);
  //solverChoice.display();

  // *****************************************************************
  // Update the cell-centered state and face-based velocity using RK3
  // Inputs:  
  //          S_old    (state on cell centers)
  //          U_old    (x-velocity on x-faces)
  //          V_old    (y-velocity on y-faces)
  //          W_old    (z-velocity on z-faces)
  //          source   (source term on cell centers)
  // Outputs:  
  //          S_new    (state on cell centers)
  //          U_new    (x-velocity on x-faces)
  //          V_new    (y-velocity on y-faces)
  //          W_new    (z-velocity on z-faces)
  // *****************************************************************

  // BJG

  int phys_bc_lovalx = phys_bc.lo(0);
  int phys_bc_lovaly = phys_bc.lo(1);
  int phys_bc_lovalz = phys_bc.lo(2);
  int phys_bc_hivalx = phys_bc.hi(0);
  int phys_bc_hivaly = phys_bc.hi(1);
  int phys_bc_hivalz = phys_bc.hi(2);

  RK3_advance(
              S_old, S_new,
              U_old, V_old, W_old,
              U_new, V_new, W_new,
              source,
              eta, zeta,kappa,
              faceflux,
              edgeflux_x, edgeflux_y, edgeflux_z,
              cenflux, geom, dx, dt,
              solverChoice, phys_bc_lovalz, phys_bc_hivalz);

  return dt;
}
