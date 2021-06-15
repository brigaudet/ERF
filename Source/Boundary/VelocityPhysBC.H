
#ifndef VELOCITY_PHYSBC_H_
#define VELOCITY_PHYSBC_H_

#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>

#include "ERF.H"
#include "prob.H"

// function to setup velocity physical boundary condition
void VelocityPhysBC(amrex::MultiFab& vel, const amrex::Geometry& geom, const amrex::BCRec& bcr, int idir, const amrex::Real& time);

inline void SetupVelocityBCTypes(const amrex::BCRec& bcr, amrex::Vector<int>& bc_lo, amrex::Vector<int>& bc_hi) {

    const int* bctype = bcr.data();

    // set to interior/periodic by default; overwrite below
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
       bc_lo[i] = bc_hi[i] = amrex::BCType::int_dir;
    }

    for (int i=0; i<AMREX_SPACEDIM; ++i) {
       // low end boundary type
       if (bctype[i] == amrex::BCType::foextrap) {
           // first-order extrapolation
           bc_lo[i] = amrex::BCType::foextrap;
       } else if (bctype[i] == amrex::BCType::ext_dir) {
           // inflow
           bc_lo[i] = amrex::BCType::ext_dir;
       }

       // high end boundary type
       if (bctype[i+AMREX_SPACEDIM] == amrex::BCType::foextrap) {
          // first-order extrapolation
          bc_hi[i] = amrex::BCType::foextrap;
       } else if (bctype[i+AMREX_SPACEDIM] == amrex::BCType::ext_dir) {
          // inflow
         bc_hi[i] = amrex::BCType::ext_dir;
       }
   }
}

#endif // VELOCITY_PHYSBC_H_ 
