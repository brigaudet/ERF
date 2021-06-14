
/*
 * setup boundary conditions for scalars
 */
#include "ScalarPhysBC.H"

void ScalarPhysBC(amrex::MultiFab& phi, const amrex::Geometry& geom, const amrex::BCRec& bcr, int scomp, int ncomp, const amrex::Real& time) {
    
    // bccomp definitions are in BCPhysToMath.cpp
    if (geom.isAllPeriodic() || phi.nGrow() == 0) {
       return;
    }

    // Physical Domain
    amrex::Box dom(geom.Domain());
    const amrex::Real* prob_lo = geom.ProbLo();
    const amrex::Real* prob_hi = geom.ProbHi();

    // grid space
    amrex::GpuArray<amrex::Real,3> dx;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        dx[d] = geom.CellSize(d);
    }

    // get the number of ghost cells
    int nghost = phi.nGrow();

    // get the boundary type for scalar
    amrex::Vector<int> bc_lo(AMREX_SPACEDIM);
    amrex::Vector<int> bc_hi(AMREX_SPACEDIM);
    SetupScalarBCTypes(bcr, bc_lo, bc_hi);

    // setup the boundary via MultiFab
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        // one ghost cell
        amrex::Box bx = mfi.growntilebox(nghost);

        const amrex::Array4<amrex::Real>& data = phi.array(mfi);

        int lo = dom.smallEnd(0);
        int hi = dom.bigEnd(0);
        
        if (bx.smallEnd(0) < lo) {
            amrex::Real x = prob_lo[0];
            if (bc_lo[0] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (i < lo) {
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = 2.0*data(lo,j,k,scomp+n) - data(lo+1,j,k,scomp+n);
                    }
                });
            }
            else if (bc_lo[0] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (i < lo) {
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = data(lo,j,k,scomp+n);
                    }
                });
            }
        }
        
        if (bx.bigEnd(0) > hi) {
            amrex::Real x = prob_hi[0];
            if (bc_hi[0] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (i > hi) {
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = 2.*data(hi,j,k,scomp+n) - data(hi-1,j,k,scomp+n);
                    }
                });
            }
            else if (bc_hi[0] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (i > hi) {
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = data(hi,j,k,scomp+n);
                    }
                });
            }
        }

#if (AMREX_SPACEDIM >= 2)
        //___________________________________________________________________________
        // y-physbc to data

        lo = dom.smallEnd(1);
        hi = dom.bigEnd(1);
        
        if (bx.smallEnd(1) < lo) {
            amrex::Real y = prob_lo[1];
            if (bc_lo[1] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (j < lo) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = 2.*data(i,lo,k,scomp+n) - data(i,lo+1,k,scomp+n);
                    }
                });
            }
            else if (bc_lo[1] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (j < lo) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = data(i,lo,k,scomp+n);
                    }
                });
            }
        }

        if (bx.bigEnd(1) > hi) {
            amrex::Real y = prob_hi[1];
            if (bc_hi[1] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (j > hi) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = 2.*data(i,hi,k,scomp+n) - data(i,hi-1,k,scomp+n);
                    }
                });
            }
            else if (bc_hi[1] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (j > hi) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
                        data(i,j,k,scomp+n) = data(i,hi,k,scomp+n);
                    }
                });
            }
        }
#endif

#if (AMREX_SPACEDIM >= 3)
        // z-physbc to data

        lo = dom.smallEnd(2);
        hi = dom.bigEnd(2);
        
        if (bx.smallEnd(2) < lo) {
            amrex::Real z = prob_lo[2];
            if (bc_lo[2] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (k < lo) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        data(i,j,k,scomp+n) = 2.*data(i,j,lo,scomp+n) - data(i,j,lo+1,scomp+n);
                    }
                });
            }
            else if (bc_lo[2] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (k < lo) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        data(i,j,k,scomp+n) = data(i,j,lo,scomp+n);
                    }
                });
            }
        }

        if (bx.bigEnd(2) > hi) {
            amrex::Real z= prob_hi[2];
            if (bc_hi[2] == amrex::BCType::foextrap) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (k > hi) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        data(i,j,k,scomp+n) = 2.*data(i,j,hi,scomp+n) - data(i,j,hi+1,scomp+n);
                    }
                });
            }
            else if (bc_hi[2] == amrex::BCType::ext_dir) {
                amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                {
                    if (k > hi) {
                        amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
                        amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
                        data(i,j,k,scomp+n) = data(i,j,hi,scomp+n);
                    }
                });
            }
        }
#endif
  } // end MFIter
}


