

#include "VelocityPhysBC.H"

//
// Setup the boundary condition for velocity
//
void VelocityPhysBC(amrex::MultiFab& vel, const amrex::Geometry& geom, const amrex::BCRec& bcr, int idir, const amrex::Real& time) {

    if (geom.isAllPeriodic()) {
        return;
    }

    // physics domain
    amrex::Box dom(geom.Domain());

    int ng = vel.nGrow();

    amrex::Vector<int> bc_vel_lo(AMREX_SPACEDIM);
    amrex::Vector<int> bc_vel_hi(AMREX_SPACEDIM);
    SetupVelocityBCTypes(bcr, bc_vel_lo, bc_vel_hi);

    // setup the boundary via MultiFab
    for (amrex::MFIter mfi(vel); mfi.isValid(); ++mfi) {

        amrex::Box bx = mfi.growntilebox(ng);

        const amrex::Array4<amrex::Real>& data = vel.array(mfi);

        // xvel
        if (idir == 0) {

           // low x
           if ((bc_vel_lo[0] == amrex::BCType::ext_dir) && (bx.smallEnd(0) <= dom.smallEnd(0))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (i < dom.smallEnd(0)) {
                    // set ghost cells to negative of interior value
                    data(i,j,k) = -data(-i,j,k);
                }           
                else if (i == dom.smallEnd(0)) {
                    // set normal velocity on boundary to zero
                    data(i,j,k) = 0.;
                }
            });
           }

           // high x
          if ((bc_vel_hi[0] == amrex::BCType::ext_dir) && (bx.bigEnd(0) >= dom.bigEnd(0)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {        
                if (i > dom.bigEnd(0)+1) {
                    data(i,j,k) = -data(2*dom.bigEnd(0)+2-i,j,k);
                }           
                else if (i == dom.bigEnd(0)+1) {
                    data(i,j,k) = 0.;
                }
            });
          }

#if (AMREX_SPACEDIM >= 2)
           // low y
          if ((bc_vel_lo[1] == amrex::BCType::ext_dir) && (bx.smallEnd(1) <= dom.smallEnd(1))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (j < dom.smallEnd(1)) {
                    data(i,j,k) = -data(i,-j,k);
                }
            });
          }

          // high y
          if ((bc_vel_hi[1] == amrex::BCType::ext_dir) && (bx.bigEnd(1) >= dom.bigEnd(1)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (j > dom.bigEnd(1)+1) {
                    data(i,j,k) = -data(i,2*dom.bigEnd(1)+2-j,k);
                }
            });
           }
#endif

#if (AMREX_SPACEDIM >= 3)
        // low z
        if ((bc_vel_lo[2] == amrex::BCType::ext_dir) && (bx.smallEnd(2) <= dom.smallEnd(2))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (k < dom.smallEnd(2)) {
                    data(i,j,k) = -data(i,j,-k);
                }
            });
        }

        // high z
        if ((bc_vel_hi[2] == amrex::BCType::ext_dir) && (bx.bigEnd(2) >= dom.bigEnd(2)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (k > dom.bigEnd(2)+1) {
                    data(i,j,k) = -data(i,j,2*dom.bigEnd(2)+2-k);
                }
            });
        }
#endif
      // yvel
     } else if (idir == 1) {
           // low x
           if ((bc_vel_lo[0] == amrex::BCType::ext_dir) && (bx.smallEnd(0) <= dom.smallEnd(0))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (i < dom.smallEnd(0)) {
                    data(i,j,k) = -data(-i,j,k);
                }
            });
           }
          // high x
          if ((bc_vel_hi[0] == amrex::BCType::ext_dir) && (bx.bigEnd(0) >= dom.bigEnd(0)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (i > dom.bigEnd(0)+1) {
                    data(i,j,k) = -data(2*dom.bigEnd(0)+2-i,j,k);
                }
            });
          }

#if (AMREX_SPACEDIM >= 2)
        // lo-y faces
        if ((bc_vel_lo[1] == amrex::BCType::ext_dir) && (bx.smallEnd(1) <= dom.smallEnd(1))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {        
                if (j < dom.smallEnd(1)) {
                    data(i,j,k) = -data(i,-j,k);
                }           
                else if (j == dom.smallEnd(1)) {
                    data(i,j,k) = 0.;
                }
            });
        }

        // hi-y faces
        if ((bc_vel_hi[1] == amrex::BCType::ext_dir) && (bx.bigEnd(1) >= dom.bigEnd(1)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {        
                if (j > dom.bigEnd(1)+1) {
                    data(i,j,k) = -data(i,2*dom.bigEnd(1)+2-j,k);
                }           
                else if (j == dom.bigEnd(1)+1) {
                    data(i,j,k) = 0.;
                }
            });
        }
#endif
        
#if (AMREX_SPACEDIM >= 3)

        // low z
        if ((bc_vel_lo[2] == amrex::BCType::ext_dir) && (bx.smallEnd(2) <= dom.smallEnd(2))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {        
                if (k < dom.smallEnd(2)) {
                    data(i,j,k) = -data(i,j,-k);
                }           
            });
        }

        // high z
        if ((bc_vel_hi[2] == amrex::BCType::ext_dir) && (bx.bigEnd(2) >= dom.bigEnd(2)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {        
                if (k > dom.bigEnd(2)+1) {
                    data(i,j,k) = -data(i,j,2*dom.bigEnd(2)+2-k);
                }           
            });
        }
#endif
        // zvel
       } else if (idir == 2) {
           // low x
           if ((bc_vel_lo[0] == amrex::BCType::ext_dir) && (bx.smallEnd(0) <= dom.smallEnd(0))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (i < dom.smallEnd(0)) {
                    data(i,j,k) = -data(-i,j,k);
                }
            });
           }

          // high x
          if ((bc_vel_hi[0] == amrex::BCType::ext_dir) && (bx.bigEnd(0) >= dom.bigEnd(0)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (i > dom.bigEnd(0)+1) {
                    data(i,j,k) = -data(2*dom.bigEnd(0)+2-i,j,k);
                }
            });
          }

#if (AMREX_SPACEDIM >= 2)
           // low y
        if ((bc_vel_lo[1] == amrex::BCType::ext_dir) && (bx.smallEnd(1) <= dom.smallEnd(1))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (j < dom.smallEnd(1)) {
                    data(i,j,k) = -data(i,-j,k);
                }
            });
        }

        // high y
        if ((bc_vel_hi[1] == amrex::BCType::ext_dir) && (bx.bigEnd(1) >= dom.bigEnd(1)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (j > dom.bigEnd(1)+1) {
                    data(i,j,k) = -data(i,2*dom.bigEnd(1)+2-j,k);
                }
            });
        }
#endif

#if (AMREX_SPACEDIM >= 3)
        // low z
        if ((bc_vel_lo[2] == amrex::BCType::ext_dir) && (bx.smallEnd(2) <= dom.smallEnd(2))) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (k < dom.smallEnd(2)) {
                    data(i,j,k) = -data(i,j,-k);
                }
                else if (k == dom.smallEnd(2)) {
                    data(i,j,k) = 0.;
                }
            });
        }
        // high z
        if ((bc_vel_hi[2] == amrex::BCType::ext_dir) && (bx.bigEnd(2) >= dom.bigEnd(2)+1)) {

            amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (k > dom.bigEnd(2)+1) {
                    data(i,j,k) = -data(i,j,2*dom.bigEnd(2)+2-k);
                }
                else if (k == dom.bigEnd(2)+1) {
                    data(i,j,k) = 0.;
                }
            });
        }
#endif
    }    
  } // end MFIter
}

