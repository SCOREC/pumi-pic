#include "Push.h"
#include <psParams.h>
#include <psAssert.h>
#include <Kokkos_Core.hpp>

void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void push_array(int np, fp_t* xs, fp_t* ys, fp_t* zs,
    int* ptcl_to_elem, elemCoords& elems,
    fp_t distance, fp_t dx, fp_t dy, fp_t dz,
    fp_t* new_xs, fp_t* new_ys, fp_t* new_zs) {
  for (int i = 0; i < np; ++i) {
    int e = ptcl_to_elem[i];
    fp_t c = elems.x[e]   + elems.y[e]   + elems.z[e]   +
               elems.x[e+1] + elems.y[e+1] + elems.z[e+1] +
               elems.x[e+2] + elems.y[e+2] + elems.z[e+2] +
               elems.x[e+3] + elems.y[e+3] + elems.z[e+3];
    c /= 4;
    new_xs[i] = xs[i] + c * distance * dx;
    new_ys[i] = ys[i] + c * distance * dy;
    new_zs[i] = zs[i] + c * distance * dz;
  }
}

#ifdef KOKKOS_ENABLED
typedef Kokkos::DefaultExecutionSpace exe_space;

//TODO Figure out how to template these helper fns
typedef Kokkos::View<fp_t*, exe_space::device_type> kkFpView;
/** \brief helper function to transfer a host array to a device view
 */
void hostToDeviceFp(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d,hv);
}
/** \brief helper function to transfer a device view to a host array
 */
void deviceToHostFp(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size(); ++i)
    h[i] = hv(i);
}

typedef Kokkos::View<fp_t(*)[3], exe_space::device_type> kkFp3View;
/** \brief helper function to transfer a host array to a device view
 */
void hostToDeviceFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size()/3; ++i) {
    hv(i,0) = h[i][0];
    hv(i,1) = h[i][1];
    hv(i,2) = h[i][2];
  }
  Kokkos::deep_copy(d,hv);
}
/** \brief helper function to transfer a device view to a host array
 */
void deviceToHostFp(kkFp3View d, fp_t (*h)[3]) {
  kkFp3View::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size()/3; ++i) {
    h[i][0] = hv(i,0);
    h[i][1] = hv(i,1);
    h[i][2] = hv(i,2);
  }
}

typedef int lid_t;
typedef Kokkos::View<lid_t*, exe_space::device_type> kkLidView;
/** \brief helper function to transfer a host array to a device view
 */
void hostToDeviceLid(kkLidView d, lid_t* h) {
  kkLidView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d,hv);
}

void push_array_kk(int np, fp_t* xs, fp_t* ys, fp_t* zs,
    int* ptcl_to_elem, elemCoords& elems,
    fp_t distance, fp_t dx, fp_t dy, fp_t dz,
    fp_t* new_xs, fp_t* new_ys, fp_t* new_zs) {
  Kokkos::Timer timer;
  kkFpView xs_d("xs_d", np);
  hostToDeviceFp(xs_d, xs);

  kkFpView ys_d("ys_d", np);
  hostToDeviceFp(ys_d, ys);

  kkLidView ptcl_to_elem_d("ptcl_to_elem_d", np);
  hostToDeviceLid(ptcl_to_elem_d, ptcl_to_elem);

  kkFpView ex_d("ex_d", elems.num_elems*elems.verts_per_elem);
  hostToDeviceFp(ex_d, elems.x);
  kkFpView ey_d("ey_d", elems.num_elems*elems.verts_per_elem);
  hostToDeviceFp(ey_d, elems.y);
  kkFpView ez_d("ez_d", elems.num_elems*elems.verts_per_elem);
  hostToDeviceFp(ez_d, elems.z);

  kkFpView zs_d("zs_d", np);
  hostToDeviceFp(zs_d, zs);

  kkFpView new_xs_d("new_xs_d", np);
  hostToDeviceFp(new_xs_d, new_xs);

  kkFpView new_ys_d("new_ys_d", np);
  hostToDeviceFp(new_ys_d, new_ys);

  kkFpView new_zs_d("new_zs_d", np);
  hostToDeviceFp(new_zs_d, new_zs);

  fp_t disp[4] = {distance,dx,dy,dz};
  kkFpView disp_d("direction_d", 4);
  hostToDeviceFp(disp_d, disp);
  fprintf(stderr, "array host to device transfer (seconds) %f\n", timer.seconds());

  #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  double totTime = 0;
  for( int iter=0; iter < NUM_ITERATIONS; iter++) {
    timer.reset();
    Kokkos::parallel_for (np, KOKKOS_LAMBDA (const int i) {
        const fp_t dir[3] = {disp_d(0)*disp_d(1),
                             disp_d(0)*disp_d(2),
                             disp_d(0)*disp_d(3)};
        int e = ptcl_to_elem_d(i);
        fp_t c = ex_d(e)   + ey_d(e)   + ez_d(e)   +
                   ex_d(e+1) + ey_d(e+1) + ez_d(e+1) +
                   ex_d(e+2) + ey_d(e+2) + ez_d(e+2) +
                   ex_d(e+3) + ey_d(e+3) + ez_d(e+3);
        c /= 4;
        new_xs_d(i) = xs_d(i) + c * dir[0];
        new_ys_d(i) = ys_d(i) + c * dir[1];
        new_zs_d(i) = zs_d(i) + c * dir[2];
    });
    totTime += timer.seconds();
  }
  printTiming("array push", totTime);
  #endif

  timer.reset();
  deviceToHostFp(new_xs_d,new_xs);
  deviceToHostFp(new_ys_d,new_ys);
  deviceToHostFp(new_zs_d,new_zs);
  fprintf(stderr, "array device to host transfer (seconds) %f\n", timer.seconds());
}
#endif //kokkos enabled

void push_scs(SellCSigma<Particle, 16>* scs,
    int* ptcl_to_elem, elemCoords& elems,
    fp_t distance, fp_t dx, fp_t dy, fp_t dz) {
  fp_t (*scs_initial_position)[3] = scs->getSCS<0>();
  fp_t (*scs_pushed_position)[3] = scs->getSCS<1>();
  for (int i = 0; i < scs->num_slices; ++i) {
    int index = scs->offsets[i];
    const int chunk = scs->slice_to_chunk[i];
    while (index != scs->offsets[i + 1]) {
      for (int j = 0; j < scs->C; ++j) {
        int row = chunk * scs->C + j;
        int e = scs->row_to_element[row];
        fp_t c = elems.x[e]   + elems.y[e]   + elems.z[e]   +
                 elems.x[e+1] + elems.y[e+1] + elems.z[e+1] +
                 elems.x[e+2] + elems.y[e+2] + elems.z[e+2] +
                 elems.x[e+3] + elems.y[e+3] + elems.z[e+3];
        c /= 4;
        int id = index++;
        scs_pushed_position[id][0] = scs_initial_position[id][0] + c * distance * dx;
        scs_pushed_position[id][1] = scs_initial_position[id][1] + c * distance * dy;
        scs_pushed_position[id][2] = scs_initial_position[id][2] + c * distance * dz;
      } // end for
    } // end while
  }
}

#ifdef KOKKOS_ENABLED

void push_scs_kk(SellCSigma<Particle, 16>* scs, int np, elemCoords& elems,
    fp_t distance, fp_t dx, fp_t dy, fp_t dz) {
  Kokkos::Timer timer;

  fp_t (*scs_initial_position)[3] = scs->getSCS<0>();
  fp_t (*scs_pushed_position)[3] = scs->getSCS<1>();  
  
  kkLidView offsets_d("offsets_d", scs->num_slices+1);
  hostToDeviceLid(offsets_d, scs->offsets);

  kkLidView slice_to_chunk_d("slice_to_chunk_d", scs->num_slices);
  hostToDeviceLid(slice_to_chunk_d, scs->slice_to_chunk);

  kkLidView num_particles_d("num_particles_d", 1);
  hostToDeviceLid(num_particles_d, &np);

  kkLidView chunksz_d("chunksz_d", 1);
  hostToDeviceLid(chunksz_d, &scs->C);

  kkLidView slicesz_d("slicesz_d", 1);
  hostToDeviceLid(slicesz_d, &scs->V);

  kkLidView num_elems_d("num_elems_d", 1);
  hostToDeviceLid(num_elems_d, &elems.num_elems);

  kkLidView row_to_element_d("row_to_element_d", elems.size);
  hostToDeviceLid(row_to_element_d, scs->row_to_element);

  kkFpView ex_d("ex_d", elems.size);
  hostToDeviceFp(ex_d, elems.x);
  kkFpView ey_d("ey_d", elems.size);
  hostToDeviceFp(ey_d, elems.y);
  kkFpView ez_d("ez_d", elems.size);
  hostToDeviceFp(ez_d, elems.z);


  kkFp3View position_d("position_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(position_d, scs_initial_position);
  kkFp3View new_position_d("new_position_d", scs->offsets[scs->num_slices]);
  hostToDeviceFp(new_position_d, scs_pushed_position);
  
  fp_t disp[4] = {distance,dx,dy,dz};
  kkFpView disp_d("direction_d", 4);
  hostToDeviceFp(disp_d, disp);
  fprintf(stderr, "kokkos scs host to device transfer (seconds) %f\n", timer.seconds());

  using Kokkos::TeamPolicy;
  using Kokkos::TeamThreadRange;
  using Kokkos::ThreadVectorRange;
  using Kokkos::parallel_for;
  typedef Kokkos::TeamPolicy<> team_policy;
  typedef typename team_policy::member_type team_member;
  #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  const int league_size = scs->num_slices;
  const int team_size = scs->C;
  const team_policy policy(league_size, team_size);

  double totTime = 0;
  for( int iter=0; iter<NUM_ITERATIONS; iter++) {
    timer.reset();
    //loop over chunks, one thread team per chunk
    parallel_for(policy, KOKKOS_LAMBDA(const team_member& thread) {
        const int slice = thread.league_rank();
        const int slice_row = thread.team_rank();
        const int rowLen = (offsets_d(slice+1)-offsets_d(slice))/chunksz_d(0);
        const int start = offsets_d(slice) + slice_row;
        const fp_t dir[3] = {disp_d(0)*disp_d(1),
                             disp_d(0)*disp_d(2),
                             disp_d(0)*disp_d(3)};
        parallel_for(TeamThreadRange(thread, chunksz_d(0)), [=] (int& j) {
          const int row = slice_to_chunk_d(slice) * chunksz_d(0) + slice_row;
          const int e = row_to_element_d(row);
          const fp_t x[4] = {ex_d(e),ex_d(e+1),ex_d(e+2),ex_d(e+3)};
          const fp_t y[4] = {ey_d(e),ey_d(e+1),ey_d(e+2),ey_d(e+3)};
          const fp_t z[4] = {ez_d(e),ez_d(e+1),ez_d(e+2),ez_d(e+3)};
          parallel_for(ThreadVectorRange(thread, rowLen), [&] (int& p) {
            const int pid = start+(p*chunksz_d(0));
            fp_t c = 0;
            for(int ei = 0; ei<4; ei++) 
              c += x[ei] + y[ei] + z[ei];
            c /= 4;
            
            new_position_d(pid,0) = position_d(pid,0) + c * dir[0];
            new_position_d(pid,1) = position_d(pid,1) + c * dir[1];
            new_position_d(pid,2) = position_d(pid,2) + c * dir[2];
            
          });
        });
    });
    totTime += timer.seconds();
  }
  printTiming("scs push", totTime);
  #endif

  timer.reset();
  
  deviceToHostFp(new_position_d,scs_pushed_position);
  
  fprintf(stderr, "array device to host transfer (seconds) %f\n", timer.seconds());
}

#endif //kokkos enabled
