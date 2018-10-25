#include "Push.h"
#include <Kokkos_Core.hpp>

void push_array(int np, double* xs, double* ys, double* zs, double distance, double dx,
                double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  for (int i = 0; i < np; ++i) {
    new_xs[i] = xs[i] + distance * dx;
    new_ys[i] = ys[i] + distance * dy;
    new_zs[i] = zs[i] + distance * dz;
  }
}

#ifdef KOKKOS_ENABLED
typedef Kokkos::DefaultExecutionSpace exe_space;

//TODO Figure out how to template these helper fns
typedef double fp_t;
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

void push_array_kk(int np, double* xs, double* ys, double* zs, double distance, double dx,
                double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  Kokkos::Timer timer;
  kkFpView xs_d("xs_d", np);
  hostToDeviceFp(xs_d, xs);

  kkFpView ys_d("ys_d", np);
  hostToDeviceFp(ys_d, ys);

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
  fprintf(stderr, "array host to device transfer %f\n", timer.seconds());

  #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  timer.reset();
  Kokkos::parallel_for (np, KOKKOS_LAMBDA (const int i) {
      new_xs_d(i) = xs_d(i) + disp_d(0) * disp_d(1);
      new_ys_d(i) = ys_d(i) + disp_d(0) * disp_d(2);
      new_zs_d(i) = zs_d(i) + disp_d(0) * disp_d(3);
    });
  fprintf(stderr, "kokkos array push %f\n", timer.seconds());
  #endif

  timer.reset();
  deviceToHostFp(new_xs_d,new_xs);
  deviceToHostFp(new_ys_d,new_ys);
  deviceToHostFp(new_zs_d,new_zs);
  fprintf(stderr, "array device to host transfer %f\n", timer.seconds());
}
#endif //kokkos enabled

void push_scs(SellCSigma* scs, double* xs, double* ys, double* zs, double distance, double dx,
              double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  for (int i = 0; i < scs->num_chunks; ++i) {
    int index = scs->offsets[i];
    while (index != scs->offsets[i + 1]) {
      for (int j = 0; j < scs->C; ++j) {
        if (scs->id_list[index] != -1) {
          int id = scs->id_list[index];
          new_xs[id] = xs[id] + distance * dx;
          new_ys[id] = ys[id] + distance * dy;
          new_zs[id] = zs[id] + distance * dz;
        }
        ++index;
      } // end for
    } // end while
  }
}

#ifdef KOKKOS_ENABLED
void push_scs_kk(SellCSigma* scs, int np, double* xs, double* ys, double* zs, double distance, double dx,
              double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  Kokkos::Timer timer;
  kkLidView offsets_d("offsets_d", scs->num_chunks+1);
  hostToDeviceLid(offsets_d, scs->offsets);

  kkLidView ids_d("ids_d", scs->offsets[scs->num_chunks]);
  hostToDeviceLid(ids_d, scs->id_list);

  kkLidView chunksz_d("chunksz_d", 1);
  hostToDeviceLid(chunksz_d, &scs->C);

  kkFpView xs_d("xs_d", np);
  hostToDeviceFp(xs_d, xs);

  kkFpView ys_d("ys_d", np);
  hostToDeviceFp(ys_d, ys);

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
  fprintf(stderr, "kokkos scs host to device transfer %f\n", timer.seconds());

  #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  timer.reset();
  Kokkos::parallel_for(scs->num_chunks, KOKKOS_LAMBDA (const int i) {
    for( int index = offsets_d(i); index != offsets_d(i+1); index+=chunksz_d(0) ) {
      Kokkos::parallel_for(chunksz_d(0), KOKKOS_LAMBDA (const int j) {
        int id = ids_d(index+j);
        if (id != -1) {
          new_xs_d(id) = xs_d(id) + disp_d(0) * disp_d(1);
          new_ys_d(id) = ys_d(id) + disp_d(0) * disp_d(2);
          new_zs_d(id) = zs_d(id) + disp_d(0) * disp_d(3);
        }
      });
    }
  });
  fprintf(stderr, "kokkos scs push %f\n", timer.seconds());
  #endif

  timer.reset();
  deviceToHostFp(new_xs_d,new_xs);
  deviceToHostFp(new_ys_d,new_ys);
  deviceToHostFp(new_zs_d,new_zs);
  fprintf(stderr, "array device to host transfer %f\n", timer.seconds());
}
#endif //kokkos enabled
