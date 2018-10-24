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
typedef double fp_t;
typedef Kokkos::DefaultExecutionSpace exe_space;
typedef Kokkos::View<fp_t*, exe_space::device_type> kkFpView;

/** \brief helper function to transfer a host array to a device view
 */
void hostToDevice(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  for (size_t i=0; i<hv.size(); ++i)
    hv(i) = h[i];
  Kokkos::deep_copy(d,hv);
}
/** \brief helper function to transfer a device view to a host array
 */
void deviceToHost(kkFpView d, fp_t* h) {
  kkFpView::HostMirror hv = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(hv,d);
  for(size_t i=0; i<hv.size(); ++i)
    h[i] = hv(i);
}

void push_array_kk(int np, double* xs, double* ys, double* zs, double distance, double dx,
                double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  kkFpView xs_d("xs_d", np);
  hostToDevice(xs_d, xs);

  kkFpView ys_d("ys_d", np);
  hostToDevice(ys_d, ys);

  kkFpView zs_d("zs_d", np);
  hostToDevice(zs_d, zs);

  kkFpView new_xs_d("new_xs_d", np);
  hostToDevice(new_xs_d, new_xs);

  kkFpView new_ys_d("new_ys_d", np);
  hostToDevice(new_ys_d, new_ys);

  kkFpView new_zs_d("new_zs_d", np);
  hostToDevice(new_zs_d, new_zs);

  fp_t disp[4] = {distance,dx,dy,dz};
  kkFpView disp_d("direction_d", 4);
  hostToDevice(disp_d, disp);

  #if defined(KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA)
  Kokkos::parallel_for (np, KOKKOS_LAMBDA (const int i) {
      // Acesss the View just like a Fortran array.  The layout depends
      // on the View's memory space, so don't rely on the View's
      // physical memory layout unless you know what you're doing.
      new_xs_d(i) = xs_d(i) + disp_d(0) * disp_d(1);
      new_ys_d(i) = ys_d(i) + disp_d(0) * disp_d(2);
      new_zs_d(i) = zs_d(i) + disp_d(0) * disp_d(3);
    });
  #endif
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
      }
    }
  }
}
