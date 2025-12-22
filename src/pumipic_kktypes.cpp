#include "pumipic_kktypes.hpp"

namespace pumipic {

  void hostToDeviceLid(kkLidView d, lid_t *h) {
    kkLidView::host_mirror_type hv = Kokkos::create_mirror_view(d);
    for (size_t i=0; i<hv.size(); ++i) {
      hv(i) = h[i];
    }
    Kokkos::deep_copy(d,hv);
  }

  void deviceToHostLid(kkLidView d, lid_t *h) {
    kkLidView::host_mirror_type hv = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(hv,d);
    for(size_t i=0; i<hv.size(); ++i) {
      h[i] = hv(i);
    }
  }

  void hostToDeviceFp(kkFpView d, fp_t* h) {
    kkFpView::host_mirror_type hv = Kokkos::create_mirror_view(d);
    for (size_t i=0; i<hv.size(); ++i)
      hv(i) = h[i];
    Kokkos::deep_copy(d,hv);
  }

  void hostToDeviceFp(kkFp3View d, fp_t (*h)[3]) {
    kkFp3View::host_mirror_type hv = Kokkos::create_mirror_view(d);
    for (size_t i=0; i<hv.size()/3; ++i) {
      hv(i,0) = h[i][0];
      hv(i,1) = h[i][1];
      hv(i,2) = h[i][2];
    }
    Kokkos::deep_copy(d,hv);
  }

  void deviceToHostFp(kkFp3View d, fp_t (*h)[3]) {
    kkFp3View::host_mirror_type hv = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(hv,d);
    for(size_t i=0; i<hv.size()/3; ++i) {
      h[i][0] = hv(i,0);
      h[i][1] = hv(i,1);
      h[i][2] = hv(i,2);
    }
  }

}
