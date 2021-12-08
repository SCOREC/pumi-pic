#pragma once
namespace pumipic {
  template <class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::sigmaSort(PairView& ptcl_pairs,
                                                    lid_t num_elems,
                                                    kkLidView ptcls_per_elem,
                                                    lid_t sigma){
    //Make temporary copy of the particle counts for sorting
    ptcl_pairs = PairView("ptcl_pairs", num_elems);
    if (sigma > 1) {
      lid_t i;
#ifdef PP_USE_CUDA
      Kokkos::View<lid_t*, typename MemSpace::device_type> elem_ids(Kokkos::ViewAllocateWithoutInitializing("elem_ids"), num_elems);
      Kokkos::View<lid_t*, typename MemSpace::device_type> temp_ppe(Kokkos::ViewAllocateWithoutInitializing("temp_ppe"), num_elems);
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
          temp_ppe(i) = -ptcls_per_elem(i);
          elem_ids(i) = i;
        });
      thrust::device_ptr<lid_t> ptcls_t(temp_ppe.data());
      thrust::device_ptr<lid_t> elem_ids_t(elem_ids.data());
      for (i = 0; i < num_elems - sigma; i+=sigma) {
        thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + i + sigma, elem_ids_t + i);
      }
      thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + num_elems, elem_ids_t + i);
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
          ptcl_pairs(i).first = -temp_ppe(i);
          ptcl_pairs(i).second = elem_ids(i);
        });
#else
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        ptcl_pairs(i).first = ptcls_per_elem(i);
        ptcl_pairs(i).second = i;
      });
      typename PairView::HostMirror ptcl_pairs_host = deviceToHost(ptcl_pairs);
      MyPair* ptcl_pair_data = ptcl_pairs_host.data();
      for (i = 0; i < num_elems - sigma; i+=sigma) {
        std::sort(ptcl_pair_data + i, ptcl_pair_data + i + sigma);
      }
      std::sort(ptcl_pair_data + i, ptcl_pair_data + num_elems);
      hostToDevice(ptcl_pairs,  ptcl_pair_data);
#endif
    }
    else {
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        ptcl_pairs(i).first = ptcls_per_elem(i);
        ptcl_pairs(i).second = i;
      });
    }
  }
}
