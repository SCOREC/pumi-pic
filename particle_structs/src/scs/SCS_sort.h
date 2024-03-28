#pragma once
namespace pumipic {
  template <class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::sigmaSort(kkLidView& ptcls,
                                                    kkLidView& index,
                                                    lid_t num_elems,
                                                    kkLidView ptcls_per_elem,
                                                    lid_t sigma){
    using TeamMem = typename PolicyType::member_type;
    ptcls = kkLidView(Kokkos::ViewAllocateWithoutInitializing("ptcls"), num_elems);
    index = kkLidView(Kokkos::ViewAllocateWithoutInitializing("index"), num_elems);
    Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
      ptcls(i) = ptcls_per_elem(i);
      index(i) = i;
    });
    if (sigma > 1) {
#ifdef PP_USE_CUDA
      lid_t i;
      Kokkos::View<lid_t*, typename MemSpace::device_type> elem_ids(Kokkos::ViewAllocateWithoutInitializing("elem_ids"), num_elems);
      Kokkos::View<lid_t*, typename MemSpace::device_type> temp_ppe(Kokkos::ViewAllocateWithoutInitializing("temp_ppe"), num_elems);
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        temp_ppe(i) = ptcls_per_elem(i);
        elem_ids(i) = i;
      });
      thrust::device_ptr<lid_t> ptcls_t(temp_ppe.data());
      thrust::device_ptr<lid_t> elem_ids_t(elem_ids.data());
      for (i = 0; i < num_elems - sigma; i+=sigma) {
        thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + i + sigma, elem_ids_t + i);
      }
      thrust::sort_by_key(thrust::device, ptcls_t + i, ptcls_t + num_elems, elem_ids_t + i);
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        ptcls(i) = temp_ppe(i);
        index(i) = elem_ids(i);
      });
#else
      sigma = Kokkos::min(sigma, Kokkos::max(num_elems, 1));
      lid_t n_sigma = num_elems/sigma;
      int vectorLen = PolicyType::vector_length_max();
      Kokkos::parallel_for( PolicyType(n_sigma, 1, vectorLen), KOKKOS_LAMBDA(const TeamMem& t){
        lid_t start = t.league_rank() * sigma;
        lid_t end = (t.league_rank() == n_sigma-1) ? num_elems : start + sigma;
        auto range = Kokkos::make_pair(start, end);
        auto ptcl_subview = Kokkos::subview(ptcls, range);
        auto index_subview = Kokkos::subview(index, range);
        Kokkos::Experimental::sort_by_key_thread(t, ptcl_subview, index_subview);
      });
#endif
    }
  }
}
