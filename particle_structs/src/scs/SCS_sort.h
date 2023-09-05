#pragma once
namespace pumipic {
  template <class DataTypes, typename MemSpace>
    void SellCSigma<DataTypes, MemSpace>::sigmaSort(kkLidView& ptcls,
                                                    kkLidView& index,
                                                    lid_t num_elems,
                                                    kkLidView ptcls_per_elem,
                                                    lid_t sigma){
    using TeamMem = typename PolicyType::member_type;
    ptcls = kkLidView("ptcls", num_elems);
    index = kkLidView("index", num_elems);
    if (sigma > 1) {
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        ptcls(i) = ptcls_per_elem(i);
        index(i) = i;
      });

      lid_t n_sigma = num_elems/sigma;
      Kokkos::parallel_for( PolicyType(n_sigma, 1), KOKKOS_LAMBDA(const TeamMem& t){
        lid_t start = t.league_rank() * sigma;
        lid_t end = (t.league_rank() == n_sigma-1) ? num_elems : start + sigma;
        auto range = Kokkos::make_pair(start, end);
        auto ptcl_subview = Kokkos::subview(ptcls, range);
        auto index_subview = Kokkos::subview(index, range);
        Kokkos::Experimental::sort_by_key_thread(t, ptcl_subview, index_subview);
      });
    }
    else {
      Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA(const lid_t& i) {
        ptcls(i) = ptcls_per_elem(i);
        index(i) = i;
      });
    }
  }
}
