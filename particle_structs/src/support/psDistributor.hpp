#pragma once

#include <mpi.h>
#include <ppTypes.h>
#include <MemberTypeLibraries.h>
#include <Kokkos_UnorderedMap.hpp>

namespace pumipic {
  template <typename Space = DefaultMemSpace>
  class Distributor {
  public:
    Distributor();
    Distributor(MPI_Comm c);
    Distributor(int nr, int* rnks,MPI_Comm c = MPI_COMM_WORLD);
    template <typename ViewT>
    Distributor(ViewT ranks, MPI_Comm c = MPI_COMM_WORLD);

    void setRanks(int nr, int* rnks);
    template <typename ViewT>
    void setRanks(ViewT rnks);
    void buildMap();

    MPI_Comm mpi_comm() const {return comm;}
    PP_INLINE bool isWorld() const {return ranks_d.size() == 0;}
    int num_ranks() const;
    int rank_host(int i) const;
    PP_DEVICE int rank(int i) const;
    PP_DEVICE int index(int process) const;
  private:
    MPI_Comm comm;
    int nranks;

    typedef Kokkos::View<int*, typename Space::device_type> IndexView;
    //List of ranks on the device
    IndexView ranks_d;
    typename IndexView::host_mirror_type ranks_h;

    //Unordered map from rank to index on device
    typedef Kokkos::UnorderedMap<lid_t, lid_t, typename Space::device_type> MapType;
    MapType mapping;
  };

  template <typename Space>
  Distributor<Space>::Distributor() : comm(MPI_COMM_WORLD), ranks_d("distributor_ranks_d", 0) {
    ranks_h = deviceToHost(ranks_d);
  }
  template <typename Space>
  Distributor<Space>::Distributor(MPI_Comm c) : comm(c),  ranks_d("distributor_ranks_d", 0) {
    ranks_h = deviceToHost(ranks_d);
  }
  template <typename Space>
  Distributor<Space>::Distributor(int nr, int* rnks, MPI_Comm c) : comm(c) {
    setRanks(nr, rnks);
  }

  template <typename Space>
  template <typename ViewT>
  Distributor<Space>::Distributor(ViewT rnks, MPI_Comm c) : comm(c) {
    setRanks(rnks);
  }

  template <typename Space>
  template <typename ViewT>
  // typename std::enable_if<std::is_same<typename Space::memory_space,
  //                                      typename ViewT::memory_space>::value>::type
  void Distributor<Space>::setRanks(ViewT rnks) {
    ranks_d = IndexView("distributor_ranks_d", rnks.size());
    Kokkos::deep_copy(ranks_d, rnks);
    ranks_h = deviceToHost(ranks_d);
    buildMap();
  }

  // template <typename Space>
  // template <typename ViewT>
  // typename std::enable_if<!std::is_same<typename Space::memory_space,
  //                                       typename ViewT::memory_space>::value>::type
  // Distributor<Space>::setRanks(ViewT rnks) {
  //   ranks_d = IndexView("distributor_ranks_d", rnks.size());
  //   ranks_h = Kokkos::create_mirror_view(ranks_d);
  //   for (int i = 0; i < rnks.size(); ++i)
  //     ranks_h(i) = rnks(i);
  //   Kokkos::deep_copy(ranks_d, ranks_h);
  //   buildMap();
  // }

  template <typename Space>
  void Distributor<Space>::setRanks(int nr, int* rnks) {
    ranks_d = IndexView("distributor_ranks_d", nr);
    ranks_h = Kokkos::create_mirror_view(ranks_d);
    for (int i = 0; i < nr; ++i) {
      ranks_h(i) = rnks[i];
    }
    Kokkos::deep_copy(ranks_d, ranks_h);

    buildMap();
  }

  template <typename Space>
  void Distributor<Space>::buildMap() {
    mapping = MapType(ranks_d.size());
    auto local_ranks = ranks_d;
    auto& local_map = mapping;
    auto mapConstruct = KOKKOS_LAMBDA(const int i) {
      local_map.insert(local_ranks(i),i);
    };
    Kokkos::parallel_for(local_ranks.size(), mapConstruct);
  }

  template <typename Space>
  int Distributor<Space>::num_ranks() const {
    if (!isWorld())
      return ranks_d.size();
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    return comm_size;
  }
  template <typename Space>
  PP_DEVICE int Distributor<Space>::rank(int i) const{
    if (!isWorld())
      return ranks_d[i];
    return i;
  }

  template <typename Space>
  int Distributor<Space>::rank_host(int i) const{
    if (!isWorld())
      return ranks_h[i];
    return i;
  }

  template <typename Space>
  PP_DEVICE int Distributor<Space>::index(int process) const {
    if (isWorld())
      return process;
    return mapping.value_at(mapping.find(process));
  }

}
