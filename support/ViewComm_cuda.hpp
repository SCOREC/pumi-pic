  /************** Cuda Communication functions **************/

#ifdef PP_USE_CUDA
  //Return type check to see if the memory space is not the host space
  template <typename Space> using IsCuda =
    typename std::enable_if<!Kokkos::SpaceAccessibility<typename Space::memory_space,
                                                        Kokkos::HostSpace>::accessible &&
                      Kokkos::SpaceAccessibility<typename Space::memory_space,
                                                 Kokkos::CudaSpace>::accessible, int>::type;
  // typename std::enable_if<std::is_same<typename Space::memory_space, Kokkos::CudaSpace>::value, int>::type;
//Cuda-aware check for OpenMPI 2.0+ taken from https://github.com/kokkos/kokkos/issues/2003
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#define PS_CUDA_AWARE_MPI
#endif

  //Send
  template <typename ViewT>
  IsCuda<ViewSpace<ViewT> > PS_Comm_Send(ViewT view, int offset, int size,
                                         int dest, int tag, MPI_Comm comm) {
    auto subview = Subview<ViewType<ViewT> >::subview(view, offset, size);

#ifdef PS_CUDA_AWARE_MPI
    return MPI_Send(subview.data(), subview.size(), MpiType<BT<ViewType<ViewT> > >::mpitype(),
                    dest, tag, comm);
#else
    auto view_host = deviceToHost(subview);
    return MPI_Send(view_host.data(), view_host.size(), MpiType<BT<ViewType<ViewT> > >::mpitype(),
                    dest, tag, comm);
#endif
  }
  //Recv
  template <typename ViewT>
  IsCuda<ViewSpace<ViewT> > PS_Comm_Recv(ViewT view, int offset, int size,
                                         int sender, int tag, MPI_Comm comm) {
    ViewT new_view("recv_view", size);
#ifdef PS_CUDA_AWARE_MPI
    int ret = MPI_Recv(new_view.data(), new_view.size(),
                       MpiType<BT<ViewType<ViewT> > >::mpitype(),
                       sender, tag, comm, MPI_STATUS_IGNORE);
#else
    typename ViewT::HostMirror view_host = Kokkos::create_mirror_view(new_view);
    int ret = MPI_Recv(view_host.data(), view_host.size(),
                       MpiType<BT<ViewType<ViewT> > >::mpitype(),
                       sender, tag, comm, MPI_STATUS_IGNORE);
    //Copy received values to device and move it to the proper indices of the view
    Kokkos::deep_copy(new_view, view_host);
#endif
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        copyViewToView(view,i+offset, new_view, i);
    });
    return ret;
  }

  //Isend
  template <typename ViewT>
  IsCuda<ViewSpace<ViewT> > PS_Comm_Isend(ViewT view, int offset, int size,
                                  int dest, int tag, MPI_Comm comm, MPI_Request* req) {
    auto subview = Subview<ViewType<ViewT> >::subview(view, offset, size);
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Isend(subview.data(), subview.size(),
                     MpiType<BT<ViewType<ViewT> > >::mpitype(), dest,
                     tag, comm, req);
#else
    auto view_host = deviceToHost(subview);
    int ret =  MPI_Isend(view_host.data(), view_host.size(),
                         MpiType<BT<ViewType<ViewT> > >::mpitype(), dest,
                         tag, comm, req);
    //Noop that will keep the view_host around until the lambda is removed
    get_map()[req] = [=]() {
      (void)view_host;
    };
    return ret;
#endif
  }
  //Irecv
  template <typename ViewT>
  IsCuda<ViewSpace<ViewT> > PS_Comm_Irecv(ViewT view, int offset, int size,
                                  int sender, int tag, MPI_Comm comm, MPI_Request* req) {
    int size_per_entry = BaseType<ViewType<ViewT> >::size;
    ViewT new_view("irecv_view", size);
#ifdef PS_CUDA_AWARE_MPI
    int ret = MPI_Irecv(new_view.data(), new_view.size(),
                        MpiType<BT<ViewType<ViewT> > >::mpitype(), sender,
                        tag, comm, req);
#else
    typename ViewT::HostMirror view_host = Kokkos::create_mirror_view(new_view);
    int ret = MPI_Irecv(view_host.data(), size * size_per_entry,
                        MpiType<BT<ViewType<ViewT> > >::mpitype(),
                        sender, tag, comm, req);
#endif
    get_map()[req] = [=]() {
#ifndef PS_CUDA_AWARE_MPI
      Kokkos::deep_copy(new_view, view_host);
#endif
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
          copyViewToView(view,i+offset, new_view, i);
      });
    };

    return ret;

  }

  //Wait
  template <typename Space>
  IsCuda<Space> PS_Comm_Wait(MPI_Request* req, MPI_Status* stat) {
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Wait(req, stat);
#else
    int ret = MPI_Wait(req, stat);
    Irecv_Map::iterator itr = get_map().find(req);
    if (itr != get_map().end()) {
      (itr->second)();
      get_map().erase(itr);
    }
    return ret;
#endif
  }

  //Waitall
  template <typename Space>
  IsCuda<Space> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Waitall(num_reqs, reqs, stats);
#else
    int ret = MPI_Waitall(num_reqs, reqs, stats);
    for (int i = 0; i < num_reqs; ++i) {
      Irecv_Map::iterator itr = get_map().find(reqs + i);
      if (itr != get_map().end()) {
        (itr->second)();
        get_map().erase(itr);
      }
    }
    return ret;
#endif

  }

  //Alltoall
  template <typename ViewT>
  IsCuda<ViewSpace<ViewT> > PS_Comm_Alltoall(ViewT send, int send_size,
                                               ViewT recv, int recv_size,
                                               MPI_Comm comm) {
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Alltoall(send.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                        recv.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(), comm);
#else
    typename ViewT::HostMirror send_host = deviceToHost(send);
    typename ViewT::HostMirror recv_host = Kokkos::create_mirror_view(recv);
    int ret = MPI_Alltoall(send_host.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                           recv_host.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(), comm);
    Kokkos::deep_copy(recv, recv_host);
    return ret;
#endif
  }

#endif
