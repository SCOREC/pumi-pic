/************** Host Communication functions **************/
template <typename Space> using IsHost =
typename std::enable_if<Kokkos::SpaceAccessibility<typename Space::memory_space, 
                                                                  Kokkos::HostSpace>::accessible, int>::type;
//Send
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Send(ViewT view, int offset, int size,
                                       int dest, int tag, MPI_Comm comm) {
  auto subview = Subview<ViewType<ViewT> >::subview(view, offset, size);
  auto view_host = deviceToHost(subview);
  return MPI_Send(view_host.data(), view_host.size(), MpiType<BT<ViewType<ViewT> > >::mpitype(),
                  dest, tag, comm);
}
//Recv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Recv(ViewT view, int offset, int size,
                                       int sender, int tag, MPI_Comm comm) {
  ViewT new_view("recv_view", size);
  typename ViewT::HostMirror view_host = create_mirror_view(new_view);
  int ret = MPI_Recv(view_host.data(), view_host.size(),
                     MpiType<BT<ViewType<ViewT> > >::mpitype(),
                     sender, tag, comm, MPI_STATUS_IGNORE);
  // Copy received values to device and move it to the proper indices of the view
  deep_copy(new_view, view_host);
  Kokkos::parallel_for(
      size, KOKKOS_LAMBDA(const int &i) {
        copyViewToView(view, i + offset, new_view, i);
      });
  return ret;
}
//Isend
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Isend(ViewT view, int offset, int size,
                                        int dest, int tag, MPI_Comm comm, MPI_Request* req) {
  auto subview = Subview<ViewType<ViewT> >::subview(view, offset, size);
  auto view_host = deviceToHost(subview);
  int ret = MPI_Isend(view_host.data(), view_host.size(),
                      MpiType<BT<ViewType<ViewT> > >::mpitype(), dest,
                      tag, comm, req);
  // Noop that will keep the view_host around until the lambda is removed
  get_map()[req] = [=](){
    (void)view_host;
  };
  return ret;
}
//Irecv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Irecv(ViewT view, int offset, int size,
                                        int sender, int tag, MPI_Comm comm, MPI_Request* req) {
  ViewT new_view("irecv_view", size);
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  typename ViewT::HostMirror view_host = create_mirror_view(new_view);
  int ret = MPI_Irecv(view_host.data(), size * size_per_entry,
                      MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      sender, tag, comm, req);
  get_map()[req] = [=](){
    Kokkos::parallel_for(
        size, KOKKOS_LAMBDA(const int &i) {
          copyViewToView(view, i + offset, new_view, i);
        });
  };
  return ret;
}

//Wait
template <typename Space>
IsHost<Space> PS_Comm_Wait(MPI_Request* req, MPI_Status* stat) {
  int ret = MPI_Wait(req, stat);
  Irecv_Map::iterator itr = get_map().find(req);
  if (itr != get_map().end()){
    (itr->second)();
    get_map().erase(itr);
  }
  return ret;
}

//Waitall
template <typename Space>
IsHost<Space> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
  int ret = MPI_Waitall(num_reqs, reqs, stats);
  for (int i = 0; i < num_reqs; ++i){
    Irecv_Map::iterator itr = get_map().find(reqs + i);
    if (itr != get_map().end()){
      (itr->second)();
      get_map().erase(itr);
    }
  }
  return ret;
}
//Alltoall
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Alltoall(ViewT send, int send_size,
                                           ViewT recv, int recv_size,
                                           MPI_Comm comm) {
  typename ViewT::HostMirror send_host = deviceToHost(send);
  typename ViewT::HostMirror recv_host = create_mirror_view(recv);
  int ret = MPI_Alltoall(send_host.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                         recv_host.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(), comm);
  deep_copy(recv, recv_host);
  return ret;
}

//Ialltoall
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Ialltoall(ViewT send, int send_size,
                                            ViewT recv, int recv_size,
                                            MPI_Comm comm, MPI_Request *request) {
  typename ViewT::HostMirror send_host = deviceToHost(send);
  typename ViewT::HostMirror recv_host = create_mirror_view(recv);
  int ret = MPI_Ialltoall(send_host.data(), send_size,
                          MpiType<BT<ViewType<ViewT> > >::mpitype(),
                          recv_host.data(), recv_size,
                          MpiType<BT<ViewType<ViewT> > >::mpitype(), comm, request);
  get_map()[request] = [=](){
    deep_copy(recv, recv_host);
  };
  return ret;
}

//reduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Reduce(ViewT send_view, ViewT recv_view, int count,
                                         MPI_Op op, int root, MPI_Comm comm) {
  typename ViewT::HostMirror send_host = deviceToHost(send_view);
  typename ViewT::HostMirror recv_host = create_mirror_view(recv_view);
  int ret = MPI_Reduce(send_host.data(), recv_host.data(), count,
                       MpiType<BT<ViewType<ViewT> > >::mpitype(),
                       op, root, comm);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  if (comm_rank == root)
    deep_copy(recv_view, recv_host);
  return ret;
}

//allreduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Allreduce(ViewT send_view, ViewT recv_view, int count,
                                            MPI_Op op, MPI_Comm comm) {
  typename ViewT::HostMirror send_host = deviceToHost(send_view);
  typename ViewT::HostMirror recv_host = create_mirror_view(recv_view);
  int ret = MPI_Allreduce(send_host.data(), recv_host.data(), count,
                          MpiType<BT<ViewType<ViewT> > >::mpitype(), op, comm);
  deep_copy(recv_view, recv_host);
  return ret;
}
