/************** Host Communication functions **************/
template <typename Space> using IsHost =
  typename std::enable_if<Kokkos::SpaceAccessibility<typename Space::memory_space,
                                                     Kokkos::HostSpace>::accessible, int>::type;
//Send
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Send(ViewT view, int offset, int size,
                                       int dest, int tag, MPI_Comm comm) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Send(view.data() + offset, size*size_per_entry,
                  MpiType<BT<ViewType<ViewT> > >::mpitype(), dest, tag, comm);
}
//Recv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Recv(ViewT view, int offset, int size,
                                       int sender, int tag, MPI_Comm comm) {
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Recv(view.data() + offset, size*size_per_entry, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                  sender, tag, comm, MPI_STATUS_IGNORE);
}
//Isend
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Isend(ViewT view, int offset, int size,
                                        int dest, int tag, MPI_Comm comm, MPI_Request* req) {
#ifdef PP_USE_CUDA
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Isend(view.data() + offset, size*size_per_entry, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                   dest, tag, comm, req);
#else
  auto subview = Subview<ViewType<ViewT> >::subview(view, offset, size);
  int ret = MPI_Isend(subview.data(), subview.size(),
                      MpiType<BT<ViewType<ViewT> > >::mpitype(), dest,
                      tag, comm, req);
  // Noop that will keep the view_host around until the lambda is removed
  get_map()[req] = [=](){
    (void)subview;
  };
  return ret;
#endif
}
//Irecv
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Irecv(ViewT view, int offset, int size,
                                        int sender, int tag, MPI_Comm comm, MPI_Request* req) {
#ifdef PP_USE_CUDA
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  return MPI_Irecv(view.data() + offset, size*size_per_entry,
                   MpiType<BT<ViewType<ViewT> > >::mpitype(),
                   sender, tag, comm, req);
#else
  ViewT new_view("irecv_view", size);
  int size_per_entry = BaseType<ViewType<ViewT> >::size;
  int ret = MPI_Irecv(new_view.data(), size * size_per_entry,
                      MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      sender, tag, comm, req);
  get_map()[req] = [=](){
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int &i) {
      copyViewToView(view, i + offset, new_view, i);
    });
  };
  return ret;
#endif
}

//Wait
template <typename Space>
IsHost<Space> PS_Comm_Wait(MPI_Request* req, MPI_Status* stat) {
#ifdef PP_USE_CUDA
  return MPI_Wait(req, stat);
#else
  int ret = MPI_Wait(req, stat);
  Irecv_Map::iterator itr = get_map().find(req);
  if (itr != get_map().end()){
    (itr->second)();
    get_map().erase(itr);
  }
  return ret;
#endif
}

//Waitall
template <typename Space>
IsHost<Space> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
#ifdef PP_USE_CUDA
  return MPI_Waitall(num_reqs, reqs, stats);
#else
  int ret = MPI_Waitall(num_reqs, reqs, stats);
  for (int i = 0; i < num_reqs; ++i){
    Irecv_Map::iterator itr = get_map().find(reqs + i);
    if (itr != get_map().end()){
      (itr->second)();
      get_map().erase(itr);
    }
  }
  return ret;
#endif
}
//Alltoall
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Alltoall(ViewT send, int send_size,
                                           ViewT recv, int recv_size,
                                           MPI_Comm comm) {
  return MPI_Alltoall(send.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      recv.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(), comm);
}

//Ialltoall
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Ialltoall(ViewT send, int send_size,
                                            ViewT recv, int recv_size,
                                            MPI_Comm comm, MPI_Request* request) {
  return MPI_Ialltoall(send.data(), send_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      recv.data(), recv_size, MpiType<BT<ViewType<ViewT> > >::mpitype(),
                      comm, request);
}

//reduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Reduce(ViewT send_view, ViewT recv_view, int count,
                                         MPI_Op op, int root, MPI_Comm comm) {
  return MPI_Reduce(send_view.data(), recv_view.data(), count,
                    MpiType<BT<ViewType<ViewT> > >::mpitype(),
                    op, root, comm);

}

//allreduce
template <typename ViewT>
IsHost<ViewSpace<ViewT> > PS_Comm_Allreduce(ViewT send_view, ViewT recv_view, int count,
                                            MPI_Op op, MPI_Comm comm) {
  return MPI_Allreduce(send_view.data(), recv_view.data(), count,
                       MpiType<BT<ViewType<ViewT> > >::mpitype(), op, comm);
}
