#pragma once

#include <Kokkos_Core.hpp>
#include "SupportKK.h"
#include "MemberTypes.h"
#include <unordered_map>
#include <mpi.h>
namespace particle_structs {
  template <typename T> struct MpiType;
#define CREATE_MPITYPE(type, mpi_type)                  \
  template <> struct MpiType<type> {                    \
    static MPI_Datatype mpitype() {return  mpi_type;}   \
  }
  CREATE_MPITYPE(char, MPI_CHAR);
  CREATE_MPITYPE(short, MPI_SHORT);
  CREATE_MPITYPE(int, MPI_INT);
  CREATE_MPITYPE(long, MPI_LONG);
  CREATE_MPITYPE(unsigned char, MPI_UNSIGNED_CHAR);
  CREATE_MPITYPE(unsigned short, MPI_UNSIGNED_SHORT);
  CREATE_MPITYPE(unsigned int, MPI_UNSIGNED);
  CREATE_MPITYPE(unsigned long int, MPI_UNSIGNED_LONG);
  CREATE_MPITYPE(float, MPI_FLOAT);
  CREATE_MPITYPE(double, MPI_DOUBLE);
  CREATE_MPITYPE(long double, MPI_LONG_DOUBLE);
  CREATE_MPITYPE(long long int, MPI_LONG_LONG_INT);
#undef CREATE_MPI_TYPE

  template <typename T> using BT = typename BaseType<T>::type;

  using Irecv_Map=std::unordered_map<MPI_Request*, std::function<void()> >;
  Irecv_Map& get_map();

  /* Routines to be abstracted
     MPI_Allreduce/NCCL
     MPI_Reduce/NCCL
     MPI_Allgather/NCCL
     MPI_Broadcast/NCCL
     MPI_Alltoallv
     MPI_Wait
     MPI_Waitany
   */

  /************** Host Communication functions **************/
  template <typename Device> using IsHost =
    typename std::enable_if<std::is_same<typename Device::memory_space, Kokkos::HostSpace>::value, int>::type;
  //Send
  template <typename T, typename Device>
  IsHost<Device> PS_Comm_Send(Kokkos::View<T*, Device> view, int offset, int size,
                                 int dest, int tag, MPI_Comm comm) {
    int size_per_entry = BaseType<T>::size;
    return MPI_Send(view.data() + offset, size*size_per_entry, MpiType<BT<T> >::mpitype(),
                    dest, tag, comm);
  }
  //Recv
  template <typename T, typename Device>
  IsHost<Device> PS_Comm_Recv(Kokkos::View<T*, Device> view, int offset, int size,
                                 int sender, int tag, MPI_Comm comm) {
    int size_per_entry = BaseType<T>::size;
    return MPI_Recv(view.data() + offset, size*size_per_entry, MpiType<BT<T> >::mpitype(),
                    sender, tag, comm, MPI_STATUS_IGNORE);
  }
  //Isend
  template <typename T, typename Device>
  IsHost<Device> PS_Comm_Isend(Kokkos::View<T*, Device> view, int offset, int size,
                                            int dest, int tag, MPI_Comm comm, MPI_Request* req) {
    int size_per_entry = BaseType<T>::size;
    return MPI_Isend(view.data() + offset, size*size_per_entry, MpiType<BT<T> >::mpitype(),
                     dest, tag, comm, req);
  }
  //Irecv
  template <typename T, typename Device>
  IsHost<Device> PS_Comm_Irecv(Kokkos::View<T*, Device> view, int offset, int size,
                                            int sender, int tag, MPI_Comm comm, MPI_Request* req) {
    int size_per_entry = BaseType<T>::size;
    return MPI_Irecv(view.data() + offset, size*size_per_entry, MpiType<BT<T> >::mpitype(),
                     sender, tag, comm, req);
  }
  //Waitall
  template <typename Device>
  IsHost<Device> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
    return MPI_Waitall(num_reqs, reqs, stats);
  }
  //Alltoall
  template <typename T, typename Device>
  IsHost<Device> PS_Comm_Alltoall(Kokkos::View<T*, Device> send, int send_size,
                                      Kokkos::View<T*, Device> recv, int recv_size,
                                      MPI_Comm comm) {
    return MPI_Alltoall(send.data(), send_size, MpiType<BT<T> >::mpitype(),
                        recv.data(), recv_size, MpiType<BT<T> >::mpitype(), comm);
  }

  /************** Cuda Communication functions **************/
#ifdef PS_USE_CUDA

  //TODO change to check if the memory space is not accessible from host and accessible from cuda
  //Return type check to see if the memory space is not the host space
  template <typename Device> using IsCuda =
  typename std::enable_if<not std::is_same<typename Device::memory_space, Kokkos::HostSpace>::value, int>::type;
//Cuda-aware check for OpenMPI 2.0+ taken from https://github.com/kokkos/kokkos/issues/2003
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#define PS_CUDA_AWARE_MPI
#endif

  //Send
  template <typename T, typename Device>
  IsCuda<Device> PS_Comm_Send(Kokkos::View<T*, Device> view, int offset, int size,
                                 int dest, int tag, MPI_Comm comm) {
    auto subview = Subview<T>::subview(view, offset, size);

#ifdef PS_CUDA_AWARE_MPI
    return MPI_Send(subview.data(), subview.size(), MpiType<BT<T> >::mpitype(),
                    dest, tag, comm);
#else
    auto view_host = deviceToHost(subview);
    return MPI_Send(view_host.data(), view_host.size(), MpiType<BT<T> >::mpitype(),
                    dest, tag, comm);
#endif
  }
  //Recv
  template <typename T, typename Device>
  IsCuda<Device> PS_Comm_Recv(Kokkos::View<T*, Device> view, int offset, int size,
                                 int sender, int tag, MPI_Comm comm) {
    Kokkos::View<T*, Device> new_view("recv_view", size);
#ifdef PS_CUDA_AWARE_MPI
    int ret = MPI_Recv(new_view.data(), new_view.size(), MpiType<BT<T> >::mpitype(),
                       sender, tag, comm, MPI_STATUS_IGNORE);
#else
    typename Kokkos::View<T*, Device>::HostMirror view_host =
      Kokkos::create_mirror_view(new_view);
    int ret = MPI_Recv(view_host.data(), view_host.size(), MpiType<BT<T> >::mpitype(),
                       sender, tag, comm, MPI_STATUS_IGNORE);
    //Copy received values to device and move it to the proper indices of the view
    Kokkos::deep_copy(new_view, view_host);
#endif
    Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
      CopyViewToView<T, Device>(view,i+offset, new_view, i);
    });
    return ret;
  }

  //Isend
  template <typename T, typename Device>
  IsCuda<Device> PS_Comm_Isend(Kokkos::View<T*, Device> view, int offset, int size,
                                  int dest, int tag, MPI_Comm comm, MPI_Request* req) {
    auto subview = Subview<T>::subview(view, offset, size);
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Isend(subview.data(),subview.size(), MpiType<BT<T> >::mpitype(), dest,
                     tag, comm, req);
#else
    auto view_host = deviceToHost(subview);
    int ret =  MPI_Isend(view_host.data(),view_host.size(), MpiType<BT<T> >::mpitype(), dest,
                         tag, comm, req);
    //Noop that will keep the view_host around until the lambda is removed
    get_map()[req] = [=]() {
      (void)view_host;
    };
    return ret;
#endif
  }
  //Irecv
  template <typename T, typename Device>
  IsCuda<Device> PS_Comm_Irecv(Kokkos::View<T*, Device> view, int offset, int size,
                                  int sender, int tag, MPI_Comm comm, MPI_Request* req) {
    int size_per_entry = BaseType<T>::size;
    Kokkos::View<T*, Device> new_view("irecv_view", size);
#ifdef PS_CUDA_AWARE_MPI
    int ret = MPI_Irecv(new_view.data(), new_view.size(), MpiType<BT<T> >::mpitype(), sender,
                        tag, comm, req);
#else
    typename Kokkos::View<T*, Device>::HostMirror view_host =
      Kokkos::create_mirror_view(new_view);
    int ret = MPI_Irecv(view_host.data(), size * size_per_entry, MpiType<BT<T> >::mpitype(),
                        sender, tag, comm, req);
#endif

    get_map()[req] = [=]() {
#ifndef PS_CUDA_AWARE_MPI
      Kokkos::deep_copy(new_view, view_host);
#endif
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int& i) {
        CopyViewToView<T, Device>(view,i+offset, new_view, i);
      });
    };

    return ret;

  }
  //Waitall
  template <typename Device>
  IsCuda<Device> PS_Comm_Waitall(int num_reqs, MPI_Request* reqs, MPI_Status* stats) {
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
  template <typename T, typename Device>
  IsCuda<Device> PS_Comm_Alltoall(Kokkos::View<T*, Device> send, int send_size,
                                               Kokkos::View<T*, Device> recv, int recv_size,
                                               MPI_Comm comm) {
#ifdef PS_CUDA_AWARE_MPI
    return MPI_Alltoall(send.data(), send_size, MpiType<BT<T> >::mpitype(),
                        recv.data(), recv_size, MpiType<BT<T> >::mpitype(), comm);
#else
    typename Kokkos::View<T*, Device>::HostMirror send_host = deviceToHost(send);
    typename Kokkos::View<T*, Device>::HostMirror recv_host = Kokkos::create_mirror_view(recv);
    int ret = MPI_Alltoall(send_host.data(), send_size, MpiType<BT<T> >::mpitype(),
                           recv_host.data(), recv_size, MpiType<BT<T> >::mpitype(), comm);
    Kokkos::deep_copy(recv, recv_host);
    return ret;
#endif
  }

#endif



}
