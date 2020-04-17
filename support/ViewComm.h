#pragma once

#include <Kokkos_Core.hpp>
#include "SupportKK.h"
#include <unordered_map>
#include <mpi.h>
namespace pumipic {
#if false //These function headers are for documentation purposes only
  /*! \file View Communications
    \brief Abstractions over various MPI routines for views located on either
    the host or device and abstract support for CUDA aware MPI.

    \note Views can be on either the host or device, but must be in the same memory
    space for matching send/recv calls

    \note Mixing of PS_Comm and direct MPI calls is not encouraged and may cause some
    failures when sending data from the device

  */
  /*!
    \brief Wrapper around MPI_Send for views

    \tparam ViewT The type of view, supports Kokkos::View & pumipic::View

    \param view The view with data on either the host or device

    \param offset The starting index of the data to send

    \param size The number of elements of the view to send

    \param dest The destination rank to send the message to

    \param tag The tag associated with the MPI_Send

    \param comm The MPI communicator

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Send(view.data() + offset, size, datatype, dest, tag, comm);

  */
  template <typename ViewT>
  int PS_Comm_Send(ViewT view, int offset, int size, int dest, int tag, MPI_Comm comm);
  /*!
    \brief Wrapper around MPI_Recv for views

    \tparam ViewT The type of view, supports Kokkos::View & pumipic::View

    \param view The view with data on either the host or device

    \param offset The starting index of where the data will be received in the view

    \param size The number of elements to receive

    \param source The source rank to receive the message from

    \param tag The tag associated with the MPI_Recv

    \param comm The MPI communicator

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Recv(view.data() + offset, size, datatype, source, tag, comm);

  */

  template <typename ViewT>
  int PS_Comm_Recv(ViewT view, int offset, int size, int source, int tag, MPI_Comm comm);

  /*!
    \brief Wrapper around MPI_Isend for views

    \tparam ViewT The type of view, supports Kokkos::View & pumipic::View

    \param view The view with data on either the host or device

    \param offset The starting index of the data to send

    \param size The number of elements of the view to send

    \param dest The destination rank to send the message to

    \param tag The tag associated with the MPI_Send

    \param comm The MPI communicator

    \param[out] request The MPI request to be filled after the MPI_Isend completes

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Isend(view.data() + offset, size, datatype, dest, tag, comm, request);
  */

  template <typename ViewT>
  int PS_Comm_Isend(ViewT view, int offset, int size, int dest, int tag,
                    MPI_Comm comm, MPI_Request* request);

  /*!
    \brief Wrapper around MPI_Irecv for views

    \tparam ViewT The type of view, supports Kokkos::View & pumipic::View

    \param view The view with data on either the host or device

    \param offset The starting index of where the data will be received in the view

    \param size The number of elements to receive

    \param source The source rank to receive the message from

    \param tag The tag associated with the MPI_Recv

    \param comm The MPI communicator

    \param[out] request The MPI request to be filled after the MPI_Irecv completes

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Irecv(view.data() + offset, size, datatype, source, tag, comm, request);

  */

  template <typename ViewT>
  int PS_Comm_Irecv(ViewT view, int offset, int size, int dest, int tag,
                    MPI_Comm comm, MPI_Request* request);

  /*!
    \brief Wrapper around MPI_Wait

    \tparam Space The memory space where the sends/recvs occurred

    \param request The requests to wait on

    \param[out] status A status filled by the MPI_Wait

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Wait(request, status);

    \note PS_Comm_Wait must be used instead of MPI_Wait if using the
    PS_Comm_Isend/Irecv functions on the device in order to finish copying the data.

  */
  template <typename Space>
  int PS_Comm_Wait(MPI_Request* request, MPI_Status* status);

  /*!
    \brief Wrapper around MPI_Waitall

    \tparam Space The memory space where the sends/recvs occurred

    \param num_requests The number of requests

    \param requests The array of requests sized `num_requests`

    \param[out] statuses An array of statuses filled by the MPI_Waitall

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Waitall(num_requests, requests, statuses);

    \note PS_Comm_Waitall must be used instead of MPI_Waitall if using the
    PS_Comm_Isend/Irecv functions on the device in order to finish copying the data.

  */
  template <typename Space>
  int PS_Comm_Waitall(int num_requests, MPI_Request* requests, MPI_Status* statuses);

  /*!
    \brief Wrapper around MPI_Alltoall for views

    \tparam ViewT The type of view, supports Kokkos::View & pumipic::View

    \param send_view The view with data on either the host or device to send

    \param send_size The number of elements to send to each process

    \param recv_view The view with data on either the host or device to receive

    \param recv_size The number of elements to recv from each process

    \param comm The MPI communicator

    \return The error value returned by the call to MPI

    \note The function call is equivalent to
    MPI_Alltoall(send_view.data(), send_size, send_datatype,
                 recv_view.data(), recv_size, recv_datatype, comm);

    \note The send_view and recv_view must be allocated on the same memory space
  */
  template <typename ViewT>
  int PS_Comm_Alltoall(ViewT send_view, int send_size, ViewT recv_view, int recv_size,
                       MPI_Comm comm);
#endif

  template <typename T> struct MpiType;
#define CREATE_MPITYPE(type, mpi_type)                  \
  template <> struct MpiType<type> {                    \
    static MPI_Datatype mpitype() {return  mpi_type;}   \
  }
  CREATE_MPITYPE(char, MPI_CHAR);
  CREATE_MPITYPE(short, MPI_SHORT);
  CREATE_MPITYPE(bool, MPI::BOOL);
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

#include "ViewComm_host.hpp"

#include "ViewComm_cuda.hpp"


}
