#include "ViewComm.h"

int comm_rank, comm_size;
template <typename Space>
int sendRecvTest(const char* name, int msg_size);
template <typename Space>
int iSendRecvWaitTest(const char* name, int msg_size);
template <typename Space>
int iSendRecvWaitAllTest(const char* name);
template <typename Space>
int allToAllTest(const char* name, int msg_size);
template <typename Space>
int reductionTest(const char* name);

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int fails = 0;

  //Test Host Functions
  fails += sendRecvTest<Kokkos::HostSpace>("Small send/recv",10);
  fails += sendRecvTest<Kokkos::HostSpace>("Large send/recv",10000);
  fails += iSendRecvWaitAllTest<Kokkos::HostSpace>("Isend/Irecv + Waitall");

  //Test Cuda Functions
  fails += sendRecvTest<Kokkos::CudaSpace>("Small send/recv",10);
  fails += sendRecvTest<Kokkos::CudaSpace>("Large send/recv",10000);
  fails += iSendRecvWaitAllTest<Kokkos::CudaSpace>("Isend/Irecv + Waitall");
  MPI_Finalize();
  Kokkos::finalize();
  return fails;
}
template <typename Space>
int sendRecvTest(const char* name, int msg_size) {
  //Setup
  if (!comm_rank)
    printf("Beginning Test %s_%s\n", name, Space::name());
  int fails = 0;
  Kokkos::View<int*, Space> device_fails("failures", 1);
  int local_rank = comm_rank;
  int local_size = comm_size;

  typedef Kokkos::RangePolicy<typename Space::execution_space> ExecPolicy;
  typename Space::execution_space exec;

  //Kokkos View Test
  if (!comm_rank)
    printf("  Kokkos View Test\n");
  // Even ranks send 1 to msg_size to Odd ranks (rank sends to rank + 1)
  if (comm_rank % 2 == 0 && comm_rank != local_size - 1) {
    Kokkos::View<int*, Space> send_view("send_view", msg_size);
    Kokkos::parallel_for(ExecPolicy(exec, 0, msg_size), KOKKOS_LAMBDA(const int i) {
        send_view(i) = i;
    });
    int ret = pumipic::PS_Comm_Send(send_view,0, msg_size, comm_rank + 1, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Send returned error code %d\n",
              comm_rank, ret);
    }
  }
  else if (comm_rank % 2 == 1) {
    Kokkos::View<int*, Space> recv_view("recv_view", msg_size);
    int ret = pumipic::PS_Comm_Recv(recv_view, 0, msg_size, comm_rank - 1, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Recv returned error code %d\n",
              comm_rank, ret);
      ++fails;
    }
    Kokkos::parallel_for(ExecPolicy(exec, 0, msg_size), KOKKOS_LAMBDA(const int i) {
      if (recv_view(i) != i) {
        printf("[ERROR] Rank %d: Recieved value is incorrect for index %d"
               "[(actual)%d != %d(should be)]\n",local_rank, i, recv_view(i), i);
        Kokkos::atomic_add(&(device_fails(0)), 1);
      }
    });
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!comm_rank)
    printf("  PUMIPic View Test\n");
  if (comm_rank % 2 == 0 && comm_rank != local_size - 1) {
    pumipic::View<int*, Space> send_view("send_view", msg_size);
    Kokkos::parallel_for(ExecPolicy(exec, 0, msg_size), KOKKOS_LAMBDA(const int i) {
        send_view(i) = i;
    });
    int ret = pumipic::PS_Comm_Send(send_view,0, msg_size, comm_rank + 1, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Send returned error code %d\n",
              comm_rank, ret);
    }
  }
  else if (comm_rank % 2 == 1) {
    pumipic::View<int*, Space> recv_view("recv_view", msg_size);
    int ret = pumipic::PS_Comm_Recv(recv_view, 0, msg_size, comm_rank - 1, 0, MPI_COMM_WORLD);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Recv returned error code %d\n",
              comm_rank, ret);
      ++fails;
    }
    Kokkos::parallel_for(ExecPolicy(exec, 0, msg_size), KOKKOS_LAMBDA(const int i) {
      if (recv_view(i) != i) {
        printf("[ERROR] Rank %d: Recieved value is incorrect for index %d"
               "[(actual)%d != %d(should be)]\n",local_rank, i, recv_view(i), i);
        Kokkos::atomic_add(&(device_fails(0)), 1);
      }
    });
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //Closing
  fails += pumipic::getLastValue<int>(device_fails);
  int final_fail;
  MPI_Allreduce(&fails, &final_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return final_fail > 0;
}

template <typename Space>
int iSendRecvWaitAllTest(const char* name) {
  //Setup
  if (!comm_rank)
    printf("Beginning Test %s_%s\n", name, Space::name());
  int fails = 0;
  Kokkos::View<int*, Space> device_fails("failures", 1);
  int local_rank = comm_rank;
  int local_size = comm_size;
  typedef Kokkos::RangePolicy<typename Space::execution_space> ExecPolicy;
  typename Space::execution_space exec;

  //Kokkos View Test
  if (!comm_rank)
    printf("  Kokkos View Test\n");
  //Send 2^comm_rank to every other rank
  {
    MPI_Request* send_requests = new MPI_Request[comm_size];
    MPI_Request* recv_requests = new MPI_Request[comm_size];
    Kokkos::View<unsigned long int*, Space> send_view("send_view", local_size);
    Kokkos::View<unsigned long int*, Space> recv_view("recv_view", local_size);
    Kokkos::parallel_for(ExecPolicy(exec,0, local_size), KOKKOS_LAMBDA(const int i) {
      send_view(i) = pow(2, local_rank);
    });
    for (int i = 0; i < local_size; ++i) {
      int ret = pumipic::PS_Comm_Isend(send_view, i, 1, i, 0, MPI_COMM_WORLD,
                                       send_requests + i);
      if (ret != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Isend to %d returned error code %d\n",
                comm_rank, i, ret);
        ++fails;
      }
      ret = pumipic::PS_Comm_Irecv(recv_view, i, 1, i, 0, MPI_COMM_WORLD, recv_requests + i);
      if (ret != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Irecv to %d returned error code %d\n",
                comm_rank, i, ret);
        ++fails;
      }
    }
    MPI_Status* statuses = new MPI_Status[comm_size];
    int ret = pumipic::PS_Comm_Waitall<Space>(comm_size, send_requests, statuses);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Waitall on send requests returned "
              "error code %d\n", comm_rank, ret);
      ++fails;
    }
    ret = pumipic::PS_Comm_Waitall<Space>(comm_size, recv_requests, statuses);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Waitall on recv requests returned "
              "error code %d\n", comm_rank, ret);
      ++fails;
    }
    Kokkos::parallel_for(ExecPolicy(exec, 0, local_size), KOKKOS_LAMBDA(const int i) {
        unsigned long int p = pow(2, i);
      if (recv_view(i) != p) {
        printf("[ERROR] Rank %d: has incorrect value on element %d"
               "[(actual) %lu != %lu (should be)]\n", local_rank, i, recv_view(i), p);
        Kokkos::atomic_add(&(device_fails(0)), 1);
      }
    });
    delete [] send_requests;
    delete [] recv_requests;
    delete [] statuses;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (!comm_rank)
    printf("  PUMIPic View Test\n");
  {
    MPI_Request* send_requests = new MPI_Request[comm_size];
    MPI_Request* recv_requests = new MPI_Request[comm_size];
    pumipic::View<unsigned long int*, Space> send_view("send_view", local_size);
    pumipic::View<unsigned long int*, Space> recv_view("recv_view", local_size);
    Kokkos::parallel_for(ExecPolicy(exec,0, local_size), KOKKOS_LAMBDA(const int i) {
      send_view(i) = pow(2, local_rank);
    });
    for (int i = 0; i < local_size; ++i) {
      int ret = pumipic::PS_Comm_Isend(send_view, i, 1, i, 0, MPI_COMM_WORLD,
                                       send_requests + i);
      if (ret != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Isend to %d returned error code %d\n",
                comm_rank, i, ret);
        ++fails;
      }
      ret = pumipic::PS_Comm_Irecv(recv_view, i, 1, i, 0, MPI_COMM_WORLD, recv_requests + i);
      if (ret != MPI_SUCCESS) {
        fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Irecv to %d returned error code %d\n",
                comm_rank, i, ret);
        ++fails;
      }
    }
    MPI_Status* statuses = new MPI_Status[comm_size];
    int ret = pumipic::PS_Comm_Waitall<Space>(comm_size, send_requests, statuses);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Waitall on send requests returned "
              "error code %d\n", comm_rank, ret);
      ++fails;
    }
    ret = pumipic::PS_Comm_Waitall<Space>(comm_size, recv_requests, statuses);
    if (ret != MPI_SUCCESS) {
      fprintf(stderr, "[ERROR] Rank %d: PS_Comm_Waitall on recv requests returned "
              "error code %d\n", comm_rank, ret);
      ++fails;
    }
    Kokkos::parallel_for(ExecPolicy(exec, 0, local_size), KOKKOS_LAMBDA(const int i) {
        unsigned long int p = pow(2, i);
      if (recv_view(i) != p) {
        printf("[ERROR] Rank %d: has incorrect value on element %d"
               "[(actual) %lu != %lu (should be)]\n", local_rank, i, recv_view(i), p);
        Kokkos::atomic_add(&(device_fails(0)), 1);
      }
    });
  }

  //Closing
  fails += pumipic::getLastValue<int>(device_fails);
  int final_fail;
  MPI_Allreduce(&fails, &final_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return final_fail > 0;
}
