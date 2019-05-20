#pragma once

#include "MemberTypeArray.h"
#include "SCS_Macros.h"
#include "SCS_Types.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdlib>

namespace particle_structs {

  template <typename T> struct MpiType;
#define CREATE_MPITYPE(type, mpi_type) \
  template <> struct MpiType<type> { \
    static constexpr int mpitype = mpi_type; \
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

  //This type represents an array of views for each type of the given DataTypes
  template <typename DataTypes> using MemberTypeViews = void*[DataTypes::size];
  template <typename DataTypes> using MemberTypeViewsConst = void* const*;
  //TODO don't use default execution space
  template <typename T> using MemberTypeView = 
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>;

  template <typename... Types> struct CreateViewsImpl;
  template <> struct CreateViewsImpl<> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<void> >, int) {}
  };
  template <typename T, typename... Types> struct CreateViewsImpl<T, Types...> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int size) {
      
      views[0] = new MemberTypeView<T>("datatype_view", size);
      MemberTypeView<T> view = *static_cast<MemberTypeView<T>*>(views[0]);
      CreateViewsImpl<Types...>(views+1, size);
    }
  };

  template <typename... Types> struct CreateViews;
  template <typename... Types> struct CreateViews<MemberTypes<Types...> > {
    CreateViews(MemberTypeViews<MemberTypes<Types...> >& views, int size) {
      CreateViewsImpl<Types...>(views, size);
    }
  };

  //TODO Make these functions device&host functions
  template <class T> struct CopyViewToViewActual {
    KOKKOS_INLINE_FUNCTION CopyViewToViewActual(MemberTypeView<T> dst, int dst_index,
                                                MemberTypeView<T> src, int src_index) {
      dst(dst_index) = src(src_index);
    }
  };
  template <class T, int N> struct CopyViewToViewActual<T[N]> {
    KOKKOS_INLINE_FUNCTION CopyViewToViewActual(MemberTypeView<T[N]> dst, int dst_index,
                                                MemberTypeView<T[N]> src, int src_index) {
      for (int i = 0; i < N; ++i)
        dst(dst_index, i) = src(src_index, i);
    }
  };
  template <class T, int N, int M> struct CopyViewToViewActual<T[N][M]> {
    KOKKOS_INLINE_FUNCTION CopyViewToViewActual(MemberTypeView<T[N][M]> dst, int dst_index,
                                                MemberTypeView<T[N][M]> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          src(src_index, i, j) = dst(dst_index, i, j);
    }
  };
  template <class T, int N, int M, int P> struct CopyViewToViewActual<T[N][M][P]> {
    KOKKOS_INLINE_FUNCTION CopyViewToViewActual(MemberTypeView<T[N][M][P]> dst, int dst_index,
                                                MemberTypeView<T[N][M][P]> src, int src_index) {
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
          for (int k = 0; k < P; ++k)
            dst(dst_index, i, j, k) = src(src_index, i, j, k);
    }
  };
 
  template <typename SCS, typename... Types> struct CopyParticlesToSendImpl;
  template <typename SCS> struct CopyParticlesToSendImpl<SCS> {
    CopyParticlesToSendImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<void> >,
                       MemberTypeViewsConst<MemberTypes<void> >,
                       typename SCS::kkLidView, typename SCS::kkLidView) {}
  };
  template <typename SCS, typename T, typename... Types> struct CopyParticlesToSendImpl<SCS, T,Types...> {
    CopyParticlesToSendImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                       MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                       typename SCS::kkLidView scs_to_array,
                       typename SCS::kkLidView array_indices) {
      enclose(scs, dsts, srcs,scs_to_array, array_indices);
    }
    void enclose(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                 typename SCS::kkLidView scs_to_array,
                 typename SCS::kkLidView array_indices) {
      int comm_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
      typename SCS::kkLidView array_indices_tmp("tmp", array_indices.size());
      Kokkos::deep_copy(array_indices_tmp, array_indices);
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      auto copySCSToArray = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const int arr_index = scs_to_array(ptcl_id);
        if (mask && arr_index != comm_rank) {
          const int index = Kokkos::atomic_fetch_add(&array_indices_tmp(arr_index), 1);
          CopyViewToViewActual<T>(dst, index, src, ptcl_id);
        }
      };
      scs->parallel_for(copySCSToArray);
      CopyParticlesToSendImpl<SCS, Types...>(scs, dsts+1, srcs+1, scs_to_array, array_indices);
    }

  };
  template <typename SCS, typename... Types> struct CopyParticlesToSend;
  template <typename SCS,typename... Types> struct CopyParticlesToSend<SCS, MemberTypes<Types...> > {
    CopyParticlesToSend(SCS* scs, MemberTypeViewsConst<MemberTypes<Types...> > dsts,  
                   MemberTypeViewsConst<MemberTypes<Types...> > srcs,
                   typename SCS::kkLidView scs_to_array,
                   typename SCS::kkLidView array_indices) {
      CopyParticlesToSendImpl<SCS, Types...>(scs, dsts, srcs, scs_to_array, array_indices);
    }
  };

  template <typename SCS, typename... Types> struct CopySCSToSCSImpl;
  template <typename SCS> struct CopySCSToSCSImpl<SCS> {
    CopySCSToSCSImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<void> >,
                     MemberTypeViewsConst<MemberTypes<void> >, typename SCS::kkLidView,
                     typename SCS::kkLidView, typename SCS::kkLidView) {}
  };
  template <typename SCS, typename T, typename... Types> struct CopySCSToSCSImpl<SCS, T,Types...> {
    CopySCSToSCSImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                     MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                     typename SCS::kkLidView new_element,
                     typename SCS::kkLidView elem_to_scs,
                     typename SCS::kkLidView scs_indices) {
      enclose(scs,dsts,srcs,new_element, elem_to_scs, scs_indices);
    }
    void enclose(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                 typename SCS::kkLidView new_element,
                 typename SCS::kkLidView elem_to_scs,
                 typename SCS::kkLidView scs_indices) {
      typename SCS::kkLidView scs_indices_tmp("tmp", scs_indices.size());
      Kokkos::deep_copy(scs_indices_tmp, scs_indices);
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      int C = scs->C;
      auto copySCSToSCS = SCS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const lid_t new_elem = new_element(ptcl_id);
        if (mask && new_elem != -1) {
          const lid_t new_row = elem_to_scs(new_elem);
          const int index = Kokkos::atomic_fetch_add(&scs_indices_tmp(new_row), C);
          CopyViewToViewActual<T>(dst, index, src, ptcl_id);
        }
      };
      scs->parallel_for(copySCSToSCS);
      CopySCSToSCSImpl<SCS, Types...>(scs, dsts+1, srcs+1, new_element, elem_to_scs, scs_indices);
    }
    
  };
  template <typename SCS, typename... Types> struct CopySCSToSCS;
  template <typename SCS,typename... Types> struct CopySCSToSCS<SCS, MemberTypes<Types...> > {
    CopySCSToSCS(SCS* scs, MemberTypeViewsConst<MemberTypes<Types...> > dsts,  
                 MemberTypeViewsConst<MemberTypes<Types...> > srcs,
                 typename SCS::kkLidView new_element,
                 typename SCS::kkLidView elm_to_scs,
                 typename SCS::kkLidView scs_indices) {
      CopySCSToSCSImpl<SCS, Types...>(scs, dsts, srcs, new_element, elm_to_scs, scs_indices);
    }
  };


  template <typename SCS, typename... Types> struct CopyNewParticlesToSCSImpl;
  template <typename SCS> struct CopyNewParticlesToSCSImpl<SCS> {
    CopyNewParticlesToSCSImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<void> >,
                     MemberTypeViewsConst<MemberTypes<void> >, typename SCS::kkLidView,
                     typename SCS::kkLidView, typename SCS::kkLidView) {}
  };
  template <typename SCS, typename T, typename... Types> struct CopyNewParticlesToSCSImpl<SCS, T,Types...> {
    CopyNewParticlesToSCSImpl(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                     MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                     typename SCS::kkLidView new_element,
                     typename SCS::kkLidView elem_to_scs,
                     typename SCS::kkLidView scs_indices) {
      enclose(scs,dsts,srcs,new_element,elem_to_scs,scs_indices);
    }
    void enclose(SCS* scs, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                 typename SCS::kkLidView new_element,
                 typename SCS::kkLidView elem_to_scs,
                 typename SCS::kkLidView scs_indices) {
      typename SCS::kkLidView scs_indices_tmp("tmp", scs_indices.size());
      Kokkos::deep_copy(scs_indices_tmp, scs_indices);
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      int C = scs->C;
      Kokkos::parallel_for(new_element.size(), KOKKOS_LAMBDA(const int& i) {
        const lid_t new_elem = new_element(i);
        const lid_t new_row = elem_to_scs(new_elem);
        const int index = Kokkos::atomic_fetch_add(&scs_indices_tmp(new_row), C);
        CopyViewToViewActual<T>(dst, index, src, i);
      });
      CopyNewParticlesToSCSImpl<SCS, Types...>(scs, dsts+1, srcs+1, new_element, elem_to_scs, scs_indices);
    }
  };
  template <typename SCS, typename... Types> struct CopyNewParticlesToSCS;
  template <typename SCS,typename... Types> struct CopyNewParticlesToSCS<SCS, MemberTypes<Types...> > {
    CopyNewParticlesToSCS(SCS* scs, MemberTypeViewsConst<MemberTypes<Types...> > dsts,  
                 MemberTypeViewsConst<MemberTypes<Types...> > srcs,
                 typename SCS::kkLidView new_element,
                 typename SCS::kkLidView elm_to_scs,
                 typename SCS::kkLidView scs_indices) {
      CopyNewParticlesToSCSImpl<SCS, Types...>(scs, dsts, srcs, new_element, elm_to_scs, scs_indices);
    }
  };

  template <typename... Types> struct SendViewsImpl;
  template <> struct SendViewsImpl<> {
    SendViewsImpl(MemberTypeViews<MemberTypes<void> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename T, typename... Types> struct SendViewsImpl<T, Types... > {
    SendViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T> v = *static_cast<MemberTypeView<T>*>(views[0]);
      typename BaseType<T>::type* data = v.data();
      int size_per_entry = BaseType<T>::size;
      MPI_Datatype mpi_type = MpiType<typename BaseType<T>::type>::mpitype;
      MPI_Isend(data + offset, size_per_entry * size, mpi_type, dest, tag, MPI_COMM_WORLD, reqs);
      SendViewsImpl<Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename... Types> struct SendViews;
  template <typename... Types> struct SendViews<MemberTypes<Types...> > {
    SendViews(MemberTypeViews<MemberTypes<Types...> > views, int offset, int size, 
              int dest, int start_tag, MPI_Request* reqs) {
      SendViewsImpl<Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  template <typename... Types> struct RecvViewsImpl;
  template <> struct RecvViewsImpl<> {
    RecvViewsImpl(MemberTypeViews<MemberTypes<void> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename T, typename... Types> struct RecvViewsImpl<T, Types... > {
    RecvViewsImpl(MemberTypeViews<MemberTypes<T, Types...> > views, int offset, int size, 
                  int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T> v = *static_cast<MemberTypeView<T>*>(views[0]);
      typename BaseType<T>::type* data = v.data();
      int size_per_entry = BaseType<T>::size;
      MPI_Datatype mpi_type = MpiType<typename BaseType<T>::type>::mpitype;
      MPI_Irecv(data + offset, size_per_entry * size, mpi_type, dest, tag, MPI_COMM_WORLD, reqs);
      RecvViewsImpl<Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename... Types> struct RecvViews;
  template <typename... Types> struct RecvViews<MemberTypes<Types...> > {
    RecvViews(MemberTypeViews<MemberTypes<Types...> > views, int offset, int size, 
              int dest, int start_tag, MPI_Request* reqs) {
      RecvViewsImpl<Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  //Implementation to deallocate views of different types
  template <typename... Types> struct DestroyViewsImpl;
  template <> struct DestroyViewsImpl<> {
    DestroyViewsImpl(MemberTypeViews<MemberTypes<void> >) {}
  };
  template <typename T, typename... Types> struct DestroyViewsImpl<T,Types...> {
    DestroyViewsImpl(MemberTypeViews<MemberTypes<T,Types...> > data) {
      delete static_cast<MemberTypeView<T>*>(data[0]);
      DestroyViewsImpl<Types...>(data+1);
    }
  };

  //Call to deallocate arrays of different types
  template <typename... Types> struct DestroyViews;
  template <typename... Types> struct DestroyViews<MemberTypes<Types...> > {
    DestroyViews(MemberTypeViews<MemberTypes<Types...> > data) {
      DestroyViewsImpl<Types...>(data+0);
    }
  };

}
