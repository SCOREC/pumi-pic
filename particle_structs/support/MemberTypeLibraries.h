#pragma once

#include "ViewComm.h"
#include "SupportKK.h"
#include "MemberTypeArray.h"
#include "PS_Macros.h"
#include "PS_Types.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdlib>

namespace particle_structs {

  //This type represents an array of views for each type of the given DataTypes
  template <typename DataTypes> using MemberTypeViews = void**;
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
      views = new void*[MemberTypes<Types...>::size];
      CreateViewsImpl<Types...>(views, size);
    }
  };

  template <typename DataTypes>
  MemberTypeViews<DataTypes> createMemberViews(int size) {
    MemberTypeViews<DataTypes> views;
    CreateViews<DataTypes>(views, size);
    return views;
  }
  template <typename DataTypes, size_t N>
  MemberTypeView<typename MemberTypeAtIndex<N, DataTypes>::type>
  getMemberView(MemberTypeViews<DataTypes> view) {
    using Type = typename MemberTypeAtIndex<N, DataTypes>::type;
    return *(static_cast<MemberTypeView<Type>*>(view[N]));
  }

  template <typename PS, typename... Types> struct CopyParticlesToSendImpl;
  template <typename PS> struct CopyParticlesToSendImpl<PS> {
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void> >,
                       MemberTypeViewsConst<MemberTypes<void> >,
                       typename PS::kkLidView, typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyParticlesToSendImpl<PS, T,Types...> {
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                       MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                       typename PS::kkLidView ps_to_array,
                       typename PS::kkLidView array_indices) {
      enclose(ps, dsts, srcs,ps_to_array, array_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                 typename PS::kkLidView ps_to_array,
                 typename PS::kkLidView array_indices) {
      int comm_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      auto copyPSToArray = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const int arr_index = ps_to_array(ptcl_id);
        if (mask && arr_index != comm_rank) {
          const int index = array_indices(ptcl_id);
          CopyViewToView<T,Kokkos::DefaultExecutionSpace::device_type>(dst, index, src, ptcl_id);
        }
      };
      ps->parallel_for(copyPSToArray);
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts+1, srcs+1, ps_to_array, array_indices);
    }

  };
  template <typename PS, typename... Types> struct CopyParticlesToSend;
  template <typename PS,typename... Types> struct CopyParticlesToSend<PS, MemberTypes<Types...> > {
    CopyParticlesToSend(PS* ps, MemberTypeViewsConst<MemberTypes<Types...> > dsts,
                   MemberTypeViewsConst<MemberTypes<Types...> > srcs,
                   typename PS::kkLidView ps_to_array,
                   typename PS::kkLidView array_indices) {
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts, srcs, ps_to_array, array_indices);
    }
  };

  template <typename PS, typename... Types> struct CopyPSToPSImpl;
  template <typename PS> struct CopyPSToPSImpl<PS> {
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void> >,
                     MemberTypeViewsConst<MemberTypes<void> >, typename PS::kkLidView,
                     typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyPSToPSImpl<PS, T,Types...> {
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                     MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                     typename PS::kkLidView new_element,
                     typename PS::kkLidView ps_indices) {
      enclose(ps,dsts,srcs,new_element, ps_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs,
                 typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      auto copyPSToPS = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const lid_t new_elem = new_element(ptcl_id);
        if (mask && new_elem != -1) {
          const int index = ps_indices(ptcl_id);
          CopyViewToView<T,Kokkos::DefaultExecutionSpace::device_type>(dst, index, src, ptcl_id);
        }
      };
      ps->parallel_for(copyPSToPS);
      CopyPSToPSImpl<PS, Types...>(ps, dsts+1, srcs+1, new_element, ps_indices);
    }

  };
  template <typename PS, typename... Types> struct CopyPSToPS;
  template <typename PS,typename... Types> struct CopyPSToPS<PS, MemberTypes<Types...> > {
    CopyPSToPS(PS* ps, MemberTypeViewsConst<MemberTypes<Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<Types...> > srcs,
                 typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      CopyPSToPSImpl<PS, Types...>(ps, dsts, srcs, new_element, ps_indices);
    }
  };


  template <typename PS, typename... Types> struct CopyNewParticlesToPSImpl;
  template <typename PS> struct CopyNewParticlesToPSImpl<PS> {
    CopyNewParticlesToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void> >,
                              MemberTypeViewsConst<MemberTypes<void> >, int,
                              typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyNewParticlesToPSImpl<PS, T,Types...> {
    CopyNewParticlesToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                              MemberTypeViewsConst<MemberTypes<T, Types...> > srcs, int ne,
                     typename PS::kkLidView ps_indices) {
      enclose(ps,dsts,srcs,ne, ps_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...> > dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > srcs, int ne,
                 typename PS::kkLidView ps_indices) {
      MemberTypeView<T> dst = *static_cast<MemberTypeView<T> const*>(dsts[0]);
      MemberTypeView<T> src = *static_cast<MemberTypeView<T> const*>(srcs[0]);
      Kokkos::parallel_for(ne, KOKKOS_LAMBDA(const int& i) {
        const int index = ps_indices(i);
        CopyViewToView<T,Kokkos::DefaultExecutionSpace::device_type>(dst, index, src, i);
      });
      CopyNewParticlesToPSImpl<PS, Types...>(ps, dsts+1, srcs+1, ne, ps_indices);
    }
  };
  template <typename PS, typename... Types> struct CopyNewParticlesToPS;
  template <typename PS,typename... Types> struct CopyNewParticlesToPS<PS, MemberTypes<Types...> > {
    CopyNewParticlesToPS(PS* ps, MemberTypeViewsConst<MemberTypes<Types...> > dsts,
                          MemberTypeViewsConst<MemberTypes<Types...> > srcs, int ne,
                 typename PS::kkLidView ps_indices) {
      CopyNewParticlesToPSImpl<PS, Types...>(ps, dsts, srcs, ne, ps_indices);
    }
  };

  //Shuffle copy currying structs
  template <typename LidView, typename... Types> struct ShuffleParticlesImpl;
  template <typename LidView> struct ShuffleParticlesImpl<LidView> {
    ShuffleParticlesImpl(MemberTypeViewsConst<MemberTypes<void> > ps,
                         MemberTypeViewsConst<MemberTypes<void> > new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {}
  };
  template <typename LidView, typename T, typename... Types>
  struct ShuffleParticlesImpl<LidView, T, Types...> {
    ShuffleParticlesImpl(MemberTypeViewsConst<MemberTypes<T, Types...> > ps,
                         MemberTypeViewsConst<MemberTypes<T, Types...> > new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {
      enclose(ps, new_particles, old_indices, new_indices, fromPS);
    }
    void enclose(MemberTypeViewsConst<MemberTypes<T, Types...> > ps,
                 MemberTypeViewsConst<MemberTypes<T, Types...> > new_particles,
                 LidView old_indices, LidView new_indices, LidView fromPS) {
      int nMoving = old_indices.size();
      MemberTypeView<T> ps_view = *static_cast<MemberTypeView<T> const*>(ps[0]);
      MemberTypeView<T> new_view;
      if (new_particles != NULL) {
        new_view = *static_cast<MemberTypeView<T> const*>(new_particles[0]);
        new_particles++;
      }

      Kokkos::parallel_for(nMoving, KOKKOS_LAMBDA(const lid_t& i) {
          const lid_t old_index = old_indices(i);
          const lid_t new_index = new_indices(i);
          const lid_t isPS = fromPS(i);
          auto src = (isPS == 1 ? ps_view : new_view);
          CopyViewToView<T, Kokkos::DefaultExecutionSpace::device_type>(ps_view, new_index,
                                                                        src, old_index);
      });
      ShuffleParticlesImpl<LidView, Types...>(ps+1, new_particles, old_indices,
                                              new_indices, fromPS);
    }
  };
  template <typename LidView, typename... Types> struct ShuffleParticles;
  template <typename LidView, typename... Types> struct ShuffleParticles<LidView, MemberTypes<Types...> > {
    ShuffleParticles(MemberTypeViewsConst<MemberTypes<Types...> > ps,
                     MemberTypeViewsConst<MemberTypes<Types...> > new_particles,
                     LidView old_indices, LidView new_indices, LidView fromPS) {
      ShuffleParticlesImpl<LidView, Types...>(ps, new_particles, old_indices, new_indices, fromPS);
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
      PS_Comm_Isend(v, offset, size, dest, tag, MPI_COMM_WORLD, reqs);
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
      PS_Comm_Irecv(v, offset, size, dest, tag, MPI_COMM_WORLD, reqs);
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
      delete [] data;
    }
  };

  template <typename DataTypes>
  void destroyViews(MemberTypeViews<DataTypes> data) {
    DestroyViews<DataTypes>(data+0);
  }


}
