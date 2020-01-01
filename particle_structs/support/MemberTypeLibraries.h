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
  template <typename DataTypes, typename Device> using MemberTypeViews = void**;
  template <typename DataTypes, typename Device> using MemberTypeViewsConst = void* const*;
  template <typename T, typename Device> using MemberTypeView =
    Kokkos::View<T*, Device>;

  /* Template Fuctions for external usage
       Note: MemorySpace defaults to the default memory space if none is provided

     To create member type views:
     auto views = createMemberViews<DataTypes, MemorySpace>(int size);

     To access one view from a member type view:
     auto view = getMemberView<DataTypes, N, MemorySpace>(views);

     To deallocate the member type views:
     destroyViews<DataTypes, MemorySpace>(views)
   */
  template <typename DataTypes, typename MemSpace = DefaultMemSpace>
    MemberTypeViews<DataTypes, typename MemSpace::device_type> createMemberViews(int size);

  template <typename DataTypes, size_t N, typename MemSpace = DefaultMemSpace>
    MemberTypeView<typename MemberTypeAtIndex<N,DataTypes>::type,typename MemSpace::device_type>
    getMemberView(MemberTypeViews<DataTypes, typename MemSpace::device_type> view);

  template <typename DataTypes, typename MemSpace = DefaultMemSpace>
    void destroyViews(MemberTypeViews<DataTypes, typename MemSpace::device_type> data);

  /* Template Structs
     CreateViews<DataTypes, Device> - creates and allocates member views
     DestryoViews<DataTypes> - deallocates member views
     CopyParticleToSend<ParticleStructure, DataTypes> - copies particle info to send arrays
     CopyPSToPS<ParticleStructure, DataTypes> - copies particle info from ps to ps
     CopyNewParticlesToPS<ParticleStructure, DataTypes> - copies particle info from
                                                          new particles to ps
     ShuffleParticles<ParticleStructure, DataTypes> - shuffles particle info within a ps
     SendViews<DataTypes> - sends views with MPI communications
     RecvViews<DataTypes> - recvs views from MPI communications
   */
  template <typename Device, typename... Types> struct CreateViews;
  template <typename Device, typename... Types> struct DestroyViews;
  template <typename PS, typename... Types> struct CopyParticlesToSend;
  template <typename PS, typename... Types> struct CopyPSToPS;
  template <typename PS, typename... Types> struct CopyNewParticlesToPS;
  template <typename PS, typename... Types> struct ShuffleParticles;
  template <typename Device, typename... Types> struct SendViews;
  template <typename Device, typename... Types> struct RecvViews;

  //Functions
  template <typename DataTypes,typename MemSpace>
    MemberTypeViews<DataTypes, typename MemSpace::device_type> createMemberViews(int size) {
    MemberTypeViews<DataTypes, typename MemSpace::device_type> views;
    CreateViews<typename MemSpace::device_type, DataTypes>(views, size);
    return views;
  }
  template <typename DataTypes, size_t N,typename MemSpace>
    MemberTypeView<typename MemberTypeAtIndex<N,DataTypes>::type,typename MemSpace::device_type>
    getMemberView(MemberTypeViews<DataTypes, typename MemSpace::device_type> view) {
    using Type = typename MemberTypeAtIndex<N, DataTypes>::type;
    return *(static_cast<MemberTypeView<Type, typename MemSpace::device_type>*>(view[N]));
  }
  template <typename DataTypes, typename MemSpace>
    void destroyViews(MemberTypeViews<DataTypes, typename MemSpace::device_type> data) {
    DestroyViews<typename MemSpace::device_type, DataTypes>(data+0);
  }

  //Create Views Templated Struct
  template <typename Device, typename... Types> struct CreateViewsImpl;
  template <typename Device> struct CreateViewsImpl<Device> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<void>, Device>, int) {}
  };
  template <typename Device, typename T, typename... Types> struct CreateViewsImpl<Device, T, Types...> {
    CreateViewsImpl(MemberTypeViews<MemberTypes<T, Types...>, Device > views, int size) {

      views[0] = new MemberTypeView<T, Device>("datatype_view", size);
      MemberTypeView<T, Device> view = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      CreateViewsImpl<Device, Types...>(views+1, size);
    }
  };

  template <typename Device, typename... Types> struct CreateViews<Device, MemberTypes<Types...> > {
    CreateViews(MemberTypeViews<MemberTypes<Types...>, Device>& views, int size) {
      views = new void*[MemberTypes<Types...>::size];
      CreateViewsImpl<Device, Types...>(views, size);
    }
  };


  //Copy Particles To Send Templated Struct
  template <typename PS, typename... Types> struct CopyParticlesToSendImpl;
  template <typename PS> struct CopyParticlesToSendImpl<PS> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void>, Device >,
                            MemberTypeViewsConst<MemberTypes<void>, Device >,
                       typename PS::kkLidView, typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyParticlesToSendImpl<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                            MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs,
                       typename PS::kkLidView ps_to_array,
                       typename PS::kkLidView array_indices) {
      enclose(ps, dsts, srcs,ps_to_array, array_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs,
                 typename PS::kkLidView ps_to_array,
                 typename PS::kkLidView array_indices) {
      int comm_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device> const*>(dsts[0]);
      MemberTypeView<T, Device> src = *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      auto copyPSToArray = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const int arr_index = ps_to_array(ptcl_id);
        if (mask && arr_index != comm_rank) {
          const int index = array_indices(ptcl_id);
          CopyViewToView<T,Device>(dst, index, src, ptcl_id);
        }
      };
      ps->parallel_for(copyPSToArray);
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts+1, srcs+1, ps_to_array,
                                            array_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyParticlesToSend<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyParticlesToSend(PS* ps, MemberTypeViewsConst<MemberTypes<Types...>, Device > dsts,
                        MemberTypeViewsConst<MemberTypes<Types...>, Device > srcs,
                   typename PS::kkLidView ps_to_array,
                   typename PS::kkLidView array_indices) {
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts, srcs, ps_to_array, array_indices);
    }
  };

  template <typename PS, typename... Types> struct CopyPSToPSImpl;
  template <typename PS> struct CopyPSToPSImpl<PS> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void>, Device >,
                   MemberTypeViewsConst<MemberTypes<void>, Device >, typename PS::kkLidView,
                     typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyPSToPSImpl<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                   MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs,
                     typename PS::kkLidView new_element,
                     typename PS::kkLidView ps_indices) {
      enclose(ps,dsts,srcs,new_element, ps_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs,
                 typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device> const*>(dsts[0]);
      MemberTypeView<T, Device> src = *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      auto copyPSToPS = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const lid_t new_elem = new_element(ptcl_id);
        if (mask && new_elem != -1) {
          const int index = ps_indices(ptcl_id);
          CopyViewToView<T,Device>(dst, index, src, ptcl_id);
        }
      };
      ps->parallel_for(copyPSToPS);
      CopyPSToPSImpl<PS, Types...>(ps, dsts+1, srcs+1, new_element, ps_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyPSToPS<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyPSToPS(PS* ps, MemberTypeViewsConst<MemberTypes<Types...>, Device> dsts,
               MemberTypeViewsConst<MemberTypes<Types...>, Device> srcs,
                 typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      CopyPSToPSImpl<PS, Types...>(ps, dsts, srcs, new_element, ps_indices);
    }
  };


  template <typename PS, typename... Types> struct CopyNewParticlesToPSImpl;
  template <typename PS> struct CopyNewParticlesToPSImpl<PS> {
    typedef typename PS::device_type Device;
    CopyNewParticlesToPSImpl(PS* ps, MemberTypeViewsConst<MemberTypes<void>, Device>,
                             MemberTypeViewsConst<MemberTypes<void>, Device>, int,
                              typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyNewParticlesToPSImpl<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyNewParticlesToPSImpl(PS* ps,
                             MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                             MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs,
                             int ne,
                             typename PS::kkLidView ps_indices) {
      enclose(ps,dsts,srcs,ne, ps_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst<MemberTypes<T, Types...>, Device> dsts,
                 MemberTypeViewsConst<MemberTypes<T, Types...>, Device> srcs, int ne,
                 typename PS::kkLidView ps_indices) {
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device> const*>(dsts[0]);
      MemberTypeView<T, Device> src = *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      Kokkos::parallel_for(ne, KOKKOS_LAMBDA(const int& i) {
        const int index = ps_indices(i);
        CopyViewToView<T,Device>(dst, index, src, i);
      });
      CopyNewParticlesToPSImpl<PS, Types...>(ps, dsts+1, srcs+1, ne, ps_indices);
    }
  };
  template <typename PS,typename... Types> struct CopyNewParticlesToPS<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyNewParticlesToPS(PS* ps, MemberTypeViewsConst<MemberTypes<Types...>, Device> dsts,
                         MemberTypeViewsConst<MemberTypes<Types...>, Device> srcs, int ne,
                         typename PS::kkLidView ps_indices) {
      CopyNewParticlesToPSImpl<PS, Types...>(ps, dsts, srcs, ne, ps_indices);
    }
  };

  //Shuffle copy currying structs
  template <typename PS, typename... Types> struct ShuffleParticlesImpl;
  template <typename PS> struct ShuffleParticlesImpl<PS> {
    typedef typename PS::device_type Device;
    typedef typename PS::kkLidView LidView;
    ShuffleParticlesImpl(MemberTypeViewsConst<MemberTypes<void>, Device> ps,
                         MemberTypeViewsConst<MemberTypes<void>, Device> new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {}
  };
  template <typename PS, typename T, typename... Types>
  struct ShuffleParticlesImpl<PS, T, Types...> {
    typedef typename PS::device_type Device;
    typedef typename PS::kkLidView LidView;
    ShuffleParticlesImpl(MemberTypeViewsConst<MemberTypes<T, Types...>, Device > ps,
                         MemberTypeViewsConst<MemberTypes<T, Types...>, Device > new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {
      enclose(ps, new_particles, old_indices, new_indices, fromPS);
    }
    void enclose(MemberTypeViewsConst<MemberTypes<T, Types...>, Device > ps,
                 MemberTypeViewsConst<MemberTypes<T, Types...>, Device > new_particles,
                 LidView old_indices, LidView new_indices, LidView fromPS) {
      int nMoving = old_indices.size();
      MemberTypeView<T, Device> ps_view = *static_cast<MemberTypeView<T, Device> const*>(ps[0]);
      MemberTypeView<T, Device> new_view;
      if (new_particles != NULL) {
        new_view = *static_cast<MemberTypeView<T, Device> const*>(new_particles[0]);
        new_particles++;
      }

      Kokkos::parallel_for(nMoving, KOKKOS_LAMBDA(const lid_t& i) {
          const lid_t old_index = old_indices(i);
          const lid_t new_index = new_indices(i);
          const lid_t isPS = fromPS(i);
          auto src = (isPS == 1 ? ps_view : new_view);
          CopyViewToView<T, Device>(ps_view, new_index, src, old_index);
      });
      ShuffleParticlesImpl<PS, Types...>(ps+1, new_particles, old_indices, new_indices, fromPS);
    }
  };
  template <typename PS, typename... Types> struct ShuffleParticles<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    typedef typename PS::kkLidView LidView;
    ShuffleParticles(MemberTypeViewsConst<MemberTypes<Types...>, Device> ps,
                     MemberTypeViewsConst<MemberTypes<Types...>, Device> new_particles,
                     LidView old_indices, LidView new_indices, LidView fromPS) {
      ShuffleParticlesImpl<PS, Types...>(ps, new_particles, old_indices,
                                                      new_indices, fromPS);
    }
  };

  template <typename Device, typename... Types> struct SendViewsImpl;
  template <typename Device> struct SendViewsImpl<Device> {
    SendViewsImpl(MemberTypeViews<MemberTypes<void>, Device> views, int offset, int size,
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename Device, typename T, typename... Types> struct SendViewsImpl<Device, T, Types...> {
    SendViewsImpl(MemberTypeViews<MemberTypes<T, Types...>, Device> views, int offset, int size,
                  int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T, Device> v = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      PS_Comm_Isend(v, offset, size, dest, tag, MPI_COMM_WORLD, reqs);
      SendViewsImpl<Device, Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename Device, typename... Types> struct SendViews<Device, MemberTypes<Types...>> {
    SendViews(MemberTypeViews<MemberTypes<Types...>, Device> views, int offset, int size,
              int dest, int start_tag, MPI_Request* reqs) {
      SendViewsImpl<Device, Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  template <typename Device, typename... Types> struct RecvViewsImpl;
  template <typename Device> struct RecvViewsImpl<Device> {
    RecvViewsImpl(MemberTypeViews<MemberTypes<void>, Device> views, int offset, int size,
                  int dest, int tag, MPI_Request* reqs) {}
  };
  template <typename Device, typename T, typename... Types> struct RecvViewsImpl<Device, T, Types...> {
    RecvViewsImpl(MemberTypeViews<MemberTypes<T, Types...>, Device > views,
                  int offset, int size, int dest, int tag, MPI_Request* reqs) {
      MemberTypeView<T, Device> v = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      PS_Comm_Irecv(v, offset, size, dest, tag, MPI_COMM_WORLD, reqs);
      RecvViewsImpl<Device, Types...>(views+1, offset, size, dest, tag + 1, reqs + 1);
    }
  };

  template <typename Device, typename... Types> struct RecvViews<Device, MemberTypes<Types...> > {
    RecvViews(MemberTypeViews<MemberTypes<Types...>, Device> views, int offset, int size,
              int dest, int start_tag, MPI_Request* reqs) {
      RecvViewsImpl<Device, Types...>(views, offset, size, dest, start_tag, reqs);
    }
  };

  //Implementation to deallocate views of different types
  template <typename Device, typename... Types> struct DestroyViewsImpl;
  template <typename Device> struct DestroyViewsImpl<Device> {
    DestroyViewsImpl(MemberTypeViews<MemberTypes<void>, Device>) {}
  };
  template <typename Device, typename T, typename... Types> struct DestroyViewsImpl<Device, T,Types...> {
    DestroyViewsImpl(MemberTypeViews<MemberTypes<T,Types...>, Device > data) {
      delete static_cast<MemberTypeView<T, Device>*>(data[0]);
      DestroyViewsImpl<Device, Types...>(data+1);
    }
  };

  //Call to deallocate arrays of different types
  template <typename Device, typename... Types> struct DestroyViews<Device, MemberTypes<Types...> > {
    DestroyViews(MemberTypeViews<MemberTypes<Types...>, Device> data) {
      DestroyViewsImpl<Device, Types...>(data+0);
      delete [] data;
    }
  };

}
