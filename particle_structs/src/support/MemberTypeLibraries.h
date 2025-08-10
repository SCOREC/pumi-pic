#pragma once
#include <cassert>
#include <ViewComm.h>
#include <SupportKK.h>
#include "MemberTypeArray.h"
#include <ppMacros.h>
#include <ppTypes.h>
#include <ppView.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdlib>
#include "ppPrint.h"

namespace pumipic {

  //This type represents an array of views for each type of the given DataTypes
  using MemberTypeViews = void**;
  using MemberTypeViewsConst = void* const*;
  template <typename T, typename Device> using MemberTypeView = View<T*, Device>;

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
    MemberTypeViews createMemberViews(int size);

  template <typename DataTypes, size_t N, typename MemSpace = DefaultMemSpace>
    MemberTypeView<typename MemberTypeAtIndex<N,DataTypes>::type,typename MemSpace::device_type>
    getMemberView(MemberTypeViews view);

  template <typename DataTypes, typename MemSpace = DefaultMemSpace>
    void destroyViews(MemberTypeViews data);

  /******* Template Structs ******/
  /* CreateViews<DataTypes, Device> - creates and allocates member views
       Usage: CreateViews<Device, DataTypes>(MemberTypeViews, size)
   */
  template <typename Device, typename... Types> struct CreateViews;
  /* DestroyViews<DataTypes> - deallocates member views
       Usage: DestroyViews<Device, MemberTypes>(MemberTypeViews + 0)
       Note: The + 0 may be required in order for the compiler to correctly understand the call
   */
  template <typename Device, typename... Types> struct DestroyViews;
  /* CopyViewsToViews<ViewType, DataTypes> - copies particle info from one view to specific
                                             indices in another view
       Usage: CopyViewsToViews<ViewType, MemberTypes>(DestiationMemberTypeViews,
                                                      SourceMemberTypeViews,
                                                      DestionationIndexPerSource);
  */
  template <typename PS, typename... Types> struct CopyViewsToViews;
  /* ShuffleParticles<ParticleStructure, DataTypes> - shuffles particle info within a ps
                                                      and add new particles
       Usage: ShuffleParticles<ParticleStructure, MemberTypes>(PSMemberTypeViews,
                                                               NewPtclMemberTypeViews,
                                                               SourceIndex,
                                                               DestinationIndex,
                                                               IfEntryIsDrawnFromPS);
   */
  template <typename Device, typename... Types> struct ShuffleParticles;
  /* SendViews<Device, DataTypes> - sends views with MPI communications
       Usage: SendViews<Device, MemberTypes>(MemberTypesViews, offsetFromStart,
                                             numberOfEntries, destinationRank, initialTag,
                                             MPI_Comm, ArrayOfRequests);
   */
  template <typename Device, typename... Types> struct SendViews;
  /* RecvViews<Device, DataTypes> - recvs views from MPI communications
       Usage: RecvViews<Device, MemberTypes>(MemberTypeViews, offsetFromStart,
                                             numberOfEntries, sendingRank, initialTag,
                                             MPI_Comm, ArrayOfRequests);
   */
  template <typename Device, typename... Types> struct RecvViews;
  /* CopyMemSpaceToMemSpace<DestinationMemSpace, SourceMemSpace, DataTypes> -
           Copies Member type views from one memory space to another memory space
      Usage: CopyMSpaceToMSpace<DestinationMemSpace, SourceMemSpace, MemberTypes>(DestinationMTV,
                                                                                  SourceMTV)
  */
  template <typename MSpace1, typename MSpace2, typename... Types> struct CopyMemSpaceToMemSpace;


  //Functions
  template <typename DataTypes,typename MemSpace>
    MemberTypeViews createMemberViews(int size) {
    MemberTypeViews views;
    CreateViews<typename MemSpace::device_type, DataTypes>(views, size);
    return views;
  }
  template <typename DataTypes, size_t N,typename MemSpace>
    MemberTypeView<typename MemberTypeAtIndex<N,DataTypes>::type,typename MemSpace::device_type>
    getMemberView(MemberTypeViews view) {
    using Type = typename MemberTypeAtIndex<N, DataTypes>::type;
    return *(static_cast<MemberTypeView<Type, typename MemSpace::device_type>*>(view[N]));
  }
  template <typename DataTypes, typename MemSpace>
    void destroyViews(MemberTypeViews data) {
    DestroyViews<typename MemSpace::device_type, DataTypes>(data+0);
  }

  //Create Views Templated Struct
  template <typename Device, typename... Types> struct CreateViewsImpl;
  template <typename Device> struct CreateViewsImpl<Device> {
    CreateViewsImpl(MemberTypeViews, int, int) {}
  };
  template <typename Device, typename T, typename... Types> struct CreateViewsImpl<Device, T, Types...> {
    CreateViewsImpl(MemberTypeViews views, int size, int num) {

      char name[100];
      sprintf(name, "datatype_view_%d", num);
      views[0] = new MemberTypeView<T, Device>(Kokkos::ViewAllocateWithoutInitializing(name), size);
      MemberTypeView<T, Device> view = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      CreateViewsImpl<Device, Types...>(views+1, size, num+1);
    }
  };

  template <typename Device, typename... Types> struct CreateViews<Device, MemberTypes<Types...> > {
    CreateViews(MemberTypeViews& views, int size) {
      views = new void*[MemberTypes<Types...>::size];
      CreateViewsImpl<Device, Types...>(views, size, 0);
    }
  };





  template <typename View, typename... Types> struct CopyViewsToViewsImpl;
  template <typename View> struct CopyViewsToViewsImpl<View> {
    typedef typename View::device_type Device;
    CopyViewsToViewsImpl(MemberTypeViewsConst,
                         MemberTypeViewsConst,
                         View) {}
  };
  template <typename View, typename T, typename... Types> struct CopyViewsToViewsImpl<View, T,Types...> {
    typedef typename View::device_type Device;
    CopyViewsToViewsImpl(MemberTypeViewsConst dsts,
                         MemberTypeViewsConst srcs,
                         View ps_indices) {
      enclose(dsts,srcs, ps_indices);
    }
    void enclose(MemberTypeViewsConst dsts,
                 MemberTypeViewsConst srcs,
                 View ps_indices) {
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device> const*>(dsts[0]);
      MemberTypeView<T, Device> src = *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      //Kokkos::View<int*, Kokkos::DefaultExecutionSpace> hasFailed(1);
      MemberTypeView<int, Device> hasFailed(1);
      int size = dst.extent(0);
      Kokkos::parallel_for(ps_indices.size(), KOKKOS_LAMBDA(const int& i) {
        const int index = ps_indices(i);
        if (index >= size || index < 0) {
          Kokkos::printf("[ERROR] copying view to view from %d to %d outside of [0-%d)\n", i, index, size);
	  hasFailed(0) = 1;
        }
        CopyViewToView<T,Device>(dst, index, src, i);
      });
      auto hasFailed_h = deviceToHost(hasFailed);
      if( hasFailed_h(0) ) {
	printError("index out of range in view-to-view copy\n");
	exit(EXIT_FAILURE);
      }
      CopyViewsToViewsImpl<View, Types...>(dsts+1, srcs+1, ps_indices);
    }
  };
  template <typename View, typename... Types> struct CopyViewsToViews<View, MemberTypes<Types...> > {
    typedef typename View::device_type Device;
    CopyViewsToViews(MemberTypeViewsConst dsts,
                     MemberTypeViewsConst srcs,
                         View ps_indices) {
      if (dsts != NULL && srcs != NULL)
        CopyViewsToViewsImpl<View, Types...>(dsts, srcs, ps_indices);
    }
  };

  template <typename MSpace1, typename MSpace2, typename... Types>
  struct CopyMemSpaceToMemSpaceImpl;

  template <typename MSpace1, typename MSpace2>
  struct CopyMemSpaceToMemSpaceImpl<MSpace1, MSpace2> {
    typedef typename MSpace1::device_type Device1;
    typedef typename MSpace2::device_type Device2;
    CopyMemSpaceToMemSpaceImpl(MemberTypeViewsConst,
                               MemberTypeViewsConst) {}
  };

  template <typename MSpace1, typename MSpace2, typename T, typename... Types>
  struct CopyMemSpaceToMemSpaceImpl<MSpace1, MSpace2, T, Types...> {
    typedef typename MSpace1::device_type Device1;
    typedef typename MSpace2::device_type Device2;
    CopyMemSpaceToMemSpaceImpl(MemberTypeViewsConst dsts,
                               MemberTypeViewsConst srcs) {
      MemberTypeView<T, Device1>* dst_view = static_cast<MemberTypeView<T, Device1>*>(dsts[0]);
      MemberTypeView<T, Device2>* src_view = static_cast<MemberTypeView<T, Device2>*>(srcs[0]);
      deep_copy(*dst_view, *src_view);
      CopyMemSpaceToMemSpaceImpl<MSpace1, MSpace2, Types...>(dsts + 1, srcs + 1);
    }
  };

  template <typename MSpace1, typename MSpace2, typename... Types>
  struct CopyMemSpaceToMemSpace<MSpace1, MSpace2, MemberTypes<Types...> > {
    typedef typename MSpace1::device_type Device1;
    typedef typename MSpace2::device_type Device2;
    CopyMemSpaceToMemSpace(MemberTypeViewsConst dsts,
                           MemberTypeViewsConst srcs) {
      CopyMemSpaceToMemSpaceImpl<MSpace1, MSpace2, Types...>(dsts, srcs);
    }

  };

  //Shuffle copy currying structs
  template <typename PS, typename... Types> struct ShuffleParticlesImpl;
  template <typename PS> struct ShuffleParticlesImpl<PS> {
    typedef typename PS::device_type Device;
    typedef typename PS::kkLidView LidView;
    ShuffleParticlesImpl(MemberTypeViewsConst ps,
                         MemberTypeViewsConst new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {}
  };
  template <typename PS, typename T, typename... Types>
  struct ShuffleParticlesImpl<PS, T, Types...> {
    typedef typename PS::device_type Device;
    typedef typename PS::kkLidView LidView;
    ShuffleParticlesImpl(MemberTypeViewsConst ps,
                         MemberTypeViewsConst new_particles,
                         LidView old_indices, LidView new_indices, LidView fromPS) {
      enclose(ps, new_particles, old_indices, new_indices, fromPS);
    }
    void enclose(MemberTypeViewsConst ps,
                 MemberTypeViewsConst new_particles,
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
    ShuffleParticles(MemberTypeViewsConst ps,
                     MemberTypeViewsConst new_particles,
                     LidView old_indices, LidView new_indices, LidView fromPS) {
      ShuffleParticlesImpl<PS, Types...>(ps, new_particles, old_indices,
                                                      new_indices, fromPS);
    }
  };

  template <typename Device, typename... Types> struct SendViewsImpl;
  template <typename Device> struct SendViewsImpl<Device> {
    SendViewsImpl(MemberTypeViews views, int offset, int size,
                  int dest, int tag, MPI_Comm comm, MPI_Request* reqs) {}
  };
  template <typename Device, typename T, typename... Types> struct SendViewsImpl<Device, T, Types...> {
    SendViewsImpl(MemberTypeViews views, int offset, int size,
                  int dest, int tag, MPI_Comm comm, MPI_Request* reqs) {
      MemberTypeView<T, Device> v = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      PS_Comm_Isend(v.view(), offset, size, dest, tag, comm, reqs);
      SendViewsImpl<Device, Types...>(views+1, offset, size, dest, tag + 1, comm, reqs + 1);
    }
  };

  template <typename Device, typename... Types> struct SendViews<Device, MemberTypes<Types...>> {
    SendViews(MemberTypeViews views, int offset, int size,
              int dest, int start_tag, MPI_Comm comm, MPI_Request* reqs) {
      SendViewsImpl<Device, Types...>(views, offset, size, dest, start_tag, comm, reqs);
    }
  };

  template <typename Device, typename... Types> struct RecvViewsImpl;
  template <typename Device> struct RecvViewsImpl<Device> {
    RecvViewsImpl(MemberTypeViews views, int offset, int size,
                  int dest, int tag, MPI_Comm comm, MPI_Request* reqs) {}
  };
  template <typename Device, typename T, typename... Types> struct RecvViewsImpl<Device, T, Types...> {
    RecvViewsImpl(MemberTypeViews views,
                  int offset, int size, int dest, int tag, MPI_Comm comm, MPI_Request* reqs) {
      MemberTypeView<T, Device> v = *static_cast<MemberTypeView<T, Device>*>(views[0]);
      PS_Comm_Irecv(v.view(), offset, size, dest, tag, comm, reqs);
      RecvViewsImpl<Device, Types...>(views+1, offset, size, dest, tag + 1, comm, reqs + 1);
    }
  };

  template <typename Device, typename... Types> struct RecvViews<Device, MemberTypes<Types...> > {
    RecvViews(MemberTypeViews views, int offset, int size,
              int dest, int start_tag, MPI_Comm comm, MPI_Request* reqs) {
      RecvViewsImpl<Device, Types...>(views, offset, size, dest, start_tag, comm, reqs);
    }
  };

  //Implementation to deallocate views of different types
  template <typename Device, typename... Types> struct DestroyViewsImpl;
  template <typename Device> struct DestroyViewsImpl<Device> {
    DestroyViewsImpl(MemberTypeViews) {}
  };
  template <typename Device, typename T, typename... Types> struct DestroyViewsImpl<Device, T,Types...> {
    DestroyViewsImpl(MemberTypeViews data) {
      delete static_cast<MemberTypeView<T, Device>*>(data[0]);
      DestroyViewsImpl<Device, Types...>(data+1);
    }
  };

  //Call to deallocate arrays of different types
  template <typename Device, typename... Types> struct DestroyViews<Device, MemberTypes<Types...> > {
    DestroyViews(MemberTypeViews data) {
      DestroyViewsImpl<Device, Types...>(data+0);
      delete [] data;
    }
  };

}
