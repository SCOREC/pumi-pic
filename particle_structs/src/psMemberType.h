#pragma once
#include "ps_for.hpp"
#include <MemberTypeLibraries.h>
namespace pumipic {
/* CopyParticleToSend<ParticleStructure, DataTypes> - copies particle info to send arrays
     Usage: CopyParticlesToSend<ParticleStructure, DataTypes>(ParticleStructure,
                                                              DestinationMemberTypeViews,
                                                              SourceMemberTypeViews,
                                                              NewProcessPerParticle,
                                                              MapFromPSToSendArray);
*/
  template <typename PS, typename... Types> struct CopyParticlesToSend;
/* CopyPSToPS<ParticleStructure, DataTypes> - copies particle info from ps to ps
     Usage: CopyPSToPS<ParticleStructure, MemberTypes>(ParticleStructure,
                                                       DestionationMemberTypeViews,
                                                       SourceMemberTypeViews,
                                                       NewRowIndexForParticle,
                                                       DestinationIndexForParticle);
*/
  template <typename PS, typename... Types> struct CopyPSToPS;

  /* CopyPSToPS2<ParticleStructure, DataTypes> - copies particle info from ps to ps
     Usage: CopyPSToPS2<ParticleStructure, MemberTypes>(ParticleStructure,
                                                       DestinationDevicePtrs,
                                                       SourceDevicePtrs,
                                                       NewRowIndexForParticle,
                                                       DestinationIndexForParticle);
  */
  template <typename PS, typename... Types> struct CopyPSToPS2;

//Copy Particles To Send Templated Struct
  template <typename PS, typename... Types> struct CopyParticlesToSendImpl;
  template <typename PS> struct CopyParticlesToSendImpl<PS> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst,
                            MemberTypeViewsConst,
                            typename PS::kkLidView, typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyParticlesToSendImpl<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl(PS* ps, MemberTypeViewsConst dsts,
                            MemberTypeViewsConst srcs,
                            typename PS::kkLidView ps_to_array,
                            typename PS::kkLidView array_indices) {
      enclose(ps, dsts, srcs,ps_to_array, array_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst dsts,
                 MemberTypeViewsConst srcs,
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
      parallel_for(ps, copyPSToArray);
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts+1, srcs+1, ps_to_array,
                                            array_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyParticlesToSend<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyParticlesToSend(PS* ps, MemberTypeViewsConst dsts,
                        MemberTypeViewsConst srcs,
                        typename PS::kkLidView ps_to_array,
                        typename PS::kkLidView array_indices) {
      CopyParticlesToSendImpl<PS, Types...>(ps, dsts, srcs, ps_to_array, array_indices);
    }
  };

  template <typename PS, typename... Types> struct CopyPSToPSImpl;
  template <typename PS> struct CopyPSToPSImpl<PS> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst,
                   MemberTypeViewsConst, typename PS::kkLidView,
                   typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyPSToPSImpl<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl(PS* ps, MemberTypeViewsConst dsts,
                   MemberTypeViewsConst srcs,
                   typename PS::kkLidView new_element,
                   typename PS::kkLidView ps_indices) {
      enclose(ps,dsts,srcs,new_element, ps_indices);
    }
    void enclose(PS* ps, MemberTypeViewsConst dsts,
                 MemberTypeViewsConst srcs,
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
      parallel_for(ps, copyPSToPS);
      CopyPSToPSImpl<PS, Types...>(ps, dsts+1, srcs+1, new_element, ps_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyPSToPS<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyPSToPS(PS* ps, MemberTypeViewsConst dsts,
               MemberTypeViewsConst srcs,
               typename PS::kkLidView new_element,
               typename PS::kkLidView ps_indices) {
      CopyPSToPSImpl<PS, Types...>(ps, dsts, srcs, new_element, ps_indices);
    }
  };

  template <typename PS, typename... Types> struct CopyPSToPSImpl2;
  template <typename PS> struct CopyPSToPSImpl2<PS> {
    typedef typename PS::device_type Device;
    PP_INLINE CopyPSToPSImpl2(void*, int i,
                              void*, int j) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyPSToPSImpl2<PS, T,Types...> {
    typedef typename PS::device_type Device;
    PP_INLINE CopyPSToPSImpl2(void* dsts, lid_t i,
                              void* srcs, lid_t j) {
      MemberTypeView<T, Device>* dst = static_cast<MemberTypeView<T, Device>*>(dsts);
      MemberTypeView<T, Device>* src = static_cast<MemberTypeView<T, Device>*>(srcs);
      CopyViewToView<T, Device>(*dst, i, *src, j);
      CopyPSToPSImpl2<PS, Types...>(dst+1, i,src+1, j);
    }
  };

  template <typename PS,typename... Types> struct CopyPSToPS2<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyPSToPS2(PS* ps, void* dsts,
                void* srcs, typename PS::kkLidView new_element,
                typename PS::kkLidView ps_indices) {
      enclose(ps, dsts, srcs, new_element, ps_indices);
    }
    void enclose(PS* ps, void* dsts, void* srcs, typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      auto copyPSToPS = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const lid_t new_elem = new_element(ptcl_id);
        if (mask && new_elem != -1) {
          const int index = ps_indices(ptcl_id);
          CopyPSToPSImpl2<PS, Types...>(dsts, index, srcs, ptcl_id);
        }
      };
      parallel_for(ps, copyPSToPS);
    }
  };

#ifdef PP_USE_CUDA
  template <typename Device, typename... Types> struct SetPtrsImpl;
  template <typename Device> struct SetPtrsImpl<Device> {
    SetPtrsImpl(void* ptrs, MemberTypeViewsConst data) {}
  };
  template <typename Device, typename T, typename... Types> struct SetPtrsImpl<Device,T, Types...> {
    SetPtrsImpl(void* ptrs, MemberTypeViewsConst data) {
      MemberTypeView<T, Device> view = *static_cast<MemberTypeView<T, Device> const*>(data[0]);
      cudaMemcpy(ptrs, &view, sizeof(MemberTypeView<T, Device>), cudaMemcpyHostToDevice);
      SetPtrsImpl<Device, Types...>(((MemberTypeView<T, Device>*)ptrs)+1, data+1);
    }
  };
#endif

  template <typename Device, typename... Types> struct SetPtrs;
  template <typename Device, typename... Types> struct SetPtrs<Device, MemberTypes<Types...> > {
    SetPtrs(void* ptrs, MemberTypeViewsConst data) {
#ifdef PP_USE_CUDA
      SetPtrsImpl<Device, Types...>(ptrs, data);
#endif
    }
  };
  /*
    Create raw pointers to device memory on device
  */
  template <typename DataTypes, typename Device>
  void createDevicePtrs(MemberTypeViewsConst data, void*& ptrs) {
#ifdef PP_USE_CUDA
      cudaMalloc(&ptrs, DataTypes::viewmemsize);
      SetPtrs<Device, DataTypes>(ptrs, data);
#endif
  }

  template <typename DataTypes, typename Device>
  void destroyPtrs(void* ptrs) {
#ifdef PP_USE_CUDA
      cudaFree(ptrs);
#endif
  }

}
