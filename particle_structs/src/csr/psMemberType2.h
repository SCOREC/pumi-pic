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
  template <typename PS, typename... Types> struct CopyParticlesToSend2;
/* CopyPSToPS<ParticleStructure, DataTypes> - copies particle info from ps to ps
     Usage: CopyPSToPS<ParticleStructure, MemberTypes>(ParticleStructure,
                                                       DestionationMemberTypeViews,
                                                       SourceMemberTypeViews,
                                                       NewRowIndexForParticle,
                                                       DestinationIndexForParticle);
*/
  template <typename PS, typename... Types> struct CopyPSToPS2;

//Copy Particles To Send Templated Struct
  template <typename PS, typename... Types> struct CopyParticlesToSendImpl2;
  template <typename PS> struct CopyParticlesToSendImpl2<PS> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl2(PS* ps, MemberTypeViewsConst,
                            MemberTypeViewsConst,
                            typename PS::kkLidView, typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyParticlesToSendImpl2<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyParticlesToSendImpl2(PS* ps, MemberTypeViewsConst dsts,
                            MemberTypeViewsConst srcs,
                            typename PS::kkLidView ps_to_array,
                            typename PS::kkLidView array_indices) {
      enclose(ps, dsts, srcs,ps_to_array, array_indices);
    }
    void enclose2(PS* ps, MemberTypeViewsConst dsts,
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
      CopyParticlesToSendImpl2<PS, Types...>(ps, dsts+1, srcs+1, ps_to_array,
                                            array_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyParticlesToSend2<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyParticlesToSend2(PS* ps, MemberTypeViewsConst dsts,
                        MemberTypeViewsConst srcs,
                        typename PS::kkLidView ps_to_array,
                        typename PS::kkLidView array_indices) {
      CopyParticlesToSendImpl2<PS, Types...>(ps, dsts, srcs, ps_to_array, array_indices);
    }
  };

  template <typename PS, typename... Types> struct CopyPSToPSImpl2;
  template <typename PS> struct CopyPSToPSImpl2<PS> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl2(PS* ps, MemberTypeViewsConst,
                   MemberTypeViewsConst, typename PS::kkLidView,
                   typename PS::kkLidView) {}
  };
  template <typename PS, typename T, typename... Types> struct CopyPSToPSImpl2<PS, T,Types...> {
    typedef typename PS::device_type Device;
    CopyPSToPSImpl2(PS* ps, MemberTypeViewsConst dsts,
                   MemberTypeViewsConst srcs,
                   typename PS::kkLidView new_element,
                   typename PS::kkLidView ps_indices) {
      enclose2(ps,dsts,srcs,new_element, ps_indices);
    }
    void enclose2(PS* ps, MemberTypeViewsConst dsts,
                 MemberTypeViewsConst srcs,
                 typename PS::kkLidView new_element,
                 typename PS::kkLidView ps_indices) {
      MemberTypeView<T, Device> dst = *static_cast<MemberTypeView<T, Device> const*>(dsts[0]);
      MemberTypeView<T, Device> src = *static_cast<MemberTypeView<T, Device> const*>(srcs[0]);
      auto copyPSToPS2 = PS_LAMBDA(int elm_id, int ptcl_id, bool mask) {
        const lid_t new_elem = new_element(ptcl_id);
        printf("new element: %d\n", new_elem);
        if (mask && new_elem != -1) {
          const int index = ps_indices(ptcl_id);
          CopyViewToView<T,Device>(dst, index, src, ptcl_id);
        }
      };
      parallel_for(ps, copyPSToPS2);
      CopyPSToPSImpl2<PS, Types...>(ps, dsts+1, srcs+1, new_element, ps_indices);
    }

  };
  template <typename PS,typename... Types> struct CopyPSToPS2<PS, MemberTypes<Types...> > {
    typedef typename PS::device_type Device;
    CopyPSToPS2(PS* ps, MemberTypeViewsConst dsts,
               MemberTypeViewsConst srcs,
               typename PS::kkLidView new_element,
               typename PS::kkLidView ps_indices) {
      CopyPSToPSImpl2<PS, Types...>(ps, dsts, srcs, new_element, ps_indices);
    }
  };
}
