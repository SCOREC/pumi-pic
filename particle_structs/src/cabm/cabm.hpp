#pragma once

#include <particle_structs.hpp>

#include <Cabana_Core.hpp>
#include <cassert>

namespace {

template <typename T, typename... Types>
struct AppendMT;

//Append type to the end
template <typename T, typename... Types>
struct AppendMT<T, Cabana::MemberTypes<Types...> > {
  static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
  using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
};


// class to append member types
template <typename T, typename... Types>
struct MemberTypesAppend;

//Append type to the end
template <typename T, typename... Types>
struct MemberTypesAppend<T, Cabana::MemberTypes<Types...> > {
  static constexpr int size = 1 + Cabana::MemberTypes<Types...>::size;
  using type = Cabana::MemberTypes<Types..., T>; //Put T before Types... to put at beginning
};

}//end anonymous

namespace pumipic {
  template <class DataTypes, typename MemSpace = DefaultMemSpace>
  class CabM : public ParticleStructure<DataTypes, MemSpace> {
  public:
    using typename ParticleStructure<DataTypes, MemSpace>::execution_space;
    using typename ParticleStructure<DataTypes, MemSpace>::memory_space;
    using typename ParticleStructure<DataTypes, MemSpace>::device_type;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidView;
    using typename ParticleStructure<DataTypes, MemSpace>::kkLidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::kkGidHostMirror;
    using typename ParticleStructure<DataTypes, MemSpace>::MTVs;

    //from https://github.com/SCOREC/Cabana/blob/53ad18a030f19e0956fd0cab77f62a9670f31941/core/src/CabanaM.hpp#L18-L19
    using CM_DT = AppendMT<int,DataTypes>;
    using AoSoA_t = Cabana::AoSoA<typename CM_DT::type,MemSpace>;

    CabM() = delete;
    CabM(const CabM&) = delete;
    CabM& operator=(const CabM&) = delete;

    CabM(lid_t num_elements, lid_t num_particles, kkLidView particles_per_element,
        kkGidView element_gids, kkLidView particle_elements = kkLidView(),
        MTVs particle_info = NULL);
    ~CabM();

    //Functions from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::nElems;
    using ParticleStructure<DataTypes, MemSpace>::nPtcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity;
    using ParticleStructure<DataTypes, MemSpace>::numRows;


    void migrate(kkLidView new_element, kkLidView new_process,
                 Distributor<MemSpace> dist = Distributor<MemSpace>(),
                 kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particle_info = NULL);

    void rebuild(kkLidView new_element, kkLidView new_particle_elements = kkLidView(),
                 MTVs new_particles = NULL);

    template <typename FunctionType>
    void parallel_for(FunctionType& fn, std::string s="");

    void printMetrics() const;

  private:
    //Variables from ParticleStructure
    using ParticleStructure<DataTypes, MemSpace>::num_elems;
    using ParticleStructure<DataTypes, MemSpace>::num_ptcls;
    using ParticleStructure<DataTypes, MemSpace>::capacity_;
    using ParticleStructure<DataTypes, MemSpace>::num_rows;
    using ParticleStructure<DataTypes, MemSpace>::ptcl_data;
    using ParticleStructure<DataTypes, MemSpace>::num_types;

    //Offsets array into CabM
    kkLidView offsets;
    AoSoA_t _aosoa;
  };

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::CabM(lid_t num_elements, lid_t num_particles,
                                kkLidView particles_per_element,
                                kkGidView element_gids,
                                kkLidView particle_elements,
                                MTVs particle_info) {
    fprintf(stderr, "[WARNING] CabM constructor not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  CabM<DataTypes, MemSpace>::~CabM() {
    fprintf(stderr, "[WARNING] CabM deconstructor not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::migrate(kkLidView new_element, kkLidView new_process,
                                         Distributor<MemSpace> dist,
                                         kkLidView new_particle_elements,
                                         MTVs new_particle_info) {
    fprintf(stderr, "[WARNING] CabM migrate(...) not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::rebuild(kkLidView new_element,
                                         kkLidView new_particle_elements,
                                         MTVs new_particles) {
    fprintf(stderr, "[WARNING] CabM rebuild(...) not implemented\n");
  }

  template <class DataTypes, typename MemSpace>
  template <typename FunctionType>
  void CabM<DataTypes, MemSpace>::parallel_for(FunctionType& fn, std::string s) {
    fprintf(stderr, "[WARNING] CabM parallel_for(...) not implemented\n");

  }

  template <class DataTypes, typename MemSpace>
  void CabM<DataTypes, MemSpace>::printMetrics() const {
    fprintf(stderr, "[WARNING] CabM printMetrics() not implemented\n");
  }
}
