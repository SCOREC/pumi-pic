#pragma once

#include <Omega_h_array.hpp>
#include <Omega_h_for.hpp>
#include "pumipic_mesh.hpp"
#include <ngraph.h>
#include <engpar_support.h>
#include <engpar_weight_input.h>
#include <engpar.h>
#include <particle_structs.hpp>

namespace pumipic {

  typedef std::set<int> Parts;
  class PartsHash {
  public:
    size_t operator()(const Parts& res) const {
      int h = 0;
      std::hash<int> hasher;
      for (auto itr = res.begin(); itr != res.end(); ++itr)
        h ^= hasher(*itr);
      return h;
    }
  };

  //Print particle imbalance statistics
  template <class PS>
  void printPtclImb(PS* ptcls, MPI_Comm comm = MPI_COMM_WORLD);

  class Mesh;
  class ParticlePlan;

  class ParticleBalancer {
  public:
    //Build Ngraph from sbars
    ParticleBalancer(Mesh& picparts);
    ~ParticleBalancer();

    /* Performs particle load balancing and redistributes particles
       picparts(in) - the picparts mesh
       ps(in) - the particle structure
       tol(in) - the target imbalance (5% would be a value of 1.05)
       new_elems(in) - the new elements each particle is moving to
       new_procs(in/out) - the new processes for each particle,
           will be changed to satisfy load balance
           Note: particles pushed outside the safe zone must have new process already set
       step_factor(in) - (optional) the rate of weight transfer
     */
    template <class PS>
    void repartition(Mesh& picparts, PS* ps, double tol,
                     typename PS::kkLidView new_elems,
                     typename PS::kkLidView new_procs,
                     double step_factor = 0.3);

    /* Performs particle load balancing on an array of particles per element
       picparts(in) - the picparts mesh
       ptcls_per_elem - the number of particles per element (size must equal number of elements in `picparts`)
       tol(in) - the target imbalance (5% would be a value of 1.05)
       step_factor(in) - (optional) the rate of weight transfer

       Returns an array of the new process per particle
     */
    template <class ViewT>
    Kokkos::View<lid_t*> partition(Mesh& picparts, ViewT ptcls_per_elem, double tol, double step_factor = 0.3, int selection_iterations = 5);

    //Access the sbar ids per element
    Omega_h::LOs getSbarIDs(Mesh& picparts) const;

    /* Steps of repartition, can be called on their own for customization */

    //adds the weight of particles in ps to graph
    template <class PS>
    void addWeights(Mesh& picparts, PS* ps, typename PS::kkLidView new_elems,
                    typename PS::kkLidView new_procs);

    //adds the weight of particles in ptcls_per_elem to graph
    template <class ViewT>
    void addWeights(Mesh& picparts, ViewT ptcls_per_elem);

    //run the weight balancer and return the plan
    ParticlePlan balance(double tol, double step_factor = 0.3);

    template <class PS>
    void selectParticles(Mesh& picparts, PS* ps, typename PS::kkLidView new_elems,
                         ParticlePlan plan, typename PS::kkLidView new_parts);

    template <typename ViewT>
    Kokkos::View<lid_t*> selectParticles(Mesh& picparts, ViewT ptcls_per_elem, ParticlePlan plan, int selection_iterations);
private:
    typedef std::unordered_map<Parts, int, PartsHash> SBarUnmap;
    int max_sbar;
    SBarUnmap sbar_ids;
    Omega_h::HostWrite<int> elm_sbar;
    agi::Ngraph* weightGraph;
    std::unordered_map<agi::gid_t,int> vert_to_sbar;
    std::unordered_map<agi::gid_t, agi::part_t> vert_to_owner;
    Kokkos::UnorderedMap<int, agi::lid_t> sbar_to_vert;

    //select particles to migrate
    void makePlan();

    SBarUnmap::iterator insert(Parts& p);
    Omega_h::HostWrite<int> buildLocalSbarMap(int comm_rank, int nelms,
                                              Omega_h::HostWrite<Omega_h::LO> buffer_ranks,
                                              Omega_h::HostWrite<Omega_h::LO> safe_per_buffer);
    void sendCoreSbars(Omega_h::CommPtr comm, Omega_h::HostWrite<Omega_h::LO> buffer_ranks);
    void globalNumberSbars(Omega_h::CommPtr comm,
                           std::unordered_map<int,int>& sbar_local_to_global);
    void cleanSbars(int comm_rank);

    void numberElements(Mesh& picparts, Omega_h::HostWrite<int> elm_sbar,
                        std::unordered_map<int, int>& map);
    void buildNgraph(Omega_h::CommPtr comm);
  };

  class ParticlePlan {
  public:
    ParticlePlan();
    ParticlePlan(std::unordered_map<Omega_h::LO, Omega_h::LO>& sbar_index_map,
                 Omega_h::Write<Omega_h::LO> tgt_parts, Omega_h::Write<Omega_h::Real> wgts);
    friend class ParticleBalancer;
  private:
    Kokkos::UnorderedMap<int, int> sbar_to_index;
    Omega_h::LOs part_ids;
    Omega_h::Write<Omega_h::Real> send_wgts;
  };

  template <class PS>
  void ParticleBalancer::addWeights(Mesh& picparts, PS* ptcls,
                                    typename PS::kkLidView new_elems,
                                    typename PS::kkLidView new_procs) {
    MPI_Comm comm = picparts.comm()->get_impl();
    int comm_rank = picparts.comm()->rank();
    // Device map of number of particles already assigned to another process
    Kokkos::UnorderedMap<int, agi::wgt_t> forcedPtcls(picparts.numBuffers(picparts->dim()));
    Omega_h::Write<Omega_h::LO> buffered_ranks(picparts.bufferedRanks(picparts->dim()));
    Kokkos::parallel_for(buffered_ranks.size(), KOKKOS_LAMBDA(const int i) {
        forcedPtcls.insert(buffered_ranks[i], 0);
    });

    //Count particles in each sbar & count particles already being migrated
    Omega_h::Write<agi::wgt_t> weights(sbar_ids.size() + 1, 0);
    Omega_h::LOs elem_sbars = getSbarIDs(picparts);
    auto sbar_to_vert_local = sbar_to_vert;
    auto accumulateWeight = PS_LAMBDA(const int elm, const int ptcl, const bool mask) {
      if (mask) {
        const int new_rank = new_procs(ptcl);
        if (new_rank == comm_rank) {
          const int e = new_elems(ptcl);
          if (e != -1) {
            int sbar_index = elem_sbars[e];
            if (sbar_to_vert_local.exists(sbar_index)) {
              auto index = sbar_to_vert_local.find(sbar_index);
              const agi::lid_t vert_index = sbar_to_vert_local.value_at(index);
              Kokkos::atomic_add(&(weights[vert_index]), 1.0);
            }
          }
        }
        else {
          const auto index = forcedPtcls.find(new_rank);
          Kokkos::atomic_add(&(forcedPtcls.value_at(index)), 1.0);
        }
      }
    };
    parallel_for(ptcls, accumulateWeight, "accumulateWeight");

    //Transfer map to host
    Omega_h::Write<int> owners(buffered_ranks.size(), "owners");
    Omega_h::Write<agi::wgt_t> wgts(buffered_ranks.size(), "owners");
    Omega_h::Write<int> index(1, 0);
    Kokkos::parallel_for(forcedPtcls.capacity(), KOKKOS_LAMBDA (uint32_t i) {
      if( forcedPtcls.valid_at(i) ) {
        const int map_index = Kokkos::atomic_fetch_add(&(index[0]), 1);
        owners[map_index] = forcedPtcls.key_at(i);
        wgts[map_index] = forcedPtcls.value_at(i);
      }
    });
    Omega_h::HostWrite<int> owners_host(owners);
    Omega_h::HostWrite<agi::wgt_t> wgts_host(wgts);

    //Send wgts to peers
    int num_peers = owners_host.size();
    agi::wgt_t* peer_wgts = new agi::wgt_t[num_peers];
    MPI_Request* send_requests = new MPI_Request[num_peers];
    MPI_Request* recv_requests = new MPI_Request[num_peers];
    for (int i = 0; i < num_peers; ++i) {
      MPI_Irecv(peer_wgts + i, 1, MPI_DOUBLE, owners_host[i],
                0, comm, recv_requests + i);
      MPI_Isend(&(wgts_host[i]), 1, MPI_DOUBLE, owners_host[i],
                0, comm, send_requests + i);
    }
    MPI_Waitall(num_peers, recv_requests, MPI_STATUSES_IGNORE);
    delete [] recv_requests;

    //Accumulate all received weight on the last vertex
    Omega_h::HostWrite<Omega_h::Real> weights_host(weights);
    for (int i = 0; i < num_peers; ++i) {
      weights_host[weights_host.size() - 1] += peer_wgts[i];
    }

    weightGraph->setWeights(weights_host.data());
    MPI_Waitall(num_peers, send_requests, MPI_STATUSES_IGNORE);
    delete [] send_requests;
    delete [] peer_wgts;
  }

  //adds the weight of particles in ptcls_per_elem to graph
  template <class ViewT>
  void ParticleBalancer::addWeights(Mesh& picparts, ViewT ptcls_per_elem) {
    //Count particles in each sbar
    Omega_h::Write<agi::wgt_t> weights(sbar_ids.size() + 1, 0);
    Omega_h::LOs elem_sbars = getSbarIDs(picparts);
    auto sbar_to_vert_local = sbar_to_vert;
    auto accumulateWeight = OMEGA_H_LAMBDA(const int elm) {
      const Omega_h::LO sbar_index = elem_sbars[elm];
      if (sbar_to_vert_local.exists(sbar_index)) {
        auto index = sbar_to_vert_local.find(sbar_index);
        const agi::lid_t vert_index = sbar_to_vert_local.value_at(index);
        Kokkos::atomic_add(&(weights[vert_index]), 1.0*ptcls_per_elem[elm]);
      }
    };
    Omega_h::parallel_for(ptcls_per_elem.size(), accumulateWeight, "accumulateWeight");

    //Apply weights to the graph
    Omega_h::HostWrite<Omega_h::Real> weights_host(weights);
    weightGraph->setWeights(weights_host.data());
  }

  template <class PS>
  void ParticleBalancer::selectParticles(Mesh& picparts, PS* ptcls,
                                         typename PS::kkLidView new_elems,
                                         ParticlePlan plan,
                                         typename PS::kkLidView new_parts) {

    int comm_size = picparts.comm()->size();
    if (comm_size == 1)
      return;
    int comm_rank = picparts.comm()->rank();
    Omega_h::LOs sbars = getSbarIDs(picparts);
    auto send_wgts = plan.send_wgts;
    auto sbar_to_index = plan.sbar_to_index;
    auto part_ids = plan.part_ids;
    auto owners = picparts.entOwners(picparts->dim());
    auto selectNonCoreParticles = PS_LAMBDA(const int elm, int ptcl, const bool mask) {
      const Omega_h::LO new_e = new_elems(ptcl);
      const Omega_h::LO new_p = new_parts(ptcl);
      if (mask && new_p == comm_rank && new_e != -1) {
        const Omega_h::LO owner = owners[new_e];
        if (owner != comm_rank) {
          const Omega_h::LO sbar = sbars[new_e];
          if (sbar_to_index.exists(sbar)) {
            const auto map_index = sbar_to_index.find(sbar);
            const Omega_h::LO index = sbar_to_index.value_at(map_index);
            const Omega_h::LO part = part_ids[index];
            const Omega_h::Real wgt = Kokkos::atomic_fetch_add(&(send_wgts[index]), -1);
            if (part >= 0) {
              if (wgt == 0)
                Kokkos::atomic_add(&(sbar_to_index.value_at(map_index)), 1);
              if (wgt > 0)
                new_parts[ptcl] = part;
            }
          }
        }
      }
    };
    parallel_for(ptcls, selectNonCoreParticles, "selectNonCoreParticles");
    auto selectParticles = PS_LAMBDA(const int elm, const int ptcl, const bool mask) {
      const Omega_h::LO new_e = new_elems(ptcl);
      const Omega_h::LO new_p = new_parts(ptcl);
      if (mask && new_p == comm_rank && new_e != -1) {
        const Omega_h::LO sbar = sbars[new_e];
        if (sbar_to_index.exists(sbar)) {
          const auto map_index = sbar_to_index.find(sbar);
          const Omega_h::LO index = sbar_to_index.value_at(map_index);
          const Omega_h::LO part = part_ids[index];
          const Omega_h::Real wgt = Kokkos::atomic_fetch_add(&(send_wgts[index]), -1);
          if (part >= 0) {
            if (wgt == 0)
              Kokkos::atomic_add(&(sbar_to_index.value_at(map_index)), 1);
            if (wgt > 0)
              new_parts[ptcl] = part;
          }
        }
      }
    };
    parallel_for(ptcls, selectParticles, "selectParticles");
  }

  template <class ViewT>
  Kokkos::View<int*> ParticleBalancer::selectParticles(Mesh& picparts, ViewT ptcls_per_elem, ParticlePlan plan, int selection_iterations) {
    int comm_size = picparts.comm()->size();
    if (comm_size == 1)
      return Kokkos::View<int*>(0);
    int comm_rank = picparts.comm()->rank();

    //Offset sum the ptcls per elem and get total number of particles
    ViewT offsets = ViewT("ptcl per elem offsets", ptcls_per_elem.size() + 1);
    Kokkos::resize(ptcls_per_elem, ptcls_per_elem.size() + 1);
    exclusive_scan(ptcls_per_elem, offsets);

    //Array of new processes per particle
    lid_t nptcls = getLastValue(offsets);
    Kokkos::View<int*> new_procs("new procs", nptcls);
    auto setSelf = OMEGA_H_LAMBDA(const int ptcl) {
      new_procs[ptcl] = comm_rank;
    };
    Omega_h::parallel_for(new_procs.size(), setSelf, "setSelf");


    //Select particles to migrate
    Omega_h::LOs sbars = getSbarIDs(picparts);
    auto send_wgts = plan.send_wgts;
    auto sbar_to_index = plan.sbar_to_index;
    auto part_ids = plan.part_ids;
    auto owners = picparts.entOwners(picparts->dim());
    for (int i = 0; i < selection_iterations; ++i) {
      auto selectParticles = OMEGA_H_LAMBDA(const int elm) {
        const Omega_h::LO sbar = sbars[elm];
        const Omega_h::LO start_ptcl = offsets[elm];
        if (sbar_to_index.exists(sbar)) {
          const auto map_index = sbar_to_index.find(sbar);
          for (Omega_h::LO i = 0; i < ptcls_per_elem[elm]; ++i) {
            const Omega_h::LO new_part = new_procs[start_ptcl+i];
            if (new_part == comm_rank) {
              const Omega_h::LO index = sbar_to_index.value_at(map_index);
              const Omega_h::LO part = part_ids[index];
              const Omega_h::Real wgt = Kokkos::atomic_fetch_add(&(send_wgts[index]), -1);
              if (part >= 0) {
                if (wgt == 0)
                  Kokkos::atomic_add(&(sbar_to_index.value_at(map_index)), 1);
                if (wgt > 0)
                  new_procs[start_ptcl+i] = part;
              }
            }
          }
        }
      };
      Omega_h::parallel_for(ptcls_per_elem.size(), selectParticles, "selectParticles");
      Omega_h::Write<Omega_h::LO> finished(1,0);
      auto checkPlan = OMEGA_H_LAMBDA(const int id) {
        if (send_wgts[id] > 0) {
          finished[0] = 1;
        }
      };
      Omega_h::parallel_for(send_wgts.size(), checkPlan);
      Omega_h::HostWrite<Omega_h::LO> finished_h(finished);
      if (finished_h[0] == 0)
        break;
    }
    return new_procs;
  }

  template <class PS>
  void ParticleBalancer::repartition(Mesh& picparts, PS* ptcls, double tol,
                                     typename PS::kkLidView new_elems,
                                     typename PS::kkLidView new_parts,
                                     double step_factor) {
    if (picparts.comm()->size() == 1)
      return;
    addWeights(picparts, ptcls, new_elems, new_parts);
    ParticlePlan plan = balance(tol, step_factor);
    selectParticles(picparts, ptcls, new_elems, plan, new_parts);
  }

  template <class ViewT>
  Kokkos::View<lid_t*> ParticleBalancer::partition(Mesh& picparts, ViewT ptcls_per_elem, double tol, double step_factor, int selection_iterations) {
    if (picparts.comm()->size() == 1) {
      lid_t np = 0;
      Kokkos::parallel_reduce(ptcls_per_elem.size(), KOKKOS_LAMBDA(const int index, int& ptcls) {
        ptcls += ptcls_per_elem[index];
      }, np);
      Kokkos::View<lid_t*> new_procs("new_procs", np);
      return new_procs;
    }
    addWeights(picparts, ptcls_per_elem);
    ParticlePlan plan = balance(tol, step_factor);
    return selectParticles(picparts, ptcls_per_elem, plan, selection_iterations);
  }

  //Print particle imbalance statistics
  template <class PS>
  void printPtclImb(PS* ptcls, MPI_Comm comm) {
    int np = ptcls->nPtcls();
    int min_p, max_p, tot_p;
    MPI_Reduce(&np, &min_p, 1, MPI_INT, MPI_MIN, 0, comm);
    MPI_Reduce(&np, &max_p, 1, MPI_INT, MPI_MAX, 0, comm);
    MPI_Reduce(&np, &tot_p, 1, MPI_INT, MPI_SUM, 0, comm);

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    if (comm_rank == 0) {
      float avg = tot_p / comm_size;
      float imb = max_p / avg;
      printf("Ptcl LB <max, min, avg, imb>: %d %d %.3f %.3f\n", max_p, min_p, avg, imb);
    }

  }

}
