//
// Created by Fuad Hasan on 2/26/25.
//

#ifndef PUMIPIC_PARTICLETRACER_H
#define PUMIPIC_PARTICLETRACER_H

#include <pumipic_adjacency.tpp>
#include <optional>

template<typename ParticleType, typename Func>
class ParticleTracer {
public:
    ParticleTracer(Omega_h::Mesh &mesh, pumipic::ParticleStructure<ParticleType> *ptcls, Func &func,
                   std::optional<double> tolerance = std::nullopt) :
            oh_mesh_(mesh), ptcls_(ptcls), func_(func) {

        elmAreas_ = Omega_h::measure_elements_real(&mesh);
        if (tolerance.has_value()) {
            tolerance_ = tolerance.value();
        } else {
            tolerance_ = pumipic::compute_tolerance_from_area(elmAreas_);
        }
    }

    bool search() {
        if (!validate_internal_data_sizes()) {
            return false;
        }

        auto particle_orig = ptcls_->template get<0>();
        auto particle_dest = ptcls_->template get<1>();
        auto particle_ids = ptcls_->template get<2>();

        bool success = pumipic::trace_particle_through_mesh(oh_mesh_, ptcls_, particle_orig, particle_dest,
                                                            particle_ids, elem_ids_, next_elem_ids_,
                                                            true, inter_faces_, inter_points_, last_exits_, 1000, true,
                                                            func_, elmAreas_,
                                                            tolerance_);

        if (!success) {
            printf("[ERROR] ParticleTracer: Failed to trace particles through the mesh.\n");
            return false;
        }
        return true;
    }


    Omega_h::Write<Omega_h::LO> GetElemIds() { return elem_ids_; }

    Omega_h::Write<Omega_h::LO> GetInterFaces() { return inter_faces_; }

    Omega_h::Write<Omega_h::Real> GetInterPoints() { return inter_points_; }

    Omega_h::Write<Omega_h::LO> GetLastExits() { return last_exits_; }

private:
    double tolerance_ = 1e-10;

    Omega_h::Mesh &oh_mesh_;
    pumipic::ParticleStructure<ParticleType> *ptcls_;
    Func &func_;

    Omega_h::Write<Omega_h::LO> elem_ids_;
    Omega_h::Write<Omega_h::LO> next_elem_ids_;
    Omega_h::Write<Omega_h::LO> inter_faces_;
    Omega_h::Write<Omega_h::Real> inter_points_;
    Omega_h::Write<Omega_h::LO> last_exits_;
    Omega_h::Reals elmAreas_;


    bool validate_internal_data_sizes() {
        int dim = oh_mesh_.dim();

        if (elem_ids_.size() != next_elem_ids_.size() || elem_ids_.size() != inter_faces_.size() ||
            elem_ids_.size() != inter_points_.size() / dim) {
            printf("[ERROR] ParticleTracer: internal data arrays are not of the appropriate size.\nSize of elem_ids: %d, "
                   "next_elem_ids: %d, inter_faces: %d, inter_points: %d\n", elem_ids_.size(), next_elem_ids_.size(),
                   inter_faces_.size(), inter_points_.size());
            return false;
        }

        // has to be either 0 or equal to the capacity of the particle structure
        if ((elem_ids_.size() != 0) && (elem_ids_.size() != ptcls_->capacity())) {
            printf("[ERROR] ParticleTracer: internal data arrays are not of appropriate size.\nSize of elem_ids has to"
                   " be equal to the capacity of the particle structure.\nSize of elem_ids: %d, capacity: %d\n",
                   elem_ids_.size(), ptcls_->capacity());
            return false;
        }

        return true;
    }

    //void reset_particle_done() {}
};

#endif //PUMIPIC_PARTICLETRACER_H
