//
// Created by Fuad Hasan on 2/26/25.
//

#ifndef PUMIPIC_PARTICLETRACER_H
#define PUMIPIC_PARTICLETRACER_H

#include <pumipic_adjacency.tpp>
#include <optional>

template<typename ElementType>
void set_write_to_zero(Omega_h::Write<ElementType> &write) {
    auto set_zero = OMEGA_H_LAMBDA(int i) {
        write[i] = 0;
    };
    Omega_h::parallel_for(write.size(), set_zero, "set_write_to_zero");
}

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
        reset_particle_done();

        auto particle_orig = ptcls_->template get<0>();
        auto particle_dest = ptcls_->template get<1>();
        auto particle_ids = ptcls_->template get<2>();

        bool success = pumipic::trace_particle_through_mesh(oh_mesh_, ptcls_, particle_orig, particle_dest,
                                                            particle_ids, elem_ids_, next_elem_ids_,
                                                            true, inter_faces_, inter_points_, last_exits_, 1000, true,
                                                            func_, elmAreas_, ptcl_done_,
                                                            tolerance_);

        if (!success) {
            printf("[ERROR] ParticleTracer: Failed to trace particles through the mesh.\n");
            return false;
        }
        return true;
    }

    [[nodiscard]]
    Omega_h::LOs getElementIds() const { return elem_ids_; }

    [[nodiscard]]
    Omega_h::LOs getIntersectionFaces() const { return inter_faces_; }

    [[nodiscard]]
    Omega_h::Reals getIntersectionPoints() const { return inter_points_; }

    [[nodiscard]]
    Omega_h::LOs GetLastExits() const { return last_exits_; }

    void updatePtclPositions() {
        auto origin = ptcls_->template get<0>();
        auto dest = ptcls_->template get<1>();
        auto updatePtclPos = PS_LAMBDA(const int &, const int &pid, const bool &) {
            origin(pid, 0) = dest(pid, 0);
            origin(pid, 1) = dest(pid, 1);
            origin(pid, 2) = dest(pid, 2);

            dest(pid, 0) = 0.0;
            dest(pid, 1) = 0.0;
            dest(pid, 2) = 0.0;
        };
        ps::parallel_for(ptcls_, updatePtclPos, "updatePtclPositions");
    }

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
    Omega_h::Write<Omega_h::LO> ptcl_done_;


    bool validate_internal_data_sizes() {
        int dim = oh_mesh_.dim();

        if (elem_ids_.size() != next_elem_ids_.size() || elem_ids_.size() != inter_faces_.size() ||
            elem_ids_.size() != inter_points_.size() / dim || elem_ids_.size() != last_exits_.size() ||
            elem_ids_.size() != ptcl_done_.size()) {
            printf("[ERROR] ParticleTracer: internal data arrays are not of the appropriate size.\nSize of elem_ids: %d, "
                   "next_elem_ids: %d, inter_faces: %d, inter_points: %d last_exits: %d ptcl_done: %d\n",
                   elem_ids_.size(), next_elem_ids_.size(), inter_faces_.size(), inter_points_.size(),
                   last_exits_.size(),
                   ptcl_done_.size());
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

    void reset_particle_done() {
        set_write_to_zero(ptcl_done_);
    }
};

#endif //PUMIPIC_PARTICLETRACER_H
