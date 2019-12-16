
#include "xgcp_types.hpp"
#include "xgcp_input.hpp"
#include <pumipic_mesh.hpp>

namespace xgcp {
  class Mesh {
  public:
    Mesh(xgcp::Input&);
    ~Mesh();

  private:
    p::Mesh* picparts;
    o::CommPtr mesh_comm, torodial_comm, group_comm;

  };
}
