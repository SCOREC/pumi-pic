#pragma once
#include "xgcp_mesh.hpp"

namespace xgcp {

  void setGyroConfig(Input& input);
  void printGyroConfig();

  void createIonGyroRingMappings(o::Mesh* mesh, o::LOs& forward_map,
                                 o::LOs& backward_map);

  void gyroScatter(Mesh& mesh, PS_I* ptcls);
  void gyroScatter(Mesh& mesh, PS_E* ptcls);

  void gyroScatter(Mesh& mesh, PS_I* ptcls, o::LOs v2v, std::string scatterTagName);

  void gyroSync(Mesh& mesh, const std::string& fwdTagName,
                const std::string& bkwdTagName, const std::string& syncTagName);

}
