#pragma once
#include <Omega_h_mesh.hpp>

namespace pumipic {

  class Mesh;

  class Input {
  public:

    //Method for creating buffers/safe zone
    enum Method {
      INVALID = -1,
      FULL, //Buffer entire mesh or safe entire buffer
      BFS, //Buffer/Safe BFS layers out
      MINIMUM, //Buffer/Safe the core part
      NONE //Do not create Safe zone (only for SafeMethod)
    };

    //Defines the type of info given in partition_vector
    enum Ownership {
      PARTITION, //partition vector holds ownership for each entitiy
      CLASSIFICATION //partition vector holds ownership for each classification id
    };

    Input(Omega_h::Mesh& mesh, char* partition_filename,
          Method bufferMethod_, Method safeMethod_,
          Omega_h::CommPtr comm = nullptr);

    Input(Omega_h::Mesh& mesh, Ownership rule, Omega_h::LOs partition_vector,
          Method bufferMethod, Method safeMethod,
          Omega_h::CommPtr comm = nullptr);

    void printInfo();
    static Method getMethod(std::string s);

    //Bridge dim for BFS (defaults to 0)
    int bridge_dim;
    //For Method = BFS, # of layers of BFS to go out to determine buffer (defaults to 3)
    int bufferBFSLayers;
    //For Method = BFS, # of layers of BFS to go out for safe zone (defaults to 1)
    int safeBFSLayers;

    friend class Mesh;
  private:
    Omega_h::Mesh& m;
    Ownership ownership_rule;
    Omega_h::LOs partition;
    Method bufferMethod;
    Method safeMethod;
    Omega_h::CommPtr comm;
  };
}
