#pragma once
#include <Omega_h_mesh.hpp>

namespace pumipic {

  class Mesh;

  class Input {
  public:

    /* Method for creating buffers/safe zone

       FULL:
         Buffer: Each picpart is composed of the entire mesh
         Safe: Each safe zone is the entire picpart*
           * If buffer is created using BFS then the safe zone
             will be some number of layers away from the boundary of
             the picpart
       BFS:
         Buffer: Each picpart is composed of each core region within
                 `bufferBFSLayers` layers of elements from the boundary
                 of the part's core region.
         Safe: The safe zone is composed of the core region plus
               elements within `safeBFSLayers` layers from the boundary
               of the part's core region.
       MINIMUM:
         Buffer: Each picpart is composed of the core region
         Safe: The safe zone is composed of each element in the core region
       NONE:
         Buffer: Invalid, uses MINIMUM instead
         Safe: The safe zone is empty
     */
    enum Method {
      INVALID = -1,
      FULL,
      BFS,
      MINIMUM,
      NONE
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

    void printMethod();
    static Method getMethod(std::string s);

    Ownership getRule() const {return ownership_rule;}
    Omega_h::LOs getPartition() const {return partition;}

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
