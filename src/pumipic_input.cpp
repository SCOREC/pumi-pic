#include "pumipic_input.hpp"
#include <fstream>
namespace pumipic {

  Input::Input(Omega_h::Mesh& mesh, char* partition_filename, 
               Method bufferMethod_, Method safeMethod_) : m(mesh) {
    partition = Omega_h::LOs(mesh.nelems(), 0);
    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (mesh.comm()->size() > 1) {
      int dot = strlen(partition_filename) - 1;
      while (dot >=0 && partition_filename[dot] != '.')
        --dot;
      if (dot < 0) {
        fprintf(stderr, "[ERROR] Filename provided has no extension (%s)", partition_filename);
        throw 1;
      }
      char* extension = partition_filename + dot + 1;
      if (strcmp(extension, "ptn")) {
        ownership_rule = PARTITION;
        Omega_h::HostWrite<Omega_h::LO> partition_vector(mesh.nelems());
        std::ifstream in_str(partition_filename);
        if (!in_str) {
          if (!comm_rank)
            fprintf(stderr,"Cannot open file %s\n", partition_filename);
          throw 2;
        }
        int own;
        int index = 0;
        while(in_str >> own) 
          partition_vector[index++] = own;
        partition = Omega_h::LOs(Omega_h::Write<Omega_h::LO>(partition_vector));
      }
      else if (strcmp(extension, "cpn")) {
        ownership_rule=CLASSIFICATION;
        std::ifstream in_str(partition_filename);
        if (!in_str) {
          if (!comm_rank)
            fprintf(stderr,"Cannot open file %s\n", partition_filename);
          throw 2;
        }
        int size;
        in_str>>size;
        Omega_h::HostWrite<Omega_h::LO> host_owners(size+1);
        int cid, own;
        while(in_str >> cid >> own) 
          host_owners[cid] = own;

        partition = Omega_h::LOs(Omega_h::Write<Omega_h::LO>(host_owners));

      }
    }
    bufferMethod = bufferMethod_;
    if (bufferMethod == NONE) {
      if (!mesh.comm()->rank())
        printf("[WARNING] bufferMethod given as NONE, setting to MINIMUM\n");
      bufferMethod=MINIMUM;
    }
    safeMethod = safeMethod_;

    bridge_dim = 0;
    bufferBFSLayers = 3;
    safeBFSLayers = 1;

    if (bufferMethod == MINIMUM)
      bufferBFSLayers = 0;
    if (safeMethod == MINIMUM)
      safeBFSLayers = 0;

  }
  Input::Input(Omega_h::Mesh& mesh, Ownership rule, Omega_h::LOs partition_vector,
               Method bufferMethod_, Method safeMethod_) : m(mesh) {
    ownership_rule = rule;
    partition = partition_vector;
    bufferMethod = bufferMethod_;
    if (bufferMethod == NONE) {
      if (!mesh.comm()->rank())
        printf("[WARNING] bufferMethod given as NONE, setting to MINIMUM\n");
      bufferMethod=MINIMUM;
    }
    safeMethod = safeMethod_;

    bridge_dim = 0;
    bufferBFSLayers = 3;
    safeBFSLayers = 1;

    if (bufferMethod == MINIMUM)
      bufferBFSLayers = 0;
    if (safeMethod == MINIMUM)
      safeBFSLayers = 0;
  }
}
