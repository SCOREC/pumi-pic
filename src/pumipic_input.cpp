#include "pumipic_input.hpp"
#include <fstream>
#include <stdexcept>
#include "ppPrint.h"

namespace {
  std::string getMethodString(pumipic::Input::Method m) {
    if( m == pumipic::Input::FULL )
      return "FULL";
    else if( m == pumipic::Input::BFS )
      return "BFS";
    else if( m == pumipic::Input::MINIMUM )
      return "MINIMUM";
    else if( m == pumipic::Input::NONE )
      return "NONE";
    else
      return "UNKNOWN";
  }
}

namespace pumipic {

  Input::Input(Omega_h::Mesh& mesh, char* partition_filename,
               Method bufferMethod_, Method safeMethod_,
               Omega_h::CommPtr c) : m(mesh) {
    partition = Omega_h::LOs(mesh.nelems(), 0);
    ownership_rule = PARTITION;
    if (!c)
      comm = mesh.library()->world();
    else
      comm = c;
    int comm_rank = comm->rank(), comm_size = comm->size();


    if (comm_size > 1) {
      int dot = strlen(partition_filename) - 1;
      while (dot >=0 && partition_filename[dot] != '.')
        --dot;
      if (dot < 0) {
        printError( "[ERROR] Filename provided has no extension (%s)", partition_filename);
        throw std::runtime_error("Filename has no extension");
      }
      char* extension = partition_filename + dot + 1;
      if (strcmp(extension, "ptn") == 0) {
        Omega_h::HostWrite<Omega_h::LO> partition_vector(mesh.nelems());
        std::ifstream in_str(partition_filename);
        if (!in_str) {
          if (!comm_rank)
            printError("Cannot open file %s\n", partition_filename);
          throw std::runtime_error("Cannot open file");
        }
        int own;
        int index = 0;
        while(in_str >> own)
          partition_vector[index++] = own;
        partition = Omega_h::LOs(Omega_h::Write<Omega_h::LO>(partition_vector));
      }
      else if (strcmp(extension, "cpn") == 0) {
        ownership_rule=CLASSIFICATION;
        int root = 0;
        if (comm_rank == root) {
          std::ifstream in_str(partition_filename);
          if (!in_str) {
            if (!comm_rank)
              printError("Cannot open file %s\n", partition_filename);
            throw std::runtime_error("Cannot open file");
          }
          int size;
          in_str>>size;
          Omega_h::HostWrite<Omega_h::LO> host_owners(size+1, "host_owners");
          int cid, own;
          while(in_str >> cid >> own)
            host_owners[cid] = own;
          int length = host_owners.size();
          MPI_Bcast(&length, 1, MPI_INT, root, comm->get_impl());
          MPI_Bcast(host_owners.data(), length, MPI_INT, root, comm->get_impl());
          partition = Omega_h::LOs(Omega_h::Write<Omega_h::LO>(host_owners));

        }
        else {
          int length;
          MPI_Bcast(&length, 1, MPI_INT, root, comm->get_impl());
          Omega_h::HostWrite<Omega_h::LO> host_owners(length, "host_owners");
          MPI_Bcast(host_owners.data(), length, MPI_INT, root, comm->get_impl());
          partition = Omega_h::LOs(Omega_h::Write<Omega_h::LO>(host_owners));

        }

      }
      else {
        printError( "[ERROR] Only .ptn and .cpn partitions are supported");
        throw std::runtime_error("Invalid partition file extension");
      }
    }
    bufferMethod = bufferMethod_;
    if (bufferMethod == NONE) {
      if (!mesh.comm()->rank())
        printInfo("[WARNING] bufferMethod given as NONE, setting to MINIMUM\n");
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
               Method bufferMethod_, Method safeMethod_,
               Omega_h::CommPtr c) : m(mesh) {
    ownership_rule = rule;
    partition = partition_vector;
    bufferMethod = bufferMethod_;
    if (!c)
      comm = mesh.library()->world();
    else
      comm = c;
    if (bufferMethod == NONE) {
      if (!comm->rank())
        printInfo("[WARNING] bufferMethod given as NONE, setting to MINIMUM\n");
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

  Input::Method Input::getMethod(std::string s) {
    const char* cs = s.c_str();
    if( !strcasecmp(cs,"FULL") )
      return Input::FULL;
    else if( !strcasecmp(cs,"BFS") )
      return Input::BFS;
    else if( !strcasecmp(cs,"MINIMUM") )
      return Input::MINIMUM;
    else if( !strcasecmp(cs,"NONE") )
      return Input::NONE;
    else
      return Input::INVALID;
  }

  void Input::printInfo() {
    std::string bname = getMethodString(bufferMethod);
    std::string sname = getMethodString(safeMethod);
    printInfo("pumipic buffer method %s\n", bname.c_str());
    printInfo("pumipic safe method %s\n", sname.c_str());
  }
}
