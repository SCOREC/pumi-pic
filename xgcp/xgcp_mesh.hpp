#pragma once
#include "xgcp_types.hpp"
#include "xgcp_input.hpp"
#include <pumipic_mesh.hpp>
#include <Omega_h_comm.hpp>

using Omega_h::MpiTraits;

/*
  XGCp Mesh encapsulates the partitioning of the mesh, torodial planes, and groups

  Partition Description:
    Mesh is partitioned into picparts based on pumipic::Mesh rules

    Torodial direction is partitioned based on planes with each process
      maintaining a major and minor plane. Every plane will be copied on two
      processes and will be major and minor on exactly one. The process
      with the plane as its major plane 'owns' the plane and will be
      responsible for solving and updating the plane values.

    Groups are copies of the same picpart/plane. Each group has a group leader
      and group members. Fields are accumulated to the group leader and scattered
      back to the members.
 */

namespace xgcp {
  class Mesh {
  public:
    Mesh(xgcp::Input&);
    ~Mesh();
    //Delete default compilers
    Mesh() = delete;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    //Directly access underlying mesh structures
    o::Mesh* operator->() {return picparts->mesh();}
    o::Mesh* omegaMesh() {return picparts->mesh();}
    p::Mesh* pumipicMesh() {return picparts;}

    /********Common mesh count information********/
    //Returns the dimension of the mesh
    int dim() const {return picparts->dim();}
    //Returns the number of entities of the picpart
    Omega_h::LO nents(int dim) const {return picparts->nents(dim);}
    //Returns the number of elements of the picpart
    Omega_h::LO nelems() const {return picparts->nelems();}

    /********Comm Rank/Size/Comm functions********/
    int worldRank() {return omegaMesh()->library()->world()->rank();}
    int worldSize() {return omegaMesh()->library()->world()->size();}
    MPI_Comm worldComm() {return omegaMesh()->library()->world()->get_impl();}

    int meshRank() {return mesh_comm->rank();}
    int meshSize() {return mesh_comm->size();}
    MPI_Comm meshComm() {return mesh_comm->get_impl();}

    int groupRank() {return group_comm->rank();}
    int groupSize() {return group_comm->size();}
    MPI_Comm groupComm() {return group_comm->get_impl();}
    bool isGroupLeader() {return group_comm->rank() == groupLeader();}
    int groupLeader() {return 0;}

    int torodialRank() {return torodial_comm->rank();}
    int torodialSize() {return torodial_comm->size();}
    MPI_Comm torodialComm() {return torodial_comm->get_impl();}
    int torodialMinorNeighbor() {return torodialRank() == 0 ?
        torodialSize()-1 : torodialRank() - 1;}
    int torodialMajorNeighbor() {return torodialRank() == torodialSize() - 1 ?
        0 : torodialRank() + 1;}

    int planeID() {return torodialRank();}
    fp_t getMajorPlaneAngle() {return major_phi;}
    fp_t getMinorPlaneAngle() {return minor_phi;}

    typedef Omega_h::Write<Omega_h::Real> GyroField;
    typedef Omega_h::Read<Omega_h::Real> GyroFieldR;
    //Get the pointers to the major and minor plane fields
    void getGyroFields(GyroField& major_plane, GyroField& minor_plane);
    //Apply the major and minor plane fields to tags on the mesh
    void applyGyroFieldsToTags();
    //Get the gyro mappings for major and minor planes for ions
    void getIonGyroMappings(Omega_h::LOs& major_map, Omega_h::LOs& minor_map);

    enum PartitionLevel {
      GROUP,
      TORODIAL,
      MESH
    };
    /*
      Accumulates field contributions across different partitioning levels
       * dim - the dimension of the mesh
       * major - the major plane contributions for the field being reduced
       * minor - the minor plane contributions for the field being reduced
       * start_level(optional) - The initial partitioning level (defaults to GROUP)
       * end_level(optional) - The final partitioning level (defaults to MESH)

       Field contributions are accumulated across the mesh entities of dimension `dim`
       from `start_level` to `end_level` inclusively. The accumulation order is
       GROUP > TORODIAL > MESH. By default all levels will be accumulated.

       GROUP gather is done by summing contributions to each group leader.
       TORODIAL gather is done by sending contriutions of each group leader's minor plane
         to the neighboring plane's major plane
       MESH gather is done using pumipic::Mesh's reduceCommArray for each group
         leader's major plane

       Note: end_level must be greater than start_level
     */
    template <class T>
    void gatherField(int dim, Omega_h::Write<T>& major, Omega_h::Read<T> minor,
                     PartitionLevel start_level = GROUP, PartitionLevel end_level = MESH);

    /*
      Scatters field values across different partitioning levels
       * dim - the dimension of the mesh
       * major - the major plane contributions for the field being reduced
       * minor - the minor plane contributions for the field being reduced
       * start_level(optional) - The initial partitioning level (defaults to GROUP)
       * end_level(optional) - The final partitioning level (defaults to MESH)

       Field contributions are scattered across the mesh entities of dimension `dim`
       from `start_level` to `end_level` inclusively. The accumulation order is
       MESH > TORODIAL > GROUP. By default all levels will be accumulated.

       MESH scatter is performed using pumipic::Mesh's reduceCommArray with accept
         operation for each group leader's major plane
       TORODIAL scatter is done by sending values of each group leader's major plane
         to the neighboring plane's minor plane
       GROUP scatter is done by scattering field values from the group leader to each group member

       Note: end_level must be greater than start_level
     */
    template <class T>
    void scatterField(int dim, Omega_h::Write<T>& major, Omega_h::Write<T>& minor,
                      PartitionLevel start_level = MESH, PartitionLevel end_level = GROUP);

  private:
    //Mesh structure for the picparts
    p::Mesh* picparts;
    o::Mesh* full_mesh;

    //Communicators for each partitioning direction
    o::CommPtr mesh_comm, torodial_comm, group_comm;

    //Torodial angle of the plane that this process is not leader of
    fp_t minor_phi;
    //Torodial angle of the plane that this process is leader of
    fp_t major_phi;

    //Major and minor ion projection mapping for each ring point to 3 mesh vertices
    Omega_h::LOs major_ion_gyro_map, minor_ion_gyro_map;

    //Major and minor electron projection mapping for each vertex to 3 mesh vertices
    Omega_h::LOs major_electron_gyro_map, minor_electron_gyro_map;

    //Tags of fields for the major and minor plane contributions on vertices
    GyroField major_plane, minor_plane;

  };

  //TODO Create methods to distringuish between CUDA_AWARE MPI and non cuda aware
  template <class T>
  void Mesh::gatherField(int dim, Omega_h::Write<T>& major, Omega_h::Read<T> minor,
                         PartitionLevel start_level, PartitionLevel end_level) {
    if (start_level <= TORODIAL && end_level >= GROUP) {
      Omega_h::HostWrite<T> major_host(major);
      Omega_h::HostRead<T> minor_host(minor);
      //Reduce to group leader
      if (start_level <= GROUP && end_level >= GROUP) {
        if(isGroupLeader())
          MPI_Reduce(MPI_IN_PLACE, major_host.data(), major_host.size(), MpiTraits<T>::datatype(),
                     MPI_SUM, groupLeader(), groupComm());
        else
          MPI_Reduce(major_host.data(), major_host.data(), major_host.size(),
                     MpiTraits<T>::datatype(), MPI_SUM, groupLeader(), groupComm());
      }
      //Send minor contributions to neighbor and recv to major plane
      if (start_level <= TORODIAL && end_level >= TORODIAL && isGroupLeader()) {
        MPI_Request minor_req, major_req;
        MPI_Isend(minor_host.data(), minor_host.size(), MpiTraits<T>::datatype(),
                  torodialMinorNeighbor(), 0, torodialComm(), &minor_req);
        Omega_h::HostWrite<T> major_recv(major_host.size(), "major_recv");
        MPI_Irecv(major_recv.data(), major_recv.size(), MpiTraits<T>::datatype(),
                  torodialMajorNeighbor(), 0, torodialComm(), &major_req);
        MPI_Status minor_stat, major_stat;
        MPI_Wait(&major_req, &major_stat);

        //??? Is it worth moving the data to the device to sum?
        for (int i = 0; i < major_recv.size(); ++i)
          major_host[i] += major_recv[i];
        MPI_Wait(&minor_req, &minor_stat);
      }
      major = Omega_h::Write<T>(major_host);
    }
    //Perform mesh reduce on mesh for each plane
    if (start_level <= MESH && end_level >= MESH && isGroupLeader()) {
      picparts->reduceCommArray(0, pumipic::Mesh::SUM_OP, major);
    }
  }

  template <class T>
  void Mesh::scatterField(int dim, Omega_h::Write<T>& major, Omega_h::Write<T>& minor,
                          PartitionLevel start_level, PartitionLevel end_level) {
    if (start_level >= MESH && end_level <= MESH && isGroupLeader()) {
      picparts->reduceCommArray(0, pumipic::Mesh::BCAST_OP, major);
    }
    if (start_level >= GROUP && end_level <= TORODIAL) {
      Omega_h::HostWrite<T> major_host(major);
      Omega_h::HostWrite<T> minor_host(minor);
      //Send major contributions to neighbor and recv to minor plane
      if (start_level >= TORODIAL && end_level <= TORODIAL && isGroupLeader()) {
        MPI_Request minor_req, major_req;
        MPI_Isend(major_host.data(), major_host.size(), MpiTraits<T>::datatype(),
                  torodialMajorNeighbor(), 0, torodialComm(), &major_req);
        Omega_h::HostWrite<T> minor_recv(minor_host.size(), "minor_recv");
        MPI_Irecv(minor_recv.data(), minor_recv.size(), MpiTraits<T>::datatype(),
                  torodialMinorNeighbor(), 0, torodialComm(), &minor_req);
        MPI_Status minor_stat, major_stat;
        MPI_Wait(&minor_req, &minor_stat);

        //??? Is it worth moving the data to the device to set values?
        for (int i = 0; i < minor_recv.size(); ++i)
          minor_host[i] = minor_recv[i];
        MPI_Wait(&major_req, &major_stat);
      }
      if (start_level >= GROUP && end_level <= GROUP) {
        MPI_Bcast(minor_host.data(), minor_host.size(), MpiTraits<T>::datatype(),
                  groupLeader(), groupComm());
        MPI_Bcast(major_host.data(), major_host.size(), MpiTraits<T>::datatype(),
                  groupLeader(), groupComm());

      }
      minor = Omega_h::Write<T>(minor_host);
      major = Omega_h::Write<T>(major_host);
    }
  }
}
