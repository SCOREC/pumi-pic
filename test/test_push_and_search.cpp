#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"
#include "Omega_h_fail.hpp" //assert

#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"
#include "Omega_h_reduce.hpp"

#include "pumipic_adjacency.hpp"
#include "unit_tests.hpp"
#include "pumipic_utils.hpp"
#include "pumipic_push.hpp"


namespace o = Omega_h;
namespace p = pumipic;

#define PRINT_DETAIL 0
int main(int argc, char** argv) 
{

  if(argc < 2) 
  {
    std::cout << "Usage: ./push_and_search mesh [nsteps] \n"
              << "Example: ./push_and_search  cube.msh 100\n";
    exit(1);
  }
   
  int nsteps = (argc>2)? atoi(argv[2]) : 100;
  
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);

  // common mesh data
  // dual, up, down are Adj ~ Graph{arrays:LOs a2ab,ab2b}
  const auto dual = mesh.ask_dual();
  const auto down_r2f = mesh.ask_down(3, 2);
  //coordinates
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs
  const auto side_is_exposed = mark_exposed_sides(&mesh);

  const auto dim = mesh.dim();
  Omega_h::Int nelems = mesh.nelems();

#if PRINT_DETAIL==1
  test_unit(&lib);
  print_mesh_stat(mesh);
  //p::print_array(&data.data()->data()[0], 3, "data");
#endif

  //TODO add tag fields
  Omega_h::Write<Omega_h::Real> eFields(3*nelems,0.1);
  Omega_h::Write<Omega_h::Real> bFields(3*nelems,0.1);
  //mesh.add_tag(0, "EField", 3);

  Omega_h::Write<Omega_h::LO> elem_nptcls(nelems, 0);

  const Omega_h::LO np = 10; //#particles in rank

  //Particle data
  Omega_h::Write<Omega_h::Real> x(np,0);
  Omega_h::Write<Omega_h::Real> y(np,0);
  Omega_h::Write<Omega_h::Real> z(np,0);
  Omega_h::Write<Omega_h::Real> xp(np,0);
  Omega_h::Write<Omega_h::Real> yp(np,0);
  Omega_h::Write<Omega_h::Real> zp(np,0);
  Omega_h::Write<Omega_h::Real> x0(np,0);
  Omega_h::Write<Omega_h::Real> y0(np,0);
  Omega_h::Write<Omega_h::Real> z0(np,0);
  Omega_h::Write<Omega_h::Real> vx(np,0);
  Omega_h::Write<Omega_h::Real> vy(np,0);
  Omega_h::Write<Omega_h::Real> vz(np,0);
  Omega_h::Write<Omega_h::Real> eFld0x(np,0);
  Omega_h::Write<Omega_h::Real> eFld0y(np,0);
  Omega_h::Write<Omega_h::Real> eFld0z(np,0);
  Omega_h::Write<Omega_h::Real> bFld0r(np,0);
  Omega_h::Write<Omega_h::Real> bFld0t(np,0);
  Omega_h::Write<Omega_h::Real> bFld0z(np,0);
  Omega_h::Write<Omega_h::Real> bccs(4*np, -1.0);
  Omega_h::Write<Omega_h::Real> xpoints(3*np, -1.0);
  Omega_h::Write<Omega_h::LO> part_flags(np, 1); // to do or not
  Omega_h::Write<Omega_h::LO> elem_ids(np); //next element to search for
  Omega_h::Write<Omega_h::LO> coll_adj_face_ids(np, -1);
  Omega_h::Write<Omega_h::LO> hitWall(np,0);

  elem_ids[0] = 87;
  double xinit[] = {-1,-1,-1};

  //initialize particles to replace this
  xp[0] = 5.0;
  yp[0] = 0.3;
  zp[0] = 0.4;
  x[0] = xp[0];
  y[0] = yp[0];
  z[0] = zp[0];
  const double xref[] ={4.406160, 0.269205, 1};
  
  //todo:load fields on mesh tags
  eFld0x[0]=2e-2;
  eFld0y[0]=1e-2;
  eFld0z[0]=1e-1;
  bFld0r[0]=1;
  bFld0t[0]=0.2;
  bFld0z[0]=1;

  Omega_h::LO gpSize=1; //per group, temporary

  Omega_h::Real dt = 5e-4;


  //Main loop
  Omega_h::LO loops = 0;
  for(int it=0; it<nsteps; ++it)
  {
    //reset search flags
    for(int i=0; i<1; ++i){ if(part_flags[i]==0){part_flags[i]=1;} }

    //todo: memcpy all ptcls
    for(int pid=0; pid<1; ++pid)
    {
      if(part_flags[pid] <=0) continue;
      x0[pid]=x[pid];
      y0[pid]=y[pid];
      z0[pid]=z[pid];
    }
    //push
    p::pushBoris(nelems, x, y,z, xp, yp, zp, vx, vy, vz, eFld0x, eFld0y, eFld0z, bFld0r, bFld0t, bFld0z, part_flags, dt);

#ifdef DEBUG
    std::cout << x0[0] << " " << y0[0] << " " <<  z0[0] << " => " << x[0] << " " << y[0] << " " <<  z[0] << "\n";
#endif
    //search
    p::search_mesh(gpSize, nelems, x0, y0, z0, x, y, z, dual, down_r2f, side_is_exposed,
       mesh2verts, coords, face_verts, part_flags, elem_ids, coll_adj_face_ids, bccs, xpoints, loops);

    if(!p::compare_array(xpoints.data(), xinit, 3, 1e-5))
    {
#ifdef DEBUG
      p::print_array(&xpoints[0], 3, "XPOINT");
#endif
      break;
    }
#ifdef DEBUG
    p::print_array(&bccs[0], 3, "BCCS");
    //if(!part_flags[0])
    std::cout << it << " in " << elem_ids[0] << " iter " << loops << "\n"; //only 1st particle
#endif
  }

  //TODO update this if Xpoint, or x,y,z changed. 
  //Valid for the rectangular block mesh (cube.msh) and the orig(5,0.3,0.4)  
  if(!(p::compare_array(xpoints.data(), xref, 3, 1e-5) ))
    return 1; //Failed

  return 0;
}



