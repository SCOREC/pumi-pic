#include <iostream>
#include <cmath>
#include <utility>

#include "Omega_h_for.hpp"
#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
//#include "Omega_h_array.hpp"
#include "Omega_h_array_ops.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"
#include "Omega_h_fail.hpp" //assert

#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_compare.hpp"


#include "gitrm_utils.hpp"
#include "gitrm_adjacency.hpp"
#include "unit_tests.hpp"

namespace g = GITRm;

#define DO_TEST 0
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world); 
  const auto dim = mesh.dim();
  
#if DO_TEST==1 
  test_unit(&lib); 
  print_mesh_stat(mesh);
  test_barycentric(mesh);  
#endif  


  //adjacent element
  const auto downs= mesh.ask_down(3, 2).ab2b;
  const auto dual = mesh.ask_dual(); 
  auto dual_faces = dual.ab2b;
  auto dual_elems = dual.a2ab;
  auto side_is_exposed = mark_exposed_sides(&mesh);
  Omega_h::Int nsides_per_elem = dim + 1;
  Omega_h::Int nelems = mesh.nelems();

  
  //Omega_h::Write<Omega_h::LO> elem_elem2elem(nelems*nsides_per_elem);

  const auto test = OMEGA_H_LAMBDA(Omega_h::LO ielem)
  {
    //Omega_h::LO ielem = 201; //some element 
    
    auto dface_ind = dual_elems[ielem];
    
    auto begin = ielem * nsides_per_elem;
    auto end = begin + nsides_per_elem;
    //std::cout << "Downs:adj for elID " << ielem << "\n";

    for(auto iface = begin; iface < end; ++iface) //faceIDs in order ?
    {
      std::cout << downs[iface] << " adj_element ";
      Omega_h::LO faceID = downs[iface];
      if(!side_is_exposed[faceID]) 
      {
         auto adj_elem  = dual_faces[dface_ind];
         std::cout << adj_elem << "\n";
         ++dface_ind;
      }
    }
  };
  //Omega_h::parallel_for(2, test, "adj"); //mesh.nelems()
  

  
  //get coordinate values
  const auto mesh2verts = mesh.ask_elem_verts(); 
  const auto coords = mesh.coords();  

  //barycentric    
  auto bc_fun = OMEGA_H_LAMBDA(Omega_h::LO ielem) 
  {
    auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
    //const auto M = Omega_h::simplex_basis<3, 3>(M);  // ??
    Omega_h::Write<Omega_h::Real> bc(4, -1.0);
    const Omega_h::Vector<3> p{3.0,0.99,0.2};

    bool res = g::find_barycentric(M[0], M[1], M[2], M[3], p, bc);
    if(g::all_positive(bc.data(), 4)) 
    {
      
      std::cout << "found in " << ielem << " \n";
      g::print_matrix(M);
    } 
    else
    {
      Omega_h::LO next = g::most_negative_index(bc.data(), 4); //only 0 or 1  ??
      //std::cout << "NEXT: " << next << " :: " << bc[0] << " " << bc[1] << " " << bc[2] << " " << bc[3]<< "\n";
      
      //get element ID for the above next
      auto dface_ind = dual_elems[ielem];
      
      auto begin = ielem * nsides_per_elem;
      auto end = begin + nsides_per_elem;
      //std::cout << "Downs:adj for elID " << ielem << "\n";
      Omega_h::LO fIndex = 0;
      for(auto iface = begin; iface < end; ++iface) //faceIDs in order ?
      {
        
        Omega_h::LO faceID = downs[iface];
        if(!side_is_exposed[faceID]) 
        {
           auto adj_elem  = dual_faces[dface_ind];
           if(fIndex == next)
           { 
             //std::cout << " Store adj_elem " << adj_elem << " for el|faceID " << ielem << "," << downs[iface] <<  "\n";
             break;
           }
           //std::cout << downs[iface] << " adj:counting.. " << adj_elem << "\n";
           ++dface_ind;
        }
        else if(fIndex == next)
        {
          std::cout << "Call Wall collision \n";
        }
        ++fIndex;
      }
    
    }
  };
  Omega_h::parallel_for(50,bc_fun, "bc");//mesh.nelems(), bc_fun, "bc");

    
  return 0;
}
//LO a....
//a.get(i)

