#include <iostream>
#include <cmath>

#include "Omega_h_file.hpp"  //gmsh
#include "Omega_h_tag.hpp"
#include "Omega_h_adj.hpp"
#include "Omega_h_array.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_scalar.hpp" //divide
#include "Omega_h_mark.hpp"

//#include "Omega_h_array_ops.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_shape.hpp"
#include "Omega_h_build.hpp"

#include <any>
#include <utility>

#include "Omega_h_align.hpp"
#include "Omega_h_array_ops.hpp"
//#include "Omega_h_bbox.hpp"
#include "Omega_h_compare.hpp"
//#include "Omega_h_confined.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_hypercube.hpp"
#include "Omega_h_int_scan.hpp"
#include "Omega_h_quality.hpp"
#include "Omega_h_recover.hpp"

#include <sstream> //?


namespace GITRm{


OMEGA_H_INLINE Omega_h::Real dot(Omega_h::Vector<3> a, Omega_h::Vector<3> b) OMEGA_H_NOEXCEPT 
{
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}

OMEGA_H_INLINE Omega_h::Reals get_barycentric(Omega_h::Vector<3> a, Omega_h::Vector<3> b, 
     Omega_h::Vector<3> c, Omega_h::Vector<3> d, Omega_h::Vector<3> p ) OMEGA_H_NOEXCEPT 
{
  Omega_h::Vector<3> bp = b - p;
  Omega_h::Vector<3> bd = b - d;
  Omega_h::Vector<3> bc = b - c;
  Omega_h::Vector<3> ap = a - p;
  Omega_h::Vector<3> ac = a - c;
  Omega_h::Vector<3> ad = a - d;
  Omega_h::Vector<3> ab = a - b;
  Omega_h::Real u = dot(bp, Omega_h::cross(bd, bc));
  Omega_h::Real v = dot(ap, Omega_h::cross(ac, ad));
  Omega_h::Real w = dot(ap, Omega_h::cross(ad, ab));
  Omega_h::Real inv_vol = 1.0/dot(ad, Omega_h::cross(ac, ab));   
  u = inv_vol * u;
  v = inv_vol * v;
  w = inv_vol * w;         
  Omega_h::Real x = 1.0 - u - v - w;
  Omega_h::Reals res(4, static_cast<Omega_h::Real>(u,v,w,x)); //wrong, only last 'x' is used

  return res;

 /*     
    \State  $u \gets \|B-P, B-D, B-C\|$ \label{alg:bcctet:tripleStart}
    \State  $v \gets \|A-P, A-C, A-D\|$
    \State  $w \gets \|A-P, A-D, A-B\|$
    \State  $V \gets \|A-D, A-C, A-B\|$
      \label{alg:bcctet:tripleEnd}
    \State  $x \gets 1.0 - u - v - w$
    \State \Return $1/V*[u,v,w,x]$
  */
}
} //namespace

void test_unit(Omega_h::Library *lib);
void print_mesh_stat(Omega_h::Mesh mesh);


int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  test_unit(&lib);
  

  auto mesh = Omega_h::gmsh::read(argv[1], world); 
  const auto mesh2verts = mesh.ask_elem_verts(); //same_as ask_verts_of(REGION), not coords
  const auto coords = mesh.coords();
  const auto dim = mesh.dim();
  print_mesh_stat(mesh);


  //adjacent element
  const auto downs= mesh.ask_down(3, 2).ab2b;
  const auto dual = mesh.ask_dual(); 
  auto dual_faces = dual.ab2b;
  auto dual_elems = dual.a2ab;
  auto side_is_exposed = mark_exposed_sides(&mesh);
  Omega_h::Int nsides_per_elem = dim + 1;
  Omega_h::Int nelems = mesh.nelems();

  
  //Omega_h::Write<Omega_h::LO> elem_elem2elem(nelems*nsides_per_elem);

  //  auto fill = OMEGA_H_LAMBDA(LO elem) { //to be used
  Omega_h::LO ielem = 201; //some element 
  
  auto dface_ind = dual_elems[ielem];
  
  auto begin = ielem * nsides_per_elem;
  auto end = begin + nsides_per_elem;
  std::cout << "Downs:adj for elID " << ielem << "\n";

  for(auto iface = begin; iface < end; ++iface) //faceIDs in order ?
  {
    std::cout << downs[iface] << " ";
    Omega_h::LO faceID = downs[iface];
    if(!side_is_exposed[faceID]) 
    {
       auto adj_elem  = dual_faces[dface_ind];
       std::cout << adj_elem << "\n";
       ++dface_ind;
    }
  }
  //} lambda
  
    
  //get coordinate values

  //Omega_h_confined.cpp 99
  auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, 3);
  auto ttv2x = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
/*     for (Omega_h::Int tte = 0; tte < 3; ++tte) 
     {
        //if (tte2b[tte]) continue;
        auto opp = Omega_h::simplex_opposite_template(Omega_h::REGION, Omega_h::EDGE, tte);
        //if (tte2b[opp]) continue;
        // at this point we have edge-edge nearness
        auto a = ttv2x[Omega_h::simplex_down_template(Omega_h::REGION, Omega_h::EDGE, tte, 0)];
        auto b = ttv2x[Omega_h::simplex_down_template(Omega_h::REGION, Omega_h::EDGE, tte, 1)];
        auto c = ttv2x[Omega_h::simplex_down_template(Omega_h::REGION, Omega_h::EDGE, opp, 0)];
        auto d = ttv2x[Omega_h::simplex_down_template(Omega_h::REGION, Omega_h::EDGE, opp, 1)];
        std::cout << " : " << a-b << "\n";
     }
  */          
 


  return 0;
}





void print_mesh_stat(Omega_h::Mesh mesh)
{
  //std::cout <<"Mesh #coords " << mesh.coords().size()  << "\n"; //3*nverts
  std::cout << "Mesh #elements " << mesh.nelems() 
            << " #faces " << mesh.nfaces() 
            << " #edges " << mesh.nedges() 
            << " #verts " << mesh.nverts() 
            << "\n";
  std::cout << "\t#elem_verts: " << mesh.ask_elem_verts().size() << "\n";
  
  std::cout << "\tdown(3,2): #ab2b " << mesh.ask_down(3, 2).ab2b.size() << "\n";
  std::cout << "\tup(2,3): #a2ab " << mesh.ask_up(2,3).a2ab.size() //NOTE: size is +1 ???
            << " ; #ab2b " << mesh.ask_up(2,3).ab2b.size() << "\n";

  const auto dual = mesh.ask_dual();
  std::cout << "\tdual: #a2ab " << dual.a2ab.size() //last is size=end_index+1 ?
            << " ; #ab2b " << dual.ab2b.size() ;
  std::cout << " ; dual.a2ab[0,1]:" << dual.a2ab[0] << "," << dual.a2ab[1] << "\n";
  
  //dual print
  std::cout <<  "Duals:\n";
  for(int i=0; i<dual.a2ab.size()-1 && i<5; ++i)
  {
     std::cout << dual.a2ab[i] << " : ";
     for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
        std::cout << dual.ab2b[j] <<", ";
     std::cout << "; ";
  }
   
  if(dual.a2ab.size()>5)
  {
     std::cout <<  ".......";   
     for(int i=dual.a2ab.size()-5; i<dual.a2ab.size()-1; ++i)
     {
       std::cout << dual.a2ab[i] << " : ";
       for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
          std::cout << dual.ab2b[j] <<", ";
       std::cout << "; ";
    }
  }
  std::cout <<  "\n";
  
  //exposed
  auto side_is_exposed = mark_exposed_sides(&mesh);
  std::cout << "size exposed faces: "<< side_is_exposed.size() <<"\n";
  std::cout <<  "Exposed:\n";
  for(int i=0; i<side_is_exposed.size() && i<10; ++i)
  {
     if(side_is_exposed[i])
       std::cout << 1 << " : ";
     else
       std::cout << 0 << " : ";        
     //   std::cout << dual.ab2b[j] <<", ";
  }
   
  if(side_is_exposed.size()>10)
  {
     std::cout <<  ".......";   
     for(int i=side_is_exposed.size()-10; i<side_is_exposed.size(); ++i)
     {
       if(side_is_exposed[i])
         std::cout << 1 << " : ";
       else
         std::cout << 0 << " : ";   
     }
  }
  std::cout <<  "\n";
    
}

void test_unit(Omega_h::Library *lib)
{
    Omega_h::Mesh m(lib);
    Omega_h::build_from_elems2verts(&m, OMEGA_H_SIMPLEX, 3, Omega_h::LOs({0, 1, 2, 3}), 4);
    print_mesh_stat(m);
    
    auto up23 = m.ask_up(2, 3);  //
    std::cout <<  "up(2,3): faceID elemID\n";
    for(int i=0; i<up23.a2ab.size()-1 ; ++i) //NOTE: size is +1 ???
     {
       auto face = up23.a2ab[i];
       std::cout << face << " " << up23.ab2b[face] << " ; "; //0 0 ::  1 0 ::  2 0 ::  3 0
     }
    std::cout <<  "\n";
    
    OMEGA_H_CHECK(m.ask_down(3, 0).ab2b == Omega_h::LOs({0, 1, 2, 3}));
    auto down32 = m.ask_down(3, 2);  // 0 1 3 2
    auto down31 = m.ask_down(3, 1);  // 0 3 1 2 4 5
    auto down21 = m.ask_down(2, 1);  //1 3 0 0 4 2 1 2 5 3 5 4 (size=12)
    auto down20 = m.ask_down(2, 0);  //  0 2 1 0 1 3 2 0 3 1 2 3
    auto down10 = m.ask_down(1, 0); // 0 1 2 0 0 3 1 2 1 3 2 3

    std::cout <<  "Down(2,0).ab2b:\n";
    for(int i=0; i<down20.ab2b.size() && i<20; ++i)
    {
      std::cout << " " << down20.ab2b[i];
    }
    //const auto coords = m.coords(); //no tag_base for coords
   
    std::cout <<  "\n----------------\n";
}


