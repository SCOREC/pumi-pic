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

//OMEGA_H_INLINE needed ?
OMEGA_H_INLINE Omega_h::Real dot(const Omega_h::Vector<3> &a, 
         const Omega_h::Vector<3> &b) OMEGA_H_NOEXCEPT 
{
  //std::cout << "DOT_a,b:"  << a[0] << " " << a[1] << " " << a[2] << " ; " 
  //          << b[0] << " " << b[1] << " " << b[2] <<  "\n";
  return (a[0]*b[0] + a[1]*b[1] + a[2]*b[2]);
}


OMEGA_H_INLINE bool find_barycentric(const Omega_h::Vector<3> &a, 
     const Omega_h::Vector<3> &b, const Omega_h::Vector<3> &c, 
     const Omega_h::Vector<3> &d,  const Omega_h::Vector<3> &p, 
     Omega_h::Write<Omega_h::Real> &bcoords ) OMEGA_H_NOEXCEPT 
{
  //Omega_h::Write<Omega_h::Real> bc(4, 0);
  Omega_h::Real u = dot(b-p, Omega_h::cross(b-d, b-c));
  Omega_h::Real v = dot(a-p, Omega_h::cross(a-c, a-d));
  Omega_h::Real w = dot(a-p, Omega_h::cross(a-d, a-b));
  Omega_h::Real vol = dot(a-d, Omega_h::cross(a-c, a-b)); 
  Omega_h::Real inv_vol = 0.0;
  if(std::abs(vol) > 0)
    inv_vol = 1.0/vol;    
  else
    return 0;
  bcoords[0] = inv_vol * u;
  bcoords[1] = inv_vol * v;
  bcoords[2] = inv_vol * w;         
  bcoords[3] = 1.0 - bcoords[0] - bcoords[1] - bcoords[2];
  //std::cout << "BC_u,v,w,inv:" << u << " " << v << " " << w <<" "<< inv_vol << "\n";

  return 1; // ?
}



} //namespace

void test_unit(Omega_h::Library *lib);
void print_mesh_stat(Omega_h::Mesh &mesh);

namespace g = GITRm;
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  //test_unit(&lib); 
  
  auto mesh = Omega_h::gmsh::read(argv[1], world); 
  const auto dim = mesh.dim();
  //print_mesh_stat(mesh);


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
  };
  Omega_h::parallel_for(2, test, "adj");
  
  //get coordinate values
  const auto mesh2verts = mesh.ask_elem_verts(); 
  const auto coords = mesh.coords();  
  
  //barycentric    
  auto bc_fun = OMEGA_H_LAMBDA(Omega_h::LO ielem) 
  {
    auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    auto ttv2x = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
    const auto M = Omega_h::simplex_basis<3, 3>(ttv2x);  // ??
    
    Omega_h::Write<Omega_h::Real> bc(4, -1.0);
    const Omega_h::Vector<3> p{1,2,3};

     auto a = M[0]; 
     auto b = M[1];
     auto c = M[2];
     //auto d = M[3]; //crash ????
     const Omega_h::Vector<3> d{0.2,0.5,1};  
    //bool res = g::find_barycentric(M[0], M[1], M[2], M[3], p, bc);
 
     bool res = g::find_barycentric(a,b,c,d,p,bc);
     std::cout << "TESTBC "  << res << " " << bc[0] << " " << bc[1] << " "<< bc[2] << " " << bc[3] << "\n";  
  };
  Omega_h::parallel_for(10, bc_fun, "bc");




    
  return 0;
}


void print_mesh_stat(Omega_h::Mesh &m)
{
  //std::cout <<"Mesh #coords " << m.coords().size()  << "\n"; //3*nverts
  std::cout << "Mesh #elements " << m.nelems() 
            << " #faces " << m.nfaces() 
            << " #edges " << m.nedges() 
            << " #verts " << m.nverts() 
            << "\n";
  std::cout << "\t#elem_verts: " << m.ask_elem_verts().size() << "\n";
  
  std::cout << "\tdown(3,2): #ab2b " << m.ask_down(3, 2).ab2b.size() << "\n";
  std::cout << "\tup(2,3): #a2ab " << m.ask_up(2,3).a2ab.size() //NOTE: size is +1 ???
            << " ; #ab2b " << m.ask_up(2,3).ab2b.size() << "\n";

  const auto dual = m.ask_dual();
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
  auto side_is_exposed = mark_exposed_sides(&m);
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
  std::cout << "\n";
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


