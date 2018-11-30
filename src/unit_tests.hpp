#ifndef GITRM_UNIT_TESTS_HPP
#define GITRM_UNIT_TESTS_HPP

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

#include "gitrm_adjacency.hpp"
#include "gitrm_utils.hpp"

namespace g = GITRm;


//static in separate file if not in class
void test_barycentric2(const Omega_h::Vector<3>a, const Omega_h::Vector<3>b,
    const Omega_h::Vector<3> c, const Omega_h::Vector<3> d, 
    const Omega_h::Vector<3>p, double v, int pos, bool intent=0)
{
  Omega_h::Write<Omega_h::Real> bc(4, -1.0);
  bool res = g::find_barycentric(a, b, c, d, p, bc);
  if(g::almost_equal(bc[pos], v)) //    OMEGA_H_CHECK(bc[pos]== v);
    std::cout << "Barycentric test passed for (" << p[0] << ","
              << p[1] << "," << p[2]<< ") => 1==" << bc[pos] << "\n";
  else if(intent)
  {
    std::cout << "Barycentric test intented to fail (" << p[0] << ","
              << p[1] << "," << p[2]<< ") => 1!=" << bc[pos] << " success :)\n";
  }
  else
    Omega_h_fail("Barycentric test failed for (%0.6f, %0.6f, %0.6f): 1 != %0.6f \n", 
        p[0], p[1], p[2], bc[pos]);
}

bool test_barycentric(Omega_h::Mesh &mesh)
{ 
  //const Omega_h::Matrix<3,4> M{{9.0,1.0,1.0}, {9.5,0.5,1.0}, 
  //  {9.5,0.0,0.5}, {9.5,0.5,0.0}};//good
  
  const Omega_h::Vector<3> p1{9.0,1.0,1.0}, p2{9.5,0.5,1.0}, 
      p3{9.5,0.0,0.5}, p4{9.5,0.5,0.0};
  
  test_barycentric2(p1, p2, p3, p4, p1, 1.0, 0);
  test_barycentric2(p1, p2, p3, p4, p2, 1.0, 1);
  test_barycentric2(p1, p2, p3, p4, p3, 1.0, 2);
  test_barycentric2(p1, p2, p3, p4, p4, 1.0, 3);
  test_barycentric2(p1, p2, p3, p4, p4, 1.0, 2, 1);  
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

#endif
