#ifndef GITRM_UNIT_TESTS_HPP
#define GITRM_UNIT_TESTS_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <exception>
#include <typeinfo>
#include <sstream>

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

#define DEBUG 0

namespace g = GITRm;

//static in separate file if not in class
bool test_barycentric_tet(const Omega_h::Matrix<3, 4> &M,
    const Omega_h::Vector<3> &p, const double *v, int pos=-1, bool intent=0)
{
  Omega_h::Write<Omega_h::Real> bcc(4, -1.0);
  bool res = g::find_barycentric_tet(M, p, bcc);
  if(pos == -1)
  {
  #ifdef DEBUG
    g::print_osh_vector(M[0], "", false);
    g::print_osh_vector(M[1], "", false);
    g::print_osh_vector(M[2], "", false);
    g::print_osh_vector(M[3], "", false);
    g::print_osh_vector(p, " P");
    g::print_array(bcc.data(), 4, "BCC");
#endif // DEBUG

    //assumes size of v is 4
    for(int i=0; i<4; ++i)
    {
      if(std::abs(bcc[i]- v[i]) > 1e-10)
      {
#ifdef DEBUG
          //Omega_h_fail format
          printf("Barycentric test failed: p=(%0.6f, %0.6f, %0.6f);"
                       " calculated_bc=(%0.6f, %0.6f, %0.6f, %0.6f): bc=%0.6f != v=%.6f \n",
                    p[0], p[1], p[2], bcc[0], bcc[1], bcc[2], bcc[3], bcc[i], v[i]);
#endif // DEBUG
        return 0;
      }
    }
#ifdef DEBUG
    std::cout << "Barycentric test passed \n";
#endif // DEBUG
  }
  else  //, v has size=1
  {
    if(g::almost_equal(bcc[pos], v[0]))
    {
#ifdef DEBUG
      std::cout << "Barycentric test passed for (" << p[0] << ","
                << p[1] << "," << p[2]<< ") => " << v[0] << "==" << bcc[pos] << "\n";
#endif // DEBUG
    }
    else if(intent)
    {
#ifdef DEBUG
      std::cout << "Barycentric test intented to fail (" << p[0] << ","
                << p[1] << "," << p[2]<< ") => " << v[0]  << "!=" << bcc[pos] << " success :)\n";
#endif // DEBUG
    }
    else
    {
#ifdef DEBUG
      g::print_matrix(M );
      g::print_osh_vector(p, "p");
      Omega_h_fail("Barycentric test failed : %0.3f != %0.3f \n", v[0], bcc[pos]);
#endif // DEBUG
      return 0;
    }
  }
  return 1;
}


bool test_barycentric1()
{
  const Omega_h::Vector<3> p1{0.0,1.0,0.0}, p2{0.5, 0.0, 0.0},
      p3{1.0,1.0,0.0}, p4{0.5,1.0,0.5};
  const Omega_h::Matrix<3, 4> M{p1, p2, p3, p4};
  //face_vert:0,2,1(a,c,b); 0,1,3(a,b,d); 1,2,3(b,c,d); 2,0,3(c,a,d)
  const Omega_h::Vector<4> bcc1{1.0, 0.0, 0.0, 0.0};
  const Omega_h::Vector<4> bcc2{0.0, 1.0, 0.0, 0.0};
  const Omega_h::Vector<4> bcc3{0.0, 0.0, 1.0, 0.0};
  const Omega_h::Vector<4> bcc4{0.0, 0.0, 0.0, 1.0};
  const Omega_h::Matrix<4, 4> bcc_mat{bcc1, bcc2, bcc3, bcc4};
  std::string bcname[] = {"u", "v", "w", "x"};
  int index = -1;
  for(int i=0; i<4; ++i)
  {
#ifdef DEBUG
    std::cout << "Barycentric test : " << bcname[i] <<  " \n";
#endif // DEBUG
    index = Omega_h::simplex_opposite_template(3, 2, i);
    if(test_barycentric_tet(M, M[index], bcc_mat[i].data()))
    {
#ifdef DEBUG
      std::cout << "Passed \n";
#endif // DEBUG
    }
    else
    {
#ifdef DEBUG
      std::cout << "Failed \n";
#endif // DEBUG
      return 0;
    }
  }

  double val = 1.0;
  if(!test_barycentric_tet(M, p2, &val, 2, 1))
    return 0;

  return 1;
}

bool test_barycentric2()
{
  const Omega_h::Vector<3> p1{0.0,0.0,0.0}, p2{1.0, 0.0, 0.0},
      p3{0.5,0.5,0.0}, p4{0.5,0.25,1.0};

  double val = 1.0;
  const Omega_h::Vector<3> p{0.2, 0.1, 0.1};
  double vals[] = {0.1, 0.15, 0.675, 0.075};

  if( !test_barycentric_tet({p1, p2, p3, p4}, p, vals))
    return 0;

  val = 0.1;
  const Omega_h::Vector<3> pt1{1.5, 0.1, 0.1};
  if( !test_barycentric_tet({p1, p2, p3, p4}, pt1, &val, 0))
    return 0;

  val = 0.35;
  const Omega_h::Vector<3> pt2{0.1, 0.2, 0.1};
  if( !test_barycentric_tet({p1, p2, p3, p4}, pt2, &val, 1))
     return 0;

  val = 1.075;
  const Omega_h::Vector<3> pt3{0.1, -0.2, 0.1};
  if( !test_barycentric_tet({p1, p2, p3, p4}, pt3, &val, 2))
     return 0;

  val = 0.325;
  const Omega_h::Vector<3> pt4{0.1, -0.2, -0.1};

  if( !test_barycentric_tet({p1, p2, p3, p4}, pt4, &val, 3))
     return 0;

  return 1;
}

bool test_barycentric_tri()
{
  const Omega_h::Vector<3> p1{0.0,0.0,0.0}, p2{2.0, 0.0, 0.0}, p3{1.0,1.0,0.0};
  const Omega_h::Matrix<3, 3> M{p1, p2, p3};
  Omega_h::Write<Omega_h::Real> bc{-1, -1, -1};

  const Omega_h::Vector<3> bcc1{1.0, 0.0, 0.0};
  const Omega_h::Vector<3> bcc2{0.0, 1.0, 0.0};
  const Omega_h::Vector<3> bcc3{0.0, 0.0, 1.0};
  const Omega_h::Matrix<3, 3> bcc_mat{bcc1, bcc2, bcc3};
  std::string bcname[] = {"u", "v", "w"};
  int index = -1;
  for(int i=0; i<3; ++i)
  {
#ifdef DEBUG
    std::cout << "Barycentric test : " << bcname[i] <<  " \n";
#endif // DEBUG
    index = Omega_h::simplex_opposite_template(2, 1, i);
    g::find_barycentric_tri_simple(M, M[index], bc);
    bool res = g::compare_array(bc.data(), bcc_mat[i].data(), 3);
#ifdef DEBUG
    g::print_array(bc.data(), 3, "BC_tri");
#endif // DEBUG
    if(!res)
    {
#ifdef DEBUG
      std::cout << "Failed \n";
#endif // DEBUG
       return 0;
    }

  }
  const Omega_h::Vector<3> xp1{0.0, -0.5, 0.0};
  const Omega_h::Vector<3> xp2{2.0, 0.5, 0.0};
  const Omega_h::Vector<3> xp3{0.0, 0.2, 0.0};

  const Omega_h::Few<Omega_h::Vector<3>, 3>& xpoints{xp1, xp2, xp3};
  for(int i=0; i<3; ++i)
  {
    g::find_barycentric_tri_simple(M, xpoints[i], bc);
#ifdef DEBUG
    g::print_array(bc.data(), 3, "BC_tri");
#endif // DEBUG
    if(bc[i] >= 0)
    {
#ifdef DEBUG
      std::cout << "Failed: BC coords expected -ve \n";
#endif // DEBUG
      return 0;
    }
  }
  return 1;
}

void test_line_tri_intx()
{
  Omega_h::Vector<3> xpoint{0, 0, 0};
  const Omega_h::Vector<3> a{0.0, 0.0, 0.0};
  const Omega_h::Vector<3> b{2.0, 0.0, 0.0};
  const Omega_h::Vector<3> c{1.0, 1.0, 0.0};
  const Omega_h::Vector<3> d{1.0, 0.0, 1.0};

  const Omega_h::Vector<3> orig{1.0, 0.5, 0.5};
  const Omega_h::Vector<3> dest{1.0, -0.2, 0.5};

  const Omega_h::Matrix<3, 4> M{a,b,c,d};

  Omega_h::Few<Omega_h::Vector<3>, 3> face; //{a,b,d};

  for(int i=0; i<4; ++i)
  {
    g::get_face_coords( M, i, face);
    Omega_h::LO edge = -1;
    bool res = g::line_triangle_intx_simple(face, orig, dest, xpoint, edge);
    if(res)
    {
#if DEBUG>0
      g::print_array(xpoint.data(), 3, "FoundXPT:");
      std::cout << "--------\n";
#endif // DEBUG
    }
  }
}

void print_mesh_stat(Omega_h::Mesh &m, bool coords=true)
{
  //std::cout <<"Mesh #coords " << m.coords().size()  << "\n"; //3*nverts
  std::cout << "Mesh #elements " << m.nelems()
            << " #faces " << m.nfaces()
            << " #edges " << m.nedges()
            << " #verts " << m.nverts()
            << "\n";

  if(coords)
  {
    const auto coords = m.coords();
    std::cout << "\n#Coords: size=" << coords.size() << " (3 x verts)\n\t";

    for(int i=0; i<coords.size() && i<20 ; ++i)
      std::cout << coords[i] << ", ";
    std::cout << "\n";

    const auto mesh2verts = m.ask_elem_verts();
    std::cout << "#ask_elem_verts: " << mesh2verts.size() << "\n"; //LOs
    for(int ielem=0; ielem<m.nelems() && ielem<2; ++ielem)
    {
      auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
      const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
      g::print_matrix(M);
      std::cout << "Vertex IDS :\n";
      for(int i=0; i<4; ++i)
      {
         std::cout << ttv2v[i] << ", ";
      }
      std::cout << "; \n";
    }
    std::cout << "\n\n";
  }

  const auto downs = m.ask_down(3, 2);
  std::cout << "#down(3,2): #ab2b " << downs.ab2b.size() << "\n";
  for(int i=0; i<downs.ab2b.size()-1 && i<10; ++i)
  {
     std::cout << downs.ab2b[i] << ", ";
   }
  std::cout << "; \n";


  std::cout << "ask_down(2,1): " <<  m.ask_down(2,1).ab2b.size();
  for(int i=0; i<20 && i<m.ask_down(2,1).ab2b.size(); ++i)
       std::cout << " " << m.ask_down(2,1).ab2b[i];
  std::cout << "\n";


  const auto ups = m.ask_up(2,3);
  std::cout << "#up(2,3): #a2ab " << ups.a2ab.size() //NOTE: size is +1 ???
            << " ; #ab2b " << ups.ab2b.size() << "\n";

  std::cout <<  "  ups:a2ab\n\t";
  for(int i=0; i<ups.a2ab.size()-1 && i<20; ++i)
  {
     std::cout << ups.a2ab[i] << ", ";
  }
  std::cout << ";\n ups:ab2b\n\t";
  for(int i=0; i<ups.ab2b.size()-1 && i<20; ++i)
  {
     std::cout << ups.ab2b[i] << ", ";
  }
   std::cout << "; \n";


  auto up12 = m.ask_up(1, 2);
  std::cout << "\nask_up(1, 2):a2ab " <<  up12.a2ab.size();
  for(int i=0; i<20 && i<up12.a2ab.size(); ++i)
     std::cout << " " << up12.a2ab[i];
  std::cout << "\n";
  std::cout << "\nask_up(1, 2):ab2b " <<  up12.ab2b.size();
  for(int i=0; i<20 && i<up12.ab2b.size(); ++i)
     std::cout << " " << up12.ab2b[i]; //12 :0 1 ;0 2; 1 2; 0 3; 1 3; 2 3
  std::cout << "\n Entries";
  for(int i=0; i<20 && i<up12.a2ab.size(); ++i)
     std::cout << " " << up12.ab2b[up12.a2ab[i]];
  std::cout << "\n";


  const auto dual = m.ask_dual();
  std::cout << "#dual: #a2ab " << dual.a2ab.size() //last is size=end_index+1 ?
            << " ; #ab2b " << dual.ab2b.size() ;
  std::cout << " ; dual.a2ab[0,1]:" << dual.a2ab[0] << "," << dual.a2ab[1] << "\n";

  //dual print
  std::cout <<  "  Duals:\n\t";
  for(int i=0; i<dual.a2ab.size()-1 && i<5; ++i)
  {
     std::cout << dual.a2ab[i] << ": ";
     for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
        std::cout << dual.ab2b[j] <<", ";
     std::cout << "; ";
  }

  if(dual.a2ab.size()>5)
  {
     std::cout <<  ".......";
     for(int i=dual.a2ab.size()-5; i<dual.a2ab.size()-1; ++i)
     {
       std::cout << dual.a2ab[i] << ": ";
       for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
          std::cout << dual.ab2b[j] <<", ";
       std::cout << "; ";
    }
  }
  std::cout <<  "\n";

  //exposed
  auto side_is_exposed = mark_exposed_sides(&m);
  std::cout << "#size exposed faces: "<< side_is_exposed.size() <<"\n";
  std::cout <<  "  Exposed:\n\t";
  for(int i=0; i<side_is_exposed.size() && i<10; ++i)
  {
     if(side_is_exposed[i])
       std::cout << 1 << ", ";
     else
       std::cout << 0 << ", ";
     //   std::cout << dual.ab2b[j] <<", ";
  }

  if(side_is_exposed.size()>10)
  {
     std::cout <<  ".......";
     for(int i=side_is_exposed.size()-10; i<side_is_exposed.size(); ++i)
     {
       if(side_is_exposed[i])
         std::cout << 1 << ", ";
       else
         std::cout << 0 << ", ";
     }
  }
  std::cout << "\n";

}

//TODO add checks
void test_unit(Omega_h::Library *lib)
{
    Omega_h::Mesh m(lib);
    Omega_h::build_from_elems2verts(&m, OMEGA_H_SIMPLEX, 3, Omega_h::LOs({0, 1, 2, 3}), 4);
    print_mesh_stat(m, false);

    auto up23 = m.ask_up(2, 3);  // // AdjPtr[][]
    std::cout <<  "up(2,3): faceID elemID\n";
    for(int i=0; i<up23.a2ab.size()-1 ; ++i) //NOTE: size is +1 ???
     {
       auto face = up23.a2ab[i];
       std::cout << face << " " << up23.ab2b[face] << " ; "; //0 0 ::  1 0 ::  2 0 ::  3 0
     }
    std::cout <<  "\n";

    auto up12 = m.ask_up(1, 2);
    std::cout << "\nask_up(1, 2):a2ab " <<  up12.a2ab.size();
    for(int i=0; i<20 && i<up12.a2ab.size(); ++i)
       std::cout << " " << up12.a2ab[i];
    std::cout << "\n";
    std::cout << "\nask_up(1, 2):ab2b " <<  up12.ab2b.size();
    for(int i=0; i<20 && i<up12.ab2b.size(); ++i)
       std::cout << " " << up12.ab2b[i]; //12 :0 1 ;0 2; 1 2; 0 3; 1 3; 2 3
    std::cout << "\n Entries";
    for(int i=0; i<20 && i<up12.a2ab.size(); ++i)
       std::cout << " " << up12.ab2b[up12.a2ab[i]];
    std::cout << "\n";


    OMEGA_H_CHECK(m.ask_down(3, 0).ab2b == Omega_h::LOs({0, 1, 2, 3})); //AdjPtr[][]
    auto down32 = m.ask_down(3, 2);  // 0 1 3 2
    auto down31 = m.ask_down(3, 1);  // 0 3 1 2 4 5
    auto down21 = m.ask_down(2, 1);  //1 3 0 ;0 4 2; 1 2 5; 3 5 4 (size=12)
    auto down20 = m.ask_down(2, 0);  //  0 2 1 0 1 3 2 0 3 1 2 3
    auto down10 = m.ask_down(1, 0); // 0 1 2 0 0 3 1 2 1 3 2 3

    std::cout <<  "Down(2,0).ab2b:\n";
    for(int i=0; i<down20.ab2b.size() && i<20; ++i)
    {
      std::cout << " " << down20.ab2b[i];
    }
    std::cout <<  "\nDown(2,1).ab2b:\n";
    for(int i=0; i<down21.ab2b.size() && i<20; ++i)
    {
      std::cout << " " << down21.ab2b[i];
    }

    std::cout <<  "\n----------------\n";
    //const auto coords = m.coords(); //no tag_base for coords
}

#endif
