#ifndef PUMIPIC_UNIT_TESTS_HPP
#define PUMIPIC_UNIT_TESTS_HPP

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

#include "pumipic_adjacency.hpp"

namespace p = pumipic;

void print_array(const double* a, int n, std::string name) {
    if(name!=" ")
      std::cout << name << ": ";
    for(int i=0; i<n; ++i)
      std::cout << a[i] << ", ";
    std::cout <<"\n";
}

template< o::LO N>
void print_osh_vector(const Omega_h::Vector<N> &v, const char* name) {
  if(N==3)
    printf("%s %g %g %g\n",  name, v[0], v[1], v[2]);
  else if(N==4)
    printf("%s %g %g %g %g\n",  name, v[0], v[1], v[2], v[3]);
  else if(N>4) {
    printf("%s ", name);
    for(o::LO i=0; i<N; ++i)
      printf("%g ", v[i]); 
    printf("\n");
  }
}

//static in separate file if not in class
//remove M
bool test_barycentric_tet(const Omega_h::Matrix<3, 4> &M,
    const o::Vector<3> &p, const double *v, int pos=-1, 
    bool intent=false, bool debug=false) {
  o::Write<o::LO>res(1,0);
  o::Write<o::LO>bcc_w(4,0);
  o::Real x = p[0];
  o::Real y = p[1];
  o::Real z = p[2];
  o::parallel_for(1, OMEGA_H_LAMBDA(int i) {
    o::Vector<3> pd;
    pd[0] = x;
    pd[1] = y;
    pd[2] = z;
    Omega_h::Vector<4> bcc_v;
    p::find_barycentric_tet(M, pd, bcc_v);
    for(int i=0; i<4; ++i)
      bcc_w[i] = bcc_v[i];
  });

  //Access p in host ??
  o::HostRead<o::LO> bcc(bcc_w);
    if(pos == -1) {
      for(int i=0; i<4; ++i) {
        if(std::abs(bcc[i]- v[i]) > 1e-10) {
          if(debug)
            printf("Barycentric test failed: p=(%0.6f, %0.6f, %0.6f);"
              " calculated_bc=(%d, %d, %d, %d): bc=%d != v=%.6f \n",
              p[0], p[1], p[2], bcc[0], bcc[1], bcc[2], bcc[3], bcc[i], v[i]);
          return false;
        }
      }
    } else if (o::are_close(bcc[pos], v[0])) {
      if(debug)
        std::cout << "Barycentric test passed for (" << p[0] << ","
          << p[1] << "," << p[2]<< ") => " << v[0] << "==" << bcc[pos] << "\n";
    }else if(intent) {
      if(debug)
        std::cout << "Barycentric test intented to fail (" << p[0] << ","
          << p[1] << "," << p[2]<< ") => " << v[0]  << "!=" << bcc[pos] << " success :)\n";
    } else {
      std::cout << "Barycentric test failed : " << v[0] << " != " <<  bcc[pos] << "\n";
      return false;
    }
  return true;// TODO check this  result[0];
}


bool test_barycentric1() {
    std::cout <<"In_test_barycentric1 \n";
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
/*
  const Omega_h::Vector<3> p1{0.0,0.0,0.0}, p2{2.0, 0.0, 0.0}, p3{1.0,1.0,0.0};
  const Omega_h::Matrix<3, 3> M{p1, p2, p3};
  Omega_h::Vector<3> bc{-1, -1, -1};

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
    p::find_barycentric_tri_simple(M, M[index], bc);
    bool res = p::compare_array(bc.data(), bcc_mat[i].data(), 3);
#ifdef DEBUG
    p::print_array(bc.data(), 3, "BC_tri");
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
    p::find_barycentric_tri_simple(M, xpoints[i], bc);
#ifdef DEBUG
    p::print_array(bc.data(), 3, "BC_tri");
#endif // DEBUG
    if(bc[i] >= 0)
    {
#ifdef DEBUG
      std::cout << "Failed: BC coords expected -ve \n";
#endif // DEBUG
      return 0;
    }
  }
*/
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
    /*
    p::get_face_from_face_index_of_tet( M, i, face);
    Omega_h::Real dp = 0;
    bool res = p::line_triangle_intx_simple(face, orig, dest, xpoint, dp);
   */
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

  if(coords)  {
    const auto coords = m.coords();
    std::cout << "\n#Coords: size=" << coords.size() << " (3 x verts)\n\t";    
    auto printCoords = OMEGA_H_LAMBDA(o::LO i) {
      if( i<20 )
        printf("coords[%2d] %.4f\n", i, coords[i]);
    };
    o::parallel_for(coords.size(), printCoords, "printCoords");
    const auto mesh2verts = m.ask_elem_verts();
    std::cout << "#ask_elem_verts: " << mesh2verts.size() << "\n"; //LOs
    auto printElems = OMEGA_H_LAMBDA(o::LO ielem) {
      if(ielem<2) {
        auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
        const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);
        for(o::LO i=0; i<4; ++i)
          printf("M%d %.4f, %.4f, %.4f\n", i, M[i][0], M[i][1], M[i][2]);
        printf("Vertex IDS :");
        for(int i=0; i<4; ++i)
        {
           printf("%4d, ", ttv2v[i]);
        }
        printf(";\n");
      }
    };
    std::cout << "\n\n";
    o::parallel_for(m.nelems(), printElems, "printElems");
  }

  const auto downs = m.ask_down(3, 2);
  std::cout << "#down(3,2): #ab2b " << downs.ab2b.size() << "\n";
  o::parallel_for(downs.ab2b.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i < 10 )
      printf("%5d %5d\n", i, downs.ab2b[i]);
  });

  std::cout << "ask_down(2,1): " <<  m.ask_down(2,1).ab2b.size();
  const auto downs21 = m.ask_down(2, 1);
  o::parallel_for(downs21.ab2b.size(), OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, downs21.ab2b[i]);
  });
  std::cout << "\n";


  const auto ups = m.ask_up(2,3);
  std::cout << "#up(2,3): #a2ab " << ups.a2ab.size() //NOTE: size is +1 ???
            << " ; #ab2b " << ups.ab2b.size() << "\n";

  std::cout <<  "  ups:a2ab\n\t";
  o::parallel_for(ups.a2ab.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, ups.a2ab[i]);
  });
  std::cout << ";\n ups:ab2b\n\t";
  o::parallel_for(ups.ab2b.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, ups.ab2b[i]);
  });
  std::cout << "; \n";


  auto up12 = m.ask_up(1, 2);
  std::cout << "\nask_up(1, 2):a2ab " <<  up12.a2ab.size();
  o::parallel_for(up12.a2ab.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, up12.a2ab[i]);
  });
  std::cout << "\n";
  std::cout << "\nask_up(1, 2):ab2b " <<  up12.ab2b.size();
  o::parallel_for(up12.ab2b.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, up12.ab2b[i]);
  });
  std::cout << "\n Entries";
  o::parallel_for(up12.a2ab.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<20 )
      printf("%5d %5d\n", i, up12.ab2b[up12.a2ab[i]]);
  });
  std::cout << "\n";


  const auto dual = m.ask_dual();
  std::cout << "#dual: #a2ab " << dual.a2ab.size() //last is size=end_index+1 ?
            << " ; #ab2b " << dual.ab2b.size() ;

  //dual print
  std::cout <<  "  Duals:\n\t";
  o::parallel_for(dual.a2ab.size()-1, OMEGA_H_LAMBDA(o::LO i) {
    if( i<5 ) {
      for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
        printf("%5d %5d %5d\n", i, dual.a2ab[i], dual.ab2b[j]);
    }
  });

  if(dual.a2ab.size()>5)
  {
    std::cout <<  ".......";
    o::parallel_for(dual.a2ab.size()-1, OMEGA_H_LAMBDA(o::LO i) {
      for(int j=dual.a2ab[i]; j<dual.a2ab[i+1] ; ++j)
        printf("%5d %5d %5d\n", i, dual.a2ab[i], dual.ab2b[j]);
    });
  }
  std::cout <<  "\n";

  //exposed
  auto side_is_exposed = mark_exposed_sides(&m);
  std::cout << "#size exposed faces: "<< side_is_exposed.size() <<"\n";
  std::cout <<  "  Exposed:\n\t";
  o::parallel_for(side_is_exposed.size(), OMEGA_H_LAMBDA(o::LO i) {
    if( i < 10 ) {
      printf("%5d %2d\n", i, side_is_exposed[i]>0);
    }
  });

  if(side_is_exposed.size()>10)
  {
    std::cout <<  "\n.......";
    o::parallel_for(side_is_exposed.size(), OMEGA_H_LAMBDA(o::LO i) {
      printf("%5d %2d\n", i, side_is_exposed[i]>0);
    });
  }

}

#endif
