#include <string>

#include "unit_tests.hpp" //=> cpp
#include "pumipic_adjacency.hpp"

namespace o = Omega_h;

//Barycentric coords associate to the opposite vertex of any face.

#define DO_TEST 0
int main(int argc, char** argv) {

  if(!(argc==2 || argc==4))
  {
    std::cout << "Usage: ./barycentric tet_points [point bcoods] \n If no point given, then all vertices are used in turn \n"
              << "Example: ./barycentric  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5  \n"
              << "Example: ./barycentric  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0:0.5,1.0,0.5  0.5,0.6,0  0,0.3,0.3,0.4 \n"
              << "Example: ./barycentric test1\n"
              << "Example: ./barycentric test2\n"
              << "Example: ./barycentric test3\n";
    exit(1);
  }

  if(std::string(argv[1]) =="test1")
  {
    if(test_barycentric1()) return 0;
    else 
      return 1;
  }
  else if(std::string(argv[1]) == "test2")
  {
    if(test_barycentric2()) return 0;
    else 
      return 1;
  }  
  else if(std::string(argv[1]) == "test3")
  {
    if(test_barycentric_tri()) return 0;
    else 
      return 1;
  }

  Omega_h::Matrix<3, 4> tet;
  Omega_h::Vector<3> point;
  Omega_h::Write<Omega_h::Real> bcc(4, -1.0);
  std::string s;

  std::stringstream ss1(argv[1]);
  std::vector<std::string> vtxs;
  std::string stemp;

  while(getline(ss1, stemp, ':'))
    vtxs.push_back(stemp);

  int i=0, j=0;
  for(auto st: vtxs)
  {
    std::stringstream sts(st);
    while(sts.good())
    {
      getline(sts, s, ',');
      tet[i][j++] = atof(s.c_str());
    }
    ++i;
    j=0;
  }


  if(argc==4)
  {
    std::stringstream ss2(argv[2]);
    i = 0;
    while(ss2.good())
    {
      getline(ss2, s, ',');
      point[i++] = atof(s.c_str());
    }
    std::stringstream ss3(argv[3]);
    i = 0;
    while(ss3.good())
    {
      getline(ss3, s, ',');
      bcc[i++] = atof(s.c_str());
    }
  }

  if(argc==2)
  {
    const Omega_h::Vector<4> bcc0{1.0, 0.0, 0.0, 0.0};
    const Omega_h::Vector<4> bcc1{0.0, 1.0, 0.0, 0.0};
    const Omega_h::Vector<4> bcc2{0.0, 0.0, 1.0, 0.0};
    const Omega_h::Vector<4> bcc3{0.0, 0.0, 0.0, 1.0};
    const Omega_h::Matrix<4, 4> bcc_mat{bcc0, bcc1, bcc2, bcc3};
    std::string bcname[] = {"u", "v", "w", "x"};
    int index = -1;
    for(int i=0; i<4; ++i)
    {
  #ifdef DEBUG
      std::cout << "Barycentric test : " << bcname[i] <<  " \n";
      pumipic::print_matrix(tet);
  #endif // DEBUG
      index = Omega_h::simplex_opposite_template(3, 2, i);
      bool res = test_barycentric_tet(tet, tet[index], bcc_mat[i].data());
      if(!res)
      {
  #ifdef DEBUG
        std::cout << "Failed \n";
  #endif // DEBUG
         return 1;
      }
    }
  }
  else
  {
  #ifdef DEBUG
      std::cout << "Barycentric test " <<  " \n";
  #endif // DEBUG
      bool res = test_barycentric_tet(tet, point, bcc.data());
      if(res)
      {
  #ifdef DEBUG
        std::cout << "Passed \n";
  #endif // DEBUG
         return 0;
      }
      else
      {
  #ifdef DEBUG
        std::cout << "Failed \n";
  #endif // DEBUG
         return 1;
      }
  }

  return 0;
}
