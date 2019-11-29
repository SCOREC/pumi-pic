
#include "unit_tests.hpp" //=> cpp
#include "pumipic_adjacency.hpp"

int main(int argc, char** argv) {

  if(!(argc==1||argc==4))
  {
    std::cout << "Usage: ./line_tri_intx [tri_points lstart lend] \n If no arguments, then test hard coded values \n"
              << "Example: ./line_tri_intx  \n"
              << "Example: ./line_tri_intx  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0  0.5,0.6,-2  0.5,0.6,2 \n";
    exit(1);
  }

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();


  if(argc==1)
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

    g::get_face_from_face_index_of_tet( M, 1, face);
    Omega_h::Real dp;
    bool res = g::line_triangle_intx_simple(face, orig, dest, xpoint, dp);
    if(res)
    {
  #if DEBUG>0
      g::print_array(xpoint.data(), 3, "FoundXPT:");
      std::cout << "--------\n";
  #endif // DEBUG
    }
    else
      return 1;
    return 0;
  }

  Omega_h::Matrix<3, 3> tri;
  Omega_h::Vector<3> orig;
  Omega_h::Vector<3> dest;

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
      tri[i][j++] = atof(s.c_str());
    }
    ++i;
    j=0;
  }

  std::stringstream ss2(argv[2]);
  i = 0;
  while(ss2.good())
  {
    getline(ss2, s, ',');
    orig[i++] = atof(s.c_str());
  }
  std::stringstream ss3(argv[3]);
  i = 0;
  while(ss3.good())
  {
    getline(ss3, s, ',');
    dest[i++] = atof(s.c_str());
  }

  Omega_h::Vector<3> xpoint;
  Omega_h::Real dp=0;
  bool res = pumipic::line_triangle_intx_simple(tri, orig, dest, xpoint, dp);

#if DEBUG>0
  if(res)
    pumipic::print_array(xpoint.data(), 3, "FoundXPT:");
  else
    std::cout << "No Intersection\n";
#endif // DEBUG

  return !res;
}
