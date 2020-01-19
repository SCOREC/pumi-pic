#include "pumipic_utils.hpp"

namespace pumipic {
void print_array(const double* a, int n, std::string name) {
    if(name!=" ")
      std::cout << name << ": ";
    for(int i=0; i<n; ++i)
      std::cout << a[i] << ", ";
    std::cout <<"\n";
}

void print_osh_vector(const Omega_h::Vector<3> &v, std::string name, 
    bool line_break) {
  std::string str = line_break ? ")\n" : "); ";
  std::cout << name << ": (" << v.data()[0]  << " " << v.data()[1] << " " << v.data()[2] << str;
}

void print_data(const Omega_h::Matrix<3, 4> &M, const Omega_h::Vector<3> &dest,
                  Omega_h::Write<Omega_h::Real> &bcc) {
    print_matrix(M);  //include file problem ?
    print_osh_vector(dest, "point");
    print_array(bcc.data(), 4, "BCoords");
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

}
