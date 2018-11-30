#ifndef GITRM_UTILS_HPP
#define GITRM_UTILS_HPP

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

namespace GITRm{

void print_matrix(const Omega_h::Matrix<3, 4> &M)
{
  std::cout << "M0  " << M[0].data()[0] << " " << M[0].data()[1] << " " << M[0].data()[2] <<"\n";     
  std::cout << "M1  " << M[1].data()[0] << " " << M[1].data()[1] << " " << M[1].data()[2] <<"\n";     
  std::cout << "M2  " << M[2].data()[0] << " " << M[2].data()[1] << " " << M[2].data()[2] <<"\n";     
  std::cout << "M3  " << M[3].data()[0] << " " << M[3].data()[1] << " " << M[3].data()[2] <<"\n";   
}

} //namespace
#endif

