#ifndef GITRM_UTILS_HPP
#define GITRM_UTILS_HPP

#include <iostream>
#include <cmath>
#include <utility>
#include <string>

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


//Problem: functions defined here are not available in other headers !
//many of them moved until it is fixed.


void print_data(const Omega_h::Matrix<3, 4> &M, const Omega_h::Vector<3> &dest,
     Omega_h::Write<Omega_h::Real> &bcc)
{
    //std::cout << "FOUND \n";
    //print_matrix(M);  //include file problem ?
    print_osh_vector(dest, "point");
    print_array(bcc.data(), 4, "BCoords");
    //Omega_h::Real dist = find_smallest_dist2face(M, dest, bcc);
    //std::cout << "Dist_to_closest_face " << dist <<  "\n";
}
} //namespace
#endif

