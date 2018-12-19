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
#include "Omega_h_reduce.hpp"

#include "gitrm_utils.hpp"
#include "gitrm_adjacency.hpp"
#include "unit_tests.hpp"

namespace o = Omega_h;
namespace g = GITRm;

#define DEBUG 1
#define DO_TEST 0
int main(int argc, char** argv) {

  auto lib = Omega_h::Library(&argc, &argv);
  const auto world = lib.world();
  auto mesh = Omega_h::gmsh::read(argv[1], world);
  const auto dim = mesh.dim();

#if DO_TEST==1
  test_unit(&lib);
  print_mesh_stat(mesh);
  test_barycentric1();
  test_barycentric2();
  test_barycentric_tri();
  test_line_tri_intx();
#endif


  //adjacent element
  const auto dual = mesh.ask_dual();
  auto dual_faces = dual.ab2b;
  auto dual_elems = dual.a2ab;
  auto up = mesh.ask_up(dim - 1, dim);
  const auto down= mesh.ask_down(dim, dim - 1);
  const auto down20 = mesh.ask_down(2,0);
  const auto downs= down.ab2b;
  auto side_is_exposed = mark_exposed_sides(&mesh);
  Omega_h::Int nsides_per_elem = dim + 1;
  Omega_h::Int nelems = mesh.nelems();

  //get coordinate values
  const auto mesh2verts = mesh.ask_elem_verts();
  const auto coords = mesh.coords();
  const auto face_verts =  mesh.ask_verts_of(2);//LOs

  Omega_h::LO nptcl = 2;

  //Simple particle data
  Omega_h::Few< Omega_h::Vector<3>, 2> dest{{0.9,0.1,0.5}, {0.26,0.2,0.12}};
  Omega_h::Few< Omega_h::Vector<3>, 2> orig{{0.8,0.3,0.7}, {0.15,0.1,0.2}};
  Omega_h::Write<Omega_h::Real> bccs(4*nptcl, -1.0);

  Omega_h::Write<Omega_h::LO> part_flags(nptcl, 1); // found or not
  //To store adj_elem for subsequent searches. Once ptcl is found/crossed, this is reset.
  Omega_h::Write<Omega_h::LO> elemIds(nptcl); //next element to search for

  //Which particle belongs to which element ?
  elemIds[0] = 26; //173{0.15,0.1,0.2}; 26{0.8,0.3,0.7}//temporary
  elemIds[1] = 150; //temporary

nptcl=1;

// Particle ownership is not yet here.

/*
Process  all particles in an element (and all elements) in parallel, from a while loop.
All particles in an element start off with the element's closure data or state.
When elements are called in parallel, each element's lambda function splits the calls 
into that for particles. All the particles are running in parallel for a single step, asynchronously.
Adjacent element IDs are stored for further search of particles. 
Wait for the completion of the step. 
At this stage, particles are to be associated with these adjacent elements, but the particle data
are still with the source elements. 
Next step is a parallel reduction  per element, which  checks if all particles are found, or collision is detected. 
If any particle remains in any element, next parallel call starts processing all elements irrespective
of particles in it are all done or not.  But in the search kernel, only remaining particles are processed. 
Omega_h::Write data set are to be updated during the run. These will be replaced by 'particle_structures',
along with the for loop inside the lambda function distributing particles. 
Omega_h::parallel_reduce doesn't work. Kokkos functions to be used when Omega_h doesn't provide it.
*/
  //particle search: adjacency + boundary crossing
  auto search_ptcl = OMEGA_H_LAMBDA( Omega_h::LO ielem)
  {
    //temporary
    if(!(ielem == elemIds[0] || ielem == elemIds[1])) return; //Assume all other elements are empty

    std::cout << "----------\n elem " << ielem << " \n";

    const auto ttv2v = Omega_h::gather_verts<4>(mesh2verts, ielem);
    const auto M = Omega_h::gather_vectors<4, 3>(coords, ttv2v);

    //Replace by particle parallel_for loop for all the remaining particles in this element
    for(Omega_h::LO iptcl = 0; iptcl < nptcl; ++iptcl)
    {
      //temporary
      std::cout << "Elem " << ielem << " part:" << iptcl << "\n";
      if(elemIds[iptcl] != ielem) continue;

      Omega_h::Write<Omega_h::Real> bcc(4*nptcl, -1.0);


      //To be DELETED. check particle origin containment in this element
      std::cout << " flag " << iptcl << " " << part_flags.data()[iptcl] << "\n";
      const bool test_res = g::find_barycentric_tet(M, orig[iptcl], bcc);
      if(g::all_positive(bcc.data(), 4))
      {
        std::cout << "ORIGIN ********detected in " << ielem << " \n";
      }
      //#end DELETE

      const bool res = g::find_barycentric_tet(M, dest[iptcl], bcc);

      if(g::all_positive(bcc.data(), 4))
      {
        part_flags.data()[iptcl] = 0;
        elemIds[iptcl] = ielem;
        for(Omega_h::LO i=0; i<4; ++i) bccs[4*nptcl+i] = bcc[i];

        std::cout << "********found in " << ielem << " \n";
        g::print_matrix(M);
        g::print_data(M, dest[iptcl], bcc);
        OMEGA_H_CHECK(g::almost_equal(bcc[0] + bcc[1] + bcc[2] +bcc[3], 1.0));
      }
      else
      {
        Omega_h::LO most_neg = g::most_negative_index(bcc.data(), 4);

        //get element ID
        auto dface_ind = dual_elems[ielem];

        auto beg_face = ielem * nsides_per_elem; //face index
        auto end_face = beg_face + nsides_per_elem;
        Omega_h::LO fIndex = 0;

        for(auto iface = beg_face; iface < end_face; ++iface) //not 0..3
        {
          Omega_h::LO faceID = downs[iface];

          auto fv2v = Omega_h::gather_verts<3>(face_verts, faceID);
          const auto face = Omega_h::gather_vectors<3, 3>(coords, fv2v);

         //g::get_face_coords( M, fIndex, abc);
          g::check_face(M, face, fIndex);

          if(!side_is_exposed[faceID]) //if most_neg but not exposed, for almost ||l to surface ???
          {
             //OMEGA_H_CHECK(side2side_elems[side + 1] - side2side_elems[side] == 2);
             auto adj_elem  = dual_faces[dface_ind];
             std::cout << "el " << ielem << " adj " << adj_elem << "\n";
             if(fIndex == most_neg) // Use elemIds[iptcl] to skip last elem, if it is set per ptcl loop
             {
               elemIds[iptcl] = adj_elem;
               std::cout << "=====> For el|faceID=" << ielem << "," << downs[iface]  << " :ADJ elem= " << adj_elem << "\n";

               // Test if element is interior, if so break. For adj search crossing surface not needed.
               // If on bdry, there could be a crossing face not yet found.
               //if(element != on_boundary) break;

             }
             ++dface_ind;
          }
          else if(bcc[fIndex] < 0) //fIndex == most_neg) TODO test this
          {
            std::cout << "********* \n Call Wall collision for el,faceID " << ielem << "," << downs[iface] << "\n";
            Omega_h::Vector<3> xpoint;
            const bool res = g::line_triangle_intx(face, orig[iptcl], dest[iptcl],   xpoint);

            if(res)
            {
              //reset adjacent element to avoid wrong use.
              elemIds[iptcl] = -1;
              part_flags.data()[iptcl] = 0;

              g::print_matrix(M);
              g::print_data(M, dest[iptcl], bcc);
              g::print_osh_vector(xpoint, "COLLISION POINT");
            }
          }
          ++fIndex;
        }
      }
    }
  };


  auto search_reduce = OMEGA_H_LAMBDA(int ielem)
  {
    // elements[ielem].nptcl
    {
      //if(part_flags[i])
        return 1;
     }
  };

  bool found = false;
  while(!found)
  {
    Omega_h::parallel_for(mesh.nelems(),  search_ptcl, "search_ptcl");//mesh.nelems()
    found = true;

    // synchronize

   // int res = Omega_h::parallel_reduce<int>(mesh.nelems(), search_reduce, "search_reduce");

    //TODO replace with the above
    for(int i=0; i<nptcl; ++i){ if(part_flags[i]) {found = false; break;} }
  }

  Omega_h::vtk::write_parallel("cube", &mesh);

  return 0;
}
//LO a....
//a.get(i)



