#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include "psTypes.h"
class SellCSigma {
 public:
  SellCSigma(int c, int sigma, int v, int ne, int np, int* ptcls_per_elem, 
	     std::vector<int>* ids, fp_t* xs, fp_t* ys, fp_t* zs, bool debug=false);
  ~SellCSigma();


  //Keep Representation public for usage by kokkos
  //Horizontal chunk size
  int C;
  //Vertical slice size
  int V;
  //Sorting chunk size
  int sigma;
  //Number of chunks
  int num_chunks;
  //Number of slices
  int num_slices;
  //Total entries
  int num_elems;
  //offsets stores an offset into each chunk of entries in id_list
  int* offsets;
  //chunk_element stores the id of the first row in the chunk
  //  This only matters for vertical slicing so that each slice can determine which row
  //  it is a part of.
  int* slice_to_chunk;
  //particle_mask true means there is a particle at this location, false otherwise
  bool* particle_mask;

  //Remains for compiling purposes
  int* id_list;
  //Particle coordinates reordered and padded to match the SCS structure
  fp_t* scs_xs;
  fp_t* scs_ys;
  fp_t* scs_zs;
  //New particle coordinate locations that are padded to match the SCS structure
  fp_t* scs_new_xs;
  fp_t* scs_new_ys;
  fp_t* scs_new_zs;

  //map from row to element
  // row = slice_to_chunk[slice] + row_in_chunk
  int* row_to_element;

  //map from array particles to scs particles
  int* arr_to_scs;

 private:
  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
  
};

#endif
