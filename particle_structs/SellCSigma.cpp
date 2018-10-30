#include "SellCSigma.h"
#include "psAssert.h"
#include <utility>
#include <functional>
#include <algorithm>

SellCSigma::SellCSigma(int c, int sig, int v, int ne, int np, int* ptcls_per_elem,
                       std::vector<int>* ids, fp_t* xs, fp_t* ys, fp_t* zs, bool debug) {
  C = c;
  sigma = sig;
  V = v;
  num_elems = ne;

  //Make temporary copy of the particle counts for sorting
  // Pair consists of <ptcl count, elem id>
  typedef std::pair<int,int> pair_t;
  pair_t* ptcls = new pair_t[num_elems];
  for (int i = 0; i < num_elems; ++i)
    ptcls[i] = std::make_pair(ptcls_per_elem[i], i);

  //Sort the entries with sigma sorting
  int i;
  for (i = 0; i < num_elems - sigma; i+=sigma) {
    std::sort(ptcls + i, ptcls + i + sigma, std::greater<pair_t>());
  }
  std::sort(ptcls + i, ptcls + num_elems, std::greater<pair_t>());
  
  //Number of chunks without vertical slicing
  num_chunks = num_elems / C + (num_elems % C != 0);
  num_slices = 0;
  int* chunk_widths = new int[num_chunks];
  row_to_element = new int[num_elems];
  //Add chunks for vertical slicing
  for (i = 0; i < num_chunks; ++i) {
    chunk_widths[i] = 0;
    for (int j = i * C; j < (i + 1) * C && j < num_elems; ++j)  {
      row_to_element[j] = ptcls[j].second;
      chunk_widths[i] = std::max(chunk_widths[i],ptcls[j].first);
    }
    int num_vertical_slices = chunk_widths[i] / V + (chunk_widths[i] % V != 0);
    num_slices += num_vertical_slices;
  }

  if(debug) {
    printf("\nSigma Sorted Particle Counts\n");
    for (i = 0; i < num_elems; ++i)
      printf("Element %d: has %d particles\n", row_to_element[i], ptcls[i].first);
  }
  

  //Create offsets into each chunk/vertical slice
  offsets = new int[num_slices + 1];
  slice_to_chunk = new int[num_slices];
  offsets[0] = 0;
  int index = 1;
  for (i = 0; i < num_chunks; ++i) {
    for (int j = V; j <= chunk_widths[i]; j+=V) {
      slice_to_chunk[index-1] = i;
      offsets[index] = offsets[index-1] + V * C;
      ++index;
    }
    int rem = chunk_widths[i] % V;
    if (rem > 0) {
      slice_to_chunk[index-1] = i;
      offsets[index] = offsets[index-1] + rem * C;
      ++index;
    }
  }

  delete [] chunk_widths;

  if(debug) {
    ALWAYS_ASSERT(num_slices+1 == index);
    printf("\nSlice Offsets\n");
    for (i = 0; i < num_slices + 1; ++i)
      printf("Slice %d starts at %d\n", i, offsets[i]);
  }
  
  //Fill the chunks
  arr_to_scs = new int[np];
  particle_mask = new bool[offsets[num_slices]];
  scs_xs = new fp_t[offsets[num_slices]];
  scs_ys = new fp_t[offsets[num_slices]];
  scs_zs = new fp_t[offsets[num_slices]];
  index = 0;
  int start = 0;
  int old_elem = 0;
  for (i = 0; i < num_slices; ++i) { //for each slice
    int elem = slice_to_chunk[i];
    if (old_elem!=elem) {
      old_elem = elem;
      start = 0;
    }
    int width = (offsets[i + 1] - offsets[i]) / C;
    for (int j = 0; j < width; ++j) { //for the width of that slice
      for (int k = elem * C; k < (elem + 1) * C; ++k) { //for each row in the slice
        if (k < num_elems && ptcls[k].first > start + j) {
          int ent_id = ptcls[k].second;
          int ptcl = ids[ent_id][start + j];
	  scs_xs[index] = xs[ptcl];
	  scs_ys[index] = ys[ptcl];
	  scs_zs[index] = zs[ptcl];
          ALWAYS_ASSERT(ptcl<np);
          arr_to_scs[ptcl] = index;
          particle_mask[index++] = true;

        }
        else {
	  scs_xs[index] = 0;
	  scs_ys[index] = 0;
	  scs_zs[index] = 0;
          particle_mask[index++] = false;
	}
      }
    }
    start+=width;
  }

  scs_new_xs = new fp_t[offsets[num_slices]];
  scs_new_ys = new fp_t[offsets[num_slices]];
  scs_new_zs = new fp_t[offsets[num_slices]];

  if(debug) {
    printf("\narr_to_scs\n");
    for (i = 0; i < np; ++i)
      printf("array index %5d -> scs index %5d\n", i, arr_to_scs[i]);
    printf("\nSlices\n");
    for (i = 0; i < num_slices; ++i){
      printf("Slice %d:", i);
      for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        printf(" %d", particle_mask[j]);
        if (j % C == C - 1)
          printf(" |");
      }
      printf("\n");
    }

    printf("\nX Coordinates\n");
    for (i = 0; i < offsets[num_slices]; ++i) {
      printf("%.2f ", scs_xs[i]);
    }
    printf("\n");
  }

  delete [] ptcls;
}

SellCSigma::~SellCSigma() {
  delete [] scs_new_xs;
  delete [] scs_new_ys;
  delete [] scs_new_zs;
  delete [] scs_xs;
  delete [] scs_ys;
  delete [] scs_zs;
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  delete [] arr_to_scs;
}
