#include "SellCSigma.h"

void sigmaSort(int num_elems, int* ptcls_per_elem, int sigma, std::pair<int, int>*& ptcl_pairs) {
  //Make temporary copy of the particle counts for sorting
  // Pair consists of <ptcl count, elem id>
  typedef std::pair<int,int> pair_t;
  ptcl_pairs = new std::pair<int,int>[num_elems];
  for (int i = 0; i < num_elems; ++i)
    ptcl_pairs[i] = std::make_pair(ptcls_per_elem[i], i);

  //Sort the entries with sigma sorting
  int i;
  if (sigma > 1) {
    for (i = 0; i < num_elems - sigma; i+=sigma) {
      std::sort(ptcl_pairs + i, ptcl_pairs + i + sigma, std::greater<pair_t>());
    }
    std::sort(ptcl_pairs + i, ptcl_pairs + num_elems, std::greater<pair_t>());
  }
}

