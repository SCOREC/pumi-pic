#include "SellCSigma.h"
#include <utility>
#include <functional>
#include <algorithm>
SellCSigma::SellCSigma(int c, int sig, int ne, int np, int* ptcls_per_elem,
                       std::vector<int>* ids) {
  C = c;
  sigma = sig;
  num_ents = ne;

  //Make temporary copy of the particle counts for sorting
  typedef std::pair<int,int> pair_t;
  pair_t* ptcls = new pair_t[ne];
  for (int i = 0; i < ne; ++i)
    ptcls[i] = std::make_pair(ptcls_per_elem[i], i);
  //Sort the entries with sigma sorting
  int i;
  for (i = 0; i < ne - sigma; i+=sigma) {
    std::sort(ptcls + i, ptcls + i + sigma, std::greater<pair_t>());
  }
  std::sort(ptcls + i, ptcls + ne, std::greater<pair_t>());

  //Create chunks and offsets
  
  //Fill the chunks
}

SellCSigma::~SellCSigma() {

}
