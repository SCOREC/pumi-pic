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

  printf("\nSigma Sorted Particle Counts\n");
  for (i = 0; i < ne; ++i)
    printf("Element %d: has %d particles\n", ptcls[i].second, ptcls[i].first);
  
  //Create chunks and offsets
  num_chunks = num_ents / C + (num_ents % C != 0);
  offsets = new int[num_chunks + 1];
  offsets[0] = 0;
  for (i = 0; i < num_chunks; ++i) {
    offsets[i + 1] = 0;
    for (int j = i * C; j < (i + 1) * C && j < num_ents; ++j) 
      offsets[i + 1] = std::max(offsets[i+1],ptcls[j].first);
    offsets[i + 1] *= C;
    offsets[i + 1] += offsets[i];
  }

  printf("\nChunk Offsets\n");
  for (i = 0; i < num_chunks + 1; ++i)
    printf("Chunk %d starts at %d\n", i, offsets[i]);

  
  //Fill the chunks
  delete ptcls;
}

SellCSigma::~SellCSigma() {
  delete [] offsets;
}
