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

#ifdef DEBUG
  printf("\nSigma Sorted Particle Counts\n");
  for (i = 0; i < ne; ++i)
    printf("Element %d: has %d particles\n", ptcls[i].second, ptcls[i].first);
#endif
  
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

#ifdef DEBUG
  printf("\nChunk Offsets\n");
  for (i = 0; i < num_chunks + 1; ++i)
    printf("Chunk %d starts at %d\n", i, offsets[i]);
#endif
  
  //Fill the chunks
  id_list = new int[offsets[num_chunks]];
  int index = 0;
  for (i = 0; i < num_chunks; ++i) {
    int width = (offsets[i + 1] - offsets[i]) / C;
    for (int j = 0; j < width; ++j) {
      for (int k = i * C; k < (i + 1) * C; ++k) {
        if (k < num_ents && ptcls[k].first > j) {
          int ent_id = ptcls[k].second;
          id_list[index++] = ids[ent_id][j];
        }
        else
          id_list[index++] = -1;
      }
    }
  }

#ifdef DEBUG
  printf("\nChunks\n");
  for (i = 0; i < num_chunks; ++i){
    printf("Chunk %d:", i);
    for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
      printf(" %d", id_list[j]);
      if (j % C == C - 1)
        printf(" |");
    }
    printf("\n");
  }
#endif

  delete [] ptcls;
}

SellCSigma::~SellCSigma() {
  delete [] offsets;
  delete [] id_list;
}
