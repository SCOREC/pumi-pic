#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>

class SellCSigma {
 public:
  SellCSigma(int c, int sigma, int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
  ~SellCSigma();


  //Keep Representation public for usage by kokkos
  //Horizontal Chunks
  int C;
  //Sorting Chunk
  int sigma;
  //Total entries
  int num_ents;
  //offsets stores an offset into each
  int* offsets;
  int* id_list;
 private:
  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
  
};

#endif
