#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>

class SellCSigma {
 public:
  SellCSigma(int c, int sigma, int ne, int np, int* ptcls_per_elem, std::vector<int>* ids);
  ~SellCSigma();


  //Keep Representation public for usage by kokkos
  //Horizontal chunk size
  int C;
  //Sorting chunk size
  int sigma;
  //Number of chunks
  int num_chunks;
  //Total entries
  int num_ents;
  //offsets stores an offset into each chunk of entries in id_list
  int* offsets;
  //id_list lists the ids in chunks as per the sell-c-sigma structure
  //   -1 represents an empty value
  int* id_list;
 private:
  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
  
};

#endif
