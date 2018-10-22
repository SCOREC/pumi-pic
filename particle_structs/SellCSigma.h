#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_

class SellCSigma {
 public:
  SellCSigma(int ne, int np, int* ptcls_per_elem);
  ~SellCSigma();


  //Keep Representation public for usage by kokkos
  
 private:
  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
  
};

#endif
