#include "SellCSigma.h"

SellCSigma::SellCSigma(int c, int sig, int ne, int np, int* ptcls_per_elem,
                       std::vector<int>* ids) {
  C = c;
  sigma = sig;
  num_ents = ne;
}

SellCSigma::~SellCSigma() {

}
