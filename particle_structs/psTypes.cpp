#include "psTypes.h"

elemCoords::elemCoords(int ne, int np) {
  num_elems = ne;
  verts_per_elem = np;
  x = new fp_t[ne*np];
  y = new fp_t[ne*np];
  z = new fp_t[ne*np];
};

elemCoords::~elemCoords() {
  delete [] x;
  delete [] y;
  delete [] z;
}
