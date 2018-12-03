#ifndef PSTYPES_H_
#define PSTYPES_H_

#include <MemberTypes.h>

#ifdef FP64
typedef double fp_t;
#endif
#ifdef FP32
typedef float fp_t;
#endif

//Particle = <current position vector, pushed position vector>
typedef MemberTypes<fp_t[3], fp_t[3]> Particle;

class elemCoords {
  public:
  int num_elems;
  int verts_per_elem;
  int size;
  fp_t* x;
  fp_t* y;
  fp_t* z;
  elemCoords(int ne, int np, int size);
  ~elemCoords();
  private:
    elemCoords() {};
};

#endif
