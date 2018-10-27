#ifndef PSTYPES_H_
#define PSTYPES_H_

#ifdef FP64
typedef double fp_t;
#endif
#ifdef FP32
typedef float fp_t;
#endif

class elemCoords {
  public:
  int num_elems;
  int verts_per_elem;
  fp_t* x;
  fp_t* y;
  fp_t* z;
  elemCoords(int ne, int np);
  ~elemCoords();
  private:
    elemCoords() {};
};

#endif
