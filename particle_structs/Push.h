#ifndef PUSH_H_
#define PUSH_H_
#include "SellCSigma.h"

void push_array(int np, double* xs, double* ys, double* zs, double distance, double dx,
                double dy, double dz, double* new_xs, double* new_ys, double* new_zs);

void push_scs(SellCSigma* scs, double* xs, double* ys, double* zs, double distance, double dx,
              double dy, double dz, double* new_xs, double* new_ys, double* new_zs);

#endif
