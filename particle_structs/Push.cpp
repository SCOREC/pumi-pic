#include "Push.h"

void push_array(int np, double* xs, double* ys, double* zs, double distance, double dx,
                double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  for (int i = 0; i < np; ++i) {
    new_xs[i] = xs[i] + distance * dx;
    new_ys[i] = ys[i] + distance * dy;
    new_zs[i] = zs[i] + distance * dz;
  }
}

void push_scs(SellCSigma* scs, double* xs, double* ys, double* zs, double distance, double dx,
              double dy, double dz, double* new_xs, double* new_ys, double* new_zs) {
  for (int i = 0; i < scs->num_chunks; ++i) {
    int index = scs->offsets[i];
    while (index != scs->offsets[i + 1]) {
      for (int j = 0; j < scs->C; ++j) {
        if (scs->id_list[index] != -1) {
          int id = scs->id_list[index];
          new_xs[id] = xs[id] + distance * dx;
          new_ys[id] = ys[id] + distance * dy;
          new_zs[id] = zs[id] + distance * dz;
        }
        ++index;
      }
    }
  }
}
