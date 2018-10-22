#ifndef DISTRIBUTE_H_
#define DISTRIBUTE_H_

#include <vector>

bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem,
                          std::vector<int>* ids);

#endif
