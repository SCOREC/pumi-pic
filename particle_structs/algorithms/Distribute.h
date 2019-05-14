#ifndef DISTRIBUTE_H_
#define DISTRIBUTE_H_

#include <vector>

namespace particle_structs {

bool distribute_elements(int ne, int strat, int comm_rank, int comm_size, int* gids);

bool distribute_particles(int ne, int np, int strat, int* ptcls_per_elem,
                          std::vector<int>* ids);

const char* distribute_name(int strat);

}

#endif
