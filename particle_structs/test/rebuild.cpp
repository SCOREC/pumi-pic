#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>

#include <psAssert.h>
#include <Distribute.h>

using particle_structs::SellCSigma;
using particle_structs::MemberTypes;
using particle_structs::distribute_particles;
using particle_structs::distribute_elements;

typedef MemberTypes<int> Type;

bool shuffleParticlesTests();
bool resortElementsTest(int comm_rank, int comm_size);

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  bool passed = true;
  if (comm_rank == 0) {
    if (!shuffleParticlesTests()) {
      passed = false;
      printf("[ERROR] shuffleParticlesTests() failed\n");
    }
  }
  if (!resortElementsTest(comm_rank, comm_size)) {
    passed = false;
    printf("[ERROR] resortElementsTest() failed\n");
  }

  Kokkos::finalize();
  MPI_Finalize();
  if (passed)
    printf("All tests passed\n");
  return 0;
}


bool shuffleParticlesTests() {
  int ne = 5;
  int np = 20;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  typedef Kokkos::DefaultExecutionSpace exe_space;
  Kokkos::TeamPolicy<exe_space> po(128, 4);
  int ts = po.team_size();
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, 5, 2, ne, np, ptcls_per_elem, ids, NULL);
  SellCSigma<Type, exe_space>* scs2 =
    new SellCSigma<Type, exe_space>(po, 5, 2, ne, np, ptcls_per_elem, ids, NULL);
  delete [] ptcls_per_elem;
  delete [] ids;

  scs->printFormat();

  int* new_element = new int[scs->size()];
  int* values = scs->getSCS<0>();
  int* values2 = scs2->getSCS<0>();
  for (int i =0; i < scs->num_slices; ++i) {
    for (int j = scs->offsets[i]; j < scs->offsets[i+1]; j+= ts) {
      for (int k = 0; k < ts; ++k) {
	if (scs->particle_mask[j+k]) {
	  new_element[j + k] = scs->row_to_element[scs->slice_to_chunk[i] * scs->C + k];
	  values[j + k] = j + k;
	  values2[j + k] = j + k;
	}
      }
    }
  }

  //Rebuild with no changes  
  scs->rebuildSCS(new_element);

  values = scs->getSCS<0>();

  for (int i =0; i < scs2->num_slices; ++i) {
    for (int j = scs2->offsets[i]; j < scs2->offsets[i+1]; j+= ts) {
      for (int k = 0; k < ts; ++k) {
	if (scs2->particle_mask[j+k]) {
	  if (values[j + k] != values2[j + k]) {
	    printf("Value mismatch at particle %d: %d != %d\n", j + k, values[j+k], values2[j+k]);
            return false;
          }
	  new_element[j + k] = scs->row_to_element[ne - (scs->slice_to_chunk[i] * scs->C + k) - 1];
	}
      }
    }
  }

  scs->rebuildSCS(new_element);

  values = scs->getSCS<0>();

  //This should be backwards
  for (int i =0; i < scs->num_slices; ++i) {
    for (int j = scs->offsets[i]; j < scs->offsets[i+1]; j+= ts) {
      for (int k = 0; k < ts; ++k) {
	if (scs->particle_mask[j+k]) {
	  printf("Particle %d has value %d\n", j + k, values[j + k]);
	}
      }
    }
  }

  scs->printFormat();

  delete [] new_element;
  delete scs;
  delete scs2;
  return true;
}


bool resortElementsTest(int comm_rank, int comm_size) {
  int ne = 5;
  int np = 20;
  int* gids = new int[ne];
  distribute_elements(ne, 0, comm_rank, comm_size, gids);
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne, np, 0, ptcls_per_elem, ids);
  typedef Kokkos::DefaultExecutionSpace exe_space;
  Kokkos::TeamPolicy<exe_space> po(128, 4);
  int ts = po.team_size();
  SellCSigma<Type, exe_space>* scs =
    new SellCSigma<Type, exe_space>(po, 5, 2, ne, np, ptcls_per_elem, ids, gids);
  delete [] ptcls_per_elem;
  delete [] ids;
  delete [] gids;

  scs->printFormat();

  int* values = scs->getSCS<0>();

  //Remove all particles from first element
  int* new_element = new int[scs->size()];
  for (int i =0; i < scs->num_slices; ++i) {
    for (int j = scs->offsets[i]; j < scs->offsets[i+1]; j+= ts) {
      for (int k = 0; k < ts; ++k) {
	if (scs->particle_mask[j+k]) {
          values[j+k] = scs->row_to_element[scs->slice_to_chunk[i] * scs->C + k];
          if (scs->slice_to_chunk[i] == 0 && (j+k - scs->offsets[i]) % scs->C == 0)
            new_element[j + k] = -1;
          else
            new_element[j + k] = scs->row_to_element[scs->slice_to_chunk[i] * scs->C + k];
        }
      }
    }
  }
  scs->rebuildSCS(new_element);

  scs->printFormat();

  values = scs->getSCS<0>();
  for (int i =0; i < scs->num_slices; ++i) {
    for (int j = scs->offsets[i]; j < scs->offsets[i+1]; j+= ts) {
      for (int k = 0; k < ts; ++k) {
	if (scs->particle_mask[j+k]) {
          int should_be = scs->row_to_element[scs->slice_to_chunk[i] * scs->C + k];
          if (values[j+k] != should_be) {
            printf("Value mismatch at particle %d: %d != %d\n", j + k, values[j+k], should_be);
            return false;

          }
        }
      }
    }
  }
  delete scs;
  delete [] new_element;
  return true;
}
