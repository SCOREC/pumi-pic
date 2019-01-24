
#include <stdio.h>
#include <Kokkos_Core.hpp>

#include <MemberTypes.h>
#include <SellCSigma.h>
#include <Distribute.h>
#include <psAssert.h>
int main(int argc, char** argv) {

  Kokkos::initialize(argc,argv);
  typedef MemberTypes<int> Type1;
  typedef MemberTypes<int,double> Type2;
  typedef MemberTypes<int[3],double[2],char> Type3;
  
  printf("Type1: %lu\n",Type1::memsize);
  ALWAYS_ASSERT(Type1::memsize == sizeof(int));
  printf("Type2: %lu\n",Type2::memsize);
  ALWAYS_ASSERT(Type2::memsize == sizeof(int) + sizeof(double));
  printf("Type3: %lu\n",Type3::memsize);
  ALWAYS_ASSERT(Type3::memsize == 3*sizeof(int) + 2*sizeof(double) + sizeof(char));

  printf("Type3 start of doubles: %lu\n",Type3::sizeToIndex<1>());
  ALWAYS_ASSERT(Type3::sizeToIndex<1>() == 3*sizeof(int));


  int ne = 5;
  int np = 10;
  int* ptcls_per_elem = new int[ne];
  std::vector<int>* ids = new std::vector<int>[ne];
  distribute_particles(ne,np, 0, ptcls_per_elem, ids);
  Kokkos::TeamPolicy<Kokkos::OpenMP> po(128, 4);
  int ts = po.team_size();
  SellCSigma<Type2, Kokkos::OpenMP>* scs = new SellCSigma<Type2, Kokkos::OpenMP>(po, 1, 10000, ne, np, ptcls_per_elem, ids, 1);

  int* scs_first = scs->getSCS<0>();
  double* scs_second = scs->getSCS<1>();

  //Loop over slices
  for (int i=0; i < scs->num_slices; ++i) {
    int start_id = scs->offsets[i];
    int end_id = scs->offsets[i+1];
    //loop through slice horizontally
    for (int j = start_id; j< end_id; j+=ts) {
      //Loop through slice vertically (C)
      for (int k=0; k < ts; ++k) {
        if (scs->particle_mask[j+k]) {
          scs_first[j+k] = scs->slice_to_chunk[i];
          scs_second[j+k] = 1.0;
        }
        else {
          scs_first[j+k] = -1;
          scs_second[j+k] = 0;
        }
      }
    }
  }

  delete scs;
  delete [] ptcls_per_elem;
  delete [] ids;

  Kokkos::finalize();
  printf("All tests passed\n");
  return 0;
}
