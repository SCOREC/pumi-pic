#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include "psAssert.h"
#include "MemberTypes.h"

template<class DataTypes, int C /*=Impl::PerformanceTraits<
                                  typename MemorySpace::kokkos_execution_space>::vector_length*/>
class SellCSigma {
 public:
  SellCSigma(int sigma, int vertical_chunk_size, int num_elements, int num_particles,
             int* particles_per_element, std::vector<int>* particle_id_bins,
             bool debug=false);
  ~SellCSigma();

  template <std::size_t N>
  typename MemberTypeAtIndex<N,DataTypes>::type* getSCS();

  //Number of Data types
  static constexpr std::size_t num_types = DataTypes::size;

  //Vertical slice size
  int V;
  //Sorting chunk size
  int sigma;
  //Number of chunks
  int num_chunks;
  //Number of slices
  int num_slices;
  //Total entries
  int num_elems;
  //chunk_element stores the id of the first row in the chunk
  //  This only matters for vertical slicing so that each slice can determine which row
  //  it is a part of.
  int* slice_to_chunk;
  //particle_mask true means there is a particle at this location, false otherwise
  bool* particle_mask;

  //offsets into the scs structure
  int* offsets;

  //map from row to element
  // row = slice_to_chunk[slice] + row_in_chunk
  int* row_to_element;

  //map from array particles to scs particles
  int* arr_to_scs;

 private: 
  //Pointers to the start of each SCS for each data type
  void* scs_data[num_types];

 private:
  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
};

//Implementation to construct SCS arrays of different types
template <typename... Types>
struct CreateSCSArraysImpl;

template <>
struct CreateSCSArraysImpl<> {
  CreateSCSArraysImpl(void* scs_data[], int size) {}
};

template <typename T, typename... Types>
struct CreateSCSArraysImpl<T,Types...> {
  CreateSCSArraysImpl(void* scs_data[], int size) {
    scs_data[0] = new T[size];
    CreateSCSArraysImpl<Types...>(scs_data+1,size);
  }
};

//Call to construct SCS arrays of different types
template <typename... Types>
struct CreateSCSArrays;

template <typename... Types>
struct CreateSCSArrays<MemberTypes<Types...> > {
  CreateSCSArrays(void* scs_data[], int size) {
    CreateSCSArraysImpl<Types...>(scs_data,size);
  }
};

//Implementation to construct SCS arrays of different types
template <typename... Types>
struct DestroySCSArraysImpl;

template <>
struct DestroySCSArraysImpl<> {
  DestroySCSArraysImpl(void* scs_data[], int) {}
};

template <typename T, typename... Types>
struct DestroySCSArraysImpl<T,Types...> {
  DestroySCSArraysImpl(void* scs_data[], int x) {
    delete [] (T*)(scs_data[0]);
    DestroySCSArraysImpl<Types...>(scs_data+1, x);
  }
};


//Call to construct SCS arrays of different types
template <typename... Types>
struct DestroySCSArrays;

template <typename... Types>
struct DestroySCSArrays<MemberTypes<Types...> > {
  DestroySCSArrays(void* scs_data[], int x) {
    DestroySCSArraysImpl<Types...>(scs_data,x);
  }
};


template<class DataTypes, int C>
SellCSigma<DataTypes,C>::SellCSigma(int sig, int v, int ne, int np,
                                    int* ptcls_per_elem, std::vector<int>* ids,
                                    bool debug) {
  sigma = sig;
  V = v;
  num_elems = ne;

  //Make temporary copy of the particle counts for sorting
  // Pair consists of <ptcl count, elem id>
  typedef std::pair<int,int> pair_t;
  pair_t* ptcls = new pair_t[num_elems];
  for (int i = 0; i < num_elems; ++i)
    ptcls[i] = std::make_pair(ptcls_per_elem[i], i);

  //Sort the entries with sigma sorting
  int i;
  for (i = 0; i < num_elems - sigma; i+=sigma) {
    std::sort(ptcls + i, ptcls + i + sigma, std::greater<pair_t>());
  }
  std::sort(ptcls + i, ptcls + num_elems, std::greater<pair_t>());
  
  //Number of chunks without vertical slicing
  num_chunks = num_elems / C + (num_elems % C != 0);
  num_slices = 0;
  int* chunk_widths = new int[num_chunks];
  row_to_element = new int[num_chunks * C];

  //Add chunks for vertical slicing
  for (i = 0; i < num_chunks; ++i) {
    chunk_widths[i] = 0;
    for (int j = i * C; j < (i + 1) * C && j < num_elems; ++j)  {
      row_to_element[j] = ptcls[j].second;
      chunk_widths[i] = std::max(chunk_widths[i],ptcls[j].first);
    }
    int num_vertical_slices = chunk_widths[i] / V + (chunk_widths[i] % V != 0);
    num_slices += num_vertical_slices;
  }

  //Set the padded row_to_element values
  for (i = num_elems; i < num_chunks * C; ++i) {
    row_to_element[i] = i;
  }

  if(debug) {
    printf("\nSigma Sorted Particle Counts\n");
    for (i = 0; i < num_elems; ++i)
      printf("Element %d: has %d particles\n", row_to_element[i], ptcls[i].first);
  }
  

  //Create offsets into each chunk/vertical slice
  offsets = new int[num_slices + 1];
  slice_to_chunk = new int[num_slices];
  offsets[0] = 0;
  int index = 1;
  for (i = 0; i < num_chunks; ++i) {
    for (int j = V; j <= chunk_widths[i]; j+=V) {
      slice_to_chunk[index-1] = i;
      offsets[index] = offsets[index-1] + V * C;
      ++index;
    }
    int rem = chunk_widths[i] % V;
    if (rem > 0) {
      slice_to_chunk[index-1] = i;
      offsets[index] = offsets[index-1] + rem * C;
      ++index;
    }
  }

  delete [] chunk_widths;

  if(debug) {
    ALWAYS_ASSERT(num_slices+1 == index);
    printf("\nSlice Offsets\n");
    for (i = 0; i < num_slices + 1; ++i)
      printf("Slice %d starts at %d\n", i, offsets[i]);
  }
  
  //Fill the chunks
  arr_to_scs = new int[np];
  particle_mask = new bool[offsets[num_slices]];
  CreateSCSArrays<DataTypes>(scs_data, offsets[num_slices]);

  
  index = 0;
  int start = 0;
  int old_elem = 0;
  for (i = 0; i < num_slices; ++i) { //for each slice
    int elem = slice_to_chunk[i];
    if (old_elem!=elem) {
      old_elem = elem;
      start = 0;
    }
    int width = (offsets[i + 1] - offsets[i]) / C;
    for (int j = 0; j < width; ++j) { //for the width of that slice
      for (int k = elem * C; k < (elem + 1) * C; ++k) { //for each row in the slice
        if (k < num_elems && ptcls[k].first > start + j) {
          int ent_id = ptcls[k].second;
          int ptcl = ids[ent_id][start + j];
          ALWAYS_ASSERT(ptcl<np);
          arr_to_scs[ptcl] = index;
          particle_mask[index++] = true;
        }
        else
          particle_mask[index++] = false;
      }
    }
    start+=width;
  }

  if(debug) {
    printf("\narr_to_scs\n");
    for (i = 0; i < np; ++i)
      printf("array index %5d -> scs index %5d\n", i, arr_to_scs[i]);
    printf("\nSlices\n");
    for (i = 0; i < num_slices; ++i){
      printf("Slice %d:", i);
      for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        printf(" %d", particle_mask[j]);
        if (j % C == C - 1)
          printf(" |");
      }
      printf("\n");
    }
  }
  delete [] ptcls;
}

template<class DataTypes, int C>
SellCSigma<DataTypes,C>::~SellCSigma() {

  DestroySCSArrays<DataTypes>((scs_data), 0);
  
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  delete [] arr_to_scs;
}


template<class DataTypes, int C>
template <std::size_t N>
typename MemberTypeAtIndex<N,DataTypes>::type* SellCSigma<DataTypes,C>::getSCS() {
  return static_cast<typename MemberTypeAtIndex<N,DataTypes>::type*>(scs_data[N]);
}

#endif
