#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include "psAssert.h"
#include "MemberTypes.h"
#include <Kokkos_Core.hpp>

template<class DataTypes, typename ExecSpace = Kokkos::DefaultExecutionSpace>
class SellCSigma {
 public:
  SellCSigma(Kokkos::TeamPolicy<ExecSpace>& p,
	     int sigma, int vertical_chunk_size, int num_elements, int num_particles,
             int* particles_per_element, std::vector<int>* particle_id_bins,
             bool debug=false);
  ~SellCSigma();

 //Gets the Nth member type SCS
  template <std::size_t N>
  typename MemberTypeAtIndex<N,DataTypes>::type* getSCS();

 //Zeroes the values of the member type N
  template <std::size_t N>
  void zeroSCS();

 //Zeroes the values of the member type N that is an array with size entries
  template <std::size_t N>
  void zeroSCSArray(int size);

  /*
    Reshuffles the scs values to the element in new_element[i]
    Calls rebuildSCS if there is not enough space for the shuffle
  */
  void reshuffleSCS(int* new_element);
  /*
    Rebuilds a new SCS where particles move to the element in new_element[i]
  */
  void rebuildSCS(int* new_element);

  //Number of Data types
  static constexpr std::size_t num_types = DataTypes::size;
  typedef ExecSpace ExecutionSpace;

  //The User defined kokkos policy
  Kokkos::TeamPolicy<ExecutionSpace> policy;
  //Chunk size
  int C;
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
  //Total particles
  int num_ptcls;
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

 //Kokkos Views
#ifdef KOKKOS_ENABLED
 void transferToDevice();

 typedef Kokkos::DefaultExecutionSpace exe_space;
 typedef Kokkos::View<int*, exe_space::device_type> kkLidView;

 kkLidView offsets_d;
 kkLidView slice_to_chunk_d;
 kkLidView num_particles_d;
 kkLidView chunksz_d;
 kkLidView slicesz_d;
 kkLidView num_elems_d;
 kkLidView row_to_element_d;

#endif
 private: 
  //Pointers to the start of each SCS for each data type
  void* scs_data[num_types];

  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
 void destroySCS();

 void constructChunks(std::pair<int,int>* ptcls, int& nchunks, int& nslices, int*& chunk_widths, int*& row_element);
 void constructOffsets(int nChunks, int nSlices, int* chunk_widths, int*& offs, int*& s2e);
};

template <class T>
struct InitializeType {
  InitializeType(T *scs_data, int size) {
    for (int i =0; i < size; ++i)
      scs_data[i] = 0;
  }
};
template <class T, int N>
struct InitializeType<T[N]> {
  InitializeType(T (*scs_data)[N], int size) {
    for (int i =0; i < size; ++i)
      for (int j =0; j < N; ++j)
        scs_data[i][j] = 0;
  }
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
    InitializeType<T>(static_cast<T*>(scs_data[0]),size);
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

void sigmaSort(int num_elems, int* ptcls_per_elem, int sigma, 
	       std::pair<int, int>*& ptcl_pairs, bool doSort = true);

template<class DataTypes, typename ExecSpace>
  void SellCSigma<DataTypes, ExecSpace>::constructChunks(std::pair<int,int>* ptcls, int& nchunks, 
							 int& nslices, int*& chunk_widths, 
							 int*& row_element) {
  nchunks = num_elems / C + (num_elems % C != 0);
  nslices = 0;
  chunk_widths = new int[nchunks];
  row_element = new int[nchunks * C];

  //Add chunks for vertical slicing
  int i;
  for (i = 0; i < nchunks; ++i) {
    chunk_widths[i] = 0;
    for (int j = i * C; j < (i + 1) * C && j < num_elems; ++j)  {
      row_element[j] = ptcls[j].second;
      chunk_widths[i] = std::max(chunk_widths[i],ptcls[j].first);
    }
    int num_vertical_slices = chunk_widths[i] / V + (chunk_widths[i] % V != 0);
    nslices += num_vertical_slices;
  }

  //Set the padded row_to_element values
  for (i = num_elems; i < nchunks * C; ++i) {
    row_element[i] = i;
  }
}

template<class DataTypes, typename ExecSpace>
 void SellCSigma<DataTypes, ExecSpace>::constructOffsets(int nChunks, int nSlices, 
							 int* chunk_widths, int*& offs, int*& s2e) {
  offs = new int[nSlices + 1];
  s2e = new int[nSlices];
  offsets[0] = 0;
  int index = 1;
  for (int i = 0; i < nChunks; ++i) {
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
}
template<class DataTypes, typename ExecSpace>
  SellCSigma<DataTypes, ExecSpace>::SellCSigma(Kokkos::TeamPolicy<ExecSpace>& p,
					       int sig, int v, int ne, int np,
					       int* ptcls_per_elem, std::vector<int>* ids,
					       bool debug)  : policy(p) {
  C = policy.team_size();
  sigma = sig;
  V = v;
  num_elems = ne;
  num_ptcls = np;
  
  printf("Building SCS with C: %d sigma: %d V: %d\n",C,sigma,V);

  //Perform sorting
  std::pair<int, int>* ptcls;
  sigmaSort(num_elems,ptcls_per_elem, sigma, ptcls);

  //Number of chunks without vertical slicing
  int* chunk_widths;
  constructChunks(ptcls, num_chunks, num_slices, chunk_widths, row_to_element);

  if(debug) {
    printf("\nSigma Sorted Particle Counts\n");
    for (int i = 0; i < num_elems; ++i)
      printf("Element %d: has %d particles\n", row_to_element[i], ptcls[i].first);
  }

  //Create offsets into each chunk/vertical slice
  constructOffsets(num_chunks, num_slices, chunk_widths, offsets, slice_to_chunk);
  delete [] chunk_widths;

  if(debug) {
    printf("\nSlice Offsets\n");
    for (int i = 0; i < num_slices + 1; ++i)
      printf("Slice %d starts at %d\n", i, offsets[i]);
  }
  
  //Allocate the SCS
  arr_to_scs = new int[np];
  particle_mask = new bool[offsets[num_slices]];
  CreateSCSArrays<DataTypes>(scs_data, offsets[num_slices]);

  //Fill the SCS
  int index = 0;
  int start = 0;
  int old_elem = 0;
  int i;
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

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes, ExecSpace>::destroySCS() {
  DestroySCSArrays<DataTypes>((scs_data), 0);
  
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  delete [] arr_to_scs;
}
template<class DataTypes, typename ExecSpace>
  SellCSigma<DataTypes, ExecSpace>::~SellCSigma() {

  DestroySCSArrays<DataTypes>((scs_data), 0);
  
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  delete [] arr_to_scs;
}


template<class DataTypes, typename ExecSpace>
template <std::size_t N>
  typename MemberTypeAtIndex<N,DataTypes>::type* SellCSigma<DataTypes,ExecSpace>::getSCS() {
  return static_cast<typename MemberTypeAtIndex<N,DataTypes>::type*>(scs_data[N]);
}

template<class DataTypes, typename ExecSpace>
template <std::size_t N>
  void SellCSigma<DataTypes,ExecSpace>::zeroSCS() {
  typename MemberTypeAtIndex<N,DataTypes>::type* scs = getSCS<N>();
  for (int i =0; i < offsets[num_slices]; ++i)
    scs[i] = 0;
}

template<class DataTypes, typename ExecSpace>
template <std::size_t N>
  void SellCSigma<DataTypes,ExecSpace>::zeroSCSArray(int size) {
  typename MemberTypeAtIndex<N,DataTypes>::type* scs = getSCS<N>();
  for (int i =0; i < offsets[num_slices]; ++i)
    for (int j=0; j < size; ++j)
      scs[i][j] = 0;
}


template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::reshuffleSCS(int* new_element) {
  //For now call rebuild
  rebuildSCS(new_element);
}


template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::rebuildSCS(int* new_element) {
  int* new_particles_per_elem = new int[num_elems];

  for (int slice = 0; slice < num_slices; ++slice) {
    int width = (offsets[slice + 1] - offsets[slice]) / C;
    int chunk = slice_to_chunk[slice];
    for (int elem = chunk * C; elem < (chunk + 1) * C; ++elem) {
      for (int particle = 0; particle < width; ++particle) {
	++new_particles_per_elem[new_element[particle]] += particle_mask[particle];
      }
    }
  }

  //Perform sorting (Disabled currently due to needing mapping from old elems -> new elems)
  std::pair<int, int>* ptcls;
  sigmaSort(num_elems, new_particles_per_elem, sigma, ptcls, false);

  //Create chunking and count slices
  int* chunk_widths;
  int new_nchunks, new_nslices;
  int* new_row_to_element;
  constructChunks(ptcls, new_nchunks,new_nslices,chunk_widths,new_row_to_element);

  //Create offsets for each slice
  int* new_offsets;
  int* new_slice_to_chunk;
  constructOffsets(new_nchunks, new_nslices, chunk_widths,new_offsets, new_slice_to_chunk);
  delete [] chunk_widths;

  //Allocate the Chunks
  int* new_arr_to_scs = new int[num_ptcls];
  bool* new_particle_mask = new bool[new_offsets[new_nslices]];
  void* new_scs_data[num_types];
  CreateSCSArrays<DataTypes>(new_scs_data, new_offsets[new_nslices]);
  
  //Fill the SCS
  int* element_index[new_nchunks];
  int chunk = -1;
  for (int i =0; i < new_nslices; ++i) {
    if ( new_slice_to_chunk[i] == chunk)
      continue;
    chunk = new_slice_to_chunk[i];
    for (int e = 0; e < C; ++e)
      element_index[chunk*C + e] = new_offsets[i] + e;
  }
  for (int slice = 0; slice < num_slices; ++slice) {
    int width = (offsets[slice + 1] - offsets[slice]) / C;
    int chunk = slice_to_chunk[slice];
    for (int elem = chunk * C; elem < (chunk + 1) * C; ++elem) {
      for (int particle = 0; particle < width; ++particle) {
	if (particle_mask[particle]) {
	  //for each type
	  int new_elem = new_element[particle];
	  new_scs_data[0][element_index[new_elem]] = scs_data[0][particle];
	  element_index[elem] += C;
	}
      }
    }
  }

  
  delete [] ptcls;
  delete [] new_particles_per_elem;

  //Destroy old scs 
  destroySCS();

  //set scs to point to new values
  num_chunks = new_nchunks;
  num_slices = new_nslices;
  row_to_element = new_row_to_element;
  offsets = new_offsets;
  slice_to_chunk = new_slice_to_chunk;
  arr_to_scs = new_arr_to_scs;
  particle_mask = new_particle_mask;
  for (int i =0; i < num_types; ++i)
    scs_data[i] = new_scs_data[i];
}



template <class T>
void hostToDevice(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type> view, T* data) {
  typename Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>::HostMirror hv = 
    Kokkos::create_mirror_view(view);
  for (size_t i = 0; i < hv.size(); ++i)
    hv(i) = data[i];
  Kokkos::deep_copy(view, hv);
}

template <class DataTypes, typename ExecSpace>
  void SellCSigma<DataTypes, ExecSpace>::transferToDevice() {
  offsets_d = kkLidView("offsets_d",num_slices+1);
  hostToDevice(offsets_d,offsets);

  slice_to_chunk_d = kkLidView("slice_to_dhunk_d",num_slices);
  hostToDevice(slice_to_chunk_d,slice_to_chunk);

  num_particles_d = kkLidView("num_particles_d",1);
  hostToDevice(num_particles_d, &num_ptcls);

  chunksz_d = kkLidView("chunksz_d",1);
  hostToDevice(chunksz_d,&C);

  row_to_element_d = kkLidView("row_to_element_d", C * num_chunks);
  hostToDevice(row_to_element_d,row_to_element);
}

#endif
