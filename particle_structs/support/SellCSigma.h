#ifndef SELL_C_SIGMA_H_
#define SELL_C_SIGMA_H_
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include "psAssert.h"
#include "MemberTypes.h"
#include "SCS_Macros.h"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <unordered_map>
namespace particle_structs {

template<class DataTypes, typename ExecSpace = Kokkos::DefaultExecutionSpace>
class SellCSigma {
 public:
#ifdef KOKKOS_ENABLED
 typedef Kokkos::View<int*, typename ExecSpace::device_type> kkLidView;
#endif
  SellCSigma(Kokkos::TeamPolicy<ExecSpace>& p,
	     int sigma, int vertical_chunk_size, int num_elements, int num_particles,
             int* particles_per_element, std::vector<int>* particle_id_bins, int* element_gids,
             bool debug=false);
  ~SellCSigma();

  //Returns the size per data type of the scs including padding
  int size() const { return offsets[num_slices];}

  //Gets the Nth member type SCS
  template <std::size_t N>
  typename MemberTypeAtIndex<N,DataTypes>::type* getSCS();

  //Zeroes the values of the member type N
  template <std::size_t N>
  void zeroSCS();

  //Zeroes the values of the member type N that is an array with size entries
  template <std::size_t N>
  void zeroSCSArray(int size);

 void printFormat() const;

  /* Migrates each particle to new_process and to new_element
     Calls rebuildSCS to recreate the SCS after migrating particles
  */
  void migrate(kkLidView new_element, kkLidView new_process);

  /*
    Reshuffles the scs values to the element in new_element[i]
    Calls rebuildSCS if there is not enough space for the shuffle
  */
  void reshuffleSCS(int* new_element);
  /*
    Rebuilds a new SCS where particles move to the element in new_element[i]
  */
  void rebuildSCS(int* new_element, bool debug = false);

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
  int* particle_mask;

  //offsets into the scs structure
  int* offsets;

  //map from row to element
  // row = slice_to_chunk[slice] + row_in_chunk
  int* row_to_element;

  //mappings from row to element gid and back to row
  int* row_to_element_gid;
 typedef std::unordered_map<int, int> GID_Mapping;
 GID_Mapping element_gid_to_row;

 //Kokkos Views
#ifdef KOKKOS_ENABLED
  void transferToDevice();

  template <typename FunctionType>
  void parallel_for(FunctionType& fn);

 kkLidView offsets_d;
 kkLidView slice_to_chunk_d;
 kkLidView num_particles_d;
 kkLidView chunksz_d;
 kkLidView slicesz_d;
 kkLidView num_elems_d;
 kkLidView row_to_element_d;
 kkLidView particle_mask_d;

#endif
 private: 
  //Pointers to the start of each SCS for each data type
  void* scs_data[num_types];

  SellCSigma() {throw 1;}
  SellCSigma(const SellCSigma&) {throw 1;}
  SellCSigma& operator=(const SellCSigma&) {throw 1;}
 void destroySCS(bool destroyGid2Row=true);

 void constructChunks(std::pair<int,int>* ptcls, int& nchunks, int& nslices, int*& chunk_widths, 
                      int*& row_element);
 void createGlobalMapping(int* row2Elm, int* elmGid, int*& row2ElmGid, GID_Mapping& elmGid2Row);
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


template <class T>
struct CopyType {
  CopyType(T *new_data, int new_index, T* old_data, int old_index) {
    new_data[new_index] = old_data[old_index];
  }
};

template <class T, int N>
struct CopyType<T[N]> {
  CopyType(T (*new_data)[N], int new_index, T (*old_data)[N], int old_index) {
    for (int i =0; i < N; ++i)
      new_data[new_index][i] = old_data[old_index][i];
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

template <typename... Types>
struct CopySCSEntriesImpl;

template <>
struct CopySCSEntriesImpl<> {
  CopySCSEntriesImpl(void* new_data[], int new_index, void* old_data[], int old_index) {}
};

template <class T, typename... Types>
struct CopySCSEntriesImpl<T, Types...> {
  CopySCSEntriesImpl(void* new_data[], int new_index, void* old_data[], int old_index) {
    CopyType<T>(static_cast<T*>(new_data[0]),new_index, 
		static_cast<T*>(old_data[0]), old_index);
    CopySCSEntriesImpl<Types...>(new_data + 1, new_index, old_data + 1, old_index);
  }
};

template <typename... Types>
struct CopySCSEntries;

template <typename... Types>
struct CopySCSEntries<MemberTypes<Types...> > {
  CopySCSEntries(void* new_data[], int new_index, void* old_data[], int old_index) {
    CopySCSEntriesImpl<Types...>(new_data, new_index, old_data, old_index);
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
	       std::pair<int, int>*& ptcl_pairs);

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
void SellCSigma<DataTypes, ExecSpace>::createGlobalMapping(int* row2Elm, int* elmGid,
                                                           int*& row2ElmGid, 
                                                           GID_Mapping& elmGid2Row) {
  row2ElmGid = new int[num_chunks*C];
  for (int i = 0; i < num_elems; ++i) {
    int gid = elmGid[row2Elm[i]];
    row2ElmGid[i] = gid;
    elmGid2Row[gid] = i;
  }
  for (int j = num_elems; j < num_chunks*C; ++j) {
    row2ElmGid[j] = -1;
  }
}

template<class DataTypes, typename ExecSpace>
 void SellCSigma<DataTypes, ExecSpace>::constructOffsets(int nChunks, int nSlices, 
							 int* chunk_widths, int*& offs, int*& s2e) {
  offs = new int[nSlices + 1];
  s2e = new int[nSlices];
  offs[0] = 0;
  int index = 1;
  for (int i = 0; i < nChunks; ++i) {
    for (int j = V; j <= chunk_widths[i]; j+=V) {
      s2e[index-1] = i;
      offs[index] = offs[index-1] + V * C;
      ++index;
    }
    int rem = chunk_widths[i] % V;
    if (rem > 0) {
      s2e[index-1] = i;
      offs[index] = offs[index-1] + rem * C;
      ++index;
    }
  }
  PS_ALWAYS_ASSERT(index == nSlices + 1);
}
template<class DataTypes, typename ExecSpace>
  SellCSigma<DataTypes, ExecSpace>::SellCSigma(Kokkos::TeamPolicy<ExecSpace>& p,
					       int sig, int v, int ne, int np,
					       int* ptcls_per_elem, std::vector<int>* ids,
                                               int* element_gids,
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

  row_to_element_gid = NULL;
  if (element_gids) {
    createGlobalMapping(row_to_element, element_gids, row_to_element_gid, element_gid_to_row);
  }

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
  particle_mask = new int[offsets[num_slices]];
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
          PS_ALWAYS_ASSERT(ptcl<np);
          particle_mask[index++] = true;
        }
        else
          particle_mask[index++] = false;
      }
    }
    start+=width;
  }

  if(debug) {
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
void SellCSigma<DataTypes, ExecSpace>::destroySCS(bool destroyGid2Row) {
  DestroySCSArrays<DataTypes>((scs_data), 0);
  
  delete [] slice_to_chunk;
  delete [] row_to_element;
  delete [] offsets;
  delete [] particle_mask;
  if (row_to_element_gid)
    delete [] row_to_element_gid;
  if (destroyGid2Row)
    element_gid_to_row.clear();
}
template<class DataTypes, typename ExecSpace>
SellCSigma<DataTypes, ExecSpace>::~SellCSigma() {
  destroySCS();
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
void SellCSigma<DataTypes, ExecSpace>::migrate(kkLidView new_element, kkLidView new_process) {
  /********* Send # of particles being sent to each process *********/
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  kkLidView num_sent_particles("num_sent_particles", comm_size);
  auto count_sending_particles = SCS_LAMBDA(int element_id, int particle_id, bool mask) {
    const int process = new_process[particle_id];
    Kokkos::atomic_fetch_add(&(num_sent_particles[process]), mask);
  };
  parallel_for(count_sending_particles);
  kkLidView num_recv_particles("num_recv_particles", comm_size);
  MPI_Alltoall(num_sent_particles.data(), 1, MPI_INT, 
               num_recv_particles.data(), 1, MPI_INT, MPI_COMM_WORLD);
  int num_new_particles = 0;

  Kokkos::parallel_reduce("sum_new_particles", comm_size, KOKKOS_LAMBDA (const int& i, int& lsum ) {
      lsum += num_recv_particles[i];
    }, num_new_particles);
  num_new_particles -= num_ptcls;
  printf("%d %d\n", comm_rank, num_new_particles);

  /********* Create new particle_elements and particle_data *********/
  int* new_particle_elements = NULL;

  void* new_particle_data[num_types];
  if (num_new_particles > 0) {
    new_particle_elements = new int[num_new_particles];
    CreateSCSArrays<DataTypes>(new_particle_data, num_new_particles);
  }

  /* /\********* Send particle information to new processes *********\/ */
  //Perform an ex-sum on num_sent_particles & num_recv_particles
  kkLidView offset_sent_particles("offset_sent_particles", comm_size+1);
  kkLidView offset_recv_particles("offset_recv_particles", comm_size+1);
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
    num += num_sent_particles[i];
    offset_sent_particles[i+1] += num*final;
  });
  Kokkos::parallel_scan(comm_size, KOKKOS_LAMBDA(const int& i, int& num, const bool& final) {
    num += num_recv_particles[i];
    offset_recv_particles[i+1] += num*final;
  });

  
  //Create an array of particles being sent
  kkLidView send_element("send_element", offset_sent_particles[comm_size]);
  //Send the particles to each neighbor

  //Recv particles from ranks that send nonzero number of particles

  /* /\********* Combine and shift particles to their new destination *********\/ */
  /* rebuildSCS(new_element, new_particles_elements, new_particle_data); */

  delete [] new_particle_elements;
  DestroySCSArrays<DataTypes>(new_particle_data, 0);
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::reshuffleSCS(int* new_element) {
  //For now call rebuild
  rebuildSCS(new_element);
}


template<class DataTypes, typename ExecSpace>
  void SellCSigma<DataTypes,ExecSpace>::rebuildSCS(int* new_element, bool debug) {

  int* new_particles_per_elem = new int[num_chunks*C];
  for (int i =0; i < num_elems; ++i)
    new_particles_per_elem[i] = 0;
  int activePtcls = 0;
  for (int slice = 0; slice < num_slices; ++slice) {
    for (int j = offsets[slice]; j < offsets[slice+1]; j+= C) {
      for (int k = 0; k < C; ++k) {
	if (particle_mask[j+k] && new_element[j+k] != -1) {
	  new_particles_per_elem[new_element[j+k]] += particle_mask[j+k];
          ++activePtcls;
	}
      }
    }
  }

  //If there are no particles left, then destroy the structure
  if(!activePtcls) {
    delete [] new_particles_per_elem;
    destroySCS();
    num_ptcls = 0;
    num_chunks = 0;
    num_slices = 0;
    row_to_element = 0;
    row_to_element_gid = 0;
    offsets = 0;
    slice_to_chunk = 0;
    particle_mask = 0;
    for (size_t i = 0; i < num_types; ++i)
      scs_data[i] = 0;
    return;
  }
  int new_num_ptcls = activePtcls;
  //Perform sorting
  std::pair<int, int>* ptcls;
  sigmaSort(num_elems, new_particles_per_elem, sigma, ptcls);


  //Create chunking and count slices
  int* chunk_widths;
  int new_nchunks, new_nslices;
  int* new_row_to_element;
  constructChunks(ptcls, new_nchunks,new_nslices,chunk_widths,new_row_to_element);

  //Create a mapping from element to new scs row index
  //Also recreate the original gid mapping
  int* element_to_new_row = new int[num_chunks * C];
  int* gid_mapping = new int[num_chunks * C];
  for (int i = 0; i < num_chunks * C; ++i) {
    element_to_new_row[new_row_to_element[i]] = i;
    if (row_to_element_gid)
      gid_mapping[row_to_element[i]] = row_to_element_gid[i];
  }

  int* new_row_to_element_gid;
  element_gid_to_row.clear();
  if (row_to_element_gid)
    createGlobalMapping(new_row_to_element, gid_mapping, 
                        new_row_to_element_gid, element_gid_to_row);
  delete [] gid_mapping;
  if(debug) {
    printf("\nSigma Sorted Particle Counts\n");
    for (int i = 0; i < num_elems; ++i)
      printf("Element %d: has %d particles\n", new_row_to_element[i], ptcls[i].first);
  }

  //Create offsets for each slice
  int* new_offsets;
  int* new_slice_to_chunk;
  constructOffsets(new_nchunks, new_nslices, chunk_widths,new_offsets, new_slice_to_chunk);
  delete [] chunk_widths;

  if(debug) {
    printf("\nSlice Offsets\n");
    for (int i = 0; i < new_nslices + 1; ++i)
      printf("Slice %d starts at %d\n", i, new_offsets[i]);
  }

  //Allocate the Chunks
  int* new_particle_mask = new int[new_offsets[new_nslices]];
  std::memset(new_particle_mask,0,new_offsets[new_nslices]*sizeof(int));
  void* new_scs_data[num_types];
  CreateSCSArrays<DataTypes>(new_scs_data, new_offsets[new_nslices]);
  
  //Fill the SCS
  int* element_index = new int[new_nchunks * C];
  int chunk = -1;
  for (int i =0; i < new_nslices; ++i) {
    if ( new_slice_to_chunk[i] == chunk)
      continue;
    chunk = new_slice_to_chunk[i];
    for (int e = 0; e < C; ++e) {
      element_index[chunk*C + e] = new_offsets[i] + e;
    }
  }

  for (int slice = 0; slice < num_slices; ++slice) {
    for (int j = offsets[slice]; j < offsets[slice+1]; j+= C) {
      for (int k = 0; k < C; ++k) {
	int particle = j + k;
	if (particle_mask[particle] && new_element[particle] != -1) {
	  int new_elem = new_element[particle];
          int new_row = element_to_new_row[new_elem];
	  int new_index = element_index[new_row];
	  CopySCSEntries<DataTypes>(new_scs_data,new_index, scs_data, particle);
	  element_index[new_row] += C;
	  new_particle_mask[new_index] = 1;
	}
      }
    }
  }
  delete [] element_index;
  delete [] element_to_new_row;
  delete [] ptcls;
  delete [] new_particles_per_elem;

  //Destroy old scs 
  destroySCS(false);

  //set scs to point to new values
  num_ptcls = new_num_ptcls;
  num_chunks = new_nchunks;
  num_slices = new_nslices;
  row_to_element = new_row_to_element;
  if (row_to_element_gid)
    row_to_element_gid = new_row_to_element_gid;
  offsets = new_offsets;
  slice_to_chunk = new_slice_to_chunk;
  particle_mask = new_particle_mask;
  for (size_t i = 0; i < num_types; ++i)
    scs_data[i] = new_scs_data[i];

  if(debug) {
    printf("\nSlices\n");
    for (int i = 0; i < num_slices; ++i){
      printf("Slice %d:", i);
      for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
        printf(" %d", particle_mask[j]);
        if (j % C == C - 1)
          printf(" |");
      }
      printf("\n");
    }
  }
}

template<class DataTypes, typename ExecSpace>
void SellCSigma<DataTypes,ExecSpace>::printFormat() const {
  char message[10000];
  char* cur = message;
  cur += sprintf(cur,"Particle Structures Sell-C-Sigma C: %d sigma: %d V: %d.\n", C, sigma, V);
  cur += sprintf(cur,"Number of Elements: %d.\nNumber of Particles: %d.\n", num_elems, num_ptcls);
  cur += sprintf(cur,"Number of Chunks: %d.\nNumber of Slices: %d.\n", num_chunks, num_slices);
  int last_chunk = -1;
  for (int i = 0; i < num_slices; ++i) {
    int chunk = slice_to_chunk[i];
    if (chunk != last_chunk) {
      last_chunk = chunk;
      cur += sprintf(cur,"  Chunk %d. Elements", chunk);
      if (row_to_element_gid)
        cur += sprintf(cur,"(GID)");
      cur += sprintf(cur,":");
      for (int row = chunk*C; row < (chunk+1)*C; ++row) {
        cur += sprintf(cur," %d", row_to_element[row]);
        if (row_to_element_gid) {
          cur += sprintf(cur,"(%d)", row_to_element_gid[row]);
        }
      }
      cur += sprintf(cur,"\n");
    }
    cur += sprintf(cur,"    Slice %d", i);
    for (int j = offsets[i]; j < offsets[i+1]; ++j) {
      if ((j - offsets[i]) % C == 0)
        cur += sprintf(cur," |");
      cur += sprintf(cur," %d", particle_mask[j]);
    }
    cur += sprintf(cur,"\n");
  }
  printf("%s", message);
}

template <class T>
void hostToDevice(Kokkos::View<T*, Kokkos::DefaultExecutionSpace::device_type>& view, T* data) {
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

  slice_to_chunk_d = kkLidView("slice_to_chunk_d",num_slices);
  hostToDevice(slice_to_chunk_d,slice_to_chunk);

  num_particles_d = kkLidView("num_particles_d",1);
  hostToDevice(num_particles_d, &num_ptcls);

  chunksz_d = kkLidView("chunksz_d",1);
  hostToDevice(chunksz_d,&C);

  row_to_element_d = kkLidView("row_to_element_d", C * num_chunks);
  hostToDevice(row_to_element_d,row_to_element);

  particle_mask_d = kkLidView("particle_mask_d", offsets[num_slices]);
  hostToDevice(particle_mask_d,particle_mask);
}

template <class DataTypes, typename ExecSpace>
template <typename FunctionType>
void SellCSigma<DataTypes, ExecSpace>::parallel_for(FunctionType& fn) {
  FunctionType* fn_d;
#ifdef SCS_USE_CUDA
  cudaMalloc(&fn_d, sizeof(FunctionType));
  cudaMemcpy(fn_d,&fn, sizeof(FunctionType), cudaMemcpyHostToDevice);
#else
  fn_d = &fn;
#endif
  const int league_size = num_slices;
  const int team_size = C;
  typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> team_policy;
  const team_policy policy(league_size, team_size);
  auto offsets_cpy = offsets_d;
  auto slice_to_chunk_cpy = slice_to_chunk_d;
  auto row_to_element_cpy = row_to_element_d;
  auto particle_mask_cpy = particle_mask_d;
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const team_policy::member_type& thread) {
    const int slice = thread.league_rank();
    const int slice_row = thread.team_rank();
    const int rowLen = (offsets_cpy(slice+1)-offsets_cpy(slice))/team_size;
    const int start = offsets_cpy(slice) + slice_row;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, team_size), [=] (int& j) {
      const int row = slice_to_chunk_cpy(slice) * team_size + slice_row;
      const int element_id = row_to_element_cpy(row);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, rowLen), [&] (int& p) {
        const int particle_id = start+(p*team_size);
        const int mask = particle_mask_cpy[particle_id];
        (*fn_d)(element_id, particle_id, mask);
      });
    });
  });
}

} // end namespace particle_structs

#endif
