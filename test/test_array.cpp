#include <fstream>

#include <Omega_h_for.hpp>
#include <pumipic_library.hpp>
#include <SupportKK.h>
#include <ppArray.h>

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);

  int fails = 0;

  pumipic::Array<int, 3> arr;
  for (int i = 0; i < arr.size(); ++i)
    arr[i] = i;

  for (int i = 0; i < 3; ++i)
    if (arr[i] != i) {
      fprintf(stderr, "[ERROR] Entry %d of array 1 is incorrect (%d != %d)\n", i, arr[i], i);
      ++fails;
    }

  pumipic::Array<int, 3, 2> arr2;
  for (int i = 0; i < arr2.size(); ++i)
    for (int j = 0; j < arr2[i].size(); ++j)
      arr2[i][j] = i * j;

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 2; ++j)
      if (arr2[i][j] != i * j) {
        fprintf(stderr, "[ERROR] Entry %d,%d of array 2 is incorrect (%d != %d)\n",
                i, j, arr2[i][j], i*j);
        ++fails;
      }

  Kokkos::View<int*, Kokkos::DefaultExecutionSpace> f("fails", 1);
  auto lamb = OMEGA_H_LAMBDA(const int& e) {
    pumipic::Array<int, 3,3,3> arr3;
    for (int i = 0; i < arr3.size(); ++i) {
      auto arr3_i = arr3[i];
      for (int j = 0; j < arr3_i.size(); ++j) {
        auto arr3_ij = arr3_i[j];
        for (int k = 0; k < arr3_ij.size(); ++k) {
          arr3_ij[k] = e + i * j * k;
        }
      }
    }

    for (int i = 0; i < 3; ++i) {
      auto arr3_i = arr3[i];
      for (int j = 0; j < 3; ++j) {
        auto arr3_ij = arr3_i[j];
        for (int k = 0; k < 3; ++k) {
          if (arr3_ij[k] != e + i * j * k) {
            printf("[ERROR] Entry %d, %d, %d of array 3 has incorrect value (%d != %d)\n",
                   i, j, k, arr3_ij[k], e + i * j * k);
            Kokkos::atomic_add(&(f[0]), 1);
          }
        }
      }
    }

  };
  Omega_h::parallel_for(10, lamb, "test_array");
  fails += pumipic::getLastValue(f);


  if (fails == 0)
    printf("All Tests Passed\n");
  return fails;
}
