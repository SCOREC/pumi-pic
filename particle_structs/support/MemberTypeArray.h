#pragma once

#include <cstdlib>
#include "MemberTypes.h"
namespace particle_structs {

  //This type represents an array of arrays for each type of the given DataTypes
  template <class DataTypes> using MemberTypeArray = void*[DataTypes::size];

  //Initializes an array of a type
  template <class T> struct SetTypeSingle {
    SetTypeSingle(T& data, typename BaseType<T>::type val) {
      data = val;
    }
  };
  template <class T, int N> struct SetTypeSingle<T[N]> {
    SetTypeSingle(T (data)[N], typename BaseType<T>::type val) {
      for (int i = 0; i < N; ++i)
        SetTypeSingle<T>(data[i], val);
    }
  };
  template <class T> struct SetTypeArray {
    SetTypeArray(T *data, int size, typename BaseType<T>::type val) {
      for (int i =0; i < size; ++i)
        SetTypeSingle<T>(data[i], val);
    }
  };

  //Implementation to allocate arrays for each data type
  template <typename... Types> struct CreateArraysImpl;
  //base case for when all types are made
  template <> struct CreateArraysImpl<> {
    CreateArraysImpl(MemberTypeArray<MemberTypes<void> >, int) {}
  };
  template <typename T, typename... Types> struct CreateArraysImpl<T,Types...> {
    CreateArraysImpl(MemberTypeArray<MemberTypes<T, Types...> > data, int size) {
      //Allocate Array
      data[0] = new T[size];
      //Initialize all values to the default constructor
      SetTypeArray<T>(static_cast<T*>(data[0]), size, T());
      //Create the remaining types
      CreateArraysImpl<Types...>(data+1,size);
    }
  };

  //Call to allocate arrays for each data type
  template <typename... Types> struct CreateArrays;
  template <typename... Types> struct CreateArrays<MemberTypes<Types...> > {
    CreateArrays(MemberTypeArray<MemberTypes<Types...> > data, int size) {
      CreateArraysImpl<Types...>(data,size);
    }
  };

  //Copy the values of a type in one array to another array
  template <class T> struct CopyType {
    CopyType(T& new_data, T& old_data) {
      new_data = old_data;
    }
  };
  template <class T, int N> struct CopyType<T[N]> {
    CopyType(T (new_data)[N], T (old_data)[N]) {
      for (int i =0; i < N; ++i)
        CopyType<T>(new_data[i], old_data[i]);
    }
  };

  //Implementation of copying entries from one MT array to another at different indices
  template <typename... Types> struct CopyEntriesImpl;
  template <> struct CopyEntriesImpl<> {
    CopyEntriesImpl(MemberTypeArray<MemberTypes<void> >, int, 
                    MemberTypeArray<MemberTypes<void> >, int) {}
  };
  template <class T, typename... Types> struct CopyEntriesImpl<T, Types...> {
    CopyEntriesImpl(MemberTypeArray<MemberTypes<T, Types...> > new_data, int new_index, 
                    MemberTypeArray<MemberTypes<T, Types...> > old_data, int old_index) {
      CopyType<T>(static_cast<T*>(new_data[0])[new_index],
                  static_cast<T*>(old_data[0])[old_index]);
      CopyEntriesImpl<Types...>(new_data + 1, new_index, old_data + 1, old_index);
    }
  };

  //Call to copy entries from one MT array to another at different indices
  template <typename... Types> struct CopyEntries;
  template <typename... Types> struct CopyEntries<MemberTypes<Types...> > {
    CopyEntries(MemberTypeArray<MemberTypes<Types...> > new_data, int new_index, 
                MemberTypeArray<MemberTypes<Types...> > old_data, int old_index) {
      CopyEntriesImpl<Types...>(new_data, new_index, old_data, old_index);
    }
  };

  //Implementation to deallocate arrays of different types
  template <typename... Types> struct DestroyArraysImpl;
  template <> struct DestroyArraysImpl<> {
    DestroyArraysImpl(MemberTypeArray<MemberTypes<void> >) {}
  };
  template <typename T, typename... Types> struct DestroyArraysImpl<T,Types...> {
    DestroyArraysImpl(MemberTypeArray<MemberTypes<T,Types...> > data) {
      delete [] static_cast<T*>(data[0]);
      DestroyArraysImpl<Types...>(data+1);
    }
  };

  //Call to deallocate arrays of different types
  template <typename... Types> struct DestroyArrays;
  template <typename... Types> struct DestroyArrays<MemberTypes<Types...> > {
    DestroyArrays(MemberTypeArray<MemberTypes<Types...> > data) {
      DestroyArraysImpl<Types...>({data});
    }
  };
}
