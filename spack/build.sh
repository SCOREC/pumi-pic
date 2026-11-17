#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:-debug}
case "$profile" in
  debug) environment="$source_dir/spack"; build_type=Debug ;;
  release) environment="$source_dir/spack"; build_type=Release ;;
  cuda) environment="$source_dir/spack/cuda"; build_type=Release ;;
  *) echo "Usage: bash spack/build.sh [debug|release|cuda]" >&2; exit 1 ;;
esac
build_dir=${PUMIPIC_BUILD_DIR:-"$source_dir/build/$profile"}
test_data_dir=${PUMIPIC_TEST_DATA_DIR:-"$source_dir/pumipic-data"}

if [[ ! -f "$test_data_dir/cube6tet.msh" ]]; then
  echo "Missing test data: run git submodule update --init pumipic-data" >&2
  exit 1
fi

# Some file-I/O tests regenerate their input fixtures. Give each build a copy.
mkdir -p "$build_dir/test-data"
tar --exclude=.git -C "$test_data_dir" -cf - . | tar -xf - -C "$build_dir/test-data"
test_data_dir="$build_dir/test-data"

# Use the concrete environment's compilers and dependencies for every step.
spack -e "$environment" build-env pumi-pic -- bash -c '
  set -euo pipefail
  cmake -S "$1" -B "$2" \
    -DCMAKE_BUILD_TYPE="$4" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DCMAKE_C_COMPILER="$(command -v mpicc)" \
    -DENABLE_CABANA=ON \
    -DIS_TESTING=ON -DPS_IS_TESTING=ON \
    -DBUILD_TESTING=ON -DTEST_DATA_DIR="$3" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "$2" --parallel "${PUMIPIC_BUILD_JOBS:-4}"
' bash "$source_dir" "$build_dir" "$test_data_dir" "$build_type"
