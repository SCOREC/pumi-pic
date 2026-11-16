#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${PUMIPIC_BUILD_DIR:-"$source_dir/build/debug"}
test_data_dir=${PUMIPIC_TEST_DATA_DIR:-"$source_dir/pumipic-data"}

if [[ ! -f "$test_data_dir/cube6tet.msh" ]]; then
  echo "Missing test data: run git submodule update --init pumipic-data" >&2
  exit 1
fi

# Use the concrete environment's compilers and dependencies for every step.
spack -e "$source_dir/spack" build-env pumi-pic -- bash -c '
  set -euo pipefail
  cmake -S "$1" -B "$2" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
    -DCMAKE_C_COMPILER="$(command -v mpicc)" \
    -DENABLE_CABANA=ON \
    -DIS_TESTING=ON -DPS_IS_TESTING=ON \
    -DBUILD_TESTING=ON -DTEST_DATA_DIR="$3" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build "$2" --parallel "${PUMIPIC_BUILD_JOBS:-4}"
' bash "$source_dir" "$build_dir" "$test_data_dir"
