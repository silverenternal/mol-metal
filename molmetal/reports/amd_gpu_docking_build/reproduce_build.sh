#!/usr/bin/env bash
set -euo pipefail

# Use a separate output directory to preserve any currently running binary.
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_root="${1:-/mnt/storage/tools/vina_gpu21_reproduce}"
commit=180272b8a5265d6ed9664178345933cebe2cd349
mkdir -p "$build_root"
curl --http1.1 -L --retry 3 --max-time 120 \
  "https://codeload.github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tar.gz/$commit" \
  -o "$build_root/source.tar.gz"
tar_sha="$(sha256sum "$build_root/source.tar.gz" | cut -d ' ' -f 1)"
if [[ "$tar_sha" != c5d8ca922f6ca4130dcc809aa620e1120be24306ca475046faef5ab7ccbf6aa5 ]]; then
  printf 'Unexpected official source archive SHA256: %s\n' "$tar_sha" >&2
  exit 1
fi
tar -xzf "$build_root/source.tar.gz" -C "$build_root"
source_dir="$build_root/Vina-GPU-2.1-$commit/QuickVina2-GPU-2.1"
cd "$source_dir"
patch --batch -p1 < "$report_dir/source_compatibility.patch"
make source WORK_DIR="$source_dir" BOOST_LIB_PATH=/usr/include OPENCL_LIB_PATH=/usr \
  OPENCL_VERSION=-DOPENCL_2_0 GPU_PLATFORM=-DAMD_PLATFORM \
  'SRC=./lib/*.cpp ./OpenCL/src/wrapcl.cpp' \
  'LIB1=-lboost_program_options -lboost_filesystem -lboost_thread -lOpenCL' \
  'OPTION=-DDISPLAY_ADDITION_INFO -DTIME_ANALYSIS -DCL_TARGET_OPENCL_VERSION=200' \
  > "$build_root/build.log" 2>&1
cc -shared -fPIC -O2 "$report_dir/trace_opencl.c" \
  -o "$build_root/trace_opencl.so" -ldl -lOpenCL
sha256sum "$build_root/source.tar.gz" ./QuickVina2-GPU-2-1
printf 'Built source-compiling executable in %s\n' "$source_dir"
