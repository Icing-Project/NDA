#!/usr/bin/env bash
#
# Fast dev loop script for Ubuntu:
# - assumes system deps and Python packages are already installed
# - configures CMake (once) and then does incremental builds
# - optionally runs the NDA binary after a successful build

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
RUN_AFTER_BUILD="${RUN_AFTER_BUILD:-1}"

echo "==> Using build directory: ${BUILD_DIR}"
echo "==> CMake build type: ${BUILD_TYPE}"

if [[ ! -d "${BUILD_DIR}" || ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "==> No existing CMake cache found, configuring build..."
    /usr/bin/cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
else
    echo "==> Reusing existing CMake configuration in ${BUILD_DIR}"
fi

echo "==> Building NDA..."
/usr/bin/cmake --build "${BUILD_DIR}" -j"$(nproc)"

if [[ "${RUN_AFTER_BUILD}" == "1" ]]; then
    echo "==> Launching NDA..."
    "./${BUILD_DIR}/NDA"
else
    echo "==> Build complete; skipping run (RUN_AFTER_BUILD=${RUN_AFTER_BUILD})"
fi

