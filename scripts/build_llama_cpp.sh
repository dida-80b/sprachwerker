#!/usr/bin/env bash
set -euo pipefail

BACKEND="${LLAMA_BACKEND:-rocm}"
REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}"
REF="${LLAMA_REF:-master}"
SRC_DIR="${LLAMA_SRC_DIR:-/work/llama.cpp}"
ARTIFACT_DIR="${LLAMA_ARTIFACT_DIR:-/artifacts}"
FORCE_REBUILD="${LLAMA_FORCE_REBUILD:-0}"
EXTRA_ARGS="${LLAMA_CMAKE_EXTRA_ARGS:-}"

mkdir -p "${ARTIFACT_DIR}"

if [[ "${FORCE_REBUILD}" != "1" ]] && [[ -x "${ARTIFACT_DIR}/llama-mtmd-cli" ]]; then
  echo "[llama-builder] artifacts already exist in ${ARTIFACT_DIR}"
  exit 0
fi

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  rm -rf "${SRC_DIR}"
  git clone --depth 1 --branch "${REF}" "${REPO}" "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --depth 1 origin "${REF}"
  git -C "${SRC_DIR}" checkout -f FETCH_HEAD
fi

CMAKE_ARGS=("-B" "build" "-DBUILD_SHARED_LIBS=ON" "-DLLAMA_BUILD_TESTS=OFF" "-DLLAMA_BUILD_EXAMPLES=ON")
case "${BACKEND}" in
  rocm)
    CMAKE_ARGS+=("-DGGML_HIP=ON")
    if [[ -n "${AMDGPU_TARGETS:-}" ]]; then
      CMAKE_ARGS+=("-DAMDGPU_TARGETS=${AMDGPU_TARGETS}")
    fi
    ;;
  cuda)
    CMAKE_ARGS+=("-DGGML_CUDA=ON")
    if [[ -n "${CUDA_DOCKER_ARCH:-}" ]]; then
      CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}")
    fi
    ;;
  cpu)
    ;;
  *)
    echo "[llama-builder] unsupported LLAMA_BACKEND=${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_SPLIT=(${EXTRA_ARGS})
  CMAKE_ARGS+=("${EXTRA_SPLIT[@]}")
fi

cd "${SRC_DIR}"
rm -rf build
cmake "${CMAKE_ARGS[@]}"
cmake --build build --config Release -j"$(nproc)"

rm -rf "${ARTIFACT_DIR:?}/"*
find build -maxdepth 3 -type f \( -name "llama-*" -o -name "*.so" -o -name "*.so.*" \) -exec cp -av {} "${ARTIFACT_DIR}/" \;

for lib in "${ARTIFACT_DIR}"/lib*.so.*; do
  [[ -e "${lib}" ]] || continue
  base="$(basename "${lib}")"
  soname="${base%.*}"
  ln -sf "${base}" "${ARTIFACT_DIR}/${soname}"
done

if [[ ! -x "${ARTIFACT_DIR}/llama-mtmd-cli" ]]; then
  echo "[llama-builder] llama-mtmd-cli missing after build" >&2
  exit 1
fi

echo "[llama-builder] finished for backend=${BACKEND}"
