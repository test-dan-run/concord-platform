# nemotronlabs-re-use: OIP v2 universal speech enhancement (NVIDIA NemotronLabs RE-USE, bidirectional Mamba) on the GPU.
# Weights are mounted at /models, never baked in.
#
# mamba-ssm publishes no wheel for this base (Python 3.12, CUDA 13.0, torch 2.14), so the first stage compiles its
# selective-scan kernel on the CUDA 13.0 devel image against the same torch, for the RTX 5090 (sm_120) only.
# causal-conv1d is not built: without it mamba-ssm uses torch's conv1d and still runs the compiled scan.
ARG BASE_IMAGE=ghcr.io/owner/py-cuda-base:0.1.1

FROM nvidia/cuda:13.0.3-devel-ubuntu24.04 AS kernels
# Must match TORCH_VERSION of py-cuda-base (custom/_bases/Dockerfile.py-cuda): the kernel links against torch's ABI.
ARG TORCH_VERSION=2.14.1
ARG MAMBA_SDIST=https://files.pythonhosted.org/packages/af/48/aa7bec0efc7453fa0512276c0069c854bf684219050294266534bfef34bb/mamba_ssm-2.3.2.post1.tar.gz
ARG MAMBA_SHA256=104cc47e9101e5401a675fa2b784f2952b9b037f3b1dd83b5ac544394e95d028
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PATH=/opt/build/bin:$PATH
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev g++ \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/build
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu130 \
 && pip install packaging==26.3 ninja setuptools wheel
WORKDIR /src
# The sdist, checked against its PyPI sha256; setup.py's nine -gencode targets are cut to sm_120 (one ninth the compile).
RUN python - <<PY
import hashlib, pathlib, re, tarfile, urllib.request
data = urllib.request.urlopen("${MAMBA_SDIST}").read()
assert hashlib.sha256(data).hexdigest() == "${MAMBA_SHA256}", "mamba-ssm sdist sha256 mismatch"
pathlib.Path("/tmp/mamba.tar.gz").write_bytes(data)
tarfile.open("/tmp/mamba.tar.gz").extractall("/src", filter="data")
setup = next(pathlib.Path("/src").glob("mamba_ssm-*/setup.py"))
text = setup.read_text()
marker = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
assert marker in text, "setup.py changed: re-check the sm_120 patch"
text = text.replace(marker, '    cc_flag = ["-gencode", "arch=compute_120,code=sm_120"]  # RTX 5090 only\n' + marker, 1)
# torch 2.14's headers need C++20 ("C++20 or later compatible compiler is required to use ATen"); setup.py asks for C++17
assert text.count('"-std=c++17"') == 4, "setup.py changed: re-check the C++ standard patch"
setup.write_text(text.replace('"-std=c++17"', '"-std=c++20"'))
PY
RUN cd /src/mamba_ssm-* \
 && MAMBA_FORCE_BUILD=TRUE MAX_JOBS=8 NVCC_THREADS=2 pip wheel --no-deps --no-build-isolation -w /wheels .

FROM ${BASE_IMAGE}
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY --from=kernels --chown=1000:1000 /wheels/ /tmp/wheels/
# mamba_ssm/__init__.py imports Mamba2/Mamba3 and the LM head, which pull in tilelang, quack and transformers;
# RE-USE uses only the Mamba (v1) block, so the package entry point exports just that.
RUN pip install --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels \
 && printf '__version__ = "2.3.2.post1"\nfrom mamba_ssm.modules.mamba_simple import Mamba\n' \
      > /opt/venv/lib/python3.12/site-packages/mamba_ssm/__init__.py
COPY server/ /app/server/
ENV WEIGHTS_DIR=/models HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "server.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
