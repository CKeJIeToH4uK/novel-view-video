# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS miniforge-base

ADD https://github.com/conda-forge/miniforge/releases/download/26.5.3-0/Miniforge3-26.5.3-0-Linux-x86_64.sh \
    /tmp/miniforge.sh

RUN bash /tmp/miniforge.sh -b -p /opt/miniforge \
    && rm /tmp/miniforge.sh


# Exact upstream trees. Git and its certificates never reach a prefix or
# runtime image; this stage is only a verified source owner for builders.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749 AS source-checkouts

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    fetch_exact() { \
      destination="$1"; repository="$2"; revision="$3"; \
      git init --quiet "${destination}"; \
      git -C "${destination}" remote add origin "${repository}"; \
      git -C "${destination}" fetch --quiet --depth=1 origin "${revision}"; \
      git -C "${destination}" checkout --quiet --detach FETCH_HEAD; \
      test "$(git -C "${destination}" rev-parse HEAD)" = "${revision}"; \
    }; \
    mkdir -p /opt/sources; \
    fetch_exact /opt/sources/gen3c \
      https://github.com/nv-tlabs/GEN3C.git \
      db2ffe12ced12ddafcec5e0422ee46ce8520746b; \
    fetch_exact /opt/sources/apex \
      https://github.com/NVIDIA/apex.git \
      6424da3b4faa6c8f062da4a48c424fff3f02d42d; \
    fetch_exact /opt/sources/transformer-engine \
      https://github.com/NVIDIA/TransformerEngine.git \
      7a7225c403bc704264e7cf437369594aeb8b3ba3; \
    test "$(git -C /opt/sources/transformer-engine rev-parse HEAD:3rdparty/cudnn-frontend)" = \
      936021bfed8c91dc416af1588b2c4eca631a9e45; \
    test "$(git -C /opt/sources/transformer-engine rev-parse HEAD:3rdparty/googletest)" = \
      f8d7d77c06936315286eb55f8de22cd23c188571; \
    fetch_exact /opt/sources/sam2 \
      https://github.com/facebookresearch/sam2.git \
      2b90b9f5ceec907a1c18123530e92e794ad901a4; \
    fetch_exact /opt/sources/dinov2 \
      https://github.com/facebookresearch/dinov2.git \
      7764ea0f912e53c92e82eb78a2a1631e92725fc8; \
    fetch_exact /opt/sources/vggt-omega \
      https://github.com/facebookresearch/vggt-omega.git \
      39a0cb8af88554f15ddcb5354cd52bde588fa014; \
    fetch_exact /opt/sources/moge \
      https://github.com/microsoft/MoGe.git \
      925b8ed835a7a9cdb7578ba15c658a0afc969030; \
    fetch_exact /opt/sources/utils3d \
      https://github.com/EasternJournalist/utils3d.git \
      3fab839f0be9931dac7c8488eb0e1600c236e183; \
    fetch_exact /opt/sources/pipeline \
      https://github.com/EasternJournalist/pipeline.git \
      866f059d2a05cde05e4a52211ec5051fd5f276d6


# The three packages below publish no accepted binary wheel. Their exact
# source archives are inputs to the Gen3C wheel builder.
FROM scratch AS gen3c-sdists

ADD https://files.pythonhosted.org/packages/72/73/b3d451dfc523756cf177d3ebb0af76dc7751b341c60e2a21871be400ae29/iopath-0.1.10.tar.gz \
    /sdists/iopath-0.1.10.tar.gz
ADD https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz \
    /sdists/antlr4-python3-runtime-4.9.3.tar.gz
ADD https://files.pythonhosted.org/packages/2d/6a/885bc91484e1aa8f618f6f0228d76d0e67000b0fdd6090673b777e311913/asciitree-0.3.3.tar.gz \
    /sdists/asciitree-0.3.3.tar.gz


# Core dependencies are downloaded first and installed from the resulting
# wheelhouse without network access.
FROM miniforge-base AS core-deps

COPY containers/diffusion/requirements/core.conda.lock /tmp/locks/core.conda.lock
COPY containers/diffusion/requirements/core.pip.lock /tmp/locks/core.pip.lock

RUN /opt/miniforge/bin/conda create --yes \
      --prefix /opt/envs/core \
      --file /tmp/locks/core.conda.lock \
    && mkdir -p /wheelhouse/core \
    && /opt/envs/core/bin/python -m pip download \
      --disable-pip-version-check \
      --no-deps \
      --only-binary=:all: \
      --destination-directory /wheelhouse/core \
      --requirement /tmp/locks/core.pip.lock \
    && /opt/envs/core/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/core \
      --requirement /tmp/locks/core.pip.lock \
    && /opt/envs/core/bin/python -m pip check


# One project wheel is the sole project payload consumed by every prefix.
FROM core-deps AS project-wheel-builder

WORKDIR /tmp/project/diffusion/code

COPY diffusion/code/pyproject.toml diffusion/code/README.md ./
COPY diffusion/code/LICENSE diffusion/code/THIRD_PARTY_NOTICES.md ./
COPY diffusion/code/src ./src

RUN /opt/envs/core/bin/python -m pip wheel \
        --disable-pip-version-check \
        --no-build-isolation \
        --no-deps \
        --wheel-dir /tmp/project-wheel \
        .


# Gen3C public wheels are resolved before source-built wheels are added.
FROM miniforge-base AS gen3c-public-deps

COPY containers/diffusion/requirements/gen3c-eval.conda.lock /tmp/locks/gen3c.conda.lock
COPY containers/diffusion/requirements/gen3c-eval.pip.lock /tmp/locks/gen3c.pip.lock

RUN /opt/miniforge/bin/conda create --yes \
      --prefix /opt/envs/gen3c \
      --file /tmp/locks/gen3c.conda.lock \
    && grep -Ev \
      '^(antlr4-python3-runtime|apex|asciitree|iopath|SAM-2|transformer_engine_torch)==' \
      /tmp/locks/gen3c.pip.lock \
      > /tmp/locks/gen3c-public.pip.lock \
    && mkdir -p /wheelhouse/gen3c \
    && /opt/envs/gen3c/bin/python -m pip download \
      --disable-pip-version-check \
      --no-deps \
      --only-binary=:all: \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      --destination-directory /wheelhouse/gen3c \
      --requirement /tmp/locks/gen3c-public.pip.lock \
    && /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/gen3c \
      --requirement /tmp/locks/gen3c-public.pip.lock


# Two official wheels declare obsolete internal tags. Repair only WHEEL
# metadata and RECORD; their Python code and shared libraries stay intact.
FROM gen3c-public-deps AS gen3c-wheel-retagger

RUN set -eux; \
    mkdir -p /tmp/retag/decord /tmp/retag/transformer_engine_cu12 \
      /wheelhouse/gen3c-retagged; \
    /opt/envs/gen3c/bin/python -m zipfile -e \
      /wheelhouse/gen3c/decord-0.6.0-py3-none-manylinux2010_x86_64.whl \
      /tmp/retag/decord; \
    decord_wheel=/tmp/retag/decord/decord-0.6.0.dist-info/WHEEL; \
    test "$(grep -Fxc 'Tag: cp36-cp36m-manylinux2010_x86_64' "${decord_wheel}")" -eq 1; \
    sed -i 's/^Tag: cp36-cp36m-manylinux2010_x86_64$/Tag: py3-none-manylinux2010_x86_64/' \
      "${decord_wheel}"; \
    test "$(grep -Fxc 'Tag: py3-none-manylinux2010_x86_64' "${decord_wheel}")" -eq 1; \
    /opt/envs/gen3c/bin/python -m wheel pack \
      --dest-dir /wheelhouse/gen3c-retagged /tmp/retag/decord; \
    /opt/envs/gen3c/bin/python -m zipfile -e \
      /wheelhouse/gen3c/transformer_engine_cu12-1.12.0-py3-none-manylinux_2_28_x86_64.whl \
      /tmp/retag/transformer_engine_cu12; \
    transformer_engine_wheel=/tmp/retag/transformer_engine_cu12/transformer_engine_cu12-1.12.0.dist-info/WHEEL; \
    test "$(grep -Fxc 'Tag: cp38-cp38-manylinux_2_28_x86_64' "${transformer_engine_wheel}")" -eq 1; \
    sed -i 's/^Tag: cp38-cp38-manylinux_2_28_x86_64$/Tag: py3-none-manylinux_2_28_x86_64/' \
      "${transformer_engine_wheel}"; \
    test "$(grep -Fxc 'Tag: py3-none-manylinux_2_28_x86_64' "${transformer_engine_wheel}")" -eq 1; \
    /opt/envs/gen3c/bin/python -m wheel pack \
      --dest-dir /wheelhouse/gen3c-retagged /tmp/retag/transformer_engine_cu12

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749 AS gen3c-build-base

COPY --from=gen3c-public-deps /opt/envs/gen3c /opt/envs/gen3c

ENV CUDA_HOME=/usr/local/cuda \
    CUDNN_HOME=/opt/envs/gen3c/lib/python3.10/site-packages/nvidia/cudnn \
    PATH="/opt/envs/gen3c/bin:/usr/local/cuda/bin:${PATH}" \
    CPATH="/opt/envs/gen3c/lib/python3.10/site-packages/nvidia/cudnn/include:${CPATH}" \
    LD_LIBRARY_PATH="/opt/envs/gen3c/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/envs/gen3c/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CC=/opt/envs/gen3c/bin/x86_64-conda-linux-gnu-cc \
    CXX=/opt/envs/gen3c/bin/x86_64-conda-linux-gnu-c++ \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM gen3c-build-base AS gen3c-python-wheel-builder

COPY --from=source-checkouts /opt/sources/sam2 /opt/sources/sam2
COPY --from=gen3c-sdists /sdists /opt/sdists

RUN set -eux; \
    mkdir -p /wheelhouse/gen3c-python; \
    python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/gen3c-python /opt/sdists/iopath-0.1.10.tar.gz; \
    python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/gen3c-python \
      /opt/sdists/antlr4-python3-runtime-4.9.3.tar.gz; \
    python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/gen3c-python /opt/sdists/asciitree-0.3.3.tar.gz; \
    SAM2_BUILD_CUDA=0 python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/gen3c-python /opt/sources/sam2

FROM gen3c-build-base AS transformer-engine-wheel-builder

COPY --from=source-checkouts /opt/sources/transformer-engine /opt/sources/transformer-engine

RUN cd /opt/sources/transformer-engine/transformer_engine/pytorch \
    && NVTE_RELEASE_BUILD=1 NVTE_CUDA_ARCHS=80 \
      TORCH_CUDA_ARCH_LIST=8.0 LDFLAGS=-Wl,--strip-all python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse .

FROM gen3c-build-base AS apex-wheel-builder

COPY --from=source-checkouts /opt/sources/apex /opt/sources/apex

RUN cd /opt/sources/apex \
    && APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
      TORCH_CUDA_ARCH_LIST=8.0 LDFLAGS=-Wl,--strip-all python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse .

FROM gen3c-public-deps AS gen3c-deps

COPY --from=gen3c-python-wheel-builder /wheelhouse/gen3c-python /wheelhouse/gen3c
COPY --from=transformer-engine-wheel-builder /wheelhouse/transformer_engine_torch-1.12.0-cp310-cp310-linux_x86_64.whl /wheelhouse/gen3c/
COPY --from=apex-wheel-builder /wheelhouse/apex-0.1-cp310-cp310-linux_x86_64.whl /wheelhouse/gen3c/
COPY --from=gen3c-wheel-retagger /wheelhouse/gen3c-retagged/ /wheelhouse/gen3c/

RUN grep -E '^(decord|transformer_engine_cu12)==' \
      /tmp/locks/gen3c.pip.lock > /tmp/locks/gen3c-retagged.pip.lock \
    && test "$(wc -l < /tmp/locks/gen3c-retagged.pip.lock)" -eq 2 \
    && /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check \
      --force-reinstall \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/gen3c \
      --requirement /tmp/locks/gen3c-retagged.pip.lock \
    && /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/gen3c \
      --requirement /tmp/locks/gen3c.pip.lock \
    && /opt/envs/gen3c/bin/python -m pip check


# VGGT is an independent dependency branch, not a Gen3C extension.
FROM miniforge-base AS vggt-public-deps

COPY containers/diffusion/requirements/vggt.conda.lock /tmp/locks/vggt.conda.lock
COPY containers/diffusion/requirements/vggt.pip.lock /tmp/locks/vggt.pip.lock

RUN /opt/miniforge/bin/conda create --yes \
      --prefix /opt/envs/vggt \
      --file /tmp/locks/vggt.conda.lock \
    && grep -Ev '^vggt-omega==' \
      /tmp/locks/vggt.pip.lock > /tmp/locks/vggt-public.pip.lock \
    && mkdir -p /wheelhouse/vggt \
    && /opt/envs/vggt/bin/python -m pip download \
      --disable-pip-version-check \
      --no-deps \
      --only-binary=:all: \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      --destination-directory /wheelhouse/vggt \
      --requirement /tmp/locks/vggt-public.pip.lock \
    && /opt/envs/vggt/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/vggt \
      --requirement /tmp/locks/vggt-public.pip.lock

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749 AS vggt-wheel-builder

COPY --from=vggt-public-deps /opt/envs/vggt /opt/envs/vggt
COPY --from=source-checkouts /opt/sources/vggt-omega /opt/sources/vggt-omega

RUN mkdir -p /wheelhouse/vggt-custom \
    && /opt/envs/vggt/bin/python -m pip wheel \
      --disable-pip-version-check \
      --no-deps \
      --no-build-isolation \
      --wheel-dir /wheelhouse/vggt-custom \
      /opt/sources/vggt-omega

FROM vggt-public-deps AS vggt-deps

COPY --from=vggt-wheel-builder /wheelhouse/vggt-custom /wheelhouse/vggt

RUN /opt/envs/vggt/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/vggt \
      --requirement /tmp/locks/vggt.pip.lock \
    && /opt/envs/vggt/bin/python -m pip check


# MoGe has its own wheelhouse and extends only the Gen3C runtime prefix.
FROM gen3c-deps AS moge-wheel-builder

COPY --from=source-checkouts /opt/sources/moge /opt/sources/moge
COPY --from=source-checkouts /opt/sources/utils3d /opt/sources/utils3d
COPY --from=source-checkouts /opt/sources/pipeline /opt/sources/pipeline

RUN set -eux; \
    mkdir -p /wheelhouse/moge-custom; \
    /opt/envs/gen3c/bin/python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/moge-custom /opt/sources/utils3d; \
    /opt/envs/gen3c/bin/python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/moge-custom /opt/sources/pipeline; \
    /opt/envs/gen3c/bin/python -m pip wheel \
      --disable-pip-version-check --no-deps --no-build-isolation \
      --wheel-dir /wheelhouse/moge-custom /opt/sources/moge

FROM gen3c-deps AS moge-wheelhouse

COPY containers/diffusion/requirements/moge-gen3c.pip.lock /tmp/locks/moge.pip.lock

RUN grep -Ev '^(moge|pipeline|utils3d)==' \
      /tmp/locks/moge.pip.lock > /tmp/locks/moge-public.pip.lock \
    && mkdir -p /wheelhouse/moge \
    && /opt/envs/gen3c/bin/python -m pip download \
      --disable-pip-version-check \
      --no-deps \
      --only-binary=:all: \
      --destination-directory /wheelhouse/moge \
      --requirement /tmp/locks/moge-public.pip.lock

COPY --from=moge-wheel-builder /wheelhouse/moge-custom /wheelhouse/moge

RUN /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --no-index \
      --only-binary=:all: \
      --find-links /wheelhouse/moge \
      --dry-run \
      --requirement /tmp/locks/moge.pip.lock


# Only Gen3C and DINOv2 source trees survive their builder owner.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS runtime-sources

COPY --from=source-checkouts /opt/sources/gen3c/cosmos_predict1 /opt/upstream/gen3c/cosmos_predict1
COPY --from=source-checkouts /opt/sources/gen3c/LICENSE /opt/sources/gen3c/ATTRIBUTIONS.md /opt/upstream/gen3c/
COPY --from=source-checkouts /opt/sources/dinov2/dinov2 /opt/upstream/dinov2/dinov2
COPY --from=source-checkouts /opt/sources/dinov2/LICENSE* /opt/upstream/dinov2/

RUN rm -rf \
      /opt/upstream/gen3c/cosmos_predict1/diffusion/training/utils/peft/lora_attn_test.py \
      /opt/upstream/gen3c/cosmos_predict1/tokenizer/modules/layers2d_test.py \
      /opt/upstream/gen3c/cosmos_predict1/tokenizer/modules/layers3d_test.py \
      /opt/upstream/gen3c/cosmos_predict1/tokenizer/test_data


# Prefix stages contain no Miniforge root, wheelhouse or build checkout.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS core-prefix

COPY --from=core-deps /opt/envs/core /opt/envs/core
COPY --from=project-wheel-builder /tmp/project-wheel /tmp/project-wheel

RUN --mount=type=bind,from=core-deps,source=/opt/miniforge/pkgs,target=/package-cache \
    --mount=type=bind,source=containers/diffusion/copy_conda_metadata.py,target=/copy_conda_metadata.py \
    /opt/envs/core/bin/python /copy_conda_metadata.py \
      /opt/envs/core /package-cache /opt/licenses/conda/core

RUN wheel_path=/tmp/project-wheel/novel_view_pipeline-0.1.0-py3-none-any.whl \
    && /opt/envs/core/bin/python -m pip install \
      --disable-pip-version-check --no-deps "${wheel_path}" \
    && /opt/envs/core/bin/python -m pip check \
    && rm -rf /tmp/project-wheel

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS gen3c-prefix

COPY --from=gen3c-deps /opt/envs/gen3c /opt/envs/gen3c
COPY --from=project-wheel-builder /tmp/project-wheel /tmp/project-wheel
COPY --from=runtime-sources /opt/upstream /opt/upstream

RUN --mount=type=bind,from=gen3c-deps,source=/opt/miniforge/pkgs,target=/package-cache \
    --mount=type=bind,source=containers/diffusion/copy_conda_metadata.py,target=/copy_conda_metadata.py \
    /opt/envs/gen3c/bin/python /copy_conda_metadata.py \
      /opt/envs/gen3c /package-cache /opt/licenses/conda/gen3c

RUN wheel_path=/tmp/project-wheel/novel_view_pipeline-0.1.0-py3-none-any.whl \
    && /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check --no-deps "${wheel_path}" \
    && site_packages="$(/opt/envs/gen3c/bin/python -c \
      'import site; print(site.getsitepackages()[0])')" \
    && printf '%s\n' /opt/upstream/gen3c > \
      "${site_packages}/distil3d-gen3c.pth" \
    && /opt/envs/gen3c/bin/python -m pip check \
    && rm -rf /tmp/project-wheel

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS vggt-prefix

COPY --from=vggt-deps /opt/envs/vggt /opt/envs/vggt
COPY --from=project-wheel-builder /tmp/project-wheel /tmp/project-wheel

RUN --mount=type=bind,from=vggt-deps,source=/opt/miniforge/pkgs,target=/package-cache \
    --mount=type=bind,source=containers/diffusion/copy_conda_metadata.py,target=/copy_conda_metadata.py \
    /opt/envs/vggt/bin/python /copy_conda_metadata.py \
      /opt/envs/vggt /package-cache /opt/licenses/conda/vggt

RUN wheel_path=/tmp/project-wheel/novel_view_pipeline-0.1.0-py3-none-any.whl \
    && /opt/envs/vggt/bin/python -m pip install \
      --disable-pip-version-check --no-deps "${wheel_path}" \
    && /opt/envs/vggt/bin/python -m pip check \
    && rm -rf /tmp/project-wheel


# Final images copy installed prefixes, never their builder layers.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS production-core

ARG DISTIL3D_SOURCE_REVISION=unknown
ARG DISTIL3D_SOURCE_DIRTY=unknown
ARG DISTIL3D_LOCK_REVISION=unknown

LABEL org.opencontainers.image.revision="${DISTIL3D_SOURCE_REVISION}" \
      io.distil3d.source.dirty="${DISTIL3D_SOURCE_DIRTY}" \
      io.distil3d.lock.revision="${DISTIL3D_LOCK_REVISION}" \
      io.distil3d.variant="core" \
      io.distil3d.gen3c.revision="db2ffe12ced12ddafcec5e0422ee46ce8520746b" \
      io.distil3d.dinov2.revision="7764ea0f912e53c92e82eb78a2a1631e92725fc8"

COPY --from=core-prefix /opt/envs/core /opt/envs/core
COPY --from=gen3c-prefix /opt/envs/gen3c /opt/envs/gen3c
COPY --from=vggt-prefix /opt/envs/vggt /opt/envs/vggt
COPY --from=runtime-sources /opt/upstream /opt/upstream
COPY --from=core-prefix /opt/licenses/conda /opt/licenses/conda
COPY --from=gen3c-prefix /opt/licenses/conda /opt/licenses/conda
COPY --from=vggt-prefix /opt/licenses/conda /opt/licenses/conda
COPY containers/diffusion/licenses /opt/licenses/distil3d

RUN groupadd --gid 1000 distil3d \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash distil3d \
    && mkdir -p /data /models /prepared /runs /cache \
    && chown distil3d:distil3d /prepared /runs /cache

ENV PATH="/opt/envs/core/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg

USER distil3d
WORKDIR /runs
ENTRYPOINT []
CMD ["/opt/envs/core/bin/python", "-m", "novel_view.cli.main", "--help"]


# The wheelhouse is mounted only while installing the MoGe delta.
FROM production-core AS production-moge

USER root

RUN --mount=type=bind,from=moge-wheelhouse,source=/wheelhouse/moge,target=/tmp/moge-wheels \
    --mount=type=bind,from=moge-wheelhouse,source=/tmp/locks/moge.pip.lock,target=/tmp/moge.pip.lock \
    /opt/envs/gen3c/bin/python -m pip install \
      --disable-pip-version-check --no-cache-dir --no-deps --no-index \
      --only-binary=:all: --find-links /tmp/moge-wheels \
      --requirement /tmp/moge.pip.lock \
    && /opt/envs/gen3c/bin/python -m pip check

LABEL io.distil3d.variant="moge"
USER distil3d


# The test image remains separate from both production targets.
# Test dependencies do not depend on project code or revision labels.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10 AS cpu-test-deps

COPY --from=core-deps /opt/envs/core /opt/envs/core

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY containers/diffusion/requirements/cpu-test.pip.lock /tmp/locks/cpu-test.pip.lock

RUN /opt/envs/core/bin/python -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --only-binary=:all: \
      --requirement /tmp/locks/cpu-test.pip.lock \
    && /opt/envs/core/bin/python -m pip check \
    && rm -rf /tmp/locks

RUN groupadd --gid 1000 distil3d \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash distil3d

FROM cpu-test-deps AS cpu-test

COPY --from=project-wheel-builder /tmp/project-wheel /tmp/project-wheel

RUN /opt/envs/core/bin/python -m pip install \
      --disable-pip-version-check --no-deps \
      /tmp/project-wheel/novel_view_pipeline-0.1.0-py3-none-any.whl \
    && /opt/envs/core/bin/python -m pip check \
    && rm -rf /tmp/project-wheel

ARG DISTIL3D_SOURCE_REVISION=unknown
ARG DISTIL3D_SOURCE_DIRTY=unknown
ARG DISTIL3D_LOCK_REVISION=unknown
ARG DISTIL3D_IMAGE_VARIANT=cpu-test

LABEL org.opencontainers.image.revision="${DISTIL3D_SOURCE_REVISION}" \
      io.distil3d.source.dirty="${DISTIL3D_SOURCE_DIRTY}" \
      io.distil3d.lock.revision="${DISTIL3D_LOCK_REVISION}" \
      io.distil3d.variant="${DISTIL3D_IMAGE_VARIANT}"

COPY --chown=distil3d:distil3d . /workspace

USER distil3d
WORKDIR /workspace/diffusion/code

# One inherited unit test proves that a component may be nested in a Git
# checkout. The test image creates a fresh metadata-only checkout; host .git
# is excluded from the build context and never copied into the image.
RUN git init --quiet /workspace \
    && git -C /workspace config user.name distil3d-test \
    && git -C /workspace config user.email distil3d-test@invalid \
    && GIT_AUTHOR_DATE=2000-01-01T00:00:00Z \
       GIT_COMMITTER_DATE=2000-01-01T00:00:00Z \
       git -C /workspace commit --quiet --allow-empty --message cpu-test

# Keep the prefix C++ runtime independent of Torch/Arrow import order.
ENV PATH="/opt/envs/core/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/envs/core/lib:${LD_LIBRARY_PATH}" \
    PYTHONDONTWRITEBYTECODE=1

CMD ["/opt/envs/core/bin/python", "-m", "pytest", "-q", "tests/unit"]
