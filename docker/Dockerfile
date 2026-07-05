# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.12

FROM ${UV_IMAGE} AS uv-bin
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        gosu \
        libgl1 \
        libegl1-mesa-dev \
        libgl1-mesa-dri \
        libglib2.0-0 \
        sudo \
        tree \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
COPY --from=uv-bin /uv /uvx /usr/local/bin/

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.42.4

COPY entrypoint.sh /usr/local/bin/cosmos-policy-entrypoint
RUN chown root:root /usr/local/bin/cosmos-policy-entrypoint && \
    chmod 0755 /usr/local/bin/cosmos-policy-entrypoint
ENTRYPOINT ["/usr/local/bin/cosmos-policy-entrypoint"]

CMD ["/bin/bash"]
