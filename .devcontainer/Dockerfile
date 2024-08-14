FROM ghcr.io/nvidia/jax:nightly-2023-09-09
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    graphviz graphviz-dev \
    libxml2-dev \
    libjpeg-dev \
    libpng-dev \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6
RUN echo 'fs.inotify.max_user_watches=524288' >> /etc/sysctl.conf
RUN --mount=type=cache,target=~/.cache/pip \
    pip install -r /tmp/requirements.txt
