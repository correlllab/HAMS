# ==============================
# Stage 1: Builder
# ==============================
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV OMNI_KIT_ACCEPT_EULA=yes

# Set HTTP/HTTPS proxy (optional)
ARG http_proxy
ARG https_proxy
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}

# Use US mirror for faster downloads
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://us.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://security.ubuntu.com/ubuntu/|g' /etc/apt/sources.list

# Install build dependencies + X11 support for GUI rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-12 g++-12 cmake build-essential unzip git-lfs wget \
    libglu1-mesa-dev vulkan-tools libvulkan1 \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxt6 libxkbcommon-x11-0 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# Accept Conda TOS and create environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n humanoid_sim_env python=3.11 -y && \
    conda clean -afy

# Switch to Conda environment
SHELL ["conda", "run", "-n", "humanoid_sim_env", "/bin/bash", "-c"]  

# Install libgcc/libstdc++ for C++17 support
RUN conda install -y -c conda-forge "libgcc-ng>=12" "libstdcxx-ng>=12" && \
    conda clean -afy

# Install PyTorch NIGHTLY for RTX 5070 Ti (50 series) support
RUN pip install --upgrade pip && \
    pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install Isaac Sim
RUN pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# Create working directory
RUN mkdir -p /home/code
WORKDIR /home/code
RUN pip install "pip==23" "setuptools==65" flatdict

# Clone and install IsaacLab
RUN git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout v2.2.0 && \
    ./isaaclab.sh --install

# Build CycloneDDS
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /cyclonedds && \
    cd /cyclonedds && mkdir build install && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

ENV CYCLONEDDS_HOME=/cyclonedds/install

# Install unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python && \
    cd unitree_sdk2_python && pip install -e .

# Clone unitree_sim_isaaclab
RUN git clone https://github.com/correlllab/CL_isaaclab_sim.git /home/code/CL_isaaclab_sim && \
    cd /home/code/CL_isaaclab_sim && pip install -r requirements.txt
    
RUN cd /home/code/CL_isaaclab_sim && git clone https://github.com/unitreerobotics/teleimager.git

RUN cd /home/code/CL_isaaclab_sim/teleimager && git reset --hard a1815d0 && sed -i 's/^requires-python = .*/requires-python = ">=3.8,<3.12"/' pyproject.toml
RUN cd /home/code/CL_isaaclab_sim/teleimager && pip install -e . && pip install aiortc

RUN cd /home/code/CL_isaaclab_sim && . fetch_assets.sh


## Clone models and ROS
#RUN git clone https://huggingface.co/datasets/unitreerobotics/unitree_model /home/code/unitree_model
#ENV UNITREE_MODEL_DIR=/home/code/unitree_model
#
#RUN git clone https://github.com/unitreerobotics/unitree_ros.git /home/code/unitree_ros
#ENV UNITREE_ROS_DIR=/home/code/unitree_ros
#
#
#RUN git clone https://github.com/unitreerobotics/unitree_rl_lab.git /home/code/unitree_rl_lab && \
#    cd /home/code/unitree_rl_lab && \
#    sed -i 's|UNITREE_MODEL_DIR = "path/to/unitree_model"|UNITREE_MODEL_DIR = os.getenv("UNITREE_MODEL_DIR", "/home/code/unitree_model")|g' \
#      ./source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py && \
#    sed -i 's|UNITREE_ROS_DIR = "path/to/unitree_ros"|UNITREE_ROS_DIR = os.getenv("UNITREE_ROS_DIR", "/home/code/unitree_ros")|g' \
#      ./source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py && \
#    ./unitree_rl_lab.sh -i


# ==============================
# Stage 2: Runtime
# ==============================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV OMNI_KIT_ALLOW_ROOT=1
ENV OMNI_KIT_ACCEPT_EULA=yes
ENV UNITREE_ROS_DIR=/home/code/unitree_ros
ENV UNITREE_MODEL_DIR=/home/code/unitree_model
ENV CYCLONEDDS_HOME=/cyclonedds/install

# Install runtime dependencies + X11 libraries for GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa git-lfs zenity unzip \
    libvulkan1 vulkan-tools \
    libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxt6 libxkbcommon-x11-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /home/code /home/code
COPY --from=builder /cyclonedds /cyclonedds
COPY --from=builder /opt/conda /opt/conda


# Initialize bashrc (removed OMNI_KIT_DISABLE_STARTUP)
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc && \
    echo 'conda activate humanoid_sim_env' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ALLOW_ROOT=1' >> ~/.bashrc && \
    echo 'export OMNI_KIT_ACCEPT_EULA=yes' >> ~/.bashrc && \
    echo 'export UNITREE_MODEL_DIR=/home/code/unitree_model' >> ~/.bashrc && \
    echo 'export UNITREE_ROS_DIR=/home/code/unitree_ros' >> ~/.bashrc && \
    echo 'export CYCLONEDDS_HOME=/cyclonedds/install' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH="/opt/conda/envs/humanoid_sim_env/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib/":$LD_LIBRARY_PATH' >> ~/.bashrc


WORKDIR /home/code

# Default to Conda environment bash
CMD ["conda", "run", "-n", "humanoid_sim_env", "/bin/bash"]

