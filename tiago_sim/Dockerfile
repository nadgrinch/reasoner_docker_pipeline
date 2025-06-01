FROM osrf/ros:noetic-desktop-full-focal

LABEL maintainer="Yue Erro <yue.erro@pal-robotics.com>"

ARG REPO_WS=/tiago_dual_public_ws
RUN mkdir -p $REPO_WS/src
WORKDIR $REPO_WS

# Install dependencies and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    libv4l-dev \
    libv4l2rds0 \
    git \
    wget \
    vim \
    locales \
    dpkg \
    ssh \
    curl \
    aptitude \
    g++ \
    gcc \
    openvpn \
    gnupg \
    bash-completion \
    vim-gtk3 \
    nano \
    psmisc \
    ccache \
    gdb \
    qtcreator \
    htop \
    man \
    meld \
    silversearcher-ag \
    terminator \
    tig \
    valgrind \
    iputils-ping \
    ipython3 \
    python-is-python3 \
    python3-scipy \
    python3-wstool \
    python3-networkx \
    python3-pip  \
    python3-vcstool \
    python3-rosinstall \
    python3-catkin-tools \
    ros-noetic-actionlib-tools \
    ros-noetic-moveit-commander \
    # Gazebo and simulation packages
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins \
    ros-noetic-controller-manager \
    ros-noetic-joint-state-controller \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-velocity-controllers \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-teleop-twist-joy \
    ros-noetic-joy \
    # GUI support
    mesa-utils \
    libgl1-mesa-glx \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/* \
  && wget https://raw.githubusercontent.com/pal-robotics/tiago_dual_tutorials/master/tiago_dual_public-noetic.rosinstall \
  && vcs import src < tiago_dual_public-noetic.rosinstall

# Add simulation related packages if not included in the rosinstall
RUN cd $REPO_WS/src && \
    if [ ! -d "tiago_simulation" ]; then \
      git clone -b noetic-devel https://github.com/pal-robotics/tiago_simulation.git || echo "Repository not found or branch doesn't exist"; \
    fi && \
    if [ ! -d "pal_gazebo_worlds" ]; then \
      git clone -b noetic-devel https://github.com/pal-robotics/pal_gazebo_worlds.git || echo "Repository not found or branch doesn't exist"; \
    fi && \
    if [ ! -d "pal_gazebo_plugins" ]; then \
      git clone -b noetic-devel https://github.com/pal-robotics/pal_gazebo_plugins.git || echo "Repository not found or branch doesn't exist"; \
    fi

# Update for correct branches if the above failed
RUN cd $REPO_WS/src && \
    for repo in tiago_simulation pal_gazebo_worlds pal_gazebo_plugins; do \
      if [ -d "$repo" ] && [ ! -d "$repo/.git" ]; then \
        rm -rf "$repo"; \
      fi; \
    done && \
    if [ ! -d "tiago_simulation" ]; then \
      git clone https://github.com/pal-robotics/tiago_simulation.git || echo "Repository not found"; \
    fi && \
    if [ ! -d "pal_gazebo_worlds" ]; then \
      git clone https://github.com/pal-robotics/pal_gazebo_worlds.git || echo "Repository not found"; \
    fi && \
    if [ ! -d "pal_gazebo_plugins" ]; then \
      git clone https://github.com/pal-robotics/pal_gazebo_plugins.git || echo "Repository not found"; \
    fi

ARG ROSDEP_IGNORE="urdf_test omni_drive_controller orocos_kdl pal_filters libgazebo9-dev pal_usb_utils speed_limit_node camera_calibration_files pal_moveit_plugins pal_startup_msgs pal_local_joint_control pal_pcl_points_throttle_and_filter current_limit_controller hokuyo_node dynamixel_cpp pal_moveit_capabilities pal_pcl dynamic_footprint gravity_compensation_controller pal-orbbec-openni2 pal_loc_measure pal_map_manager joint_impedance_trajectory_controller ydlidar_ros_driver ament_lint_common rclcpp ament_cmake urdf_test humanoid_nav_msgs pal_hardware_interfaces ament_cmake_pytest"

# Install ROS dependencies
RUN apt-get update && rosdep install --from-paths src --ignore-src -y --rosdistro noetic --skip-keys="${ROSDEP_IGNORE}"

# Build the workspace
RUN bash -c "source /opt/ros/noetic/setup.bash \
    && catkin build -DCATKIN_ENABLE_TESTING=0 -j $(expr `nproc` / 2) \
    && echo 'source $REPO_WS/devel/setup.bash' >> ~/.bashrc"

# Set up environment variables for GUI support
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1
ENV DEBIAN_FRONTEND=noninteractive

# Copy and prepare entrypoint script
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
