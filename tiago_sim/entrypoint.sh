#!/bin/bash
set -e

# Source the ROS environment
source /opt/ros/noetic/setup.bash

# Go to your workspace directory
cd /tiago_dual_public_ws

# Build user_packages if it exists
USER_PKG_DIR="/tiago_dual_public_ws/src/user_packages/"
ALL_PACKAGES=$(catkin list)
USER_PACKAGE_PATHS=$(find "$USER_PKG_DIR" -mindepth 1 -maxdepth 3 -type f -name 'package.xml' -exec dirname {} \;)
PACKAGES_TO_BUILD=()

for pkg_path in $USER_PACKAGE_PATHS; do
  pkg_name=$(basename "$pkg_path")
  if echo "$ALL_PACKAGES" | grep -q "$pkg_name"; then
    PACKAGES_TO_BUILD+=("$pkg_name")
  fi
done

echo "Packages to build: ${PACKAGES_TO_BUILD[@]}"

if [ ${#PACKAGES_TO_BUILD[@]} -gt 0 ]; then
  catkin build "${PACKAGES_TO_BUILD[@]}"
else
  echo "No user packages found to build."
fi

# Source the workspace setup file if it exists
if [ -f "/tiago_dual_public_ws/devel/setup.bash" ]; then
  source /tiago_dual_public_ws/devel/setup.bash
fi

# Set default ROS environment variables if not already set
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
export ROS_IP=${ROS_IP:-127.0.0.1}
export GAZEBO_MASTER_URI=${GAZEBO_MASTER_URI:-http://localhost:11345}

# Set display variable if running in a container with X11 forwarding
if [ -n "$DISPLAY" ]; then
  echo "Using display: $DISPLAY"
else
  echo "No display detected. GUI applications may not work."
fi

exec "$@"
