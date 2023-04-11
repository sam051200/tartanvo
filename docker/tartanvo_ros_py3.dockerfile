FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
ENV TZ=Asia/Taipei
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_DPDS=apt_packages.txt
ARG PY_DPDS=requirements.txt

WORKDIR /tmp

USER root

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# install apt dependencies
RUN apt update
COPY ./${APT_PKGS} ./
RUN xargs apt install \
    --yes \
    --no-install-recommends \
    < ${APT_DPDS}

# install python dependencies
COPY ./${PY_DPDS} ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    --requirement ${PY_DPDS}

# Clean up
RUN apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR /app

CMD ["bash"]

# TODO: add python2 venv? (for eval tool: https://github.com/uzh-rpg/rpg_trajectory_evaluation)
# RUN apt install -y -no-recom python3-venv=3.8.2-0ubuntu2
# ADD https://bootstrap.pypa.io/pip/2.7/get-pip.py
# RUN python3 -m venv myenv
# RUN source myenv/bin/activate