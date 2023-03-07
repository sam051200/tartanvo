FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
ENV TZ=Etc/UTC
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
RUN xargs apt install --yes --no-install-recommends < ${APT_DPDS}

# install python dependencies
COPY ./${PY_DPDS} ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir --requirement ${PY_DPDS}

# Clean up
RUN apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR /app

CMD ["bash"]
