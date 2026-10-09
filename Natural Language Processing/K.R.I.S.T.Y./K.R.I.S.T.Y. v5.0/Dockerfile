FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG ROS2_ENABLED=false
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV ROS2_ENABLED=${ROS2_ENABLED}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "$ROS2_ENABLED" = "true" ] ; then \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-humble-ros-base python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/* ; \
    fi

WORKDIR /opt/kristy_v5

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /opt/kristy_v5/checkpoints \
    && python3 -c "from module import generate_synthetic_manifest, KristyConfig; generate_synthetic_manifest('./manifest.json', 20, KristyConfig())"

HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import sqlite3; conn=sqlite3.connect('./checkpoints/kristy_webui_security.db'); conn.execute('SELECT 1'); conn.close()"

EXPOSE 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "--timeout", "120", "webui:app"]