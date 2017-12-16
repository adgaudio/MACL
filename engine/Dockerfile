# NOTE: This Dockerfile makes an unnecessarily large image, and should be
# broken up.
FROM ubuntu:latest

# Install miniconda, apt dependencies, and conda libraries
RUN apt-get update && apt-get install -y \
    wget ca-certificates bzip2 \
    libgl1 libglu1 libgtk2.0-0 libasound2 \
    less \
    g++ \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3.sh \
  && /bin/bash /miniconda3.sh -b -p /opt/conda \
  && rm /miniconda3.sh \
  && export PATH=/opt/conda/bin:$PATH
RUN export PATH=/opt/conda/bin:$PATH \
  && conda install -y \
    python=3.5 \
    numpy pyopengl pyopengl-accelerate ipython fontconfig pip mkl \
  && conda install -c conda-forge tensorflow keras \
  && pip install pyglet \
  && chmod -R o+r /opt/conda/
ENV PATH /opt/conda/bin:$PATH



# UID of the user should match that of host user for X11 socket
ARG UID=1000
RUN useradd -ms /bin/bash --uid ${UID} alex
ENV PYTHONPATH /app/:$PYTHONPATH
WORKDIR /app

#Testing
RUN true \
  && apt-get update \
  && apt-get install -y gcc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && pip install mypy \
  && conda install -y pytest
RUN echo '#!/usr/bin/env bash\npy.test engine/ $@ && mypy --ignore-missing-imports engine/' \
  | tee -a test ; chmod +x test

# Visualization libraries for plotting and debugging
RUN true \
  && apt-get update \
  && apt-get install -y libsm6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN conda install -y \
    pyqt ipython-qtconsole ipython-notebook \
    matplotlib seaborn graphviz \
  && mkdir -p /home/alex/.local/share/jupyter/ \
  && chown -R alex:alex /home/alex/

# TMP: temp deps to add above
RUN apt-get update && apt-get install -y openmpi-bin && apt-get clean
# install CNTK gpu version for python3.5
RUN pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.0-cp35-cp35m-linux_x86_64.whl
RUN pip install -U keras

USER alex
CMD python engine/examples.py
COPY . /app/engine/

# Visualization, part 2: suppresses annoying matplotlib warning msgs
RUN ipython --matplotlib -c '"1"' \
  && ln -s \
    ~/.cache/matplotlib/fontList.py3k.cache \
    ~/.cache/matplotlib/fontList.cache

ENV KERAS_BACKEND cntk
