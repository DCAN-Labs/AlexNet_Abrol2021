FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure nibabel is installed:"
RUN python -c "import nibabel"

#INSTALL THESE OUTSIDE OF ENVIRONMENT.YML?
RUN pip install hypopt nipy pytorch-lightning

# Demostrate that pytorch-lightning package is installed
RUN echo "Make sure pytorch-lightning is installed"
RUN python -c "import pytorch_lightning as pl"

# The code to run when container is started:
RUN mkdir /code
COPY bin /code/bin
COPY data /code/data
COPY env /code/env
COPY results /code/results
COPY src /code/src
COPY README.md /code/README.md

# Add /code folder and sub-folders to PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/code:/code/bin:/code/data:/code/env:/code/results:/code/src"

RUN chmod 555 -R /code

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "/code/src/dcan/inference/infer.py"]
