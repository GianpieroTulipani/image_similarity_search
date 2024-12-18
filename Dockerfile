FROM continuumio/miniconda3:24.9.2-0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONDA_AUTO_UPDATE_CONDA=0

COPY environment.yml /tmp/environment.yml

RUN conda init bash \
    && . ~/.bashrc \
    && conda env update --name image_similarity_search --file /tmp/environment.yml --prune \
    && conda clean --all --yes \
    && conda activate image_similarity_search

WORKDIR /image_similarity_search
COPY pyproject.toml .
COPY README.md .

COPY models ./models
COPY data/interim/dataset.csv data/interim/dataset.csv


WORKDIR /image_similarity_search/image_similarity_search
COPY image_similarity_search/ .
EXPOSE 55000


WORKDIR /image_similarity_search/
RUN pip install -e .

# Use bash as the shell to enable sourcing .bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV LD_LIBRARY_PATH=/opt/conda/envs/image_similarity_search/lib:/opt/conda/lib/opt/conda/lib
# Set the entrypoint to activate the Conda environment and run the FastAPI app
ENTRYPOINT ["bash", "-c", "source ~/.bashrc && conda activate image_similarity_search && fastapi run image_similarity_search/api/main.py --port 55000"]

