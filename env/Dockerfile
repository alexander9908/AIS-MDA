# Multi-stage to keep image small
FROM mambaorg/micromamba:1.5.8 as base

# Create env from environment.yml
WORKDIR /workspace
COPY env/environment.yml /tmp/environment.yml
RUN micromamba create -y -n ais -f /tmp/environment.yml &&     micromamba clean --all --yes
SHELL ["micromamba", "run", "-n", "ais", "/bin/bash", "-lc"]

# Runtime stage
FROM mambaorg/micromamba:1.5.8
WORKDIR /workspace
COPY --from=base /opt/conda/envs/ais /opt/conda/envs/ais
COPY . /workspace
ENV MAMBA_DOCKERFILE_ACTIVATE=1
SHELL ["micromamba", "run", "-n", "ais", "/bin/bash", "-lc"]
CMD ["/bin/bash"]
