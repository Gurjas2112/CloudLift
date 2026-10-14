# Hugging Face Space (Docker SDK, CPU). Same content as docker/Dockerfile.cpu, run as uid 1000.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CLOUDLIFT_DEVICE=cpu CLOUDLIFT_HOST=0.0.0.0 CLOUDLIFT_PORT=7860 GRADIO_ANALYTICS_ENABLED=False \
    HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /app
# libexpat1 is required by the GDAL bundled in the Rasterio wheel
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY --chown=user pyproject.toml README.md LICENSE ./
COPY --chown=user cloudlift ./cloudlift
COPY --chown=user configs ./configs
COPY --chown=user scripts ./scripts
COPY --chown=user assets ./assets
COPY --chown=user third_party ./third_party
RUN pip install --no-deps -e . && chown -R user /app
USER user

# Bake the SYNTHETIC hero scenes at build time so the demo starts instantly.
RUN python scripts/bake_demo.py

EXPOSE 7860
CMD ["python", "-m", "cloudlift.ui.console"]
