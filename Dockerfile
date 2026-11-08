# CPU-only image. The model is downloaded at run time into a cache volume; converted / quantised models are written to ./models.
FROM gcc:13-bookworm AS cpp
ARG ORT_VERSION=1.20.1
RUN set -e; arch="$(uname -m)"; \
    case "$arch" in x86_64) a=x64;; aarch64) a=aarch64;; *) echo "unsupported CPU $arch"; exit 1;; esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${a}-${ORT_VERSION}.tgz" | tar xz -C /opt; \
    mv /opt/onnxruntime-linux-${a}-${ORT_VERSION} /opt/ort
RUN apt-get update && apt-get install -y --no-install-recommends cmake && rm -rf /var/lib/apt/lists/*
COPY cpp/ /src/
RUN cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DORT_DIR=/opt/ort && cmake --build /build -j2

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PIP_NO_CACHE_DIR=1 HB_AUDIO=/usr/local/bin/hb_audio
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=cpp /opt/ort/lib /opt/ort/lib
COPY --from=cpp /build/hb_audio /usr/local/bin/hb_audio
COPY gender_bench.py .
ENV LD_LIBRARY_PATH=/opt/ort/lib
EXPOSE 8000 8001
CMD ["python", "gender_bench.py"]
