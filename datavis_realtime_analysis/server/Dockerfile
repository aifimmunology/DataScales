FROM python:3.12-slim

# gcloud CLI + ssh client: the backend drives GPU jobs via gcloud compute ssh
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg openssh-client \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./server

EXPOSE 8000

ENTRYPOINT ["/app/server/entrypoint.sh"]
