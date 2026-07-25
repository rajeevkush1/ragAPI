#!/bin/bash
# ── RAG CPU VM PROVISIONING STARTUP SCRIPT ──────────────────────────────────

# Redirect stdout/stderr to a log file for debugging
exec > /var/log/startup-script-execution.log 2>&1
echo "Starting system provisioning (CPU Mode) at $(date)"

# 1. Update APT and install common dependencies
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 \
    git \
    software-properties-common

# 2. Install Docker
echo "Installing Docker..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# 3. Clone Repository
echo "Cloning codebase repository..."
mkdir -p /opt/rag-app
cd /opt/rag-app
git clone https://github.com/rajeevkush1/rag-advanced-research.git .

# 4. Create default .env file
echo "Configuring .env file..."
cp .env.example .env

# 5. Spin up Docker Compose stack
echo "Starting Docker Compose services (CPU mode)..."
docker compose up -d --build

echo "System provisioning (CPU Mode) completed at $(date)"
