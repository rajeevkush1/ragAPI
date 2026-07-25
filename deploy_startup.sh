#!/bin/bash
# ── RAG GPU VM PROVISIONING STARTUP SCRIPT ──────────────────────────────────

# Redirect stdout/stderr to a log file for debugging
exec > /var/log/startup-script-execution.log 2>&1
echo "Starting system provisioning at $(date)"

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

# 3. Install NVIDIA Drivers (Headless Server Driver)
echo "Installing NVIDIA Drivers..."
add-apt-repository -y ppa:graphics-drivers/ppa
apt-get update
apt-get install -y --no-install-recommends nvidia-driver-535-server

# 4. Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 5. Clone Repository
echo "Cloning codebase repository..."
mkdir -p /opt/rag-app
cd /opt/rag-app
git clone https://github.com/rajeevkush1/rag-advanced-research.git .

# 6. Uncomment GPU Configuration in docker-compose.yml
echo "Configuring docker-compose for GPU..."
python3 -c "
with open('docker-compose.yml', 'r') as f:
    lines = f.readlines()

new_lines = []
uncomment = False
for line in lines:
    if '# If running with GPU support (NVIDIA)' in line:
        uncomment = True
        new_lines.append(line)
        continue
    if uncomment:
        if line.strip().startswith('#'):
            new_lines.append(line.replace('# ', '', 1))
        else:
            uncomment = False
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('docker-compose.yml', 'w') as f:
    f.writelines(new_lines)
"

# 7. Create default .env file
echo "Configuring .env file..."
cp .env.example .env

# 8. Spin up Docker Compose stack
echo "Starting Docker Compose services..."
docker compose up -d --build

echo "System provisioning completed at $(date)"
