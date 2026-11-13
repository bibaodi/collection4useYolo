# On Debian 13
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Create project
mkdir -p ~/mcp-demo && cd ~/mcp-demo
python3 -m venv .venv
source .venv/bin/activate

pip install mcp

