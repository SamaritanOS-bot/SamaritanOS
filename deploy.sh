#!/bin/bash
# NullArchitect Bot - Oracle Cloud Server Setup
# Run: bash deploy.sh
set -e

echo "=== NullArchitect Server Setup ==="

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt update -qq
sudo apt install -y python3 python3-pip python3-venv git

# 2. Clone repo (or pull if exists)
APP_DIR="$HOME/AiBottester"
if [ -d "$APP_DIR" ]; then
    echo "[2/6] Updating existing repo..."
    cd "$APP_DIR"
    git pull
else
    echo "[2/6] Cloning repo..."
    git clone https://github.com/SamaritanOS-bot/SamaritanOS.git "$APP_DIR"
    cd "$APP_DIR"
fi

# 3. Python venv
echo "[3/6] Setting up Python environment..."
cd "$APP_DIR/backend"
python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements-server.txt

# 4. .env check
if [ ! -f .env ]; then
    echo "[4/6] WARNING: .env file missing!"
    echo "  You need to create backend/.env with:"
    echo "    LLAMA_API_KEY=your_openrouter_key"
    echo "    MOLTBOOK_API_BASE=https://moltbook.com/api/v1"
    echo "    MOLTBOOK_API_KEY=your_moltbook_key"
    echo "    MOLTBOOK_AGENT_NAME=NullArchitect"
    echo "    AGENT_LOOP_INTERVAL_SEC=1800"
    echo "    AGENT_MAX_DAILY_POSTS=48"
    echo ""
    echo "  Run: nano $APP_DIR/backend/.env"
    echo ""
else
    echo "[4/6] .env found."
fi

# 5. Install systemd service
echo "[5/6] Installing systemd service..."
sudo tee /etc/systemd/system/nullarchitect.service > /dev/null <<EOF
[Unit]
Description=NullArchitect Moltbook Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR/backend
ExecStart=$APP_DIR/backend/venv/bin/python agent_runner.py --loop
Restart=always
RestartSec=30
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nullarchitect

# 6. Auto-update cron (git pull + restart every 6 hours)
echo "[6/6] Setting up auto-update cron..."
CRON_CMD="cd $APP_DIR && git pull --quiet && sudo systemctl restart nullarchitect"
(crontab -l 2>/dev/null | grep -v "AiBottester"; echo "0 */6 * * * $CRON_CMD") | crontab -

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start nullarchitect"
echo "  Stop:    sudo systemctl stop nullarchitect"
echo "  Logs:    journalctl -u nullarchitect -f"
echo "  Status:  sudo systemctl status nullarchitect"
echo "  Update:  cd $APP_DIR && git pull && sudo systemctl restart nullarchitect"
echo ""
echo "Auto-update: Every 6 hours (git pull + restart)"
echo ""
if [ ! -f .env ]; then
    echo ">>> NEXT: Create .env file first, then start the service <<<"
fi
