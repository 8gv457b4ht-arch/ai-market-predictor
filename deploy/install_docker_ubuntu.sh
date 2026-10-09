#!/usr/bin/env bash
# Ubuntu/Debian VPS bootstrap: Docker Engine + compose plugin, enabled at boot, basic firewall.
set -euo pipefail
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo systemctl enable --now docker containerd   # containers with restart: unless-stopped come back after reboot
sudo usermod -aG docker "$USER" || true
if command -v ufw >/dev/null; then
  sudo ufw allow OpenSSH
  sudo ufw allow 8000/tcp   # dashboard over HTTP (close it when you use the HTTPS profile)
  sudo ufw allow 80/tcp
  sudo ufw allow 443/tcp
  sudo ufw --force enable
fi
docker --version && docker compose version
echo "Log out and back in once so the docker group applies, then run ./scripts/start.sh"
