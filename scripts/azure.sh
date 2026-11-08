#!/usr/bin/env bash
# Run the campaign and the live site on an Azure VM, for when people are watching.
#
#   scripts/azure.sh up       create the VM (first time) or start it again
#   scripts/azure.sh push     copy runs/campaign from here to the VM and start training
#   scripts/azure.sh status   what the VM and the campaign are doing
#   scripts/azure.sh pull     stop training on the VM and copy runs/campaign back here
#   scripts/azure.sh down     pull, then stop the VM (compute billing stops; the disk stays)
#   scripts/azure.sh delete   remove everything, disk included
#
# One VM (4 vCPUs, the most an Azure for Students region allows), Caddy for
# HTTPS at https://$NAME.$REGION.cloudapp.azure.com, and two systemd services.
# Training runs at low priority so the site's two channels keep near 15 decisions/s.
# Only one machine should train at a time: stop the campaign here before `push`.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${SMARTFLY_DNS:-smartfly-live}"
REGION="${SMARTFLY_REGION:-centralindia}"
SIZE="${SMARTFLY_SIZE:-Standard_D4as_v4}"
GROUP=smartfly
VM=smartfly
USER_=fly
HOST="$NAME.$REGION.cloudapp.azure.com"
REPO=https://github.com/Puneeth-RV/Smart-Fly.git
KEY="${SMARTFLY_KEY:-$HOME/.ssh/id_ed25519}"
RSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"
SSH=($RSH "$USER_@$HOST")

cloud_init() {
  cat <<EOF
#cloud-config
package_update: true
packages: [git, build-essential, python3-venv, python3-dev, caddy, rsync]
write_files:
  - path: /etc/smartfly/Caddyfile
    content: |
      $HOST {
        encode gzip
        reverse_proxy 127.0.0.1:8090 {
          flush_interval -1
        }
      }
  - path: /etc/systemd/system/smartfly-site.service
    content: |
      [Unit]
      Description=Smart-Fly live site
      After=network-online.target
      [Service]
      User=$USER_
      WorkingDirectory=/home/$USER_/Smart-Fly
      ExecStart=/home/$USER_/Smart-Fly/.venv/bin/python -u -m smartfly.web.server
      Nice=-5
      CPUWeight=1000
      Restart=always
      RestartSec=5
      [Install]
      WantedBy=multi-user.target
  - path: /etc/systemd/system/smartfly-campaign.service
    content: |
      [Unit]
      Description=Smart-Fly campaign
      After=network-online.target
      [Service]
      User=$USER_
      WorkingDirectory=/home/$USER_/Smart-Fly
      ExecStart=/home/$USER_/Smart-Fly/.venv/bin/python -u -m smartfly.campaign
      StandardOutput=append:/home/$USER_/Smart-Fly/runs/campaign/campaign.log
      StandardError=inherit
      Nice=19
      CPUWeight=10
      Restart=on-failure
      RestartSec=30
      [Install]
      WantedBy=multi-user.target
runcmd:
  - sudo -u $USER_ git clone --depth 1 $REPO /home/$USER_/Smart-Fly
  - sudo -u $USER_ mkdir -p /home/$USER_/Smart-Fly/runs/campaign
  - cd /home/$USER_/Smart-Fly && sudo -u $USER_ python3 -m venv .venv
  - cd /home/$USER_/Smart-Fly && sudo -u $USER_ .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  - cd /home/$USER_/Smart-Fly && sudo -u $USER_ .venv/bin/pip install -q -e .
  - cd /home/$USER_/Smart-Fly && sudo -u $USER_ .venv/bin/python -m smartfly.compat
  - cp /etc/smartfly/Caddyfile /etc/caddy/Caddyfile
  - systemctl daemon-reload
  - systemctl enable --now smartfly-site
  - systemctl restart caddy
EOF
}

state() { az vm get-instance-view -g "$GROUP" -n "$VM" --query "instanceView.statuses[1].code" -o tsv 2>/dev/null; }

up() {
  if [[ -z "$(az vm show -g "$GROUP" -n "$VM" --query id -o tsv 2>/dev/null)" ]]; then
    echo "creating $SIZE in $REGION (takes a few minutes)"
    az group create -n "$GROUP" -l "$REGION" -o none
    az vm create -g "$GROUP" -n "$VM" -l "$REGION" --size "$SIZE" \
      --image Ubuntu2404 --admin-username "$USER_" --ssh-key-values "$KEY.pub" \
      --os-disk-size-gb 32 --storage-sku StandardSSD_LRS \
      --public-ip-sku Standard --public-ip-address-dns-name "$NAME" \
      --custom-data <(cloud_init) -o none
    az vm open-port -g "$GROUP" -n "$VM" --port 80,443 --priority 900 -o none
  elif [[ "$(state)" != "PowerState/running" ]]; then
    echo "starting the VM"
    az vm start -g "$GROUP" -n "$VM" -o none
  fi
  echo "waiting for setup to finish"
  until "${SSH[@]}" true 2>/dev/null; do sleep 10; done
  "${SSH[@]}" "cloud-init status --wait >/dev/null; cd Smart-Fly && git pull -q --ff-only"
  echo "up: https://$HOST/"
}

push() {
  if pgrep -f smartfly.campaign >/dev/null; then
    echo "the campaign is running on this machine; stop it first" >&2
    exit 1
  fi
  "${SSH[@]}" "sudo systemctl stop smartfly-campaign"
  rsync -az --delete -e "$RSH" runs/campaign/ "$USER_@$HOST:Smart-Fly/runs/campaign/"
  "${SSH[@]}" "sudo systemctl enable --now smartfly-campaign && sudo systemctl restart smartfly-site"
  echo "training on the VM: https://$HOST/"
}

pull() {
  "${SSH[@]}" "sudo systemctl disable --now smartfly-campaign"
  rm -rf runs/campaign-before-pull
  cp -R runs/campaign runs/campaign-before-pull
  rsync -az --delete -e "$RSH" "$USER_@$HOST:Smart-Fly/runs/campaign/" runs/campaign/
  echo "pulled into runs/campaign (the previous copy is in runs/campaign-before-pull)"
}

status() {
  echo "VM: $(state)  https://$HOST/"
  "${SSH[@]}" "systemctl is-active smartfly-site smartfly-campaign caddy | paste -sd ' '; \
    tail -n 2 Smart-Fly/runs/campaign/campaign.log; \
    grep '^update' Smart-Fly/runs/campaign/train.log | tail -n 1" 2>/dev/null || true
}

down() {
  if [[ "$(state)" == "PowerState/running" ]]; then
    pull
  fi
  az vm deallocate -g "$GROUP" -n "$VM" -o none
  echo "stopped; the disk and address are kept (about \$6 a month), \`up\` starts it again"
}

delete() {
  az group delete -n "$GROUP" --yes
}

case "${1:-}" in
  up | push | pull | status | down | delete) "$1" ;;
  *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
