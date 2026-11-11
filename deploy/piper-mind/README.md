# piper-mind: Piper's research mind on a Proxmox VM

The research loop, its memory (SQLite + embeddings) and the dashboard run here; the Orin keeps the
real-time work (voice, vision, the head). Models and embeddings stay on the PC's Ollama (`llm.base_url`).

VM: Ubuntu Server 24.04 LTS, 2 vCPU, 4 GB, 32 GB, fixed IP (DHCP reservation), timezone = the Orin's
(the research rhythm runs on the local clock).

## Set up

    sudo apt install -y git python3-venv python3-pip sqlite3 qemu-guest-agent
    git clone https://github.com/stvenmobile/piper_assistant.git && cd piper_assistant
    python3 -m venv .venv && .venv/bin/pip install -r requirements-mind.txt

Copy `.env` from the Orin (OPENALEX_API_KEY) and `config.yaml` if it has overrides.

## Move the memory from the Orin

1. On the Orin: stop Piper (research must not be writing), then make a consistent copy:
   `sqlite3 data/memory/piper_memory.db ".backup /tmp/piper_memory.db"`
2. Copy it, and the research folder (wiki cache, reports), to piper-mind:
   `scp /tmp/piper_memory.db steve@piper-mind:piper_assistant/data/memory/`
   `scp -r data/research steve@piper-mind:piper_assistant/data/`
3. On the Orin, turn research and the dashboard off so there is only one research mind - in its `.env`
   (per machine; config.yaml is shared through git):
   `PIPER_RESEARCH_ENABLED=false` and `PIPER_DASHBOARD_ENABLED=false`

## Services

    sudo cp deploy/piper-mind/piper-research.service deploy/piper-mind/piper-dashboard.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now piper-research piper-dashboard
    journalctl -u piper-research -f        # its log

Dashboard: http://piper-mind:8080/ (home network only).
