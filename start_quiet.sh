#!/usr/bin/env bash
# Piper in do-not-disturb mode: no microphone, no speech (research, vision and the head still run).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start_piper.sh" --quiet "$@"
