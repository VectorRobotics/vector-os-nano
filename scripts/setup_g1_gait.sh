#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Setup the G1 real-gait assets (campaign #5, DQ-9 approved 2026-06-12).
#
# Downloads the pretrained 12-DOF G1 walking policy (motion.pt, TorchScript,
# 144 KB) and the G1 MJCF model (scene.xml + g1_12dof.xml + 64 meshes, ~57 MB)
# from unitree_rl_gym (BSD-3, HangZhou YuShu TECHNOLOGY) at a PINNED commit
# into assets/g1_gait/ (gitignored — heavy assets never enter this repo).
# Idempotent: verified assets are left untouched.
set -euo pipefail

REPO_URL="https://github.com/unitreerobotics/unitree_rl_gym.git"
PIN="276801e46c5d433564f24658bac64f254b7d2d4b"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/assets/g1_gait"
POLICY_SHA_FILE="$DEST/.policy.sha256"

if [ -f "$DEST/motion.pt" ] && [ -f "$DEST/scene.xml" ] \
   && [ -f "$POLICY_SHA_FILE" ] \
   && sha256sum -c "$POLICY_SHA_FILE" --status 2>/dev/null; then
    echo "g1_gait assets already present and verified: $DEST"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "fetching unitree_rl_gym @ ${PIN:0:12} ..."
git clone --quiet --depth 1 "$REPO_URL" "$TMP/rl_gym"
git -C "$TMP/rl_gym" fetch --quiet --depth 1 origin "$PIN"
git -C "$TMP/rl_gym" checkout --quiet "$PIN"

mkdir -p "$DEST"
cp "$TMP/rl_gym/deploy/pre_train/g1/motion.pt" "$DEST/"
cp -r "$TMP/rl_gym/resources/robots/g1_description/." "$DEST/"
cp "$TMP/rl_gym/LICENSE" "$DEST/LICENSE.unitree_rl_gym"
( cd "$DEST" && sha256sum motion.pt > "$POLICY_SHA_FILE" )

echo "g1_gait assets installed: $DEST"
ls "$DEST" | head -5
