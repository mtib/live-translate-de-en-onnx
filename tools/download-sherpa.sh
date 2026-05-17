#!/usr/bin/env bash
# Downloads sherpa-onnx arm64 shared dylib + all ONNX models needed for
# the German→English pipeline:
#   - Silero-VAD
#   - German streaming zipformer ASR (sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06)
#   - campplus speaker embedding (English-compatible, 28 MB)
#   - kitten-mini-en-v0_8 TTS
#
# Idempotent — skips anything already on disk.
set -euo pipefail
cd "$(dirname "$0")/.."

SHERPA_VERSION="1.13.2"
SHERPA_DIR="external/sherpa-onnx"
MODELS_DIR="build/sherpa-models"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

mkdir -p "${MODELS_DIR}"

# ── dylib ────────────────────────────────────────────────────────────────────
if [[ ! -f "${SHERPA_DIR}/lib/libsherpa-onnx-c-api.dylib" ]]; then
  echo "→ downloading sherpa-onnx ${SHERPA_VERSION} osx-arm64 shared"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/sherpa.tar.bz2" \
    "${BASE}/v${SHERPA_VERSION}/sherpa-onnx-v${SHERPA_VERSION}-osx-arm64-shared.tar.bz2"
  mkdir -p "$(dirname "${SHERPA_DIR}")"
  tar -xjf "${TMP}/sherpa.tar.bz2" -C "$(dirname "${SHERPA_DIR}")"
  # rename extracted dir
  EXTRACTED=$(find "$(dirname "${SHERPA_DIR}")" -maxdepth 1 -name "sherpa-onnx-v*-osx-arm64-shared" -type d | head -1)
  if [[ -n "${EXTRACTED}" && "${EXTRACTED}" != "${SHERPA_DIR}" ]]; then
    mv "${EXTRACTED}" "${SHERPA_DIR}"
  fi
  rm -rf "${TMP}"
  echo "✓ sherpa-onnx dylib"
else
  echo "✓ sherpa-onnx dylib already present"
fi

# ── Silero-VAD ───────────────────────────────────────────────────────────────
VAD_MODEL="${MODELS_DIR}/silero_vad.onnx"
if [[ ! -f "${VAD_MODEL}" ]]; then
  echo "→ downloading silero_vad.onnx"
  curl -L --fail --progress-bar \
    -o "${VAD_MODEL}" \
    "${BASE}/asr-models/silero_vad.onnx"
  echo "✓ silero_vad.onnx"
else
  echo "✓ silero_vad.onnx already present"
fi

# ── German streaming ASR ─────────────────────────────────────────────────────
DE_MODEL_NAME="sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"
DE_DIR="${MODELS_DIR}/${DE_MODEL_NAME}"
if [[ ! -d "${DE_DIR}" ]]; then
  echo "→ downloading German streaming zipformer (${DE_MODEL_NAME})"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/asr-de.tar.bz2" \
    "${BASE}/asr-models/${DE_MODEL_NAME}.tar.bz2"
  tar -xjf "${TMP}/asr-de.tar.bz2" -C "${MODELS_DIR}"
  rm -rf "${TMP}"
  echo "✓ German ASR: ${DE_DIR}/"
else
  echo "✓ German ASR model already present"
fi

# ── campplus speaker embedding (English-compatible, 28 MB) ──────────────────
SPKR_MODEL="${MODELS_DIR}/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
if [[ ! -f "${SPKR_MODEL}" ]]; then
  echo "→ downloading campplus speaker embedding"
  curl -L --fail --progress-bar \
    -o "${SPKR_MODEL}" \
    "${BASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
  echo "✓ campplus speaker model"
else
  echo "✓ campplus speaker model already present"
fi

# ── kitten-mini TTS ──────────────────────────────────────────────────────────
KITTEN_DIR="${MODELS_DIR}/kitten-mini-en-v0_8"
if [[ ! -d "${KITTEN_DIR}" ]]; then
  echo "→ downloading kitten-mini-en-v0_8 TTS model"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/tts.tar.bz2" \
    "${BASE}/tts-models/kitten-mini-en-v0_8.tar.bz2"
  tar -xjf "${TMP}/tts.tar.bz2" -C "${MODELS_DIR}"
  rm -rf "${TMP}"
  echo "✓ kitten-mini TTS: ${KITTEN_DIR}/"
else
  echo "✓ kitten-mini TTS already present"
fi

echo ""
echo "All sherpa-onnx assets ready."
echo "Next: ./build.sh"
