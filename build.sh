#!/usr/bin/env bash
# Build LiveTranslate and wrap it into a proper .app bundle so macOS
# treats it as a real app (entitlements, menu bar, permissions prompts).
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${CONFIG:-release}"
APP_NAME="LiveTranslate"
APP_DIR="build/${APP_NAME}.app"
SHERPA_LIB="external/sherpa-onnx/lib"
MODELS_DIR="build/sherpa-models"

# Download the sherpa-onnx dylib + all ONNX models (idempotent).
./tools/download-sherpa.sh

echo "→ swift build -c ${CONFIG}"
# Tell the linker where to find libsherpa-onnx-c-api.dylib and
# libonnxruntime.dylib at link time. They'll also be copied into
# Frameworks/ below so the dynamic loader finds them at runtime.
LIBRARY_PATH="${SHERPA_LIB}" swift build -c "${CONFIG}"

BIN_PATH="$(swift build -c "${CONFIG}" --show-bin-path)/${APP_NAME}"

rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"
mkdir -p "${APP_DIR}/Contents/Frameworks"

cp "${BIN_PATH}" "${APP_DIR}/Contents/MacOS/${APP_NAME}"
cp Info.plist "${APP_DIR}/Contents/Info.plist"

# Embed the Frameworks/ RPATH so the dynamic loader finds the sherpa dylibs.
install_name_tool -add_rpath "@executable_path/../Frameworks" \
    "${APP_DIR}/Contents/MacOS/${APP_NAME}" 2>/dev/null || true

# ── sherpa-onnx dylibs into Frameworks/ ─────────────────────────────────────
# Copy both the C API dylib and the ONNX Runtime dylib (a transitive dep of
# the C API dylib) into the bundle's Frameworks/ directory.  The binary was
# linked with -rpath @executable_path/../Frameworks so the dynamic loader
# picks them up from there at runtime.
for DYLIB in \
    "${SHERPA_LIB}/libsherpa-onnx-c-api.dylib" \
    "${SHERPA_LIB}/libonnxruntime.1.24.4.dylib"
do
    if [[ -f "${DYLIB}" ]]; then
        cp "${DYLIB}" "${APP_DIR}/Contents/Frameworks/"
        echo "  bundled $(basename "${DYLIB}")"
    else
        echo "WARNING: ${DYLIB} not found — run tools/download-sherpa.sh first"
    fi
done

# Create an unversioned symlink that the C API dylib may reference.
ONNX_VERSIONED="${APP_DIR}/Contents/Frameworks/libonnxruntime.1.24.4.dylib"
if [[ -f "${ONNX_VERSIONED}" ]]; then
    ln -sf "libonnxruntime.1.24.4.dylib" \
        "${APP_DIR}/Contents/Frameworks/libonnxruntime.dylib" 2>/dev/null || true
fi

# ── ONNX models into Resources/ ─────────────────────────────────────────────
# (Silero VAD was removed — replaced by energy + zero-cross-rate VAD in
# SherpaTranscriber.)

# German ASR model directory
ASR_DIR="${MODELS_DIR}/sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"
if [[ -d "${ASR_DIR}" ]]; then
    cp -R "${ASR_DIR}" "${APP_DIR}/Contents/Resources/"
else
    echo "WARNING: ASR model dir not found: ${ASR_DIR}"
fi

# kitten-mini TTS model directory
TTS_DIR="${MODELS_DIR}/kitten-mini-en-v0_8"
if [[ -d "${TTS_DIR}" ]]; then
    cp -R "${TTS_DIR}" "${APP_DIR}/Contents/Resources/"
else
    echo "WARNING: TTS model dir not found: ${TTS_DIR}"
fi

# ── App icon ─────────────────────────────────────────────────────────────────
./tools/make-icon.sh build/icon
cp build/icon/icon.icns "${APP_DIR}/Contents/Resources/icon.icns"

# ── Code-sign ────────────────────────────────────────────────────────────────
# By default ad-hoc (sign id "-"), which means every rebuild produces a fresh
# cdhash and macOS prompts for permissions again. Set
# `LIVETRANSLATE_SIGN_IDENTITY` to the name of a self-signed code-signing
# certificate in your login keychain (see README) to persist TCC grants.
SIGN_IDENTITY="${LIVETRANSLATE_SIGN_IDENTITY:--}"
codesign --force --deep --sign "${SIGN_IDENTITY}" "${APP_DIR}" >/dev/null
if [[ "${SIGN_IDENTITY}" != "-" ]]; then
    echo "  signed with identity: ${SIGN_IDENTITY}"
fi

echo "✓ built ${APP_DIR}"
echo "  run with: open ${APP_DIR}"
