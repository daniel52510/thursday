#!/usr/bin/env bash
# One-time setup for retraining the "Hey THURSDAY" wake word on the Mac (CPU; no GPU needed).
# Everything lands in wake-train/ (git-ignored). About 20 GB of downloads, most of it
# openWakeWord's precomputed negative features (17.3 GB).
#
# Two venvs because the pieces disagree on numpy:
#   venv-gen  piper-sample-generator (numpy 2): synthetic "Hey THURSDAY" clips from 900 voices
#   venv      openWakeWord 0.6 trainer (speechbrain 0.5, acoustics: numpy 1): augment + train
# They only share WAV files.
#
#   scripts/wake_train/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
W=wake-train
PY=${PYTHON:-/opt/homebrew/bin/python3.12}
mkdir -p $W/data

[ -d $W/piper-sample-generator ] || git clone -q https://github.com/rhasspy/piper-sample-generator $W/piper-sample-generator
[ -d $W/openWakeWord ] || git clone -q --depth 1 --branch v0.6.0 https://github.com/dscripka/openWakeWord $W/openWakeWord

echo "venv-gen: sample generator"
[ -d $W/venv-gen ] || $PY -m venv $W/venv-gen
$W/venv-gen/bin/pip install -q --upgrade pip
$W/venv-gen/bin/pip install -q -e $W/piper-sample-generator

echo "venv: openWakeWord trainer"
[ -d $W/venv ] || $PY -m venv $W/venv
$W/venv/bin/pip install -q --upgrade pip
# torch 2.2.2: speechbrain 0.5 calls torchaudio.set_audio_backend, removed after 2.2.
$W/venv/bin/pip install -q "numpy<2" "scipy<1.14" "torch==2.2.2" "torchaudio==2.2.2" torchinfo torchmetrics \
    "speechbrain==0.5.16" "audiomentations==0.33.0" "torch-audiomentations==0.11.0" "acoustics==0.2.6" \
    mutagen pronouncing "pyarrow<18" soundfile pyyaml tqdm onnx onnxruntime
$W/venv/bin/pip install -q -e $W/openWakeWord
# The melspectrogram + speech-embedding models every openWakeWord model sits on.
$W/venv/bin/python -c "import openwakeword.utils as u; u.download_models(model_names=['_none_'])"

echo "datasets"
H=https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main
cd $W/data
curl -sSL -C - -o validation_set_features.npy $H/validation_set_features.npy
curl -sSL -C - -o en_US-libritts_r-medium.pt \
    https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt
cp ../piper-sample-generator/models/en_US-libritts_r-medium.pt.json .  # the generator reads it next to the .pt
F=openwakeword_features_ACAV100M_2000_hrs_16bit.npy
curl -sSL -C - -o $F $H/$F
# A broken or doubled download would silently feed the trainer garbage; the size is exact.
size=$(stat -f %z $F)
[ "$size" = 17280000128 ] || { echo "$F is $size bytes, expected 17280000128: delete it and re-run" >&2; exit 1; }
# Two AudioSet shards (~1.4 GB) as background noise; decoded to 16 kHz WAVs in the next slice.
for s in 00 01; do
    curl -sSL -C - -o audioset_bal_$s.parquet \
        https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train/$s.parquet
done
cd - >/dev/null

$W/venv-gen/bin/python -c "import piper, torch, audiomentations; print('venv-gen ok')"
$W/venv/bin/python -c "import openwakeword, speechbrain, acoustics, torch_audiomentations; print('venv ok')"
pip_freeze() { "$1/bin/pip" freeze --exclude-editable; }
pip_freeze $W/venv-gen > scripts/wake_train/requirements-gen.lock.txt
pip_freeze $W/venv > scripts/wake_train/requirements-train.lock.txt
echo "setup done"
