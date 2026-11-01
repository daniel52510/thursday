#!/usr/bin/env bash
# Retrains "Hey THURSDAY" on the Mac and scores it against the current model.
# Needs scripts/wake_train/setup.sh first, and recordings in wake-data/
# (satellite/record_wake.py). Each step is skipped when its output exists;
# delete wake-train/run/<step> to redo one.
#
#   scripts/wake_train/train.sh
#
# THURSDAY is slower while this runs (CPU shared with Ollama/TTS): run it when idle.
# Every path is quoted: the repo lives under "GitHub Projects".
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
RUN="$ROOT/wake-train/run"
PY="$ROOT/wake-train/venv/bin/python"
GEN_PY="$ROOT/wake-train/venv-gen/bin/python"
VOICES="$ROOT/wake-train/data/en_US-libritts_r-medium.pt"
# The generator reads <model>.pt.json next to the model; it ships in the repo's models/.
[ -f "$VOICES.json" ] || cp "$ROOT/wake-train/piper-sample-generator/models/en_US-libritts_r-medium.pt.json" "$VOICES.json"
mkdir -p "$RUN/synth" "$RUN/rirs" "$RUN/stub"
t0=$(date +%s)
stamp() { echo "[$(( $(date +%s) - t0 ))s] $*"; }

# Mono 16 kHz room recordings only: speechbrain's reverb can't take stereo (the generator's
# own impulses/ folder is stereo 44.1 kHz studio effects, so it isn't used).
if [ -z "$(ls "$RUN/rirs")" ]; then
    stamp "room impulse responses (MIT, 270 files)"
    H=https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses
    curl -s "https://huggingface.co/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz" \
        | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin)]" \
        | while read -r f; do curl -sSL -o "$RUN/rirs/$(basename "$f")" "$H/resolve/main/$f"; done
fi

# Synthetic speech from ~800 LibriTTS voices at three speeds. Phrase variants change the
# prosody; near-miss phrases teach it what not to wake on.
gen() {  # gen <dir> <count> <text>
    [ -n "$(ls "$RUN/synth/$1" 2>/dev/null)" ] && return
    stamp "synthetic: $1 ($2 x \"$3\")"
    # Run from the repo: the .pt voice model needs its piper_train package.
    (cd "$ROOT/wake-train/piper-sample-generator" && "$GEN_PY" -m piper_sample_generator "$3" \
        --model "$VOICES" --max-samples "$2" --batch-size 50 --max-speakers 800 \
        --length-scales 0.85 1.0 1.15 --output-dir "$RUN/synth/$1" 2>&1 | grep -vE "Warning|warn" || true)
    [ -n "$(ls "$RUN/synth/$1" 2>/dev/null)" ] || { echo "no clips generated for $1" >&2; exit 1; }
}
gen pos_train_1 2500 "hey thursday"
gen pos_train_2 1500 "hey, thursday!"
gen pos_train_3 1000 "hey thursday?"
gen pos_test_1 500 "hey thursday"
i=0
for p in "thursday" "hey there" "hey thirsty" "hey siri" "hey google" "first day" "this thursday" \
         "hey tuesday" "on thursday" "hey thurston" "hey thirty" "they're thirsty" "hey, it's today" "thursday night"; do
    i=$((i + 1)); gen "neg_train_$i" 280 "$p"
done
gen neg_test_1 250 "hey tuesday"
gen neg_test_2 250 "this thursday"

stamp "assembling training folders"
"$PY" scripts/wake_train/prepare.py

echo 'def generate_samples(*args, **kwargs): raise RuntimeError("train.sh generates the clips")' \
    > "$RUN/stub/generate_samples.py"
# Paths quoted in the YAML for the same reason.
cat > "$RUN/hey_thursday.yml" <<EOF
model_name: hey_thursday
target_phrase: ["hey thursday"]
custom_negative_phrases: []
n_samples: 5000
n_samples_val: 500
tts_batch_size: 50
augmentation_batch_size: 16
piper_sample_generator_path: "$RUN/stub"
output_dir: "$RUN"
rir_paths: ["$RUN/rirs"]
background_paths: ["$RUN/background"]
background_paths_duplication_rate: [1]
false_positive_validation_data_path: "$ROOT/wake-train/data/validation_set_features.npy"
augmentation_rounds: 1
feature_data_files:
  ACAV100M_sample: "$ROOT/wake-train/data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
model_type: dnn
layer_size: 32
steps: 50000
max_negative_weight: 1500
target_false_positives_per_hour: 0.2
EOF

TRAIN="$ROOT/wake-train/openWakeWord/openwakeword/train.py"
# train.py skips augmenting when positive_features_train.npy exists, even a partial one
# from a crashed run; so redo (--overwrite) unless all four feature files are there.
F="$RUN/hey_thursday"
if [ -f "$F/positive_features_train.npy" ] && [ -f "$F/negative_features_train.npy" ] \
   && [ -f "$F/positive_features_test.npy" ] && [ -f "$F/negative_features_test.npy" ]; then
    stamp "features already computed"
else
    stamp "augmenting clips and computing features"
    (cd "$RUN" && "$PY" "$ROOT/scripts/wake_train/run_train.py" "$TRAIN" --training_config hey_thursday.yml \
        --augment_clips --overwrite)
fi
stamp "training"
# The last step converts to tflite with TensorFlow, which isn't installed (the satellite
# uses the .onnx), so that one error is expected once the .onnx exists.
(cd "$RUN" && "$PY" "$ROOT/scripts/wake_train/run_train.py" "$TRAIN" --training_config hey_thursday.yml --train_model) \
    || [ -f "$RUN/hey_thursday.onnx" ]
[ -f "$RUN/hey_thursday.onnx" ] || { echo "training produced no model" >&2; exit 1; }

stamp "old vs new on your held-out recordings"
"$PY" scripts/wake_train/evaluate.py "$RUN/hey_thursday.onnx"
stamp "done: $RUN/hey_thursday.onnx"
