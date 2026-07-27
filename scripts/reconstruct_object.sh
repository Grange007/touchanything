#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAME="touchanything-camera"
DATA_ROOT="examples/data/record_printed_camera_sample20"
JSON_FILE="sample_20_noaxis_8.json"
DATA_ROOT_STAGE2=""
JSON_FILE_STAGE2=""
PROMPT="a camera"
CONFIG_STAGE1="configs/touchanything/stage1_real.yaml"
CONFIG_STAGE2="configs/touchanything/stage2_real.yaml"
OUTPUT_DIR="outputs/touchanything"
MESH_RESOLUTION=256
USE_WANDB=true
MAX_STEPS=""
CONVERT_MESH_SCALE=""
ELEVATION_RANGE=""
DRY_RUN=false

if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${PYTHONWARNINGS},ignore::FutureWarning,ignore:pkg_resources is deprecated as an API:UserWarning"
else
    export PYTHONWARNINGS="ignore::FutureWarning,ignore:pkg_resources is deprecated as an API:UserWarning"
fi

show_help() {
    cat <<'EOF'
Usage: scripts/reconstruct_object.sh [options]

Run the two-stage TouchAnything reconstruction pipeline for one object.

Options:
  -n, --name NAME              Experiment name prefix.
  -d, --data-root DIR          Record directory containing rgb/depth/normal/mask files.
  -j, --json FILE              Metadata JSON inside the record directory.
  -p, --prompt TEXT            Text prompt for diffusion guidance.
  -c1, --config-stage1 FILE    Stage-1 config.
  -c2, --config-stage2 FILE    Stage-2 config.
  -o, --output-dir DIR         Output root directory.
  -r, --resolution NUM         Mesh export resolution.
  -w, --wandb                  Enable Weights & Biases logging.
      --data-root-stage2 DIR   Optional stage-2 record directory.
      --json-stage2 FILE       Optional stage-2 metadata JSON.
      --convert-mesh-scale VAL Override stage-2 mesh conversion scale.
      --elevation-range RANGE  Override random camera elevation range, e.g. "[-15,30]".
      --max-steps NUM          Override trainer.max_steps for both stages.
      --dry-run                Print commands without running them.
  -h, --help                   Show this help message.

Example:
  bash scripts/reconstruct_object.sh \
    --data-root examples/data/record_printed_camera_sample20 \
    --json sample_20_noaxis_8.json \
    --prompt "a camera"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)
            NAME="$2"; shift 2 ;;
        -d|--data-root)
            DATA_ROOT="$2"; shift 2 ;;
        -j|--json)
            JSON_FILE="$2"; shift 2 ;;
        -p|--prompt)
            PROMPT="$2"; shift 2 ;;
        -c1|--config-stage1)
            CONFIG_STAGE1="$2"; shift 2 ;;
        -c2|--config-stage2)
            CONFIG_STAGE2="$2"; shift 2 ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -r|--resolution)
            MESH_RESOLUTION="$2"; shift 2 ;;
        -w|--wandb)
            USE_WANDB=true; shift ;;
        --data-root-stage2)
            DATA_ROOT_STAGE2="$2"; shift 2 ;;
        --json-stage2)
            JSON_FILE_STAGE2="$2"; shift 2 ;;
        --convert-mesh-scale)
            CONVERT_MESH_SCALE="$2"; shift 2 ;;
        --elevation-range)
            ELEVATION_RANGE="$2"; shift 2 ;;
        --max-steps)
            MAX_STEPS="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            show_help; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            show_help
            exit 1 ;;
    esac
done

abspath_under_root() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s/%s\n' "$PROJECT_ROOT" "$value"
    fi
}

run_step() {
    local label="$1"
    shift

    printf '\n==> %s\n' "$label"
    if [[ "$DRY_RUN" == true ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
        return
    fi
    "$@"
}

CONFIG_STAGE1="$(abspath_under_root "$CONFIG_STAGE1")"
CONFIG_STAGE2="$(abspath_under_root "$CONFIG_STAGE2")"
DATA_ROOT="$(abspath_under_root "$DATA_ROOT")"
OUTPUT_DIR="$(abspath_under_root "$OUTPUT_DIR")"

if [[ -z "$DATA_ROOT_STAGE2" ]]; then
    DATA_ROOT_STAGE2="$DATA_ROOT"
else
    DATA_ROOT_STAGE2="$(abspath_under_root "$DATA_ROOT_STAGE2")"
fi
if [[ -z "$JSON_FILE_STAGE2" ]]; then
    JSON_FILE_STAGE2="$JSON_FILE"
fi

if [[ ! -f "$CONFIG_STAGE1" ]]; then
    echo "Missing stage-1 config: $CONFIG_STAGE1" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_STAGE2" ]]; then
    echo "Missing stage-2 config: $CONFIG_STAGE2" >&2
    exit 1
fi
if [[ ! -f "$DATA_ROOT/$JSON_FILE" ]]; then
    echo "Missing metadata JSON: $DATA_ROOT/$JSON_FILE" >&2
    exit 1
fi
if [[ ! -f "$DATA_ROOT_STAGE2/$JSON_FILE_STAGE2" ]]; then
    echo "Missing stage-2 metadata JSON: $DATA_ROOT_STAGE2/$JSON_FILE_STAGE2" >&2
    exit 1
fi

PROMPT_HYDRA="$(python3 - "$PROMPT" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
)"

EXPERIMENT_NAME="${NAME}-$(date +%Y%m%d-%H%M)"
TAG_STAGE1="stage1-geo-neus"
TAG_STAGE1_EXPORT="stage1-export-mesh"
TAG_STAGE2="stage2-refine-dmtetra"
TAG_STAGE2_EXPORT="stage2-export-mesh"
STAGE1_CKPT_PATH="$OUTPUT_DIR/$EXPERIMENT_NAME/$TAG_STAGE1/ckpts/last.ckpt"
STAGE1_PARSED_CONFIG="$OUTPUT_DIR/$EXPERIMENT_NAME/$TAG_STAGE1/configs/parsed.yaml"
STAGE2_CKPT_PATH="$OUTPUT_DIR/$EXPERIMENT_NAME/$TAG_STAGE2/ckpts/last.ckpt"
STAGE2_PARSED_CONFIG="$OUTPUT_DIR/$EXPERIMENT_NAME/$TAG_STAGE2/configs/parsed.yaml"

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$OUTPUT_DIR"
fi
cd "$PROJECT_ROOT"

printf 'TouchAnything reconstruction\n'
printf '  data:   %s/%s\n' "$DATA_ROOT" "$JSON_FILE"
printf '  prompt: %s\n' "$PROMPT"
printf '  output: %s/%s\n' "$OUTPUT_DIR" "$EXPERIMENT_NAME"

STAGE1_ARGS=(
    python3 launch.py
    --config "$CONFIG_STAGE1"
    --train
    use_timestamp=False
    exp_root_dir="$OUTPUT_DIR"
    name="$EXPERIMENT_NAME"
    tag="$TAG_STAGE1"
    data.root_dir="$DATA_ROOT"
    data.json_path="$JSON_FILE"
    system.prompt_processor.prompt="$PROMPT_HYDRA"
    system.loggers.wandb.enable="$USE_WANDB"
    system.loggers.wandb.project="touchanything"
)

if [[ -n "$MAX_STEPS" ]]; then
    STAGE1_ARGS+=("trainer.max_steps=$MAX_STEPS")
fi
if [[ -n "$ELEVATION_RANGE" ]]; then
    STAGE1_ARGS+=("data.random_camera.elevation_range=$ELEVATION_RANGE")
fi

run_step "[1/4] Stage 1 training" "${STAGE1_ARGS[@]}"

if [[ "$DRY_RUN" == false && ! -f "$STAGE1_CKPT_PATH" ]]; then
    echo "Stage-1 checkpoint was not created: $STAGE1_CKPT_PATH" >&2
    exit 1
fi

run_step "[2/4] Stage 1 mesh export" \
    python3 launch.py \
    --config "$STAGE1_PARSED_CONFIG" \
    --export \
    resume="$STAGE1_CKPT_PATH" \
    system.guidance_type=none \
    system.exporter_type=mesh-exporter \
    system.geometry.isosurface_resolution="$MESH_RESOLUTION" \
    system.exporter.context_type=cuda \
    exp_root_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME" \
    tag="$TAG_STAGE1_EXPORT"

STAGE2_ARGS=(
    python3 launch.py
    --config "$CONFIG_STAGE2"
    --train
    use_timestamp=False
    exp_root_dir="$OUTPUT_DIR"
    name="$EXPERIMENT_NAME"
    tag="$TAG_STAGE2"
    data.root_dir="$DATA_ROOT_STAGE2"
    data.json_path="$JSON_FILE_STAGE2"
    system.prompt_processor.prompt="$PROMPT_HYDRA"
    system.geometry_convert_from="$STAGE1_CKPT_PATH"
    system.loggers.wandb.enable="$USE_WANDB"
    system.loggers.wandb.project="touchanything"
)

if [[ -n "$MAX_STEPS" ]]; then
    STAGE2_ARGS+=("trainer.max_steps=$MAX_STEPS")
fi
if [[ -n "$CONVERT_MESH_SCALE" ]]; then
    STAGE2_ARGS+=("system.geometry.convert_mesh_scale=$CONVERT_MESH_SCALE")
fi
if [[ -n "$ELEVATION_RANGE" ]]; then
    STAGE2_ARGS+=("data.random_camera.elevation_range=$ELEVATION_RANGE")
fi

run_step "[3/4] Stage 2 training" "${STAGE2_ARGS[@]}"

if [[ "$DRY_RUN" == false && ! -f "$STAGE2_CKPT_PATH" ]]; then
    echo "Stage-2 checkpoint was not created: $STAGE2_CKPT_PATH" >&2
    exit 1
fi

run_step "[4/4] Stage 2 mesh export" \
    python3 launch.py \
    --config "$STAGE2_PARSED_CONFIG" \
    --export \
    resume="$STAGE2_CKPT_PATH" \
    system.guidance_type=none \
    system.exporter_type=mesh-exporter \
    system.geometry.isosurface_resolution="$MESH_RESOLUTION" \
    system.exporter.context_type=cuda \
    exp_root_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME" \
    tag="$TAG_STAGE2_EXPORT"

printf '\nDone. Outputs: %s/%s\n' "$OUTPUT_DIR" "$EXPERIMENT_NAME"
