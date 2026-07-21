#!/usr/bin/env bash
# Repeatable RoboCasa camera scan -> EmbodiedAgent memory validation workflow.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(dirname -- "$REPO_ROOT")"
SPATIAL_ENV_FILE="${SPATIAL_ENV_FILE:-$REPO_ROOT/docker/.env}"

# Keep local credentials out of command history and Git. This mirrors
# docker_run.sh, while still allowing an explicitly exported variable to be
# used when no local env file exists. Values loaded from docker/.env take
# precedence over stale variables in the invoking shell.
if [[ -f "$SPATIAL_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$SPATIAL_ENV_FILE"
    set +a
fi

COMPOSE_FILE="$REPO_ROOT/docker/docker-compose.yml"
MEMORY_DOCKERFILE="$REPO_ROOT/docker/SpatialMemoryDockerfile"
EMBODIED_ROOT="${EMBODIED_AGENT_ROOT:-$WORKSPACE_ROOT/EmbodiedAgent}"
SCAN_ROOT="${SPATIAL_SCAN_ROOT:-$REPO_ROOT/container_cache/spatial_memory_scans}"
BENCHMARK_ROOT="${SPATIAL_BENCHMARK_ROOT:-$REPO_ROOT/container_cache/spatial_memory_benchmarks}"
HF_CACHE="${SPATIAL_HF_CACHE:-$REPO_ROOT/container_cache/huggingface}"
MEMORY_IMAGE="${SPATIAL_MEMORY_IMAGE:-hams_spatial_memory:latest}"
MEMORY_RUNTIME_SCHEMA=2

TASK=Kitchen
LAYOUT=9
STYLE=9
SEED=42
MAX_POSITIONS=16
HEADINGS="0,90,180,270"
IMAGE_SIZE=512
CAMERA_HEIGHT=1.25
PITCH=-12
SPACING=0.5
FOOTPRINT_RADIUS=0.30
MODEL=siglip_base
DEVICE=auto
TOP_K=5
LIVE_INTERVAL=3
LIVE_DURATION=24
LIVE_SESSION=""
EXPECT_LIVE_SESSION=""
RERANK_RECALL_K=12
RERANK_PERMUTATIONS=3
VLM_MODEL="${SPATIAL_VLM_MODEL:-gemini-3.5-flash}"
RERANK_REPLAY=""
RERANK_CASES=()
RERANK_OPTIONS_USED=0
RERANK_CASE_OPTIONS_USED=0
VLM_MODEL_OPTION_USED=0
SCAN_NAME=""
SCAN_NAME_EXPLICIT=0
REPLACE=0
REBUILD=0
FORCE_BUILD=0
QUERIES=()
BENCHMARK_NAME=""
BENCHMARK_EPISODES=12
BENCHMARK_OBJECTS="mug,bowl"
BENCHMARK_ROUTE_POINTS=8
BENCHMARK_CAPTURE_INTERVAL=2
BENCHMARK_ADAPTER="embodied_agent"
BENCHMARK_ADAPTER_KWARGS=""
BENCHMARK_ADAPTER_OPTION_USED=0
BENCHMARK_ADAPTER_KWARGS_OPTION_USED=0
BENCHMARK_MAX_EPISODES=""
BENCHMARK_KEEP_STATE=0
BENCHMARK_COMPARE_ADAPTERS="latest_only,embodied_agent,embodied_agent_recency"

usage() {
    cat <<'EOF'
RoboCasa camera-only spatial-memory validation

Usage:
  docker/scripts/spatial_memory_camera.sh all [options]
  docker/scripts/spatial_memory_camera.sh scan [options]
  docker/scripts/spatial_memory_camera.sh live [options]
  docker/scripts/spatial_memory_camera.sh memory [options]
  docker/scripts/spatial_memory_camera.sh rerank-setup [options]
  docker/scripts/spatial_memory_camera.sh rerank [options]
  docker/scripts/spatial_memory_camera.sh benchmark [options]
  docker/scripts/spatial_memory_camera.sh benchmark-setup [options]
  docker/scripts/spatial_memory_camera.sh benchmark-eval [options]
  docker/scripts/spatial_memory_camera.sh benchmark-suite [options]
  docker/scripts/spatial_memory_camera.sh benchmark-compare [options]
  docker/scripts/spatial_memory_camera.sh build
  docker/scripts/spatial_memory_camera.sh where [options]
  docker/scripts/spatial_memory_camera.sh benchmark-where [options]

Commands:
  all      replace the selected scan, then rebuild and query its memory index
  scan     generate only RGB + x/y/yaw (add --replace to replace an existing scan)
  live     append a timed closed/open-fridge session, then index only its new frames
  memory   import/index/query an existing scan (add --rebuild for a clean index)
  rerank-setup  create a dedicated 64-frame + 9-frame frozen benchmark dataset
  rerank   compare frozen FAISS candidates with opt-in Gemini VLM reranking
  benchmark  generate object-relocation episodes, then run streaming evaluation
             (add --replace to replace an existing selected dataset and its reports)
  benchmark-setup  generate only the algorithm-neutral two-lap episodes
  benchmark-eval   evaluate an existing dataset through a selected adapter
  benchmark-suite  run three baselines + Gemini, then compare all four
  benchmark-compare compare latest matching runs on identical episodes
  build    build the small memory-only Docker image
  where    print the selected host scan directory
  benchmark-where  print the selected host benchmark directory

Scene options:
  --task NAME             default: Kitchen (fixtures only, no task objects)
  --layout ID             default: 9
  --style ID              default: 9
  --seed N                default: 42
  --scan-name NAME        override the generated dataset directory name
  --positions N           camera positions, four headings each (default: 16)
  --headings DEG,...      default: 0,90,180,270
  --image-size PX         default: 512
  --replace               replace only scanner-owned files in this scan

Live options:
  --interval SEC          wall-clock seconds between captures (default: 3)
  --duration SEC          capture span from first through final deadline (default: 24)
  --session-id NAME       optional stable session name
  --validate-session NAME validate an already captured live session during memory import

Camera options:
  --camera-height M       default: 1.25
  --pitch DEG             default: -12
  --spacing M             candidate-grid spacing (default: 0.5)
  --footprint-radius M    floor-clearance radius (default: 0.30)

Memory options:
  --model NAME            default: siglip_base
  --device auto|cpu|cuda  default: auto
  --query TEXT            repeatable; defaults: refrigerator, sink, stove
  --top-k N               default: 5
  --rebuild               replace generated metadata and FAISS index
  --force-build           rebuild the memory image even if it exists

VLM rerank options:
  --recall-k N            FAISS recall depth before VLM/recency (default: 12)
  --vlm-model NAME        default: gemini-3.5-flash
  --case NAME             repeatable controlled case selector (default: all)
  --permutations N        candidate-order trials per case (default: 3)
  --response-replay PATH  test-only replay JSON path inside the memory container

Object-relocation benchmark options:
  --benchmark-name NAME   dataset directory name (generated by default)
  --episodes N            episode count (default: 12)
  --objects GROUP,...     RoboCasa object groups (default: mug,bowl)
  --route-points N        observations per camera lap (default: 8)
  --capture-interval SEC  synthetic sensor interval (default: 2)
  --adapter NAME|MOD:CLS  evaluator adapter (default: embodied_agent)
                           sanity baseline: latest_only
                           temporal baseline: embodied_agent_recency
                           VLM built-in: embodied_agent_vlm
  --adapter-kwargs JSON   constructor kwargs for a custom adapter
  --max-episodes N        evaluate only the first N episodes
  --keep-state            retain generated memory and FAISS files in the report
  --compare-adapters A,... latest runs to compare (default: latest_only,
                           embodied_agent,embodied_agent_recency)

Examples:
  docker/scripts/spatial_memory_camera.sh all
  docker/scripts/spatial_memory_camera.sh live --interval 3 --duration 24
  docker/scripts/spatial_memory_camera.sh rerank-setup
  docker/scripts/spatial_memory_camera.sh rerank --case open_refrigerator
  docker/scripts/spatial_memory_camera.sh benchmark --episodes 12
  docker/scripts/spatial_memory_camera.sh benchmark-eval --max-episodes 1
  docker/scripts/spatial_memory_camera.sh benchmark-eval \
    --adapter embodied_agent_vlm --recall-k 12 --top-k 3 --max-episodes 1
  docker/scripts/spatial_memory_camera.sh benchmark-eval \
    --adapter latest_only --top-k 3
  docker/scripts/spatial_memory_camera.sh benchmark-eval \
    --adapter embodied_agent_recency --recall-k 12 --top-k 3
  docker/scripts/spatial_memory_camera.sh benchmark-suite \
    --benchmark-name object_relocation_layout09_style09_seed42 --top-k 3
  docker/scripts/spatial_memory_camera.sh benchmark-compare
  docker/scripts/spatial_memory_camera.sh all --layout 10 --style 10
  docker/scripts/spatial_memory_camera.sh memory --query "blue cabinets"
EOF
}

die() {
    echo "[spatial-camera] ERROR: $*" >&2
    exit 1
}

note() {
    echo "[spatial-camera] $*"
}

is_integer() {
    [[ "$1" =~ ^-?[0-9]+$ ]]
}

is_number() {
    [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

slugify() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_'
}

parse_options() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task) [[ $# -ge 2 ]] || die "--task needs a value"; TASK="$2"; shift 2 ;;
            --layout) [[ $# -ge 2 ]] || die "--layout needs a value"; LAYOUT="$2"; shift 2 ;;
            --style) [[ $# -ge 2 ]] || die "--style needs a value"; STYLE="$2"; shift 2 ;;
            --seed) [[ $# -ge 2 ]] || die "--seed needs a value"; SEED="$2"; shift 2 ;;
            --scan-name) [[ $# -ge 2 ]] || die "--scan-name needs a value"; SCAN_NAME="$2"; SCAN_NAME_EXPLICIT=1; shift 2 ;;
            --positions) [[ $# -ge 2 ]] || die "--positions needs a value"; MAX_POSITIONS="$2"; shift 2 ;;
            --headings) [[ $# -ge 2 ]] || die "--headings needs a value"; HEADINGS="$2"; shift 2 ;;
            --image-size) [[ $# -ge 2 ]] || die "--image-size needs a value"; IMAGE_SIZE="$2"; shift 2 ;;
            --camera-height) [[ $# -ge 2 ]] || die "--camera-height needs a value"; CAMERA_HEIGHT="$2"; shift 2 ;;
            --pitch) [[ $# -ge 2 ]] || die "--pitch needs a value"; PITCH="$2"; shift 2 ;;
            --spacing) [[ $# -ge 2 ]] || die "--spacing needs a value"; SPACING="$2"; shift 2 ;;
            --footprint-radius) [[ $# -ge 2 ]] || die "--footprint-radius needs a value"; FOOTPRINT_RADIUS="$2"; shift 2 ;;
            --model) [[ $# -ge 2 ]] || die "--model needs a value"; MODEL="$2"; shift 2 ;;
            --device) [[ $# -ge 2 ]] || die "--device needs a value"; DEVICE="$2"; shift 2 ;;
            --query) [[ $# -ge 2 ]] || die "--query needs a value"; QUERIES+=("$2"); shift 2 ;;
            --top-k) [[ $# -ge 2 ]] || die "--top-k needs a value"; TOP_K="$2"; shift 2 ;;
            --interval) [[ $# -ge 2 ]] || die "--interval needs a value"; LIVE_INTERVAL="$2"; shift 2 ;;
            --duration) [[ $# -ge 2 ]] || die "--duration needs a value"; LIVE_DURATION="$2"; shift 2 ;;
            --session-id) [[ $# -ge 2 ]] || die "--session-id needs a value"; LIVE_SESSION="$2"; shift 2 ;;
            --validate-session) [[ $# -ge 2 ]] || die "--validate-session needs a value"; EXPECT_LIVE_SESSION="$2"; shift 2 ;;
            --recall-k) [[ $# -ge 2 ]] || die "--recall-k needs a value"; RERANK_RECALL_K="$2"; RERANK_OPTIONS_USED=1; shift 2 ;;
            --vlm-model) [[ $# -ge 2 ]] || die "--vlm-model needs a value"; VLM_MODEL="$2"; RERANK_OPTIONS_USED=1; VLM_MODEL_OPTION_USED=1; shift 2 ;;
            --case) [[ $# -ge 2 ]] || die "--case needs a value"; RERANK_CASES+=("$2"); RERANK_OPTIONS_USED=1; RERANK_CASE_OPTIONS_USED=1; shift 2 ;;
            --permutations) [[ $# -ge 2 ]] || die "--permutations needs a value"; RERANK_PERMUTATIONS="$2"; RERANK_OPTIONS_USED=1; RERANK_CASE_OPTIONS_USED=1; shift 2 ;;
            --response-replay) [[ $# -ge 2 ]] || die "--response-replay needs a value"; RERANK_REPLAY="$2"; RERANK_OPTIONS_USED=1; RERANK_CASE_OPTIONS_USED=1; shift 2 ;;
            --benchmark-name) [[ $# -ge 2 ]] || die "--benchmark-name needs a value"; BENCHMARK_NAME="$2"; shift 2 ;;
            --episodes) [[ $# -ge 2 ]] || die "--episodes needs a value"; BENCHMARK_EPISODES="$2"; shift 2 ;;
            --objects) [[ $# -ge 2 ]] || die "--objects needs a value"; BENCHMARK_OBJECTS="$2"; shift 2 ;;
            --route-points) [[ $# -ge 2 ]] || die "--route-points needs a value"; BENCHMARK_ROUTE_POINTS="$2"; shift 2 ;;
            --capture-interval) [[ $# -ge 2 ]] || die "--capture-interval needs a value"; BENCHMARK_CAPTURE_INTERVAL="$2"; shift 2 ;;
            --adapter) [[ $# -ge 2 ]] || die "--adapter needs a value"; BENCHMARK_ADAPTER="$2"; BENCHMARK_ADAPTER_OPTION_USED=1; shift 2 ;;
            --adapter-kwargs) [[ $# -ge 2 ]] || die "--adapter-kwargs needs a value"; BENCHMARK_ADAPTER_KWARGS="$2"; BENCHMARK_ADAPTER_KWARGS_OPTION_USED=1; shift 2 ;;
            --max-episodes) [[ $# -ge 2 ]] || die "--max-episodes needs a value"; BENCHMARK_MAX_EPISODES="$2"; shift 2 ;;
            --keep-state) BENCHMARK_KEEP_STATE=1; shift ;;
            --compare-adapters) [[ $# -ge 2 ]] || die "--compare-adapters needs a value"; BENCHMARK_COMPARE_ADAPTERS="$2"; shift 2 ;;
            --replace) REPLACE=1; shift ;;
            --rebuild) REBUILD=1; shift ;;
            --force-build) FORCE_BUILD=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
}

validate_options() {
    [[ "$TASK" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid task: $TASK"
    for value in "$LAYOUT" "$STYLE" "$SEED" "$MAX_POSITIONS" "$IMAGE_SIZE" "$TOP_K"; do
        is_integer "$value" || die "expected an integer, got: $value"
    done
    for value in "$CAMERA_HEIGHT" "$PITCH" "$SPACING" "$FOOTPRINT_RADIUS" \
                 "$LIVE_INTERVAL" "$LIVE_DURATION"; do
        is_number "$value" || die "expected a number, got: $value"
    done
    ((MAX_POSITIONS > 0)) || die "positions must be positive"
    ((IMAGE_SIZE > 0)) || die "image size must be positive"
    ((TOP_K > 0)) || die "top-k must be positive"
    awk -v value="$LIVE_INTERVAL" 'BEGIN { exit !(value > 0) }' \
        || die "interval must be positive"
    awk -v value="$LIVE_DURATION" 'BEGIN { exit !(value > 0) }' \
        || die "duration must be positive"
    [[ "$DEVICE" == auto || "$DEVICE" == cpu || "$DEVICE" == cuda ]] \
        || die "device must be auto, cpu, or cuda"
    [[ "$MODEL" == siglip_base || "$MODEL" == siglip || "$MODEL" == dinov2_base \
       || "$MODEL" == dinov2 ]] \
        || die "unsupported model: $MODEL (CLIP is disabled because upstream projections are incomplete)"

    for value in "$BENCHMARK_EPISODES" "$BENCHMARK_ROUTE_POINTS"; do
        is_integer "$value" || die "expected a benchmark integer, got: $value"
    done
    ((BENCHMARK_EPISODES > 0)) || die "episodes must be positive"
    ((BENCHMARK_ROUTE_POINTS >= 2)) || die "route-points must be at least 2"
    is_number "$BENCHMARK_CAPTURE_INTERVAL" \
        || die "capture-interval must be a number"
    awk -v value="$BENCHMARK_CAPTURE_INTERVAL" 'BEGIN { exit !(value > 0) }' \
        || die "capture-interval must be positive"
    if [[ -n "$BENCHMARK_MAX_EPISODES" ]]; then
        is_integer "$BENCHMARK_MAX_EPISODES" || die "max-episodes must be an integer"
        ((BENCHMARK_MAX_EPISODES > 0)) || die "max-episodes must be positive"
    fi
    [[ "$BENCHMARK_OBJECTS" =~ ^[A-Za-z0-9_,-]+$ ]] \
        || die "objects must be comma-separated RoboCasa group names"
    [[ "$BENCHMARK_ADAPTER" =~ ^[A-Za-z0-9_.:-]+$ ]] \
        || die "invalid adapter name or import path"
    [[ "$BENCHMARK_COMPARE_ADAPTERS" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)+$ ]] \
        || die "compare-adapters must contain at least two comma-separated adapter names"

    if [[ -z "$SCAN_NAME" ]]; then
        SCAN_NAME="$(slugify "$TASK")_layout$(printf '%02d' "$LAYOUT")_style$(printf '%02d' "$STYLE")_seed${SEED}"
    else
        SCAN_NAME="$(slugify "$SCAN_NAME")"
    fi
    if [[ -z "$BENCHMARK_NAME" ]]; then
        BENCHMARK_NAME="object_relocation_layout$(printf '%02d' "$LAYOUT")_style$(printf '%02d' "$STYLE")_seed${SEED}"
    else
        BENCHMARK_NAME="$(slugify "$BENCHMARK_NAME")"
    fi
    [[ -n "$BENCHMARK_NAME" && "$BENCHMARK_NAME" != "." && "$BENCHMARK_NAME" != ".." ]] \
        || die "invalid benchmark name"
    [[ -n "$SCAN_NAME" && "$SCAN_NAME" != "." && "$SCAN_NAME" != ".." ]] \
        || die "invalid scan name"
    if [[ -n "$LIVE_SESSION" ]]; then
        LIVE_SESSION="$(slugify "$LIVE_SESSION")"
        [[ "$LIVE_SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] \
            || die "invalid live session id"
    fi
    if [[ -n "$EXPECT_LIVE_SESSION" ]]; then
        EXPECT_LIVE_SESSION="$(slugify "$EXPECT_LIVE_SESSION")"
        [[ "$EXPECT_LIVE_SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] \
            || die "invalid validation session id"
    fi
}

validate_rerank_options() {
    is_integer "$RERANK_RECALL_K" || die "recall-k must be an integer"
    is_integer "$RERANK_PERMUTATIONS" || die "permutations must be an integer"
    ((RERANK_RECALL_K > 0)) || die "recall-k must be positive"
    ((RERANK_PERMUTATIONS > 0)) || die "permutations must be positive"
    [[ "$VLM_MODEL" =~ ^[A-Za-z0-9._/-]+$ ]] || die "invalid VLM model name"
    local rerank_case
    for rerank_case in "${RERANK_CASES[@]}"; do
        [[ "$rerank_case" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid rerank case: $rerank_case"
    done
}

vlm_key_env_name() {
    if [[ -n "${GEMINI_API_KEY:-}" \
          && "${GEMINI_API_KEY}" != "your-gemini-api-key-here" ]]; then
        printf '%s\n' GEMINI_API_KEY
    elif [[ -n "${GOOGLE_API_KEY:-}" \
            && "${GOOGLE_API_KEY}" != "your-google-api-key-here" ]]; then
        printf '%s\n' GOOGLE_API_KEY
    else
        return 1
    fi
}

require_vlm_key() {
    vlm_key_env_name >/dev/null || die \
        "Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY; put it in $SPATIAL_ENV_FILE"
}

validate_rerank_setup_scene() {
    [[ "$TASK" == Kitchen && "$LAYOUT" -eq 9 && "$STYLE" -eq 9 && "$SEED" -eq 42 ]] \
        || die "rerank-setup is frozen to Kitchen layout=9 style=9 seed=42"
    [[ "$MODEL" == siglip_base ]] \
        || die "rerank-setup requires model=siglip_base for text retrieval"
    [[ "$MAX_POSITIONS" -eq 16 && "$HEADINGS" == "0,90,180,270" \
       && "$IMAGE_SIZE" -eq 512 ]] \
        || die "rerank-setup requires the default 16 positions, headings, and image size"
    awk -v value="$CAMERA_HEIGHT" 'BEGIN { exit !(value == 1.25) }' \
        || die "rerank-setup requires camera-height=1.25"
    awk -v value="$PITCH" 'BEGIN { exit !(value == -12) }' \
        || die "rerank-setup requires pitch=-12"
    awk -v value="$SPACING" 'BEGIN { exit !(value == 0.5) }' \
        || die "rerank-setup requires spacing=0.5"
    awk -v value="$FOOTPRINT_RADIUS" 'BEGIN { exit !(value == 0.30) }' \
        || die "rerank-setup requires footprint-radius=0.30"
    awk -v interval="$LIVE_INTERVAL" -v duration="$LIVE_DURATION" \
        'BEGIN { exit !(interval == 3 && duration == 24) }' \
        || die "rerank-setup requires the default live interval=3 duration=24"
}

validate_live_options() {
    awk -v value="$LIVE_INTERVAL" 'BEGIN { exit !(value >= 1) }' \
        || die "live interval must be at least 1 second"
    awk -v duration="$LIVE_DURATION" -v interval="$LIVE_INTERVAL" \
        'BEGIN { exit !(duration >= 3 * interval) }' \
        || die "live duration must cover at least four captures (duration >= 3 * interval)"
}

scan_host_dir() {
    printf '%s/%s\n' "$SCAN_ROOT" "$SCAN_NAME"
}

benchmark_host_dir() {
    printf '%s/%s\n' "$BENCHMARK_ROOT" "$BENCHMARK_NAME"
}

acquire_scan_lock() {
    command -v flock >/dev/null 2>&1 || die "flock is required for safe scan updates"
    mkdir -p "$SCAN_ROOT/.locks"
    exec 9>"$SCAN_ROOT/.locks/$SCAN_NAME.lock"
    flock -n 9 || die "another scan or memory process is updating $SCAN_NAME"
}

acquire_benchmark_lock() {
    command -v flock >/dev/null 2>&1 || die "flock is required for safe benchmark updates"
    mkdir -p "$BENCHMARK_ROOT/.locks"
    exec 8>"$BENCHMARK_ROOT/.locks/$BENCHMARK_NAME.lock"
    flock -n 8 || die "another benchmark process is updating $BENCHMARK_NAME"
}

build_memory_image() {
    local installed_schema=""
    if docker image inspect "$MEMORY_IMAGE" >/dev/null 2>&1; then
        installed_schema="$(docker image inspect \
            --format '{{ index .Config.Labels "org.hams.spatial-memory-runtime" }}' \
            "$MEMORY_IMAGE" 2>/dev/null || true)"
    fi
    if [[ "$FORCE_BUILD" == 1 || "$installed_schema" != "$MEMORY_RUNTIME_SCHEMA" ]]; then
        note "building memory-only image $MEMORY_IMAGE"
        docker build -f "$MEMORY_DOCKERFILE" -t "$MEMORY_IMAGE" "$REPO_ROOT"
    fi
}

run_scan() {
    mkdir -p "$SCAN_ROOT"
    local -a command=(
        docker compose -f "$COMPOSE_FILE" --profile robocasa run --rm --no-deps
        --user "$(id -u):$(id -g)"
        -e HOME=/tmp
        -e MUJOCO_GL=egl
        -v "$SCAN_ROOT:/data"
        robocasa
        python -u spatial_memory_scan.py
        --task "$TASK"
        --layout "$LAYOUT"
        --style "$STYLE"
        --seed "$SEED"
        --output "/data/$SCAN_NAME"
        --max-positions "$MAX_POSITIONS"
        --headings "$HEADINGS"
        --image-width "$IMAGE_SIZE"
        --image-height "$IMAGE_SIZE"
        --camera-height "$CAMERA_HEIGHT"
        --pitch "$PITCH"
        --spacing "$SPACING"
        --footprint-radius "$FOOTPRINT_RADIUS"
    )
    [[ "$REPLACE" == 1 ]] && command+=(--overwrite)
    note "scanning RoboCasa $TASK layout=$LAYOUT style=$STYLE seed=$SEED"
    "${command[@]}"
}

run_live_scan() {
    local host_scan
    host_scan="$(scan_host_dir)"
    [[ -f "$host_scan/scan_manifest.json" ]] \
        || die "baseline scan not found: $host_scan (run the all command first)"
    [[ "$REPLACE" == 0 ]] || die "live capture cannot be combined with --replace"
    [[ "$REBUILD" == 0 ]] || die "live capture cannot be combined with --rebuild"
    if [[ -z "$LIVE_SESSION" ]]; then
        LIVE_SESSION="live_$(date -u +%Y%m%d_%H%M%S)"
    fi

    local -a command=(
        docker compose -f "$COMPOSE_FILE" --profile robocasa run --rm --no-deps
        --user "$(id -u):$(id -g)"
        -e HOME=/tmp
        -e MUJOCO_GL=egl
        -v "$SCAN_ROOT:/data"
        robocasa
        python -u spatial_memory_live.py
        --task "$TASK"
        --layout "$LAYOUT"
        --style "$STYLE"
        --seed "$SEED"
        --output "/data/$SCAN_NAME"
        --session-id "$LIVE_SESSION"
        --interval "$LIVE_INTERVAL"
        --duration "$LIVE_DURATION"
        --image-size "$IMAGE_SIZE"
        --camera-height "$CAMERA_HEIGHT"
        --pitch "$PITCH"
        --spacing "$SPACING"
        --footprint-radius "$FOOTPRINT_RADIUS"
    )
    note "capturing live session=$LIVE_SESSION interval=${LIVE_INTERVAL}s duration=${LIVE_DURATION}s"
    "${command[@]}"
    EXPECT_LIVE_SESSION="$LIVE_SESSION"
}

run_memory() {
    local host_scan
    host_scan="$(scan_host_dir)"
    [[ -d "$EMBODIED_ROOT/.git" ]] \
        || die "EmbodiedAgent not found at $EMBODIED_ROOT"
    [[ -d "$host_scan/color" ]] \
        || die "scan not found: $host_scan (run the scan command first)"
    mkdir -p "$HF_CACHE"
    build_memory_image

    local -a gpu_args=(--gpus all)
    [[ "$DEVICE" == cpu ]] && gpu_args=()
    local -a command=(
        docker run --rm
        "${gpu_args[@]}"
        --user "$(id -u):$(id -g)"
        -e HOME=/tmp
        -e PYTHONPATH=/opt/EmbodiedAgent
        -e HF_HOME=/cache/huggingface
        -v "$EMBODIED_ROOT:/opt/EmbodiedAgent:ro"
        -v "$host_scan:/data"
        -v "$HF_CACHE:/cache/huggingface"
        "$MEMORY_IMAGE"
        python -u /opt/EmbodiedAgent/tools/robocasa_memory_smoke.py /data
        --model "$MODEL"
        --device "$DEVICE"
        --top-k "$TOP_K"
    )
    [[ "$REBUILD" == 1 ]] && command+=(--rebuild)
    [[ -n "$EXPECT_LIVE_SESSION" ]] \
        && command+=(--expect-live-session "$EXPECT_LIVE_SESSION")
    local query
    for query in "${QUERIES[@]}"; do
        command+=(--query "$query")
    done
    note "validating EmbodiedAgent memory with model=$MODEL"
    "${command[@]}"
}

run_rerank() {
    local host_scan
    host_scan="$(scan_host_dir)"
    [[ -d "$EMBODIED_ROOT/.git" ]] \
        || die "EmbodiedAgent not found at $EMBODIED_ROOT"
    [[ -d "$host_scan/color" && -f "$host_scan/memory/memory_index.json" ]] \
        || die "indexed scan not found: $host_scan (run the all command first)"

    local -a key_args=()
    if [[ -z "$RERANK_REPLAY" ]]; then
        local key_name
        key_name="$(vlm_key_env_name)" || die \
            "rerank requires GEMINI_API_KEY or GOOGLE_API_KEY; put it in $SPATIAL_ENV_FILE"
        key_args=(-e "$key_name")
    fi
    mkdir -p "$HF_CACHE"
    build_memory_image

    local -a gpu_args=(--gpus all)
    [[ "$DEVICE" == cpu ]] && gpu_args=()
    local -a command=(
        docker run --rm
        "${gpu_args[@]}"
        --user "$(id -u):$(id -g)"
        -e PYTHONPATH=/opt/EmbodiedAgent
        -e HF_HOME=/cache/huggingface
        "${key_args[@]}"
        -v "$EMBODIED_ROOT:/opt/EmbodiedAgent:ro"
        -v "$host_scan:/data"
        -v "$HF_CACHE:/cache/huggingface"
        "$MEMORY_IMAGE"
        python -u /opt/EmbodiedAgent/tools/robocasa_memory_rerank_eval.py /data
        --model "$MODEL"
        --device "$DEVICE"
        --recall-k "$RERANK_RECALL_K"
        --permutations "$RERANK_PERMUTATIONS"
        --vlm-model "$VLM_MODEL"
    )
    local rerank_case
    for rerank_case in "${RERANK_CASES[@]}"; do
        command+=(--case "$rerank_case")
    done
    [[ -n "$RERANK_REPLAY" ]] && command+=(--response-replay "$RERANK_REPLAY")
    note "evaluating VLM reranking model=$VLM_MODEL recall_k=$RERANK_RECALL_K"
    "${command[@]}"
    note "rerank reports: $host_scan/vlm_rerank"
}

run_rerank_setup() {
    local host_scan
    host_scan="$(scan_host_dir)"
    [[ ! -e "$host_scan" ]] \
        || die "dedicated rerank dataset already exists: $host_scan (use rerank, or choose another --scan-name)"

    REPLACE=1
    REBUILD=1
    run_scan
    run_memory

    REPLACE=0
    REBUILD=0
    LIVE_SESSION="rerank-benchmark"
    run_live_scan
    run_memory
    note "rerank benchmark ready: $host_scan"
    note "next: docker/scripts/spatial_memory_camera.sh rerank --scan-name $SCAN_NAME"
}

run_benchmark_setup() {
    mkdir -p "$BENCHMARK_ROOT"
    local host_benchmark
    host_benchmark="$(benchmark_host_dir)"
    local -a command=(
        docker compose -f "$COMPOSE_FILE" --profile robocasa run --rm --no-deps
        -e HOME=/tmp
        -e MUJOCO_GL=egl
        -v "$BENCHMARK_ROOT:/data"
        robocasa
        python -u spatial_memory_benchmark.py
        --output "/data/$BENCHMARK_NAME"
        --output-owner "$(id -u):$(id -g)"
        --episodes "$BENCHMARK_EPISODES"
        --objects "$BENCHMARK_OBJECTS"
        --layout "$LAYOUT"
        --style "$STYLE"
        --seed "$SEED"
        --route-points "$BENCHMARK_ROUTE_POINTS"
        --capture-interval "$BENCHMARK_CAPTURE_INTERVAL"
        --image-size "$IMAGE_SIZE"
        --camera-height "$CAMERA_HEIGHT"
        --pitch "$PITCH"
        --spacing "$SPACING"
        --footprint-radius "$FOOTPRINT_RADIUS"
    )
    [[ "$REPLACE" == 1 ]] && command+=(--overwrite)
    note "generating $BENCHMARK_EPISODES relocation episodes at $host_benchmark"
    "${command[@]}"
    note "benchmark contact sheets: $host_benchmark/episodes/*/contact_sheet.jpg"
}

run_benchmark_eval() {
    local host_benchmark
    host_benchmark="$(benchmark_host_dir)"
    [[ -d "$EMBODIED_ROOT/.git" ]] \
        || die "EmbodiedAgent not found at $EMBODIED_ROOT"
    [[ -f "$host_benchmark/benchmark_manifest.json" ]] \
        || die "benchmark dataset not found: $host_benchmark (run benchmark-setup first)"
    mkdir -p "$HF_CACHE"
    build_memory_image

    local adapter_kwargs="$BENCHMARK_ADAPTER_KWARGS"
    if [[ -z "$adapter_kwargs" ]]; then
        if [[ "$BENCHMARK_ADAPTER" == embodied_agent ]]; then
            adapter_kwargs="{\"model\":\"$MODEL\",\"device\":\"$DEVICE\"}"
        elif [[ "$BENCHMARK_ADAPTER" == embodied_agent_recency ]]; then
            adapter_kwargs="{\"model\":\"$MODEL\",\"device\":\"$DEVICE\",\"recall_k\":$RERANK_RECALL_K}"
        elif [[ "$BENCHMARK_ADAPTER" == embodied_agent_vlm ]]; then
            adapter_kwargs="{\"model\":\"$MODEL\",\"device\":\"$DEVICE\",\"recall_k\":$RERANK_RECALL_K,\"vlm_model\":\"$VLM_MODEL\"}"
        else
            adapter_kwargs="{}"
        fi
    fi
    local -a key_args=()
    if [[ "$BENCHMARK_ADAPTER" == embodied_agent_vlm ]]; then
        local key_name
        key_name="$(vlm_key_env_name)" || die \
            "embodied_agent_vlm requires GEMINI_API_KEY or GOOGLE_API_KEY; put it in $SPATIAL_ENV_FILE"
        key_args=(-e "$key_name")
    fi
    local -a gpu_args=(--gpus all)
    [[ "$DEVICE" == cpu || "$BENCHMARK_ADAPTER" == latest_only ]] && gpu_args=()
    local -a command=(
        docker run --rm
        "${gpu_args[@]}"
        --user "$(id -u):$(id -g)"
        -e HOME=/tmp
        -e PYTHONPATH=/opt/EmbodiedAgent:/opt/Humanoid_Simulation
        -e HF_HOME=/cache/huggingface
        "${key_args[@]}"
        -v "$EMBODIED_ROOT:/opt/EmbodiedAgent:ro"
        -v "$REPO_ROOT:/opt/Humanoid_Simulation:ro"
        -v "$host_benchmark:/data"
        -v "$HF_CACHE:/cache/huggingface"
        "$MEMORY_IMAGE"
        python -u -m benchmarks.spatial_memory.evaluate
        --dataset /data
        --adapter "$BENCHMARK_ADAPTER"
        --adapter-kwargs "$adapter_kwargs"
        --top-k "$TOP_K"
    )
    [[ -n "$BENCHMARK_MAX_EPISODES" ]] \
        && command+=(--max-episodes "$BENCHMARK_MAX_EPISODES")
    [[ "$BENCHMARK_KEEP_STATE" == 1 ]] && command+=(--keep-state)
    note "evaluating streaming memory adapter=$BENCHMARK_ADAPTER top_k=$TOP_K"
    "${command[@]}"
    note "benchmark reports: $host_benchmark/reports"
    note "each run contains report.html, summary.md, and results.json"
}

run_benchmark_compare() {
    local host_benchmark
    host_benchmark="$(benchmark_host_dir)"
    [[ -f "$host_benchmark/benchmark_manifest.json" ]] \
        || die "benchmark dataset not found: $host_benchmark (run benchmark-setup first)"
    command -v python3 >/dev/null 2>&1 || die "python3 is required for comparison reports"

    local -a compare_adapters=()
    IFS=',' read -r -a compare_adapters <<< "$BENCHMARK_COMPARE_ADAPTERS"
    local -a command=(
        python3 -m benchmarks.spatial_memory.compare
        --dataset "$host_benchmark"
    )
    local adapter
    for adapter in "${compare_adapters[@]}"; do
        command+=(--adapter "$adapter")
    done
    note "comparing latest identical runs: $BENCHMARK_COMPARE_ADAPTERS"
    "${command[@]}"
    note "comparison reports: $host_benchmark/comparisons"
}

run_benchmark_suite() {
    [[ "$BENCHMARK_ADAPTER_OPTION_USED" == 0 \
       && "$BENCHMARK_ADAPTER_KWARGS_OPTION_USED" == 0 ]] \
        || die "benchmark-suite selects its four adapters; do not pass --adapter or --adapter-kwargs"
    require_vlm_key

    local adapter
    for adapter in \
        latest_only \
        embodied_agent \
        embodied_agent_recency \
        embodied_agent_vlm; do
        BENCHMARK_ADAPTER="$adapter"
        BENCHMARK_ADAPTER_KWARGS=""
        run_benchmark_eval
    done
    BENCHMARK_COMPARE_ADAPTERS="latest_only,embodied_agent,embodied_agent_recency,embodied_agent_vlm"
    run_benchmark_compare
}

main() {
    local command="${1:-help}"
    shift || true
    parse_options "$@"
    validate_options
    command -v docker >/dev/null 2>&1 || die "docker is required"

    if [[ "$command" == rerank || "$command" == rerank-setup ]]; then
        if [[ "$SCAN_NAME_EXPLICIT" == 0 ]]; then
            SCAN_NAME="${SCAN_NAME}_rerank_benchmark"
        fi
    fi
    if [[ "$command" == rerank ]]; then
        validate_rerank_options
    elif [[ "$command" == benchmark || "$command" == benchmark-eval \
            || "$command" == benchmark-suite ]]; then
        [[ "$RERANK_CASE_OPTIONS_USED" == 0 ]] \
            || die "case, permutations, and response-replay are accepted only by rerank"
        if [[ "$command" == benchmark-suite ]]; then
            validate_rerank_options
        elif [[ "$RERANK_OPTIONS_USED" == 1 ]]; then
            validate_rerank_options
            if [[ "$BENCHMARK_ADAPTER" == embodied_agent_recency ]]; then
                [[ "$VLM_MODEL_OPTION_USED" == 0 ]] \
                    || die "--vlm-model requires --adapter embodied_agent_vlm"
            else
                [[ "$BENCHMARK_ADAPTER" == embodied_agent_vlm ]] \
                    || die "benchmark recall/VLM options require a recency or VLM adapter"
            fi
        fi
    elif [[ "$command" != help && "$command" != -h && "$command" != --help \
            && "$RERANK_OPTIONS_USED" == 1 ]]; then
        die "VLM options are accepted only by rerank, benchmark, benchmark-eval, or benchmark-suite"
    fi

    case "$command" in
        all)
            [[ -z "$EXPECT_LIVE_SESSION" && -z "$LIVE_SESSION" ]] \
                || die "all does not accept live-session options"
            acquire_scan_lock
            REPLACE=1
            REBUILD=1
            run_scan
            run_memory
            note "complete; scan and query sheets: $(scan_host_dir)"
            ;;
        scan)
            [[ "$REBUILD" == 0 && -z "$EXPECT_LIVE_SESSION" && -z "$LIVE_SESSION" ]] \
                || die "scan does not accept --rebuild or live-session options"
            acquire_scan_lock
            run_scan
            ;;
        live)
            [[ -z "$EXPECT_LIVE_SESSION" ]] \
                || die "live creates a new session; do not pass --validate-session"
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 ]] \
                || die "live cannot be combined with --replace or --rebuild"
            validate_live_options
            acquire_scan_lock
            run_live_scan
            run_memory
            note "live temporal memory complete; session=$LIVE_SESSION"
            note "session contact sheet: $(scan_host_dir)/live_sessions/$LIVE_SESSION/contact_sheet.jpg"
            ;;
        memory)
            [[ "$REPLACE" == 0 && -z "$LIVE_SESSION" ]] \
                || die "memory does not accept --replace or --session-id"
            acquire_scan_lock
            run_memory
            ;;
        rerank-setup)
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 && -z "$LIVE_SESSION" \
               && -z "$EXPECT_LIVE_SESSION" && ${#QUERIES[@]} -eq 0 ]] \
                || die "rerank-setup does not accept replace, rebuild, session, validation, or query options"
            validate_rerank_setup_scene
            validate_live_options
            acquire_scan_lock
            run_rerank_setup
            ;;
        rerank)
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 && -z "$LIVE_SESSION" \
               && -z "$EXPECT_LIVE_SESSION" && ${#QUERIES[@]} -eq 0 ]] \
                || die "rerank does not accept scan replacement, live-session, rebuild, or query options"
            acquire_scan_lock
            run_rerank
            ;;
        benchmark)
            [[ "$REBUILD" == 0 && -z "$LIVE_SESSION" && -z "$EXPECT_LIVE_SESSION" \
               && ${#QUERIES[@]} -eq 0 ]] \
                || die "benchmark does not accept rebuild, live-session, validation, or query options"
            acquire_benchmark_lock
            run_benchmark_setup
            run_benchmark_eval
            ;;
        benchmark-setup)
            [[ "$REBUILD" == 0 && -z "$LIVE_SESSION" && -z "$EXPECT_LIVE_SESSION" \
               && ${#QUERIES[@]} -eq 0 ]] \
                || die "benchmark-setup does not accept rebuild, live-session, validation, or query options"
            acquire_benchmark_lock
            run_benchmark_setup
            ;;
        benchmark-eval)
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 && -z "$LIVE_SESSION" \
               && -z "$EXPECT_LIVE_SESSION" && ${#QUERIES[@]} -eq 0 ]] \
                || die "benchmark-eval does not accept replace, rebuild, session, validation, or query options"
            acquire_benchmark_lock
            run_benchmark_eval
            ;;
        benchmark-suite)
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 && -z "$LIVE_SESSION" \
               && -z "$EXPECT_LIVE_SESSION" && ${#QUERIES[@]} -eq 0 ]] \
                || die "benchmark-suite does not accept replace, rebuild, session, validation, or query options"
            acquire_benchmark_lock
            run_benchmark_suite
            ;;
        benchmark-compare)
            [[ "$REPLACE" == 0 && "$REBUILD" == 0 && -z "$LIVE_SESSION" \
               && -z "$EXPECT_LIVE_SESSION" && ${#QUERIES[@]} -eq 0 ]] \
                || die "benchmark-compare does not accept replace, rebuild, session, validation, or query options"
            acquire_benchmark_lock
            run_benchmark_compare
            ;;
        build) build_memory_image ;;
        where) scan_host_dir ;;
        benchmark-where) benchmark_host_dir ;;
        help|-h|--help) usage ;;
        *) usage >&2; die "unknown command: $command" ;;
    esac
}

main "$@"
