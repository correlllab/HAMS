# RoboCasa spatial-memory camera baseline

This is the smallest repeatable test of the EmbodiedAgent memory core in a
RoboCasa kitchen. It uses a fixed-height kinematic camera instead of H1
locomotion, so a falling robot cannot interrupt the scan.

## Run the complete test

Keep `EmbodiedAgent` beside this repository:

```text
workspace/
├── Humanoid_Simulation/
└── EmbodiedAgent/
```

Then run one command:

```bash
cd /home/tanxuan/workspace/Humanoid_Simulation
docker/scripts/spatial_memory_camera.sh all
```

The command:

1. creates RoboCasa `Kitchen` layout 9 / style 9 / seed 42;
2. hides and parks H1, then captures 16 safe camera positions in four headings;
3. writes 64 RGB frames and matching `[x, y, yaw]` sidecars;
4. records a timezone-aware ISO8601 capture timestamp for every frame;
5. imports them into `EpisodicMemory`;
6. builds a SigLIP / FAISS index; and
7. queries `refrigerator`, `sink`, and `stove` and validates every returned
   image, `memory_id`, and pose.

The first memory run may take longer while it builds the small memory image and
downloads the SigLIP weights. Both are reused on later runs.

## Inspect the result

Print the selected result directory:

```bash
docker/scripts/spatial_memory_camera.sh where
```

The important outputs are:

```text
contact_sheet.jpg                  baseline scan views
query_results/refrigerator.jpg     top retrieval results
query_results/sink.jpg
query_results/stove.jpg
smoke_results.json                 machine-readable validation report
color/000042.png                   captured RGB frame
robot_xy/000042.txt                x y yaw_rad pose
frame_meta/000042.json             capture time, source, episode, camera metadata
```

Frame `000042.png` always maps to `memory_id="mem_000042"`.

## Test live temporal memory

After the baseline exists, run:

```bash
docker/scripts/spatial_memory_camera.sh live
```

The default session captures nine frames from `t=0` through `t=24` at real
three-second wall-clock intervals. The camera remains at the same safe pose,
facing the refrigerator.
The first half records it closed; the test harness then opens it and records the
second half. Expected closed/open labels stay only in the test-session ledger and
are never passed to EmbodiedAgent memory.

The live command is append-only. It continues after the highest existing frame,
preserves all older memory IDs, and then warm-starts the existing FAISS index to
add only the new frames. Capture and indexing run sequentially so two processes
never write the metadata or FAISS files at the same time. This first temporal
baseline therefore supports retrieval immediately after each finite live session,
not queries while the camera is still recording.

Use a shorter session while developing:

```bash
docker/scripts/spatial_memory_camera.sh live --interval 1 --duration 4
```

Each session writes:

```text
live_sessions/<session-id>/session.json
live_sessions/<session-id>/contact_sheet.jpg
```

The validator checks that frame IDs are contiguous, timestamps are monotonic,
the before/after poses are identical, the visual scene really changed, the recent
time window excludes the old view, and metadata and FAISS counts remain aligned.
Once a dataset has live history, the destructive `all` reset is intentionally
refused; choose another `--scan-name` for a fresh experiment.

Before live history exists, rerunning `all` explicitly replaces that baseline
scan and rebuilds its generated memory/index. Running the wrapper with no command
only prints help; it never starts a destructive reset implicitly.

## Evaluate VLM reranking (opt in)

The offline memory command proves recall but does not call a paid model. The
controlled labels require exact frame IDs, so first create a dedicated benchmark
dataset. This is separate from the append-only development scan:

```bash
docker/scripts/spatial_memory_camera.sh rerank-setup
```

That command performs the reproducible sequence `64-frame baseline -> one current
9-frame live session`, indexes all 73 frames, and records the live session as
`rerank-benchmark`. It defaults to the separate scan name
`kitchen_layout09_style09_seed42_rerank_benchmark` and refuses to replace an
existing directory.

Store the Gemini key once in the local, Git-ignored environment file and run the
evaluator:

```bash
cp docker/.env.example docker/.env  # first time only
chmod 600 docker/.env
# Edit docker/.env and replace the GEMINI_API_KEY placeholder.
docker/scripts/spatial_memory_camera.sh rerank
```

`spatial_memory_camera.sh` sources `docker/.env` automatically. The key is
forwarded to the container by environment-variable name; its value is not
written to the repository, report, or Docker command line. An explicitly
exported key remains a fallback when `docker/.env` does not exist. The normal
`all`, `live`, and `memory` commands remain VLM-free.

The default benchmark freezes frames `mem_000000` through `mem_000072`, so later
append-only live sessions cannot silently change the comparison. It evaluates
controlled open/closed-refrigerator, sink, stove, and absent-target cases. Each
candidate pool is small and de-duplicated; expected labels are evaluator-only and
are never included in a prompt or `MemoryEntry`.

This first benchmark is a **controlled discrimination test**: candidate membership
comes from the frozen case file, then embedding scores determine the baseline order.
It is not the production FAISS top-12 candidate list. Full-snapshot known-positive
Hit@12 is reported separately to verify that FAISS could recall the target before
reranking. In `results.json`, `evaluation_scope` is therefore
`controlled_candidate_pool`; `live_vlm_schema_verified=true` means a real Gemini
backend returned valid complete responses, not that end-to-end retrieval improved.

For every positive case the report records FAISS known-positive Hit@12, embedding
and VLM P@1/AP/nDCG, candidate-order sensitivity, JSON/permutation validity, and
latency. The absent-target case records the highest VLM confidence to expose
hallucinated matches. Reports and visual order sheets are written below:

```text
vlm_rerank/<run-id>/results.json
vlm_rerank/<run-id>/<case>/embedding_order.jpg
vlm_rerank/<run-id>/<case>/vlm_permutation_XX.jpg
```

Run one inexpensive sanity case first:

```bash
docker/scripts/spatial_memory_camera.sh rerank \
  --case open_refrigerator --permutations 1
```

FAISS deliberately recalls more frames than the final result count before VLM
reranking. The VLM decides what is visibly present; capture timestamps remain the
authority for which matching observation is newer.

## Object-relocation benchmark

Use this benchmark for quantitative comparison of memory algorithms, including
whether they update after an object moves. It is separate from the fixed
refrigerator reranking dataset and does not require Gemini:

```bash
docker/scripts/spatial_memory_camera.sh benchmark --episodes 12
```

For each episode, RoboCasa samples one target (`mug` or `bowl` by default) on a
counter surface A. A virtual camera follows a deterministic eight-view route
that covers every eligible counter. After lap 1, the simulator moves the object
to a different, well-separated surface B using RoboCasa's native placement
sampler. The camera then repeats exactly the same route for lap 2. The relocation
itself is not captured.

The route is based only on static kitchen geometry. It does not use the current
target pose, visibility labels, or retrieval results, so each memory algorithm
receives the same sensor stream. Captures have deterministic two-second
timestamps; `--capture-interval 1`, `2`, or `3` can model a different sensor
rate without waiting in wall-clock time.

To separate generation from evaluation:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-setup --episodes 12
docker/scripts/spatial_memory_camera.sh benchmark-eval
```

The default evaluator returns FAISS Top K directly. To test the production-style
two-stage path on the **same frozen episodes**, put the Gemini key in
`docker/.env` as shown above and select the VLM adapter:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-eval \
  --benchmark-name object_relocation_layout09_style09_seed42 \
  --adapter embodied_agent_vlm \
  --recall-k 12 \
  --top-k 3 \
  --max-episodes 1
```

This runs `SigLIP query -> real FAISS recall (up to Top-12) -> Gemini image
rerank -> final Top-3`. A streaming checkpoint with fewer than 12 ingested frames
uses every available frame. Start with one episode because each query is one
multimodal API call.
After checking the report and quota, omit `--max-episodes 1` for the complete
dataset. The key is forwarded by environment-variable name and is not saved in
the report.

The VLM must return every recalled `memory_id` exactly once plus one complete
analysis per candidate. Invalid or incomplete output falls back to the original
FAISS order. The report shows VLM valid-response/fallback rates, each result's
original FAISS rank and score, FAISS recall-pool hit/coverage, VLM confidence and
reasoning, so a recall failure and an invalid rerank cannot silently count as a
VLM failure or successful rerank.

Use one episode for a quick end-to-end check:

```bash
docker/scripts/spatial_memory_camera.sh benchmark \
  --episodes 1 --route-points 4 --image-size 256 --top-k 3
```

Each episode stores algorithm input and ground truth separately:

```text
episodes/<episode-id>/
├── color/                  RGB sensor observations
├── robot_xy/               x y yaw sidecars
├── frame_meta/             timestamps and camera metadata
├── observations.jsonl      neutral streaming-ingest contract
├── queries.jsonl           evaluator-owned query checkpoints and relevance IDs
├── oracle/episode.json     simulator-only object poses and visibility labels
└── contact_sheet.jpg       both route laps in time order
```

The adapter receives only one `observations.jsonl` record and its RGB image at a
time. It never receives `queries.jsonl` relevance fields or `oracle/`. The first
lap checks static retrieval. During lap 2, a query runs after each frame starting
with the first frame where B is actually visible; this prevents camera travel
time from being counted as memory update delay.

`results.json` reports:

- static Recall@K and MRR;
- current-location Recall@K and top-1 accuracy after relocation (any correct B view);
- current visible-view coverage@K and latest-visible-frame Recall@K, which expose
  correct B frames that the ranking still misses;
- latest-visible-frame Top-1 accuracy, which distinguishes returning the newest
  frame somewhere in Top K from ranking it first;
- stale old-location top-1 rate and stale fraction in the top K;
- update lag in frames, measured from first visible B observation;
- historical old-location recall;
- absent-query top-1 raw score, plus confidence and false-positive rate at 0.5
  for adapters that expose calibrated `confidence_0_1`; and
- FAISS recall-pool hit/coverage and VLM valid-response/fallback rates when using
  the VLM adapter; and
- ingest/query p50 and p95 latency.

Reports are written under:

```text
container_cache/spatial_memory_benchmarks/
  <benchmark-name>/reports/<adapter>/<run-id>/
  ├── report.html    browsable metrics, episode contact sheets, and Top-K images
  ├── summary.md     compact report for the IDE, email, or meeting notes
  └── results.json   complete machine-readable result
```

Open `report.html` in a browser. Its overview shows the main static/live metrics;
expand an episode to inspect its A-to-B relocation, two-lap contact sheet, query
timeline, and ranked candidates. Relevant/current images are outlined in green
and stale old-location images in red. Images are referenced relative to the
benchmark directory, so keep the report with its dataset when sharing it.

Every new `benchmark` or `benchmark-eval` run generates all three formats. To
render HTML and Markdown for an older `results.json` without rerunning embeddings:

```bash
python3 -m benchmarks.spatial_memory.report \
  --results /path/to/results.json \
  --dataset /path/to/the/benchmark-dataset
```

The `embodied_agent` adapter is the current EmbodiedAgent SigLIP + live FAISS
pipeline. `embodied_agent_vlm` adds the repository's Gemini precision stage
without changing the dataset, query schedule, or metrics.

Two built-in sanity/comparison baselines are also available:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-eval \
  --benchmark-name object_relocation_layout09_style09_seed42 \
  --adapter latest_only --top-k 3

docker/scripts/spatial_memory_camera.sh benchmark-eval \
  --benchmark-name object_relocation_layout09_style09_seed42 \
  --adapter embodied_agent_recency --recall-k 12 --top-k 3
```

`latest_only` ignores the query and returns the newest frames. It is a deliberate
sanity baseline: it should do well on immediate live updates but poorly on static
and historical retrieval. `embodied_agent_recency` retrieves up to 12 FAISS
candidates, min-max normalizes their similarity within that recall pool, then
blends it with linear frame recency using a fixed default weight of `0.25`. The
prior is query-independent, so it cannot special-case phrases such as "before it
moved". To evaluate a preselected alternative weight, pass for example:

```bash
--adapter-kwargs '{"model":"siglip_base","device":"auto","recall_k":12,"recency_weight":0.1}'
```

After all methods have run on the same frozen episodes, generate one comparison:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-compare \
  --benchmark-name object_relocation_layout09_style09_seed42
```

The command selects the latest `latest_only`, `embodied_agent`, and
`embodied_agent_recency` runs by default. Use `--compare-adapters` to change the
list. It refuses to compare different Top-K values, episode subsets, query text,
or oracle labels; consequently, a one-episode Gemini smoke run cannot silently be
compared with a 20-episode baseline. Outputs are written below
`comparisons/<run-id>/` as `comparison.html`, `comparison.md`, and
`comparison.json`, with means and 95% confidence intervals across episodes.

For the normal four-method evaluation, the separate evaluator commands can be
replaced by one opt-in suite command:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-suite \
  --benchmark-name object_relocation_layout09_style09_seed42 \
  --recall-k 12 --top-k 3
```

It runs `latest_only`, `embodied_agent`, `embodied_agent_recency`, and
`embodied_agent_vlm` sequentially on the same frozen dataset, then generates a
comparison containing all four. `--max-episodes` applies to every method, which
is useful for a one-episode quota check. The suite requires the local key before
starting and counts the scheduled VLM queries before running any method. The
default safety limit is 20 calls; on the current 20-episode dataset the guard
reports 153 scheduled calls and recommends the largest safe prefix (three
episodes and 15 calls). Set a smaller subset with `--max-episodes`, or explicitly
raise `--vlm-call-limit` only after checking the available quota. It is not the
default because every streaming VLM query consumes API quota.

To compare another method, implement
`benchmarks.spatial_memory.adapter.MemoryAdapter` (`reset`, `ingest`, `query`,
and `close`) and select it by import path:

```bash
docker/scripts/spatial_memory_camera.sh benchmark-eval \
  --adapter my_package.my_adapter:MyAdapter \
  --adapter-kwargs '{"config":"small"}'
```

The wrapper's memory image must contain that adapter's dependencies. Algorithms
with another runtime can mount the same dataset and invoke
`python -m benchmarks.spatial_memory.evaluate` in their own environment. This
keeps the dataset, query schedule, oracle, and metrics identical across methods.

## Common reruns

Query the existing scan without rescanning or rebuilding its index:

```bash
docker/scripts/spatial_memory_camera.sh memory --query "blue cabinets"
```

Revalidate a named live session after restarting the memory container:

```bash
docker/scripts/spatial_memory_camera.sh memory \
  --validate-session live_20260717_210000
```

Try a different RoboCasa kitchen:

```bash
docker/scripts/spatial_memory_camera.sh all --layout 10 --style 10
```

Generate frames without running EmbodiedAgent memory:

```bash
docker/scripts/spatial_memory_camera.sh scan --layout 1 --style 1
```

Show every option:

```bash
docker/scripts/spatial_memory_camera.sh help
```

To keep `EmbodiedAgent` elsewhere, point the wrapper to it:

```bash
EMBODIED_AGENT_ROOT=/path/to/EmbodiedAgent \
  docker/scripts/spatial_memory_camera.sh memory
```

## Scope

This baseline tests the portable memory path:

```text
RoboCasa RGB + [x, y, yaw] + capture timestamp
  -> EpisodicMemory metadata
  -> SigLIP embeddings
  -> FAISS retrieval
  -> MemoryCandidate image + memory_id + pose
```

The default `all`, `live`, and `memory` path intentionally does not use H1
walking, ROS, Nav2, SLAM, Habitat, VLM reranking, or the complete agent loop. The
optional `rerank` command exercises the existing Gemini precision stage without
starting those systems. The current live test performs a deterministic
refrigerator update at one camera pose; a roaming frontier policy can later feed
the same capture contract without changing the memory core. The object-relocation
benchmark adds a deterministic roaming camera and true A-to-B live updates while
remaining independent of H1 walking.
