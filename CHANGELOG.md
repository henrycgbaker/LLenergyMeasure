# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x pre-release series).
Minor version bumps (`0.x.0`) mark milestone completions. Breaking changes can occur between any 0.x release.

## [Unreleased]

> **Results-bundle format break (unreleased, ships with v0.7.0):** results-bundle
> format 2.0, anchored at commit `09ec455e`. Every v0.7.0 format addition rides
> this single untagged 2.0 stamp - there is no `2.1`. `v0.6.0` is the rollback
> anchor. The clean breaks folded into 2.0 (no alias translation):
>
> - Runner provenance is unified into one `RunnerProvenance` model and `bundle_version` stamps `"2.0"` ([#869]).
> - The per-experiment `environment.json` sidecar is renamed `system.json` ([#879]).
> - The runner `mode` is renamed `local`/`docker` -> `process`/`container`, and the no-spec `source` sentinel `local` -> `implicit` ([#880]).
> - Result fields are realigned: `measurement_config_hash` -> `declared_config_hash`, `thermal_throttle` -> the symmetric `throttle` object, `mj_per_tok_*` -> `energy_per_token_mj_*` ([#881]).
>
> Older bundles read best-effort with a single warning, except a pre-v0.7 runner
> `mode` value, which fails validation loudly on read. See
> [the results-schema reference](docs/reference/results-schema.md) for the full
> read-tolerance contract.

<!-- towncrier release notes start -->

## [0.7.0] - 2026-08-07

### Added

- CI now fails a bare engine version-pin bump. A PR that advances `engine_versions/<engine>/current.yaml` must also ship the regenerated knowledge (`make absorb` outputs plus the packaged `src/llenergymeasure/engines/<engine>/` config, rules, and schema copies), so a pin bump can never merge with a config typed against the old engine surface. ([#849](https://github.com/henrycgbaker/llenergymeasure/pull/849))
- A new `BundleWriter` (`llenergymeasure.results.bundle`) owns the per-experiment results-bundle write policy in one place, driven by a declarative artefact registry (`ARTEFACTS` in `llenergymeasure.domain.bundle_artefacts`). A completed experiment that references a timeseries whose Parquet did not land in the bundle now warns at write time rather than only at read time; adding a future bundle artefact is one registry entry plus one writer method. ([#853](https://github.com/henrycgbaker/llenergymeasure/pull/853))
- A `BundleReader` (`llenergymeasure.results.bundle`) is the read-side counterpart to `BundleWriter`: `BundleReader.read(bundle_dir)` discovers artefacts via the same `ARTEFACTS` registry and returns a `LoadedBundle`, with `read_sidecar(bundle_dir, key)` for single-artefact access. A shared `parse_experiment_result_payload` parser now backs both result read paths (the tolerant docker exchange-read and the strict bundle-read), so there is one parser with two tolerance settings instead of duplicated logic. ([#854](https://github.com/henrycgbaker/llenergymeasure/pull/854))
- The host HuggingFace cache directory the docker runner bind-mounts into experiment containers is now configurable via `LLEM_DOCKER_HF_CACHE` (default `$HOME/.cache/huggingface`), mirroring `LLEM_DOCKER_SHM_SIZE`. Point it at shared storage or a large scratch disk when the default home lives on a small volume. ([#861](https://github.com/henrycgbaker/llenergymeasure/pull/861))
- Re-added the Speculative Decoding and Static KV Cache rows to the engine capability matrix, now derived from the mined engine surface (a field-presence probe over the generated config) rather than hand-authored prose, so the cells self-update on every engine-bump absorb. Docs-only: no runtime, CLI, or dispatch path reads these functions. ([#875](https://github.com/henrycgbaker/llenergymeasure/pull/875))
- A top-level `serving_mode` field on `ExperimentConfig`, a closed `Literal["offline", "server"]` defaulting to `"offline"`. `"offline"` is the only mode with an execution path today; `"server"` is accepted by the config model (online-serving measurement ships with the v0.7.0 server-mode work) but has no execution path yet. It is a conditioning identity axis: it enters the declared, resolved, and observed config-hash families, so an offline config and a server config never deduplicate together. Additive - every existing offline config stays valid and omitting the field is identical to `serving_mode: offline`. Because the field enters the wholesale config dump, the absolute `declared_config_hash` (and `study_design_hash`) of every pre-existing config shifts by one key; this is benign pre-1.0 since dedup is within-study (all configs shift together). ([#885](https://github.com/henrycgbaker/llenergymeasure/pull/885))
- Server-mode config surface: a top-level `server:` namespace (legal only when `serving_mode: server`) carrying a `traffic:` spec - `rate` (requests/sec, required), `arrival` (poisson/gamma with optional `burstiness`), a measurement window as exactly one of `window_seconds` or `window_requests`, `concurrency_cap`, optional `slo` bounds (`ttft_ms`, `tpot_ms` at a shared `percentile`), and `seed`. A generic mode-section match validator binds the namespace to its serving_mode, mirroring the engine-section match: a `server:` section under `serving_mode: offline`, or `serving_mode: server` with no `server:` section, fails loudly. `traffic.rate` is a hashed identity axis (sweepable via `server.traffic.rate: [...]`); `traffic.slo` is a post-hoc overlay excluded from both config-hash families but stamped in the config sidecar and result provenance. ([#888](https://github.com/henrycgbaker/llenergymeasure/pull/888))
- Open-loop traffic generation for server mode: a `TrafficSource` seam (the single window-manager-facing interface for driving online load) and a built-in async Poisson/gamma issuer (`OpenLoopPoissonSource`) that implements it. The issuance schedule is precomputed from the arrival process and the seed, so it is deterministic and fully open-loop - a stalled or concurrency-capped transport never slows issuance. Latency bookkeeping follows the MLPerf LoadGen schedule-anchored convention: each request records `issued_at` (its ideal scheduled time, the latency anchor), `dispatched_at`, and `completed_at`, so cap-induced queue time counts against the system under test rather than hiding from tail percentiles (coordinated omission). A `concurrency_cap` gates dispatch of already-issued requests without touching the schedule; a materially binding cap is disclosed via `cap_bound_fraction` on the issuer's report rather than silently absorbed. Adds the `server` optional-dependency extra (`pip install 'llenergymeasure[server]'`) providing the pure-Python `httpx` client, lazily imported at the transport use site with an actionable error naming the extra. ([#893](https://github.com/henrycgbaker/llenergymeasure/pull/893))
- A `ServerCapable` engine-plugin protocol extension (`launch` / `await_ready` / `shutdown`) for online-serving measurement, added additively as a sibling of the offline `run_inference` contract (the single-call surface is unchanged). An engine claims server support only by implementing all three methods. Readiness is gated by a REAL inference request driven through the serving path after a liveness poll: a passing `/health` never satisfies readiness on its own. The first adapter is vLLM (`vllm serve`), which serves either as a host subprocess (process mode) or in the pinned upstream `vllm/vllm-openai` container. The server container runs with `--network host` unconditionally (co-located client and server, the standard serving-benchmark topology). Shutdown is idempotent and leak-free, escalating from a graceful stop to a hard kill, and the docker-outside-of-docker topology (llenergymeasure itself in a container reaching a `--network host` sibling) surfaces an actionable error rather than a generic timeout. TensorRT-LLM and transformers server adapters follow later. ([#894](https://github.com/henrycgbaker/llenergymeasure/pull/894))
- The TensorRT-LLM server adapter for online-serving measurement, extending the `ServerCapable` protocol added earlier. TRT-LLM serves via `trtllm-serve <model> --port <port>`, invoked explicitly on both legs because the NGC `tensorrt-llm/release` image is the documented `trtllm-serve` vehicle but does not bake it as the entrypoint; readiness polls `/health` then drives a real `/v1/completions` request. Every server-container launch now bind-mounts the host HuggingFace cache and sets `HF_HOME` (the same `LLEM_DOCKER_HF_CACHE`-driven mount the offline docker dispatch uses), so a launched server reuses downloaded weights instead of re-pulling the full model each run; this also applies to the vLLM adapter's container leg. Transformers server mode is deferred to a fast-follow (`transformers serve` at the pinned version is upstream-scoped to moderate load with no first-class health endpoint, not the sustained load a measurement harness drives): a config with `engine=transformers` and `serving_mode=server` is rejected at config validation with an actionable error pointing at vllm or tensorrt. vLLM and TensorRT-LLM are the server-mode engines for this release. ([#895](https://github.com/henrycgbaker/llenergymeasure/pull/895))
- A `ServerSession` that makes online-serving measurement runnable end to end: a context-managed sibling of the offline subprocess/container sessions whose one server lifetime produces N window results (one per measured window). `__enter__` launches the engine server and drives it to readiness with a real probe request drawn from the measured traffic shape; `run()` drives the window manager per rate level (warm up, exclude the ramp, run the contiguous measured windows, drain for latency); `__exit__` shuts the server down and reaps it exactly once on the normal, interrupt, and exception paths alike. Server configs route through the study runner via a new dispatch path (`StudyRunner._run_one` -> `ServerSession`); the offline subprocess and container paths are byte-unchanged. The measured loop runs in-process on the host (the traffic issuer plus host-side energy/thermal sampling), with only the engine server out-of-process, so no serialized config crosses a boundary and the user-config-resolved warmup protocol is read directly. Client-counted output tokens flow to the energy denominator and the stability gate through the token-receipt seam; at this stage they are the server-reported completion-token counts (an interim stand-in, stamped loudly in the session result, until client-side canonical counting and the request log land). A warmup-traffic failure aborts the session; any other level failure is recorded invalid-with-reason and the session continues, with each measured window carrying its level's warmup outcome and pre-window protocol for the offline-vs-server comparability label. Progress display gains a server surface, and the internal step-list surface identifiers are realigned to the process/container runner vocabulary. ([#905](https://github.com/henrycgbaker/llenergymeasure/pull/905))
- Server-mode measurement now persists a results bundle per measured window, and every bundle carries the session facts of the run it belongs to. A server session (one server lifetime: launch, warm up, measure, drain) writes one bundle per window through the same writer the offline path uses, each stamped with a `session` block (a shared session id, the window and level counts, and the raw launch-to-ready, per-level warmup, and drain durations and energies) and a per-window `server` provenance block (level and window position, level validity, warmup outcome, and the pre-window protocol label). The launch, warmup, and drain phases are instrumented with the same measurement bracket and sampler the windows use; a phase whose energy cannot be measured records null rather than zero. Window bundles are written at level close and finalized at session close, so an interrupt mid-session keeps the completed windows on disk (with the drain fields null) instead of losing them. Server results are now first-class in the study result: the mapped per-window results enter `StudyResult.experiments` and their paths join `result_files`, replacing the interim side channel. Consecutive grid points that differ only in `server.traffic.rate` fold into one session (one launch, a rate level per grid point), and each grid point still resolves its own manifest entry. The manifest also records each experiment's resolved-config hash, and resume now rejects a study whose resolved protocol (for example a user-config warmup overlay) changed since the original run rather than silently skipping a differently-resolved cell. The `session` block is additive and rides the existing untagged bundle 2.0 break: it is present in both modes (an offline experiment stamps a fresh session id, `window_count` 1, and null phase raws), and a bundle written before it stays loadable with the block absent. ([#907](https://github.com/henrycgbaker/llenergymeasure/pull/907))
- Server-mode measurement now writes a `requests.parquet` per-window request log and counts output tokens client-side. Each measured window's bundle gains one row per issued request (issue, dispatch, first-token, and completion timestamps; TTFT and end-to-end latency; the per-token receipt times; the client-counted output tokens; the engine's self-reported input/output usage as auxiliary; the stream's `finish_reason` so a length-truncation is distinguishable from a natural stop; status; and the boundary-attribution flags for measurement-window, ramp, and drain-tail), so latency percentiles, goodput, and SLO attainment can be derived offline without re-running. The HTTP transport now streams the OpenAI-compatible completions response and counts the streamed deltas in llem's own callback, identically across engines: that client-side count is the canonical J/token denominator (energy denominator, stability gate, and the mapped result token fields), and the engine's `usage` block rides only as auxiliary provenance. A mid-stream failure preserves the tokens delivered before it died, so a request that streamed in-span tokens then errored or timed out still counts them in the denominator (its GPU compute is in the window energy either way); every row carries its physical facts regardless of status (first-token time and TTFT are real whenever a token physically arrived, an error row carries its to-failure latency), and only `finish_reason` stays null unless an actual finish chunk arrived, so consumers filter latency percentiles by status themselves. The per-window `server` provenance discloses the counting mechanism (`token_counting`) and the auxiliary server-reported total (`server_reported_output_tokens`). The window's measured monotonic span bounds ride as `requests.parquet` file metadata, and per-row token counts are receipt-unclipped (issue-partitioned), so an alternative attribution is re-derivable by clipping each row's receipt series to the span; the authoritative span-clipped denominator remains `result.json`'s `output_tokens`. `requests.parquet` is a registered bundle artefact swept by the writer's finalize backstop: a server bundle missing it warns, offline bundles skip it silently, and a bundle written after an abort (whose per-request bookkeeping was lost) writes none. Additive on the existing untagged bundle 2.0 break; no bundle_version change. Client-side input/prefill token counting is not done (it would need a host-side tokenizer); the engine's `prompt_tokens` is preserved in the request log for cross-checking. ([#908](https://github.com/henrycgbaker/llenergymeasure/pull/908))
- Server-mode results now carry per-window derived metrics (`ExperimentResult.server_metrics`): TTFT / ITL / TPOT / end-to-end latency percentiles, request throughput, completion / error / timeout counts with a completion rate, and a post-hoc SLO overlay (attainment fraction, `slo_pass` verdict, goodput tok/s, and an energy-at-operating-point validity flag). The metrics are derived at persist time from the same per-request records that feed `requests.parquet` with no re-sampling, and the SLO overlay is a pure function of `(records, bounds)` so a window is re-judgeable offline against any bounds via `llenergymeasure.results.server_metrics.evaluate_slo` over `requests.parquet` (loaded with `request_log.rows_from_parquet`). Attainment is the share of completed requests meeting all configured bounds jointly (MLPerf server-scenario reading); a length-truncated completion is attainment-eligible and disclosed. Server results now report `input_tokens` / `total_tokens` as `null` (client-side prefill counting is post-v0.7) rather than a placeholder zero, and the CLI result summary gains a Server section. Offline results are unaffected (the new fields stay `null`). ([#909](https://github.com/henrycgbaker/llenergymeasure/pull/909))
- A YAML/CLI-loaded study must use one `serving_mode` at this release: a study whose expanded experiments mix `offline` and `server` is rejected at study-config loading with an actionable error that names the modes found and tells the user to split the study so every experiment shares one `serving_mode`. This is a deliberately-deletable staging restriction (one loader-edge check, no model change) that a later release removes to admit the engine x `serving_mode` grid crossing; the data model stays mixed-legal, so direct `StudyConfig(experiments=[...])` construction is unaffected. Separately, loading a foldable server rate sweep under `experiment_order: sequential` with more than one cycle now emits a one-line info hint that sequential order launches one server per cell per cycle and that `experiment_order: interleave` reuses one launch per sweep pass (both orders are correct; the hint is a did-you-know, not a warning). ([#911](https://github.com/henrycgbaker/llenergymeasure/pull/911))
- Per-window server provenance now discloses `cap_bound_fraction` in `result.json` (under the `server` block): the fraction of a rate level's scheduled issuances the concurrency cap delayed beyond a small tolerance, sourced from the level's issuer report and shared across the level's windows. It is `0.0` when the level ran uncapped or the cap never materially bound, and `null` when the level aborted before its issuer report was recorded. This closes the disclosure the traffic issuer already computed but never persisted, so a materially binding cap is stamped for result provenance rather than silently absorbed. ([#918](https://github.com/henrycgbaker/llenergymeasure/pull/918))
- Maintainer sign-off of an engine-rule can now carry fresh evidence into the shipped rule. A residue entry in `absorb_signoff.yaml` may add an optional `citation:` (a file:line you verified at the new engine version) and an optional `note:` (a one-line reviewer remark) alongside `human_confirmed: true`; both are written into the promoted rule's provenance instead of being discarded at promotion. Engine-rule provenance gains a matching optional `note:` field for reviewer legibility. The corpus schema is unchanged and existing rules load exactly as before.
- Per-mode warmup namespaces and a principled config-resolution core. Warmup is now a per-mode protocol: `offline.warmup` carries the offline prompt-loop convergence protocol (the settings migrated verbatim from the retired `measurement.warmup`, thermal floor included), and a new `server.warmup` block configures server-mode warmup - `mode: composite` (the default: warm with issuer-driven traffic and open the measured window only once GPU power has plateaued, temperature has settled, and no thermal throttle is active, with a `timeout_seconds` failsafe, default 900s, that proceeds and stamps `convergence: timed_out` rather than hanging) or `mode: fixed` (the explicit opt-out: a `duration_seconds` warmup with no gate, default 300s). `server.warmup` has no thermal-floor knob by design (the server's loaded equilibrium is the measured thermal posture). Both warmup blocks join the config identity in both hash families via the mode-section projection (`server.traffic.slo` remains the sole declared-hash exclusion). Adds `config/precedence.py`, the layered config-resolution core: an `UNSET` sentinel ("use the layer below", distinct from an explicit `None`), `prune_unset`, `resolve_layers`, and a `PrecedenceChain` naming the precedence order (call-site > env > study YAML > user config > pydantic defaults), plus `resolve_server_warmup`, the resolver that layers a server-warmup protocol through that chain. The core and the resolver land here; connecting them to a real construction path (a tool-wide user-config warmup default feeding the resolved config view while the declared hash stays user-intent-only) is deferred to the server-session work and the setup-UX workstream (#886), which add the user-config home and the resolved-vs-declared overlay this needs.
- Server-mode documentation: a serving-study tutorial, a server-mode measurement section in the methodology page (windows, the warmup convergence gate, and the offline-vs-server comparability caveats), a server-measurement architecture reference (traffic source, server session, window manager), and a results-schema extension covering the per-window bundles, `requests.parquet`, and the derived server metrics with the SLO overlay. Also corrects stale `serving_mode` descriptions across config, result, and reference docs that still described offline as the only mode with an execution path.
- Server-mode warmup execution (`harness/server_warmup.py`), filling the warmup-hook seam the window manager reserved. Before each rate level's measured windows open, a `ServerWarmup` warms the server with issuer-driven traffic at the target rate drawn from the MEASURED traffic's shape distribution (through the traffic source, never a canned-prompt loop), re-warming per level. In the default composite mode it opens the window only once all three thermal-equilibrium observables hold together, each read from the SAME power/thermal sampler poll and gating independently: power plateaued (windowing's stable-through-end CoV <= 0.05, reused unchanged), every GPU's temperature settled (trailing-90s range below 2C, which also imposes a ~90s loaded-observation floor), and no thermal throttle active in the trailing window. A hard timeout failsafe proceeds and stamps `timed_out` rather than hanging or silently passing. Fixed mode runs the same traffic path for a fixed duration with no gate (duration 0 skips warmup traffic). There is no idle cooldown anywhere in the server warmup path (the loaded equilibrium is the measured thermal posture). Each level's outcome and a per-mode pre-window protocol description are recorded for result provenance (the offline-vs-server divergence label), and the readiness probe's request shape is drawn from the same traffic distribution.
- Server-mode window object and multi-level window manager (`harness/window_manager.py`). A first-class `WindowSpec` (`rate`, duration-or-count, ramp exclusion, disclosed attribution policy) defines one measured window; the `WindowManager` drives a rate sweep level by level. Each level runs `windows_per_level` (default 3) consecutive measured windows at the configured rate, contiguous with no re-warm between them: it runs the warmup hook once (a seam the server warmup protocol fills, no-op today), starts the open-loop traffic, excludes the ramp PROSPECTIVELY once (the first measured span starts after the ramp, never trimmed afterwards), then for each window emits an explicit start-window event to open energy measurement, holds the measured span, and emits stop-window to close it - finally draining every in-flight request to completion for its latency record. Delineation is event-driven (a `WindowEnergySink` seam with start/stop/abort events; the default reuses the existing `MeasurementBracket`, one bracket per window), never inferred from timestamps. The manager owns the sink lifecycle, so a level failure preserves whatever was measured: a window open at failure has its live sampler released via an explicit abort event exactly once, while a failure during the post-measurement drain or in a window's own close keeps the measured cores of the cleanly-closed windows; the partial state (with the failure site disclosed) is attached to the re-raised, unchanged exception (a CancelledError stays a CancelledError). The two boundary policies stay distinct: the energy denominator counts client-counted tokens received within the measured span, while latency records cover every request issued in the span followed to completion past the close - so a boundary-straddling request appears in latency with its full latency yet contributes only its in-span tokens to the energy denominator. The stability gate is calibrated on J/token, reusing windowing.py's coefficient-of-variation, stable-through-end, clean, and clip machinery plus the trapezoidal integrator: per window it discloses a diagnostic coefficient of variation over 4 sub-window J/token values (the empirical window-calibration constant), and the level GATE passes iff the window-level J/token values agree within 0.05 over every 3 consecutive windows, stable through the end of the level; a failing level is stamped invalid-with-reason, never dropped. Adds config knobs under the server namespace: `server.traffic.window_seconds` now defaults to 240s when omitted (it was required), a new `server.traffic.ramp_exclusion_seconds` (default 30s), and a new `server.cooldown_seconds` inter-level pause (default 0), all joining the config identity in both hash families with no new exclusion (`server.traffic.slo` remains the sole declared-hash exclusion). Count-bound windows (`server.traffic.window_requests`) are rejected in a server config at v0.7 with an actionable error, since the server-mode measurement path is duration-grounded.
- Tool-wide server warmup defaults in the user config, wired through the layered config-resolution core. The user config gains a `server.warmup` block (same shape as the study-level `server.warmup`): a machine-local default for the server warmup protocol, overlaid PER FIELD beneath the study YAML. Precedence is study YAML > user config > built-in default, so a warmup field the study wrote always wins and a field it left unset takes the user-config value; a user field set to the built-in default's value still counts as supplied. The overlay lands in the RESOLVED config hash, never the declared one - a shared study file keeps its declared identity across machines (reproducibility by file sharing), while dedup binds on the realised protocol, so two runs of one study under different user-config warmups are distinct measurements. Resume and drift-detection stay declared-hash-only for now, so a resumed study is blind to a user-config warmup change between runs (the resolved-hash resume guard is banked for a later release). This connects `resolve_server_warmup` to the study load path (`llem` loads the user config in `load_study` and `finalise_study` applies the overlay before dedup), the construction path the per-mode-warmup-namespaces note had staged as deferred. Offline warmup is not overlaid at this release (its execution reads the declared config directly); env and call-site chain layers stay supported-but-unfed.

### Changed

- The Renovate config now documents its one-engine-per-bump-PR policy: each engine's regex-managed pin keeps its own package rule with no shared `groupName`, so a version bump opens a separate reviewable PR per engine rather than batching several engines into one diff. ([#850](https://github.com/henrycgbaker/llenergymeasure/pull/850))
- Consolidated per-engine facts (identity package, dtypes, plugin dispatch, parallelism model, default-image version source) behind a single `ssot.ENGINES` descriptor registry; the previously hand-rolled per-engine branch sites now route through it. Internal refactor, no behavior change. ([#851](https://github.com/henrycgbaker/llenergymeasure/pull/851))
- Config and engine layer tidy (no behavior change): the three copy-paste engine-section validators collapse to a loop, the sweep and scaffold series writers share one order-preserving dedupe helper, the `EnginePlugin.load_model` protocol return type tightens to `tuple[Any, Any]`, and two unreachable TensorRT plugin guards are removed. ([#852](https://github.com/henrycgbaker/llenergymeasure/pull/852))
- **Breaking (results bundle):** the three independent per-artefact `schema_version` counters are replaced by a single `bundle_version` stamped into `result.json`, `config.json`, and the system sidecar, so one number covers the on-disk layout, the artefact set, and each artefact's schema as one contract. `ExperimentResult.schema_version` is renamed `bundle_version`. Earlier bundles read best-effort (the retired `schema_version` key is dropped on load); no converter tooling is provided. ([#853](https://github.com/henrycgbaker/llenergymeasure/pull/853))
- Internal (no behavior or results change): `results.persistence.load_result` is now a thin wrapper over `BundleReader.read` (public API and behaviour unchanged), and `report-gaps` reads each per-experiment `config.json` provenance sidecar through `BundleReader.read_sidecar` rather than a bare file read, so the on-disk sidecar location and encoding are owned in one place. ([#854](https://github.com/henrycgbaker/llenergymeasure/pull/854))
- The TensorRT HF pre-quantised-checkpoint gate (AWQ/GPTQ rejection unless a prebuilt `engine_path` is supplied) is folded into the tensorrt plugin's `check_hardware` hook, so preflight routes every engine uniformly through one compatibility seam (behaviour unchanged: same error, same message). Separately, the vLLM `gpu_memory_utilization` pre-allocation heuristic is extracted from `run_inference` to a named, unit-tested `_peak_matches_vllm_prealloc` helper. ([#855](https://github.com/henrycgbaker/llenergymeasure/pull/855))
- Internal restructure (no behavior or results change): the measured-window mechanics are extracted into a mode-agnostic `MeasurementBracket` (`llenergymeasure.harness.bracket`), and per-model-load state is split from per-window state, groundwork for a future server measurement mode. ([#856](https://github.com/henrycgbaker/llenergymeasure/pull/856))
- Internal restructure (no behavior or results change): the measurement core is decomposed. Result assembly is split by measurement source (an offline producer plus a mode-agnostic assembler) so a future server mode adds a sibling producer without touching assembly, and the 1431-line `harness/measurement.py` is split into `lifecycle` / `window` / `result_assembly` / `persistence` behind the `MeasurementHarness` facade. The contradictory per-engine batch-size knowledge is resolved to one SSOT `BatchSizeModel` on each `ssot.ENGINES` descriptor. ([#857](https://github.com/henrycgbaker/llenergymeasure/pull/857))
- Internal restructure (no behavior or results change): the 786-line `config/grid.py` is split into a grid-orchestration facade plus `config.sweep_expansion` and `config.cycle_ordering`; the public `llenergymeasure.config.grid` import surface is unchanged. ([#860](https://github.com/henrycgbaker/llenergymeasure/pull/860))
- Internal restructure (no behavior or results change): the 1291-line `infra/docker_runner.py` god-module is decomposed into an `infra/docker/` package (`command`, `lifecycle`, `exchange`, `diagnostics`), with `DockerRunner` staying the facade with unchanged signatures. The container dependency-import probe previously embedded as a bash heredoc is extracted to a real package-data module (`infra/_container/probe_imports.py`) that lint and mypy cover. ([#861](https://github.com/henrycgbaker/llenergymeasure/pull/861))
- Internal restructure (no behavior or results change): the progress-step vocabulary is now a single registry in `llenergymeasure.domain.progress`, with `register_step()` the extension point for future modes; the label map, phase map, and ordered step lists all derive from it. Separately, the 1322-line `cli/_step_display.py` is split into `_step_render` / `_experiment_display` / `_study_display` behind a facade, with byte-identical display output. ([#862](https://github.com/henrycgbaker/llenergymeasure/pull/862))
- Internal restructure (no behavior or results change): one experiment dispatch is now an `ExperimentSession` context manager (`llenergymeasure.study.session`) whose lifetime (acquire -> produce -> release) is separable from result production, groundwork for a future server session. The study orchestration cluster moves from `api/_impl.py` to `llenergymeasure.study.orchestration.orchestrate_study`; `api` becomes a thin adapter and the public `run_experiment` / `run_study` signatures and behaviour are unchanged. ([#863](https://github.com/henrycgbaker/llenergymeasure/pull/863))
- Internal restructure (no behavior or results change): the generated per-engine config models moved from `src/llenergymeasure/engines/<engine>/config.py` to `src/llenergymeasure/config/generated/<engine>.py`, beside their only importers. This removes the two `config -> engines.*.config` import-linter exceptions entirely; the generated modules are byte-identical and still regenerated by the codegen script. ([#868](https://github.com/henrycgbaker/llenergymeasure/pull/868))
- **Breaking (results bundle):** `bundle_version` bumps to `"2.0"` for the provenance-unification break. The two sibling runner-provenance models merge into one `RunnerProvenance` (`llenergymeasure.domain.provenance`); the top-level `ExperimentResult.baseline_power_w` copy is retired in favour of `energy_breakdown.baseline_power_w`; four never-populated environment fields are dropped and the driver-reported CUDA field is renamed `driver_supported_version`. A new `ExperimentResult.serving_mode` mirrors the config-side offline/server discriminator. Earlier bundles read best-effort with a single warning; see [the results-schema reference](docs/reference/results-schema.md) for the full read-tolerance contract. ([#869](https://github.com/henrycgbaker/llenergymeasure/pull/869))
- **Breaking (results bundle):** the per-experiment hardware/runtime sidecar is renamed `environment.json` -> `system.json` (MLPerf system-description alignment), across both the bundle sidecar and the study-level snapshot. This rides the same untagged `bundle_version` `"2.0"` break. Clean break with no fallback: a pre-rename `environment.json` is not read, so on an older bundle the system sidecar is simply treated as absent. ([#879](https://github.com/henrycgbaker/llenergymeasure/pull/879))
- **Breaking (runner vocabulary):** the runner mode is renamed `local`/`docker` -> `process`/`container` (image shorthand `docker:<image>` -> `container:<image>`) and the no-spec `source` sentinel `local` -> `implicit`, naming the packaging axis symmetrically. Clean break, not an alias: every entry point that parses a user-supplied runner value (study YAML `runners:`, `LLEM_RUNNER_<ENGINE>`, user config) rejects the old values with a migration error, and `RunnerProvenance.mode` is now a closed `Literal`, so a pre-v0.7 bundle carrying the old mode fails validation loudly on read. Update any pinned `runners:` values to the new vocabulary. ([#880](https://github.com/henrycgbaker/llenergymeasure/pull/880))
- **Breaking (config):** the top-level `harness:` config key is retired; its execution knobs (`batch_size`, `torch_compile*`, `allow_tf32`, `autocast_*`) move into a per-engine `transformers.llem_execution:` sub-section (only transformers has these knobs). Clean break with no alias: a config still carrying a top-level `harness:` key is rejected with an error naming the new location. The sweep-axis key becomes `transformers.llem_execution.batch_size`; update any config that set these knobs. Because the knobs move inside the engine section, the declared-config-hash of any config that sets them shifts, and the resolved/observed hash-view slot is renamed `harness` -> `llem_execution`, shifting those hash values too (dedup is within-study, so this is benign pre-1.0). ([#881](https://github.com/henrycgbaker/llenergymeasure/pull/881))
- **Breaking (results bundle, rides 2.0 - no version bump):** result-field nomenclature alignment. `ExperimentResult.measurement_config_hash` is renamed `declared_config_hash` (the Parquet timeseries metadata key follows); `thermal_throttle` becomes the symmetric `throttle` object (`throttle.thermal` and `throttle.power`, each an axis with `any`/`hw`/`sw`), which adds the previously-missing combined power indicator; and `mj_per_tok_total` / `mj_per_tok_adjusted` become `energy_per_token_mj_total` / `energy_per_token_mj_adjusted` (the study manifest's `mj_per_tok` becomes `energy_per_token_mj`). Clean break: the old names are not tolerated on read. See [the results-schema reference](docs/reference/results-schema.md) for the field contracts. ([#881](https://github.com/henrycgbaker/llenergymeasure/pull/881))
- BREAKING: `serving_mode` is now a required config field with no default (it previously defaulted to `offline`). The serving regime is a primary experimental condition, so an implicit mode is no longer allowed. Every config must declare `serving_mode: offline` (batch inference over a fixed prompt set) or `serving_mode: server` (online serving); a config omitting it fails to load with a migration message. Migration: add `serving_mode: offline` to existing configs to preserve current behavior. Declared-config-hash shift: the required `serving_mode` and the new `server:` key entering the config dump move the declared-config hash of every experiment. This affects within-study dedup grouping only (accepted pre-1.0). ([#888](https://github.com/henrycgbaker/llenergymeasure/pull/888))
- The transformers engine pin advances from 5.7.0 to 5.14.1 with fully re-mined engine knowledge: rediscovered schema (three new generation parameters), a regenerated typed config, and a reconciled validation-rule corpus - 18 re-verified rules now carry 5.14.1 source citations, 10 hand-verified sampling-bound rules cover parameters the automated mining could not reach, and mis-mined or engine-unenforced rules were retired. ([#906](https://github.com/henrycgbaker/llenergymeasure/pull/906))
- Server-mode goodput (`server_metrics.goodput_tokens_s`) is now the literature-exact direct join instead of an `attainment x throughput` product: the in-span output tokens of the requests that both completed and met every configured SLO bound, divided by the span duration (DistServe, OSDI'24, arXiv:2401.09670; Wang et al., arXiv:2410.14257 Eq. 5). No failed request's tokens enter it at any weight, closing the earlier overstatement at a high failure rate. The numerator is in-span-clipped (each qualifying request's receipts up to the window's `span_end`), so a drain straddler's tail tokens are excluded while its full latency still judges its SLO compliance, and goodput stays `<= avg_tokens_per_second`. The overlay remains re-judgeable offline: `evaluate_slo` now takes the window span (read from the `requests.parquet` file metadata via the new `request_log.span_from_parquet`) rather than a precomputed throughput. Server-mode metric only; offline results are unchanged. ([#915](https://github.com/henrycgbaker/llenergymeasure/pull/915))
- Two unwired `server.traffic` fields are removed from the v0.7 config surface before release: `min_query_count` and the traffic-level `passthrough_kwargs`. Both shipped as complete no-ops (zero consumers) yet entered the resolved and observed server config-hash identity, so removing them changes those hashes. This is a clean break with no compatibility cost: no released version ever produced server-mode hashes, and a config that still sets either key now fails validation loudly under the strict `server.traffic` schema. ([#916](https://github.com/henrycgbaker/llenergymeasure/pull/916))
- Internal cleanup (no behavior or results change): dead server-arc internals accumulated across the v0.7 build are removed. Gone are the superseded `WindowManager.run_levels` multi-level driver (and its unused inter-level `cooldown_seconds` constructor knob; the live cooldown stays config-driven in the session), the write-only `WindowRecord.start_event`/`stop_event` fields, the unused latency half of `WindowBookkeeping` (the `LatencyRecord` class and the `latency_records` / `issued_in_span_count` / `completed_in_span_count` / `straddling_count` fields; the span-clipped energy denominator and attribution policy stay), the test-only `ServerSessionResult.window_count` property, and the orphaned `describe_offline_warmup_protocol` helper. Two stale docstrings are corrected to match the wired code (`build_window_bookkeeping` and the server-warmup precedence resolver). ([#917](https://github.com/henrycgbaker/llenergymeasure/pull/917))
- Internal restructure (no behavior or results change): the resolved `RunnerSpec` value object moved from `infra.runner_resolution` to the config layer (`llenergymeasure.config.runner_spec`), so the study layer no longer imports infra for a config-derived fact. The engine-installed host check is now a public `harness.preflight.check_engine_installed`, and the `LLEM_DOCKER_GPUS` vs `study_execution.gpu_indices` conflict warning is emitted from a single choke point in `orchestrate_study`.
- Offline warmup convergence (`convergence_detection: true`) now uses stable-through-end semantics instead of a single trailing window: the loop converges only once the trailing eligible region is a SUSTAINED plateau (every window-sized slice below `cv_threshold`), not merely when the last window happens to be quiet. This shares the coefficient-of-variation detector math with the server warmup gate (the configurable `cv_threshold` is preserved). The stability comparison is now `CoV <= cv_threshold` (was strictly `<`), matching the shared detector's boundary semantics - a value exactly at the threshold now counts as stable. A convergence run that would previously have stopped on a single lucky quiet window now keeps warming until the plateau holds - a behavior change for offline convergence mode only (the default fixed-count warmup is unchanged).
- Study `experiments:` list entries now DEEP-merge onto the fixed study config, matching the sweep-axis path (which already deep-merged). Previously an explicit entry shallow-replaced whole top-level sections, so an entry that re-declared any part of a nested section (for example `server.traffic.rate`) silently dropped that section's fixed-level siblings (for example a fixed `server.warmup`); the two ways of expressing the same study disagreed, and a dropped block read as study-unset. Now an entry overrides only the keys it names and inherits the rest from the fixed level. This is a clean break in merge semantics: a study that relied on an entry wholesale-replacing a section must now write the full replacement section explicitly in that entry. Top-level scalar overrides are unaffected (a scalar in an entry still replaces the fixed value).
- The `measurement.warmup` config section moved to `offline.warmup` (a clean break, no alias): `measurement:` now holds only mode-invariant methodology (`baseline`, `energy_sampler`, the measurement-window knobs). A config that still nests `warmup` under `measurement` fails with an actionable error naming the new home. As a deliberate consequence of the mode-namespace projection, an offline config's declared and resolved config hashes SHIFT (the warmup content leaves the `measurement` view slot; a default-offline study that sets no warmup knobs projects an empty mode-section, so its fingerprint moves). This is a within-study-consistent re-fingerprinting - all offline configs shift uniformly, so dedup relationships are unchanged - and it folds into the same untagged bundle-2.0 break. The shipped example, tutorial, and CI configs are migrated to `offline.warmup`.

### Removed

- Deleted the orphaned `scripts/_drift.py` invariant-drift checker (493 LoC) and its test. It had no CI, Makefile, or `absorb.py` consumer. ([#850](https://github.com/henrycgbaker/llenergymeasure/pull/850))
- Trimmed the engine capability matrix and dropped the Runtime Limitations table to what the mined engine schema can back: eight hand-authored capability rows and all seven `get_runtime_limitations()` rows are de-claimed, keeping the five field-presence-derivable rows. `docs/reference/engines/invalid-combos.md` is regenerated accordingly. Docs-only: no runtime, CLI, or dispatch path reads these functions. ([#866](https://github.com/henrycgbaker/llenergymeasure/pull/866))
- Dropped the unused `extra_mounts` field from `RunnerSpec`; it was never populated by the runner-resolution chain, so it was wrong-altitude state on a resolved value object. The facade-level `DockerRunner(extra_mounts=...)` parameter (the legitimate mount injection point, used for the baseline-cache bind mount) is unchanged. ([#872](https://github.com/henrycgbaker/llenergymeasure/pull/872))

### Fixed

- `TransformersEngine.load_model` now wraps model/tokenizer construction failures in `EngineError` (naming the model and engine), at parity with the vllm and tensorrt plugins, so a missing/gated model, OOM, or bad dtype surfaces as one actionable error rather than a bare traceback. Separately, when energy auto-selection finds no available sampler, the per-sampler probe reasons are now folded into the structured `energy_measurement_unavailable` warning, not only logged. ([#855](https://github.com/henrycgbaker/llenergymeasure/pull/855))
- The GPU CI in-container suite job no longer livelocks under co-tenant host load: OpenMP/MKL thread pools are capped and set to passive wait, the suite container gets a CPU quota, a hung test dumps all thread stacks after 30 minutes instead of dying silently at the job timeout, and containers orphaned by a job cancellation are killed by an always-run cleanup step. ([#864](https://github.com/henrycgbaker/llenergymeasure/pull/864))
- The dispatch-metadata hermeticity stub now covers the whole unit suite rather than only `tests/unit/docker/`: the baseline-container and engine docker-path tests no longer fail with `PackageNotFoundError` when the suite runs in-container from a source tree with no install. The real-metadata assertions keep their skip-when-absent behaviour. ([#867](https://github.com/henrycgbaker/llenergymeasure/pull/867))
- Test and docstring tidy with one new orchestration regression test: the GPU-selector conflict warning choke point now has coverage (it fires once per dispatch when a Docker runner is present and never for an all-local study). Stale harness test docstrings name the current `build_result` seam, and the `WarmupConfig.thermal_floor_seconds` docstring no longer misattributes its default to an MLPerf Power mandate. Value and validation unchanged. ([#871](https://github.com/henrycgbaker/llenergymeasure/pull/871))
- The subprocess measurement session no longer leaks resources when its `__enter__` fails mid-acquisition: a raise from the pre-dispatch GPU-residual check now releases what was acquired (consumer thread, staging tmpdir, both pipe ends) and re-raises, and teardown is hardened so a `_cleanup` consumer-stop raise can no longer skip the pipe close and staging-dir removal. Related doc/comment references were also refreshed to the current API (`write_environment(*, host_snapshot, runner=...)`, `build_offline_metrics`, the unified runner provenance block). ([#872](https://github.com/henrycgbaker/llenergymeasure/pull/872))
- Runner auto-detection is now container-self-aware. When llenergymeasure runs inside a container without a usable Docker socket, auto-detection selects process mode instead of attempting docker-in-docker (previously a stray `docker` CLI on PATH would make it try); with a mounted socket it keeps container mode (docker-outside-of-docker siblings via the host daemon). Multi-engine elevation gained the same awareness, failing with an actionable error instead of attempting DinD. In the socket-mounted (docker-outside-of-docker) topology, Docker pre-flight no longer hard-fails on a missing local NVIDIA Container Toolkit, since GPU injection is performed by the host daemon. Explicit runner pins (env var, study YAML, user config) are unchanged. ([#892](https://github.com/henrycgbaker/llenergymeasure/pull/892))
- The docker-outside-of-docker (DooD) topology error in the server readiness path no longer fires on the first connection failure, which aborted valid launches in the supported DooD topology because a sibling server still loading its model is not yet listening and that transient connection refusal is indistinguishable from permanent unreachability. Topology is now diagnosed only at the readiness deadline: it is raised as the terminal explanation (with the container's `docker logs` tail attached) only when the whole budget is spent on the container leg with every probe having failed at the transport level under DooD, while any HTTP response anywhere makes an exhausted deadline an ordinary readiness timeout. ([#897](https://github.com/henrycgbaker/llenergymeasure/pull/897))
- A composite server warmup that reaches its timeout now proceeds with the documented `timed_out` disclosure instead of failing the experiment when the warmup traffic schedule ends exactly at the timeout boundary.
- A grouped server session that ran to completion but whose every rate level failed its stability gate (or aborted during warmup) now reports all N of its cells as failed in the study summary, not one. The session-level failure dict carries its group size so the cell accounting stays balanced (completed plus failed equals the number of experiments); previously only the whole-group launch-failure path carried it, so a fully-invalid grouped session under-counted its failures.
- Docker-runner rescue tests (`TestConfigSidecarRescue`, `TestTimeseriesParquetRescue`) no longer flake under randomized test ordering: the `tempfile.mkdtemp` mock now routes by prefix instead of assuming a fixed call count, so the process-cached dispatch-asset materialisation can no longer steal the rescue tempdir slot.


## [v0.6.0] - 2026-07-18

### Added

- Docker dispatch can now be scoped to specific host GPUs from config via the new
  `study_execution.gpu_indices` field (a list of host device indices, e.g. `[2, 3]`),
  translated to `docker run --gpus device=2,3`. Scoping at the docker level keeps CUDA and
  NVML indices consistent inside the container (both re-enumerate from 0), so energy
  attribution stays correct. The process-global `LLEM_DOCKER_GPUS` env var overrides this
  field (env>config); when both are set the env wins and a warning is logged. The field is
  placement metadata and is excluded from the declared-config and study-design hashes, so
  pinning a study to different physical GPUs never changes dedup grouping. The same selector
  drives per-GPU advisory-lock naming, the baseline container's `--gpus`, and the per-target
  baseline cache key, so config-pinned studies lock, baseline, and cache the correct physical
  devices. ([#838])
- The Docker `--shm-size` for llem-launched containers is now configurable via the
  `LLEM_DOCKER_SHM_SIZE` env var (default `8g`, the previous hardcoded value). Raise it for
  very large tensor-parallel runs or lower it on memory-constrained hosts. ([#838])
- The per-experiment `environment.json` sidecar now records a `runner` block
  restoring runner provenance to every result: `mode` (`docker` vs `local`),
  `image`, `image_digest` (the resolved registry digest `repo@sha256:...`,
  pinning the full software stack as the cross-run reproducibility anchor), and
  `source` (the precedence layer that selected the runner). The sidecar also
  gains its own `schema_version` (`"1.0"`, independent of `result.json`), its
  first explicit version. The digest is resolved host-side via
  `docker image inspect`; resolution is best-effort and records `null` (never
  fails a run) for local runs, locally-built images, or when docker is
  unavailable. Older sidecars load with `runner: null`. ([#837])
- Generic-environment documentation. A new how-to page, "Running on a cloud GPU
  VM", covers AWS/GCP/Azure GPU instances end to end: prerequisites (linking the
  canonical NVIDIA driver, Docker, and NVIDIA Container Toolkit guides),
  known-good provider images and network egress needs, `pip install
  llenergymeasure` through `llem doctor`, a first measurement, and a multi-engine
  study. It states the supported-environment matrix (supported: bare-metal GPU
  hosts, any Docker-capable GPU host, cloud GPU VMs; out of scope for now:
  Slurm/apptainer, Windows native, fractional-GPU power measurement) and adds MIG
  operational guidance (`LLEM_DOCKER_GPUS=device=MIG-<uuid>` slice pinning with
  the per-physical-GPU power-telemetry caveat). The Docker setup guide gains a
  conservative rootless-Docker/Podman note (detected but untested) and a MIG
  cross-reference. This documents capabilities the dispatch code already had.
  ([#840])
- Distributions are now published to PyPI automatically on tagged releases via OIDC trusted
  publishing (no API tokens or stored secrets). The release pipeline builds the sdist and
  wheel once, attaches them to the GitHub Release, and a separate `publish-pypi` job uploads
  those same bytes to PyPI, so both carry identical artefacts. `pip install llenergymeasure`
  becomes the supported install path from the next tag onward. This is the package's first
  PyPI onboarding: the maintainer configures the trusted publisher on pypi.org before the
  first tagged publish, and a not-yet-configured publisher fails only the publish step (the
  build and GitHub Release still succeed). The `package-validation` CI check was also upgraded
  from an import-only probe to a real CPU-only smoke that installs the built wheel into a clean
  virtualenv, imports the public API, and exercises the CLI to a zero exit. Release-process and
  install docs were corrected to describe the actual trusted-publishing mechanism (the previous
  release doc claimed a publish step that was never implemented). ([#826])

### Changed

- Image resolution now warns when a locally-built `llenergymeasure:{engine}` tag
  wins over the version-pinned default. That precedence is intentional (fast local
  iteration), but a months-stale dev tag could hijack resolution invisibly: on one
  host a stale local vLLM tag even failed the schema handshake as a hard mismatch
  while the user had no idea a local image was being preferred. The warning names
  the local tag in use, the version-pinned default it bypassed, and the remedy
  (`docker rmi llenergymeasure:{engine}` to restore the pinned default, or pin an
  explicit image via `runners.<engine>` / `LLEM_IMAGE_<ENGINE>`); it is emitted
  once per resolution. `llem doctor` surfaces the same shadowing fact on the
  affected engine's row. Resolution behaviour is unchanged. ([#843])
- Study preparation now pulls missing Docker engine images concurrently (one
  `docker pull` per thread, capped at 3) instead of serialising them. A
  multi-engine study on a fresh box no longer waits for several multi-GB pulls
  back to back. Locally cached images are still inspected first, so a cached
  image never triggers a remote call, and progress output stays coherent (each
  image's lines are serialised, never interleaved). A single failing pull no
  longer cancels its siblings: every pull runs to completion and any failures
  are reported together as one aggregate error that names each image and its
  cause (registry-unreachable vs image-absent). ([#832])
- `llem doctor` is now the single environment health check. It reports GPU/driver,
  per-engine availability (importable locally or via Docker, with image-cache state),
  energy samplers (NVML/Zeus/CodeCarbon), Docker (CLI/daemon/NVIDIA Container Toolkit),
  `HF_TOKEN` presence (detect-and-advise - the value is never printed), the resolved user
  configuration with per-setting provenance, and the image schema handshake folded in as a
  section. Every line is prefixed `[ok]`/`[warn]`/`[fail]` with a `-> fix` hint. `--check`
  exits 0/1/2 (ok/warnings/errors) for CI scripting and `--json` emits the full report as
  machine-readable JSON. Plain `llem doctor` still exits non-zero on a hard failure - an
  image schema mismatch or an unparseable/invalid user-config file. ([#834])
- Multi-engine Docker elevation is now precedence-based. An engine whose runner is
  explicitly pinned (env var, the study `runners:` section, or user config) keeps that pin;
  only engines left on auto-detection are elevated to Docker for isolation. Engines pinned
  to `local` in a multi-engine study are checked for host importability at preflight, with a
  specific error naming the engine, the missing package, and the two fixes (install the
  engine extra, or drop the explicit local pin). Docker is required only when an
  auto-resolved engine actually needs elevating, so an all-explicit-local multi-engine study
  now runs without Docker. Previously every engine in a multi-engine study was
  unconditionally elevated to Docker, which failed when Docker was absent even for engines
  the user had pinned to local. Runner choice is machine-binding and recorded per result.
  ([#835])
- `configs/example-study-full.yaml` is now a lean, runnable multi-engine example (100 lines,
  down from 585). The previous file was a 52,032-run reference marked "not intended for
  end-to-end execution" and carried a "KNOWN-BAD crosses" block plus stale TODOs. It now
  runs as-is (17 experiments) and covers every top-level section of a study spec. For the
  exhaustive per-engine field surface it previously duplicated, the header points readers at
  `llem study init` (schema-derived, so it cannot drift). ([#839])

### Removed
- `llem config` is removed. Its environment-diagnostics role is subsumed by the broadened
  `llem doctor` above. There is no deprecation shim (pre-PyPI); scripts should call
  `llem doctor` (or `llem doctor --check` / `llem doctor --json`). ([#834])

### Fixed

- The container dependency-priming probe now verifies each runtime requirement
  actually imports in the container interpreter, not just that its distribution
  metadata is present. Metadata presence does not prove importability: a package
  can be installed yet fail to import when a compiled extension was built for the
  wrong ABI or the install is otherwise broken, and the old presence-only check
  left such a dependency unprimed. The probe now resolves each distribution's
  top-level import name (a small override table covers the known non-identity
  cases `nvidia-ml-py` to `pynvml`, `pyyaml` to `yaml`, and `python-dotenv` to
  `dotenv`, then `top_level.txt` / `packages_distributions`, falling back to the
  normalised distribution name) and imports it, priming any requirement whose
  import raises. Absent metadata still short-circuits as missing with no import
  attempt, and the requirements-hash fast-path stamp is unchanged. ([#845])
- Fresh `pip install` + Docker dispatch no longer crashes the transformers and
  TensorRT-LLM engines inside their containers. The package bind-mount now
  exposes only the `llenergymeasure` package (mounted at
  `/llem-src/llenergymeasure`) instead of the package's parent directory. For a
  wheel install that parent is the venv's entire `site-packages`, and because
  the container entrypoint prepends `/llem-src` to `PYTHONPATH` (which precedes
  the container's own site-packages on `sys.path`), every host third-party
  package used to shadow the image's native copy - a host `pydantic_core` C
  extension built for a different Python minor broke transformers, and a fresh
  `huggingface-hub` broke TensorRT-LLM's version guard. Mounting the package
  directory alone makes `/llem-src` contain nothing but `llenergymeasure`, so
  host dependencies can no longer shadow container-native ones. One uniform
  mount serves editable and wheel installs alike. ([#844])
- `llem run -o/--output-dir` is now honored for fresh (non-resume) studies. The flag
  was documented as "Output directory for results" but was silently a no-op on a fresh
  run: `run_study` only consumed `output_dir` as the auto-detect-resume search base and
  never threaded it into the results-dir resolution, so results always landed in the YAML
  `output.results_dir` (default `./results`). Fresh runs now resolve the base with
  precedence `-o` override > YAML `output.results_dir` > user config > `./results`, and the
  preflight/dry-run panel's "Study results path" reflects the same precedence so the
  preview matches where results actually land. Resume semantics are unchanged (`-o` stays
  the search base for `--resume`). The results path is placement metadata, excluded from
  the declared-config, study-design, and dedup hashes, so pointing a study at a different
  output directory never changes dedup grouping. ([#842])
- Experiment failures now surface their real cause. Under `-v`, a Docker container
  failure prints the traceback the container entrypoint captured (the actual
  engine/CUDA error), instead of the uninformative host-side traceback of the
  DockerRunner raise site. Local and subprocess dispatch failures (single- and
  multi-experiment) now persist their captured traceback to
  `failed-runs/{config_hash}_cycle{N}_traceback.txt` and record a `log_file`
  pointer in the manifest, matching the Docker path so a failure is debuggable
  regardless of dispatch mode. ([#836])
- Resolved-config and observed-config hashes now canonicalise integral numerics onto a
  single form, closing an int-vs-float gap of the same class #822 fixed for the declared
  hash. A field typed `float` but valued as an int (e.g. vLLM `cpu_offload_gb = 0`) stayed
  a python `int` in the resolved view's `mode="python"` dump but was a genuine `float` in
  the native engine object the observed pipeline captured, so semantically identical
  configs hashed differently. The shared `_normalise` helper (`domain/hashing.py`) now
  folds any integral-valued float onto its `int` form, so all three hash paths (declared,
  resolved, observed) agree. Folding toward `int` (not `int -> float`) keeps genuine
  integer identity fields (seeds, token counts) bit-exact. This corrects both dedup
  grouping (`resolve_library_effective`, which decides how many experiments physically run)
  and observed-collision gap detection (`find_observed_collisions`, which feeds the rules
  corpus). Hash values change only for configs containing an int-valued float field;
  pre-1.0, old persisted bundles keep their recorded hashes (no migration) and a study
  resumed across the boundary may regroup. ([#833])
- `.env.example` and `docker-compose.yml` no longer point at bootstrap tooling that does
  not exist. The file header and PUID/PGID guidance referenced a `setup.sh` auto-generator
  that was never shipped, and the `docker compose` PUID/PGID error told users to run
  `llem doctor` to auto-generate `.env` (doctor never writes `.env`). All three now give
  the real instructions: `cp .env.example .env`, then set `PUID`/`PGID` from `id -u` /
  `id -g`. Also removed the dead `LEM_ENGINE=pytorch` block from `.env.example` (wrong
  prefix, pre-rename engine name, and no consumer anywhere in the codebase). ([#831])
- Corrected stale user-facing strings a pip-installed user could encounter. Shipped error
  messages in `llem doctor`, the host engine-import guard, and the run preflight now point at
  the published documentation site instead of a `docs/development.md` filesystem path that was
  never packaged. The install docs' `git clone` URL was corrected to the real repository name
  (`llenergymeasure.git`), and stale v0.9.0-era version strings in example output across the
  install, FAQ, troubleshooting, and docker-setup how-tos were refreshed to the current
  release. Build-time benchmark figures that named an internal host are reframed against a
  generic reference machine. The wheel no longer ships the developer `README.md` files it
  previously bundled (`*.md` excluded from the wheel build; runtime data and `py.typed` still
  ship). ([#827])
- A TensorRT-LLM config that sets a prebuilt `engine_path` without `backend: trt` is now
  rejected at config validation with an actionable error, instead of silently constructing the
  pytorch flow against a compiled-engine directory. `engine_path` points the loader at a
  directory of compiled `rank*.engine` files that only the trt constructor can read; the
  pytorch constructor (the default when `backend` is unset) treats its model argument as a
  HuggingFace checkpoint, so the mismatch previously surfaced as an opaque model-load failure
  deep in the engine rather than a clear config error. The guard fires whenever `engine_path`
  is set and `backend` is not `trt` (covering unset, explicit `pytorch`, and explicit null);
  set `backend: trt`, or drop `engine_path` to build from the model checkpoint. ([#828])
- `cycle_gap_seconds`, the longer thermal-equalisation pause between cycles, no longer fires
  mid-repetition under `sequential` experiment order. The runner placed the cycle gap using
  positional modulo math that is only valid when experiments are laid out as full passes over
  the configs (`interleave`/`reverse`/`shuffle`/`latin_square`); under `sequential` order the
  sequence is `[A,A,A,B,B,B,...]`, so the gap landed inside a config's repetition block rather
  than at a cycle boundary. Under `sequential` order the cycle gap now fires between the
  per-config repetition blocks (once per transition to a different config); back-to-back
  identical repetitions are separated only by the smaller `experiment_gap_seconds`. All
  pass-structured orders are unchanged. ([#829])
- Container dispatch (`runner: docker`) now works from a plain `pip install`, not only from a
  source checkout. The dispatch inputs (the container entrypoint script and the runtime
  dependency list) were previously resolved by walking up from the repo layout, which does not
  exist under a site-packages install: docker then bind-mounted nonexistent sources, the
  entrypoint mount became an empty directory, and every engine's docker run failed immediately.
  The entrypoint script now ships as package data and both dispatch inputs are materialised
  from the installed package, so dispatch behaves identically from a checkout and from an
  installed wheel. A preflight error is raised before `docker run` if a dispatch asset cannot
  be materialised, replacing the silent empty-directory mount. ([#830])

## [v0.5.1] - 2026-07-17

### Added

- `result.json` now exposes `input_tokens` and `output_tokens` alongside `total_tokens`
  (the actual tokenised counts as observed by the engine, where
  `total_tokens = input_tokens + output_tokens`). The split was already computed in the
  harness for the FLOPs-per-token fields but never persisted; exposing it enables auditing
  declared-vs-actual input lengths. Rides the unreleased schema 5.0 (additive, no version
  bump). ([#819])
- The `timeseries.parquet` sidecar now carries `experiment_id` and
  `measurement_config_hash` as Parquet file-level key-value metadata (not columns), so the
  artefact stays attributable if separated from its result directory. This mirrors the
  identity fields the JSON sidecars already carry and completes the per-experiment bundle
  identity rationalisation. Data columns and schema are unchanged; readers that ignore file
  metadata are unaffected. ([#813])
- The per-experiment `config.json` sidecar now carries its own `schema_version`
  (`"2.0"`), independent of `result.json`'s schema version. It succeeds the retired
  `_resolution.json` sidecar (`"1.0"`), whose per-field provenance now lives in this file.
  ([#811])
- `equivalence_groups.json` now records `study_name` alongside `study_id`, so the
  study-level artefact stays attributable if separated from its parent directory. ([#811])

### Changed

- Per-field config provenance (which fields were overridden and why: CLI flag, sweep, or
  YAML) is folded into the `config.json` sidecar under a new `provenance` section instead
  of a standalone `_resolution.json` file. `config.json` is now the single home for both
  declared/observed config and its provenance. Consumers (including `llem report-gaps`)
  read the `provenance` section; the `_resolution.json` file is no longer written or read.
  Pre-1.0, bundles produced by older versions are not backfilled: their `_resolution.json`
  is simply ignored. `config.json` now materialises in the experiment directory on every
  successful run, including the docker (multi-engine) path and runs with
  `save_timeseries` off: the docker runner rescues `config.json` from the container
  exchange dir alongside `timeseries.parquet`, and the local path always stages an output
  dir for the sidecar. If a completed experiment ends without a `config.json`, the runner
  logs a warning rather than dropping the provenance silently. ([#811])
- Docs: purged stale `effective_config` terminology and pre-consolidation flat-layout
  references across the methodology, how-to, tutorial, and generated reference pages,
  aligning them with the `config.json` sidecar and the current bundle layout. ([#820])
- Docs: refreshed version and project-status references (citation, release process,
  roadmap, landing page) to the post-v0.11.0 state. ([#824])
- `result.json` schema bumped `4.0` -> `5.0` (breaking). `result.json` is now measurement
  output: the configuration/methodology fields `engine_version`, `measurement_methodology`,
  `steady_state_window`, `measurement_window_discard_fraction`, and
  `steady_state_not_detected` are removed from it and live in the `config.json` sidecar as
  top-level fields. `result.json` keeps `model_name` and `engine` as convenience copies -
  deliberate small duplication so a result file stays self-describing when separated from
  its directory; the authoritative home for both is `config.json`. In the sidecar, the
  engine's library version is now named `engine_version` (was `library_version`); the config
  sidecar schema stays `2.0` because it is still unreleased. Pre-1.0, old bundles are not
  migrated in place: they load as their own `schema_version` and are unaffected on disk.
  Consumers read methodology and authoritative identity from `config.json` (see the
  results-schema and `run_study` reference docs for the join pattern). The `config.json`
  sidecar is guaranteed to materialise next to every `result.json` on all successful runs -
  including the docker (multi-engine) path and runs with `save_timeseries` off - so the
  join never dangles. ([#812])

### Fixed

- The `config.json` and `environment.json` sidecars now materialise in the experiment
  directory under docker dispatch. Atomic writes (`result.json`, both sidecars, the study
  manifest, and equivalence groups) are now created world-readable (0644) instead of the
  0600 that `tempfile.mkstemp` produces regardless of umask. Previously a root container
  wrote these two sidecars 0600, and the non-root host then hit `PermissionError` reading
  them during the rescue step - swallowed at debug level - so both sidecars were silently
  dropped from every docker-dispatched bundle (`result.json` and `timeseries.parquet`
  survived because they were already written 0644). Sidecar-rescue failures now log a
  warning naming the path and reason rather than vanishing. ([#823])
- `llem run` warnings now reach the terminal. The package installs a `NullHandler` at
  import time, which the logging setup mistook for an already-configured handler and so
  never attached the real stream handler - suppressing every `WARNING`-level message
  (including the sidecar-rescue backstops above) on the normal run path. The setup now
  ignores the placeholder and attaches the stream handler. ([#823])
- Declared-config and study-design hashes are now computed on a json-mode
  `model_dump`, so a field typed float but defaulted to an int literal (e.g. vllm
  `cpu_offload_gb = 0`) hashes identically whether or not the config has been
  through a JSON round-trip. Previously the host hashed the in-memory int (`0`)
  while the container re-validated the config the host wrote and hashed the
  coerced float (`0.0`), producing different hashes; the host then named the
  result file with a hash the container never wrote and reported "Container
  exited 0 but no result file found" for a successful run. Configs that pinned
  such fields explicitly (e.g. `cpu_offload_gb: 0.0`) no longer need to. ([#822])
- `environment.json` now records the environment the experiment actually ran in for
  docker-dispatched experiments, instead of the dispatching host's environment. The
  container entrypoint collects the environment snapshot inside the container, threads it
  into the harness, and persists it to the exchange dir; the docker runner rescues it
  alongside `config.json` and `timeseries.parquet`, and the study layer prefers the rescued
  in-container snapshot over its host-collected cache. Previously the host snapshot (wrong
  python version, `cuda_version` null, `container.detected` false) was written for every
  docker run, defeating reproducibility metadata on the multi-engine path. Local
  in-process runs are unchanged. A docker run that completes without a rescued snapshot now
  logs a warning rather than silently recording host values. ([#821])

## [v0.5.0] - 2026-07-16

### Added

- GPU CI engine matrix: `gpu-ci.yml` now boots each pinned upstream engine container
  (vllm, tensorrt both backends) and runs one tiny inference through the real `llem run`
  docker-dispatch path, auto-triggered when a PR touches engine pins, engine plugins, or
  the dispatch surface. GPU jobs serialize to respect the shared runner's free devices;
  the transformers test container now mounts project source per the image contract. ([#806])
- `ExperimentResult.engine_build_cache_hit`: whether the tensorrt trt-backend engine build
  was served from the on-disk build cache (`true`) or compiled fresh (`false`); `null` when
  the cache is not in play (pytorch backend, other engines, an `engine_path` override, or the
  cache disabled). Detected from TRT-LLM's own `llm_build_stats.engine_dir`, which is
  populated only on the cache-reuse path (the sibling `cache_hitted` flag is unusable - it is
  `True` on both the reuse and fresh-build paths). Annotates `model_load_time_sec` (a hit
  skips the compile). Additive optional field; result schema_version stays 4.0. ([#804])
- `llem doctor` now reports the TensorRT-LLM engine build cache: location, engine-entry count,
  total size, and the manual clean command. Informational only (never affects the exit code) -
  the cache lifecycle is manual and visible, and llem never auto-evicts. ([#804])
- `ExperimentResult.model_load_time_sec`: wall-clock seconds spent in `engine.load_model()`
  (model load plus any engine build/compile performed there - the tensorrt trt backend's
  TRT engine build, vLLM torch.compile / CUDA-graph capture). Captured by the harness
  around the load call, so all three engines gain it uniformly. Non-energy run metadata:
  the phase completes before the NVML energy window opens. Additive optional field;
  result schema_version stays 4.0. ([#803])
- `LLEM_DOCKER_GPUS`: docker `--gpus` request for llem-launched experiment and baseline
  containers (default `all`, the historical behaviour). On shared multi-GPU hosts, pin llem
  to free devices (e.g. `device=2`); restricting at the docker level keeps CUDA and NVML
  indices consistent inside the container. ([#803])
- TensorRT per-request latency metrics under `latency_profiling`: the plugin now sets
  `SamplingParams(return_perf_metrics=True)` when latency profiling is enabled and extracts
  per-request TTFT / E2E / average TPOT from `RequestOutput.metrics_dict` (the TRT-LLM 1.x
  surface, live-verified on both backend legs at 1.2.1; the vLLM-shaped
  `RequestOutput.metrics` namespace does not exist there). Capture mode is
  `per_request_batch`. Default runs are unchanged: the flag is never set without
  latency profiling, and the `latency_profiling_unsupported` degradation warning now
  fires only when profiling was requested but the engine returned no metrics. ([#803])
- TensorRT-LLM schema discovery now mines BOTH backend args classes and unions them into
  one discovered engine-params surface (63 -> 95 fields): `TrtLlmArgs` (the `trt` backend)
  and `TorchLlmArgs` (the `pytorch` backend). Each field carries a `backends` applicability
  list (`[pytorch, trt]`, `[trt]`, or `[pytorch]`) recording which backend's args class
  carries it. Codegen keeps a single `EngineParams`; the metadata is descriptive (surfaced
  as a Backends column in the schema reference), and cross-backend applicability is enforced
  by loud validation rules, never silent dropping. Absence of the key means "all backends",
  so the single-class vllm/transformers schemas are byte-unchanged. ([#801])
- Exposure-time field narrowing via a new optional `exposure_overrides` block in
  `curated.yaml`: narrows a generated config field (`enum` -> `Literal`, `default`) without
  touching the mined schema. Used to expose tensorrt `backend` as `Literal["pytorch", "trt"]`
  defaulting to `"pytorch"` while the mined type stays `str`. ([#801])
- Backend-applicability rules in the tensorrt corpus (27 -> 29): `fast_build` and
  `quant_config` require `backend='trt'` (both exist only on `TrtLlmArgs`; the pytorch
  backend's `TorchLlmArgs` rejects them under `extra='forbid'`). Live-verified by direct
  construction at 1.2.1; enforced at the config-expansion grain, complementing the plugin's
  construction-grain `ConfigError` guards (defense in depth at both grains). ([#801])
- The deterministic cross-field extractor's target table now walks the `TorchLlmArgs`
  validator tree (`stream_interval`, `batch_wait_*`, `ray_*` preconditions,
  `speculative_config`), joining the pytorch-backend validators to the standing candidate
  surface. ([#801])
- Standing plugin-kwarg lint: `scripts/check_plugin_kwargs.py` cross-checks the literal
  constructor-kwarg names each engine plugin's translation layer hand-types against that
  engine's discovered schema at the current pin, with a rationale-carrying allowlist for
  genuine off-surface kwargs (transformers `from_pretrained` open `**kwargs`). Catches
  upstream kwarg renames the mined schema already knows about but the hand-written glue
  code missed (the `quantization` -> `quant_config` case). Wired as
  `make check-plugin-kwargs` and a `plugin-kwarg-check` job in ci.yml. ([#800])
- Standing deterministic cross-field extractor (`scripts/cross_field_extractor.py`): the
  value core of the retired per-version invariant miners - the AST walk extracting
  cross-field raise/normalisation conditions from validator bodies - generalised into one
  engine-generic proposer driven by per-engine descriptors. No LLM, no network, no engine
  import: it reads the pinned source tree on the host, so it covers CUDA-binding engines
  and is byte-stable. Wired into absorb as a third standing pool source alongside the
  analyst cold read and manual seeds; the verification ladder still adjudicates every
  candidate. Pool and corpus dedup now also collapse candidates on the canonical match
  spec (severity + fields + operators + values), dropping two shipped duplicate tensorrt
  rule pairs (29 -> 27). ([#795])

### Changed

- TensorRT-LLM user docs rewritten to match the activated engine at 1.2.1: both measured
  backends as a config axis, the build-cache lifecycle and result annotations,
  single-process multi-GPU with `LLEM_DOCKER_GPUS` pinning and `NCCL_*` forwarding, and
  hardware/quantisation constraints. Stale claims contradicting the code were fixed: the
  invented `build_metadata` result field, the removed `mpirun` entrypoint wrapping,
  "latency always null for tensorrt", broken flat-layout YAML examples, and five 1.0.0
  image pins. ([#815])
- Energy-methodology doc updated for fields that had moved: the GPU-telemetry section now
  documents study-level `output.save_timeseries` (its experiment-level `gpu_telemetry`
  predecessor was renamed when `OutputConfig` was extracted), and `energy_sampler` is shown
  under `measurement:` rather than as a flat top-level key. ([#808])
- The upstream-image engine-version handshake probe (a ~60s cold `docker run` per engine
  per study) is now cached on disk keyed by the image content digest, under
  `platformdirs.user_cache_dir("llem")/image-probe/`. A warm image resolves from the cache
  (a `docker image inspect`) instead of re-probing, so the per-study cost the F2 plan flagged
  drops to near zero. The in-process memo remains the first tier; a corrupt, unreadable, or
  unwritable cache degrades to a fresh probe and never crashes. Entries never go stale: a
  rebuilt or re-pulled image gets a new digest and a fresh probe, so no TTL is needed. ([#805])
- TensorRT-LLM backends are now selected by constructor class, not a kwarg: `backend='trt'`
  resolves `tensorrt_llm._tensorrt_engine.LLM` and `pytorch`/unset resolves `tensorrt_llm.LLM`,
  validated live at 1.2.1 (the base `LLM` rejects `backend='trt'` at model load). `backend` is
  no longer forwarded as a kwarg; an unsupported value raises `ConfigError` naming
  `{pytorch, trt}`. ([#797])
- TensorRT-LLM pin advanced 1.0.0 -> 1.2.1 through the full bump pipeline: schema re-mined
  byte-stably, the 20-field curation carried forward with no discovery debt, typed config
  regenerated (`max_num_tokens` default now 8192), and the shipped rules corpus grown 15 -> 29
  (construction-confirmed additions plus human-signed residue). ([#792])

### Fixed

- vLLM per-request latency was never captured at 0.19.1: the extractor read the V0
  `RequestOutput.metrics` surface, which no longer exists under V1, and the offline `LLM`
  entrypoint forces `disable_log_stats=True`, so `latency_stats` came back null on every
  vLLM run. The plugin now enables engine stats only when latency profiling is requested
  (mirroring the tensorrt `return_perf_metrics` gate, so default energy runs are untouched)
  and reads the V1 `RequestStateStats` surface: TTFT from `first_token_latency`, E2E from
  the monotonic decode interval, and a decode-average ITL; capture mode is `proportional`.
  The stale V0 extractors were removed. ([#817])
- Memory metrics are no longer a silent 0.0 for out-of-process engines. vLLM V1 runs its
  model in the EngineCore child process and TensorRT-LLM in its executor process, so torch's
  per-process allocator in the driver process saw nothing and `extended_metrics.memory`
  `peak_memory_mb` / `model_memory_mb` came back as exactly 0.0 on every vLLM and TRT-LLM
  experiment (real values only for in-process Transformers). Both capture points now fall
  back to NVML device-used memory when the torch reading is implausible (`== 0.0`): the
  harness model-memory baseline, and the plugin peak-memory read (vLLM's NVML fallback was
  previously gated on the already-broken `peak > 0` torch value so it never fired; TRT-LLM
  had no fallback at all). Transformers keeps its authoritative torch reading and never
  consults NVML. The cascade fields that were nulled by the zero (`tokens_per_gb_vram`,
  `model_memory_utilisation`, `kv_cache_memory_ratio`) now populate. NVML `used` is a
  whole-device reading (it includes this process's CUDA context and any co-tenants), so the
  absolute peak/model figures are an upper bound; the derived `inference_memory_mb` delta
  cancels the shared context term and stays meaningful. Under `LLEM_DOCKER_GPUS` pinning the
  container sees only the pinned device(s), so NVML index resolution matches the experiment's
  own GPUs. The raw memory fields are now nullable and any residual 0.0 is coerced to null at
  the domain boundary - the fields are always either a real measurement or null, never a
  silently-wrong zero. ([#816])
- llem now forwards every `NCCL_*` host environment variable into both the experiment
  container (`infra.docker_runner`) and the baseline container (`study.baseline_container`),
  so NCCL tuning/workaround settings set on the host reach the engine process, which runs
  inside the container. The motivating case is `NCCL_P2P_DISABLE=1` on PCIe multi-GPU hosts
  whose topology lacks functional peer-to-peer (P2P, e.g. an inter-GPU link reported as `SYS`
  in `nvidia-smi topo -m`, often because ACS is enabled): without it, tensor-parallel runs
  hang at the first NCCL collective. Vars are emitted as explicit `-e KEY=VALUE` args
  (matching the existing `LLEM_*` forwarding idiom) in sorted key order for a deterministic
  command. ([#814])
- Multi-device GPU pinning (`LLEM_DOCKER_GPUS=device=1,3`) was rejected by docker: the bare
  `--gpus` value parses as CSV, so it split at the comma into a device id plus a GPU count
  ("cannot set both Count and DeviceIDs on device request"). Comma-bearing `device=`
  selectors are now wrapped in literal double quotes via `env_config.docker_gpus_arg()`;
  the `all`, count, and single-device forms pass through verbatim, and lock-id parsing
  still sees the raw selector. ([#810])
- TensorRT-LLM tensor-parallel runs are no longer launched under `mpirun`: the LLM API
  self-manages TP from a single process at 1.2.x, and the external `mpirun -n {tp}` wrap
  made every rank re-run the whole experiment entrypoint (under `python -m` every rank
  passes the `__main__` guard), so TP=2 either corrupted the shared build cache in the
  upstream write-guard race or OOMed on duplicate executors. The mpirun branch and
  `LLEM_MPI_NP` are removed; the container entrypoint always execs a single `python3` and
  the LLM API spawns its own ranks. Verified live at TP=2 on the trt backend, cold build
  and warm cache hit. ([#809])
- Study GPU advisory locks are now named by the PHYSICAL device a study occupies, not the
  in-container logical index. Lock names came from `_resolve_gpu_indices` (which enumerates
  from 0), but the physical device is chosen at the docker level by `LLEM_DOCKER_GPUS`
  (`--gpus device=N`), which the lock logic never saw. Under pinning the container always sees
  its GPU as logical 0, so a study pinned to `device=2` locked `gpu-0.lock`, and two studies
  pinned to DIFFERENT physical GPUs both contended on `gpu-0.lock` and spuriously serialised.
  A new `env_config.pinned_gpu_lock_ids()` parses the docker selector into physical lock ids
  (`device=2` -> `["2"]`, `device=2,3` -> `["2","3"]`, a `GPU-<uuid>` selector used verbatim
  as a stable per-device id); `all` / unset / unrecognised shapes fall back to the logical
  indices (logical == physical when every GPU is visible). Lock naming only - measurement-side
  index resolution is unchanged. ([#807])
- TensorRT-LLM build cache silently died across containers unless
  `LLEM_TRT_BUILD_CACHE_PATH` was set by hand: the docker runner bind-mounts the host
  `~/.cache/trt-llm` at `/root/.cache/trt-llm`, but TRT-LLM's own default cache root
  (`/tmp/.cache/tensorrt_llm/llmapi/`) is unmounted and evaporates with each ephemeral
  container. The runner now defaults `LLEM_TRT_BUILD_CACHE_PATH` to the mount target (via the
  mount's `extra_env`, mirroring the `HF_HOME` pattern), so compiled engines persist out of
  the box; a host-set value still wins (forwarded last, docker `-e` is last-wins). Verified
  live: upstream `BuildCacheConfig` keying is sound - identical config hits across containers,
  and tensor-parallel / max-shape / quantisation / dtype changes each key to a distinct
  engine hash. ([#804])
- The TRT-LLM pre-quantised-checkpoint preflight message now names what was live-verified at
  1.2.1: AutoAWQ / AutoGPTQ community-format HF checkpoints load on neither backend (the trt
  backend raises `NotImplementedError` on the `quant_method`; the pytorch backend rejects the
  weight layout), and TRT-LLM's supported pre-quantised path is its own ModelOpt export
  (`hf_quant_config.json`). The rejection stands for both backends; the `engine_path` escape
  hatch is unchanged. ([#804])
- vLLM containerized runs at 0.19.1 crashed at engine start with "Cannot re-initialize CUDA
  in forked subprocess": the harness touches torch.cuda (hardware preflight) before the
  plugin constructs the engine, and vLLM's EngineCore worker forks by default. The plugin now
  sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` (setdefault, so a user override wins) - vLLM's
  own prescription for embedding contexts. Found by the first live containerized vllm run at
  this pin. ([#803])
- Container deps-cache stamp is now keyed per engine: the pyproject-verified stamp was shared
  across all images of one python minor, so whichever engine dispatched first suppressed the
  missing-deps probe for the others (the TRT-LLM NGC image, which ships every llem dep,
  masked the vllm image's missing pyarrow -> parquet write crash). ([#803])
- The standing cross-field extractor emitted citations the citation checker rejects
  (single-line spans with no quote), crashing `make absorb` at the `citation_pass_rate`
  stage whenever the candidate pool held an extractor candidate. Fixed at the emitter:
  each citation now spans the outermost contributing guard down to the raise/assignment
  site and carries the verbatim source quote of that span. Claim ids and output stay
  byte-stable; the shipped corpus is unchanged. ([#802])
- Transformers image seed and dev cache hygiene: `make docker-seed-transformers` no longer runs
  a harmful second push that wrote plain `transformers:latest` / `transformers:v<version>`
  images and a clobber-prone `mode=max` cache to `:latest`. The `-buildcache` ref is now the
  single cache manifest; the canonical tags are written only by the promotion and release
  tag-copies. `docker-compose.yml`'s `cache_from` now targets that per-version buildcache ref
  (the engine pin is exported as `TRANSFORMERS_VERSION` by the `make docker-build` wrapper and
  also passed as the build arg, so local builds install the pinned version and actually reuse
  the seeded FA3 layers), and the dead `LLEM_PKG_VERSION` / `LLEM_EXPCONF_SCHEMA_FINGERPRINT`
  build-args and Makefile exports are removed (the Dockerfile stopped consuming them in 0.10.0).
  The Dockerfile pins its `ghcr.io/astral-sh/uv` base (was `:latest`, so every uv
  release invalidated every subsequent builder layer) and records that `MAX_JOBS` must never
  be overridden via build-arg in CI, since it participates in the FA3 layer's cache key. ([#799])
- Release image publish is now a registry-side tag-copy of the promoted seed digest, not a
  hosted rebuild. `docker-publish.yml` points `transformers:<version>` at the already-promoted
  `transformers:transformers-<pin>` via `docker buildx imagetools create`, eliminating the
  flash-attention FA3 compile that OOM'd the hosted runner; the workflow aborts loudly when the
  seed/promotion source is missing. ([#798])
- TensorRT-LLM quantisation is passed as the native `quant_config` kwarg, not the
  long-removed `quantization` name (`TrtLlmArgs` is `extra='forbid'` at 1.2.1, so the old
  name crashed any quantised run at construction). ([#797])
- The TensorRT-LLM plugin no longer forwards TRT-build-only knobs (`fast_build`,
  `quant_config`, the engine build cache) to the pytorch backend, whose `TorchLlmArgs` rejects
  them under `extra='forbid'` - this crashed the default pytorch-backend config. Declaring
  `quant_config` or `fast_build=True` on the pytorch backend now raises `ConfigError` rather
  than silently measuring a different configuration. ([#797])
- The four silent `ImportError` fallbacks for TensorRT-LLM sub-configs (QuantConfig,
  BuildCacheConfig, KvCacheConfig, SchedulerConfig) now raise `EngineError` when the user
  declared that sub-config and the native class cannot be imported, instead of silently
  dropping it and measuring a different configuration than declared. ([#797])
- Corrected stale TensorRT-LLM version references: the `RequestOutput.metrics` comment (metrics
  are absent at 1.2.1 without `return_perf_metrics`, not "usually absent in 0.21.0") and the
  container entrypoint's engine-version list (now vLLM 0.19.1 / TRT-LLM 1.2.1 / Python 3.12).
  ([#797])
- Absorb sign-off records now carry the full withheld rule body, so a maintainer's
  `human_confirmed` mark re-ships a rule even after the withholding run dropped it from the
  corpus; a bodyless mark fails loudly instead of silently skipping. ([#792])
- The construction-probe gate rejects pydantic type-coercion noise instead of confirming
  false-positive rules, and bare present-flag claims are unprobeable by construction. ([#792])
- TensorRT discovery and probe containers route through the NVIDIA entrypoint: the 1.2.1 NGC
  image moved `LD_LIBRARY_PATH` setup into `/etc/shinit_v2`, so bypassing the entrypoint broke
  `import tensorrt`. ([#792])

### Removed

- Vendored per-version invariant-miner machinery, superseded by the standing cross-field
  extractor: 13 static/dynamic miner bodies under `engine_versions/*/producers/`, their
  orphaned proposed/validated outputs, the orchestrator and shim modules, and miner-only
  tests (net -25k lines). The drift check drops the retired miner rows and keeps the live
  schema-producer path. ([#796])

## [v0.4.1] - 2026-07-13

The engine-knowledge-as-data milestone. Hand-curated per-engine config was replaced by
typed Pydantic configs code-generated from validation rules mined directly from engine
source; the engine-coupling restructure, per-engine SSOT version pins, a Docusaurus docs
site, study-authoring commands (`llem study init` / bounds mode / sweep idioms / plan
preview), the absorb conductor, and a rewritten hosted-byte-verification CI all landed on
this line. Engine pins advanced to vLLM 0.19.1, TensorRT-LLM 1.0.0, and Transformers 5.7.0.

### Breaking Changes

- **Hand-curated `engine_configs.py` deleted; per-engine typed configs are now
  code-generated.** The ~1100-line hand-maintained `engine_configs.py` was removed. Each
  engine's typed Pydantic config surface is now code-generated from validation rules mined
  directly from that engine's source, keyed to the pinned version under `engine_versions/`.
  A parity gate guarded the deletion. Author-facing YAML is unchanged; only the internal
  config construction path moved to codegen. ([#733], [#734], [#735], [#736], [#738])

- **Engine validation vocabulary renamed from "invariants" to "rules".** The mined corpus,
  its loader surface, and the CLI/docs terminology now say "rules". A single shipped rules
  corpus with a closed severity enum replaces the prior split, and the invariant-era
  compatibility shims were dropped. YAML that referenced the old `engine_invariants` naming
  must move to the `rules` vocabulary. ([#480], [#572], [#737], [#740])

- **`llem run` reduced to session flags only.** The semantic-override flags were removed;
  experiment parameters now live in the YAML config (author one with `llem study init`), and a
  config path is now required. Flags describe the session; YAML describes the experiment. Removed
  flags and their YAML equivalents:
  - `--model` / `-m` -> `task.model`
  - `--engine` / `-e` -> `engine`
  - `--dataset` / `-d` -> `task.dataset.source`
  - `--n-prompts` / `-n` -> `task.dataset.n_prompts`
  - `--cycles` -> `study_execution.n_cycles`
  - `--order` -> `study_execution.experiment_order`
  - `--no-gaps` -> `study_execution.experiment_gap_seconds: 0` (and `cycle_gap_seconds: 0`)
  - `--timeout` -> `study_execution.wall_clock_timeout_hours`
  - `--no-circuit-breaker` -> `study_execution.max_consecutive_failures: 0`
  - `--fail-fast` -> `study_execution.max_consecutive_failures: 1`
  - `--no-dedup` -> `study_execution.deduplicate_equivalent: false`

  Migrate the quick single-run path:
  ```bash
  # before
  llem run --model gpt2 --engine transformers
  # after
  llem study init -m gpt2 --defaults
  llem run study.yaml
  ```

  Retained session flags: `--output`, `--quiet`, `--verbose`, `--resume`, `--resume-dir`,
  `--dry-run`, `--no-lock`, `--skip-preflight`. Passing a removed flag now gives Typer's standard
  "No such option" (exit 2). ([#749])

- **`engine: pytorch` renamed to `engine: transformers`** throughout YAML, CLI, and Python API.
  The `pytorch` identifier has been renamed to `transformers` - the engine runs HuggingFace
  Transformers `.generate()`. PyTorch is the tensor substrate, not the engine, and renaming aligns
  with `pip install transformers` and the library that owns the inference API.

  Migrate with:
  ```bash
  sed -i 's/engine: pytorch/engine: transformers/g; s/^pytorch:/transformers:/g' your-study.yaml
  ```

  Affected: YAML engine value, YAML section key, `PyTorchConfig` class, `ENGINE_PYTORCH` constant,
  `[pytorch]` extra, `LLEM_RUNNER_PYTORCH`/`LLEM_IMAGE_PYTORCH` env vars, Docker image tags.
  Preserved (PyTorch the library - unchanged): `import torch`, `torch_dtype`, `pytorch/pytorch:*`
  base image, `PYTORCH_VERSION` build args, `torch_compile_backend` field. ([#261])

- **`backend:` field and `--backend` flag renamed to `engine:`** in YAML configs, CLI, and result
  JSON. Aligns terminology with how vLLM, TRT-LLM, and HuggingFace use "engine" natively.

  Migrate with:
  ```bash
  sed -i 's/^\(\s*\)backend:/\1engine:/g' your-study.yaml
  ```

  Affected: YAML field, CLI flag (`-b` becomes `-e`), result JSON fields `"backend"` and
  `"backend_version"`, Python symbols `BackendPlugin`, `BackendError`, `BACKEND_*` constants,
  `get_backend()`, `detect_default_backend()`. ([#260])

- **`tensorrt.tp_size` renamed to `tensorrt.tensor_parallel_size`** to match `TrtLlmArgs` native
  naming. `transformers.tp_size` is unchanged (follows the `accelerate` convention). ([#269])

- **Typed-field curation for engine configs.** Applies the maximalist rubric "type anything with a
  plausible energy/throughput/latency path" to each engine's Pydantic surface. Dropped fields
  remain settable via YAML (`extra="allow"` passthrough unless noted). ([#270])

  Transformers: drops `revision` (reproducibility metadata) and `trust_remote_code` (security
  toggle); adds `allow_tf32`, `autocast_enabled`, `autocast_dtype`, `low_cpu_mem_usage`.

  vLLM: drops `sampling.max_tokens` and `beam_search.max_tokens` (duplicates of
  `ExperimentConfig.max_output_tokens`); adds `num_scheduler_steps`, `max_seq_len_to_capture`,
  `distributed_executor_backend`; replaces flat speculative fields with nested `VLLMSpeculativeConfig`.

  TensorRT-LLM: drops `engine_path`, `TensorRTCalibConfig`, `TensorRTBuildCacheConfig`,
  `sampling.return_perf_metrics`, and `backend: Literal["trt"]`; adds `pipeline_parallel_size`
  and `max_num_tokens`.

- **Engines (vLLM, TensorRT-LLM) now run exclusively inside Docker.** Host extras `[vllm]` and
  `[tensorrt]` removed. Only `[transformers]` remains host-installable. ([#498])

- **`dtype:` and `decoder:` fields migrated into per-engine sub-configs.** Top-level
  `ExperimentConfig.dtype` and `ExperimentConfig.decoder` have moved to each engine's own
  configuration section. ([#290], [#291])

- **`--dtype` and `--batch-size` CLI flags removed.** Both fields are now set via YAML config only.
  ([#292])

- **`precision:` field renamed to `dtype:`** with standard value strings (e.g. `float16`, `bfloat16`
  instead of the prior enum). ([#196])

### Added

- `llem doctor` CLI command reports per-engine image status (OK / MISMATCH / UNVERIFIED /
  UNREACHABLE) and exits non-zero on mismatch for CI gating. ([#256])
- Host/container schema fingerprint verification: Docker images stamped at build time with a
  `llem.expconf.schema.fingerprint` OCI label. Mismatches abort with a rebuild hint. Bypassable
  via `LLEM_SKIP_IMAGE_CHECK=1`. ([#256])
- `SchemaLoader` class (`llenergymeasure.config.SchemaLoader`) reads vendored engine schemas via
  `importlib.resources` with per-instance caching and major-version envelope validation. ([#268])
- Engine parameter discovery introspects installed engine packages inside their Docker images.
  The initial standalone script (`scripts/discover_engine_schemas.py`, #266) was later superseded
  by the `scripts/engine_producers/` codegen toolkit (see Breaking Changes). ([#266])
- Vendored engine parameter schemas at `src/llenergymeasure/engines/{vllm,tensorrt,transformers}/`.
  Regenerate with `make discover-schema ENGINE=<engine>`. ([#266])
- Per-engine sub-package layout (`src/llenergymeasure/engines/<engine>/`) co-locating runtime data,
  schema JSON, and engine invariants YAML. ([#570])
- Per-engine SSOT for library version pins (`engine_versions/`) used by Renovate, Dockerfiles,
  and the invariant-mining pipeline. ([#477])
- Engine invariants mining pipeline: static and dynamic miners for all three engines extract
  validation rules as a reproducible corpus. ([#375], [#434], [#444])
- Vendor-replay CI gate validates corpus against live engine packages; TensorRT gate runs on
  self-hosted GPU runner. ([#414], [#440], [#447])
- `probe` primitive for binary miner reusability check. ([#482])
- `ConfigProbe` protocol and per-engine `probe_config()` implementations. ([#293])
- Configurable per-experiment timeout via `study_execution.experiment_timeout_seconds` (default
  600 s), replacing the previous `max(n_prompts * 2, 600)` heuristic. Both local and Docker paths
  honour the same field. ([#250])
- Disk-persisted baseline power cache with configurable strategy and TTL enforcement. ([#242], [#243])
- Per-study JSONL log capturing runtime warnings and container stderr. ([#395])
- `llem report-gaps` command proposes corpus rules from runtime observations. ([#397])
- Study robustness features: circuit breaker, resume-on-failure, GPU locks, container lifecycle
  management. ([#214])
- Live per-experiment progress display with Rich panels and sub-bullet heartbeats. ([#152], [#165])
- `.env`-based runtime config and configurable `device_map` default. ([#275])
- `trust_remote_code` opt-in via `LLEM_TRUST_REMOTE_CODE` env var. ([#274])
- TRT-LLM build cache configurable via `LLEM_TRT_BUILD_CACHE_{ENABLED,DIR}` env vars. ([#277])
- Tensor parallelism fields (`tp_plan`, `tp_size`) for the Transformers engine. ([#161])
- Cross-field operators in vendored-rules loader. ([#410])
- Docusaurus documentation site at `website/` serving user, methodology, API, and architecture
  docs. ([#566])
- Per-engine discovered-schema Markdown digest rendered to `docs/`. ([#560])
- Architecture documentation suite in `docs/architecture/`. ([#433])
- Per-engine engine-invariants and engine-schemas CI workflows with cross-pipeline coordination
  (consolidated from predecessor mine + vendor + parameter-discovery workflows). ([#484], [#486])
- Engine-pipeline orchestrator (`engine-pipeline.yml`) as single reusable workflow entry point.
  ([#514], [#573])
- Cloudflare Pages PR preview deploy workflow. ([#575])
- SSOT audit trail and GHCR image retention policies. ([#546])
- Engine-knowledge SSOT workspace under `engine_versions/` with per-version mined producer
  snapshots, a shared schema-extraction substrate, and workspace-driven codegen for the typed
  engine config models. ([#733], [#734], [#735], [#736])
- Vendored per-version producer snapshots for all three engines (vLLM 0.7.3 / 0.16.0 / 0.18.1 /
  0.19.1, TensorRT-LLM 0.21.0 / 1.0.0 / 1.2.0 / 1.2.1, Transformers 4.57.3 / 5.3.0 / 5.6.2 / 5.7.0).
  ([#599], [#600], [#601], [#636], [#637], [#638], [#639], [#640], [#641], [#642])
- Schema-vs-source drift tool (coverage and added-direction probes, `EXCLUSIONS.yaml`, sticky-comment
  gate) replacing the prior probe primitive. ([#635], [#643], [#644], [#649], [#650])
- `llem study init` scaffold command for authoring a study YAML. ([#745])
- `llem study plan` preview command showing the expanded experiment grid. ([#750])
- Numeric sweep-axis idioms (`span`, `log`, `pow2`) for concise study axes. ([#746])
- Bounds mode with series policies and per-axis overrides. ([#747])
- Sweep configs record the rejecting rule id when a candidate is skipped. ([#741])
- Absorb conductor orchestrates engine-rule refresh across all engines. ([#756], [#771])
- Upstream validator coverage check confirms mined rules cover each engine's validators. ([#757])
- Cold-read analyst proposer for engine rules. ([#755])
- Citation checker (tier 1 of the rule verification ladder). ([#753])
- Construction and identity probe kernel for rule verification. ([#754])
- Observed-collision miner surfaces dormant rule candidates from runtime observations. ([#752])
- Self-hosted Renovate cron; engine-version scanning revived and its config migrated. ([#732],
  [#775], [#776])
- Generated-doc drift gate in the docs-freshness workflow. ([#760], [#761])
- Fan-in gate making the engine-rules-check requireable without deadlock; bump gates made
  reachable and requireable. ([#769], [#772])
- Runtime-literal discovery stage: string-literal candidates pooled from corpus cross-refs,
  upstream AST/docstring scans, LLM proposals, and previous-schema carry-forward, each verified
  by a two-leg construction probe in the pinned container; confirmed literals are recorded in
  the discovered schema and code-generated as union types. Standing census via
  `make check-corpus-literals`. ([#789])
- Miner final-run recall check with report, a probe-confirmed nested vLLM compilation-config
  cross-field rule, and nested-path rule firing tests. ([#788])

### Changed

- Re-typed `tensorrt.backend` as `Literal["trt", "pytorch", "_autodeploy"] | None` (reverses a
  prior incorrect curation-pass drop; `None` lets TRT-LLM auto-pick the runtime path). ([#276])
- Engine-invariants pipeline consolidated from separate mine + vendor + parameter-discovery
  workflows into a single orchestrated flow with sequential downstream pipelines. ([#484], [#573])
- `study_execution` field names updated (execution fields renamed, `reverse`/`latin_square`
  ordering modes added). ([#190])
- Dataset restructured into nested `DatasetConfig` sub-model. ([#195])
- `OutputConfig` extracted from `ExperimentConfig` as a separate sub-model. ([#203])
- `EnergyConfig` flattened to `energy_sampler` + `gpu_telemetry` fields. ([#201])
- `study_name` field replaces generic `name` field in study configs. ([#182])
- `n_prompts` default reduced to 50; `max_output_tokens` default bumped to 256. ([#175], [#213])
- Renovate customManager retargeted from Dockerfile ARGs to `engine_versions/` SSOT. ([#481])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` replaced with upstream-direct images
  plus volume mounts. ([#509])
- Advanced engine pins to vLLM 0.19.1, TensorRT-LLM 1.0.0, and Transformers 5.7.0, adopting the
  generated nested configs at those versions. ([#738])
- Engine pipeline rewritten to hosted byte-verification (the mined corpus and generated configs
  are verified byte-for-byte against a fresh mine, no human source-diffing). ([#758])
- Documentation aligned to the byte-verification engine-knowledge flow and the rules vocabulary;
  stale engine and contributing pages rewritten. ([#759], [#764], [#774])
- Discovery write path inverted: container discovery writes only under `engine_versions/`, with
  `make promote-schemas` as the sole writer of the packaged copies. ([#785])
- Knowledge-production scripts gated by ruff and mypy in Makefile and CI; import-linter layer
  contracts made honest; bundle artefact filenames centralised as constants. ([#784])
- Dead code deleted and duplicated helpers folded across study and scripts layers. ([#786])
- Docs: heading anchors match the live slugifier, version references pinned to `current.yaml`
  sources, and curation-era drift rewritten. ([#787])

### Fixed

- `ImportError: cuKernelGetName` when importing `tensorrt_llm`: LD_LIBRARY_PATH ordering placed
  the bundled compat CUDA 12.2 library ahead of the host-driver mount. Fixed by prepending
  `/usr/local/cuda/compat/lib` so the host-driver mount takes precedence. ([#264])
- Miner `added_at` timestamp lost on re-mine; f-string `message_template` fields now rendered
  correctly. ([#523])
- `Dockerfile.transformers` stale references to the old `[pytorch]` extra and header comments
  corrected. ([#265])
- Config hash mismatch in Docker study runs resolved. ([#176])
- Config-identity hash now covers the active engine's `harness` block (`batch_size`,
  `torch_compile`, `allow_tf32`, `autocast`). These drive execution but were omitted, so a
  `harness.batch_size` sweep collapsed to a single resolved hash and default dedup ran one
  experiment instead of the full sweep. ([#783])
- `measurement.*` methodology fields now join the config-identity hash, so sweeping warmup,
  baseline, energy sampler, or windowing produces distinct runs rather than deduping to one. ([#783])
- `--no-dedup` no longer crashes with a `KeyError` when a sweep canonicalises two grid points
  to the same declared config: manifest entries are built from actual per-hash occurrence
  counts, keeping the manifest aligned with the runner's per-occurrence cycle counter. ([#783])
- Non-matching engine sections stripped correctly during multi-engine grid expansion. ([#171])
- Docker auto-elevation enforced for multi-engine studies. ([#172])
- Baseline cache path resolved before Docker bind-mount. ([#248])
- Purged false-positive and phantom rules from the shipped corpus. ([#767])
- Rule message rendering and loader guards corrected. ([#768])
- Study dedup fallback, grid edge cases, and dormant-candidate visibility repaired. ([#770])
- Schema discovery retargeted at the `current.yaml` pins. ([#765])
- Pin-driven Transformers publish and GHCR prune exemption. ([#766])
- Rules follow-ups: probe operator canonicalisation, loader operator allowlist, message-template
  residue, and `{invariant_id}` renamed to `{rule_id}`. ([#779])
- Per-engine upstream default images resolve from the pinned engine version instead of a
  never-published GHCR ref; `llem doctor` verifies them. ([#780])
- Energy-measurement failure is loud instead of a silent zero: sampler auto-selection warns,
  absent energy stays `None` rather than coercing to `0.0`, sampler failures set a measurement
  warning, and NVML init failures log debug traces. ([#781])
- Documented top-level `images:` study key no longer leaks into experiment configs and rejects
  the whole study; user-facing config copy de-jargonised. ([#782])
- Self-hosted Renovate aborted before opening update PRs: the stability commit-status POST the
  App token cannot make is now disabled in config. ([#791])

### Removed

- Internal helper `llenergymeasure.study.runner._calculate_timeout` (replaced by direct config
  reads). ([#529])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` engine images. ([#509])
- Dead invariant-mining pipeline and the `refresh-invariants` / `schema-diff` scripts. ([#762],
  [#763])
- Orphaned files, dead references, and retired mined-invariants documentation pages. ([#760],
  [#773])
- Predecessor CI workflows: `auto-mine.yml`, `vendor-tensorrt.yml`, `vendor-vllm.yml`,
  `parameter-discovery.yml`, and predecessors. ([#483], [#485])


## [v0.4.0] - 2026-03-20

Docker infrastructure, vLLM engine, TensorRT-LLM engine, package restructure, test hardening, and CI.

### Added

- NVML GPU memory residual check before experiment dispatch (threshold 1 GB), preventing
  stale-process contamination. ([#24], [#26])
- Docker runner infrastructure: container lifecycle management, volume mounts, GPU index
  resolution. ([#27], [#124])
- Docker pre-flight environment checks. ([#28])
- TensorRT-LLM Docker image rewrite with CUDA 12.6.2 upgrade. ([#114])
- `TensorRTConfig` expanded to full TRT-LLM parameter schema. ([#115])
- `mpirun` injection for TensorRT-LLM tensor parallelism. ([#116])
- `BackendPlugin.validate_config` protocol method. ([#121])
- `TensorRTBackend` implementation registered in `get_backend()`. ([#122])
- `TensorRTConfig.engine_path` for pre-compiled engine loading. ([#143])
- 9-layer import-linter architecture enforcement in CI. ([#135], [#144])

### Changed

- Package restructured with file moves, import rewrites, and layer boundary fixes. ([#133], [#134])
- Prompt loading moved outside the NVML measurement window. ([#145])
- Shared backend helpers extracted; dead warmup code removed. ([#140])
- Test suite restructured; `importorskip` guards added for optional dependencies. ([#137], [#138])

### Fixed

- `accelerate` restored as a `[pytorch]` optional dependency (accidentally dropped). ([#132])
- Runner mode auto-detection (local vs Docker) on startup. ([#146])
- Silent `NVMLError`, payload detection, and empty `gpu_indices` guard. ([#141])

### Removed

- Dead code, stale type annotations, and unused dependencies. ([#130])


## [v0.3.0] - 2026-02-27

Multi-experiment study sweeps.

### Added

- `run_study()` public API for multi-experiment studies. ([#23])
- `StudyConfig` with sweep grammar (grid and cycle ordering). ([#23])
- YAML-driven parameter sweeps across models, engines, and precisions. ([#23])
- `StudyRunner` with sequential experiment dispatch. ([#23])
- Study-level aggregation and result collection. ([#23])
- Manifest-based progress tracking with resume support. ([#23])


## [v0.2.0] - 2026-02-27

First end-to-end single-experiment release.

### Added

- `run_experiment()` public API. ([#22])
- `ExperimentConfig` to `ExperimentResult` pipeline. ([#22])
- Energy measurement via CodeCarbon and Zeus backends. ([#22])
- Extended metrics: TPOT, TEI, memory efficiency. ([#22])
- Streaming latency measurement (TTFT / ITL). ([#22])
- Results persistence in Parquet format. ([#22])


## [v0.1.1] - 2025-12-29

Post-thesis re-founding. One December development burst rebuilt the frozen thesis
prototype into an installable, tested package.

- Package renamed `llm-bench` -> `lem`; all imports moved to `llenergymeasure`.
- Energy-backend plugin registry with automatic CodeCarbon registration; `FlopsEstimator`
  with a three-strategy fallback chain (calflops, architecture, parameter estimate), each
  returning a confidence level; results aggregation with temporal-overlap detection and
  GPU-attribution verification; CSV/JSON export; structured logging replacing `print()`.
- Typer-based CLI with `experiment`, `aggregate`, `config`, `results`, and `datasets`
  subcommands; `ExperimentOrchestrator` with protocol-based dependency injection; the
  earlier `MAIN_*.py` entry points removed.
- A 416-test suite (unit, integration, and end-to-end) runnable without GPU access via
  mocked data; `requirements.txt` retired in favour of the Poetry lockfile; methodology
  documentation added.
- Production containerisation: a multi-stage Dockerfile on a CUDA 12.4 base, Docker Compose
  production and dev profiles, a VS Code devcontainer with GPU passthrough, and Makefile
  targets for common Docker operations.

## [v0.1.0] - 2025-05-17

Thesis research prototype complete: stable multi-model benchmarking on production
hardware. The code was frozen at this point for the maintainer's thesis (submitted
~2025-07).

### Added

- Multi-model experiment support with scenario-based configuration.
- Experiment suite CSV export with consistent naming conventions.
- Failed experiment detection with cycle tracking and automatic retry.
- Minimum output token enforcement for comparable generation lengths.
- Large model stability improvements (gradient checkpointing, CUDA cache clearing).
- Data wrangling pipelines for experiment result analysis (Pandas-based).
- Plotting functionality for efficiency metrics visualisation.
- FLOPs caching preventing redundant calculations.

## [v0.0.1] - 2025-03-22

Origin: first measurement scaffolding (multi-GPU aggregation, FLOPs, Optimum-benchmark).

### Added

- Distributed results aggregation across multiple GPUs with per-process JSON files.
- FLOPs calculation with quantisation awareness and `calflops` integration.
- Robust process cleanup with signal handlers and distributed barrier synchronisation.
- Optimum benchmark integration for standardised measurements.

### Changed

- Distributed execution stability improved: proper NCCL initialisation and teardown.
- Major directory restructuring separating config, core, and result handling.


[Unreleased]: https://github.com/henrycgbaker/llenergymeasure/compare/v0.6.0...HEAD
[v0.6.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.6.0
[v0.5.1]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.5.1
[v0.5.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.5.0
[v0.4.1]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.4.1
[v0.4.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.4.0
[v0.3.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.3.0
[v0.2.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.2.0
[v0.1.1]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.1.1
[v0.1.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.1.0
[v0.0.1]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.0.1

[#22]: https://github.com/henrycgbaker/llenergymeasure/pull/22
[#23]: https://github.com/henrycgbaker/llenergymeasure/pull/23
[#24]: https://github.com/henrycgbaker/llenergymeasure/pull/24
[#26]: https://github.com/henrycgbaker/llenergymeasure/pull/26
[#27]: https://github.com/henrycgbaker/llenergymeasure/pull/27
[#28]: https://github.com/henrycgbaker/llenergymeasure/pull/28
[#113]: https://github.com/henrycgbaker/llenergymeasure/pull/113
[#114]: https://github.com/henrycgbaker/llenergymeasure/pull/114
[#115]: https://github.com/henrycgbaker/llenergymeasure/pull/115
[#116]: https://github.com/henrycgbaker/llenergymeasure/pull/116
[#121]: https://github.com/henrycgbaker/llenergymeasure/pull/121
[#122]: https://github.com/henrycgbaker/llenergymeasure/pull/122
[#124]: https://github.com/henrycgbaker/llenergymeasure/pull/124
[#130]: https://github.com/henrycgbaker/llenergymeasure/pull/130
[#132]: https://github.com/henrycgbaker/llenergymeasure/pull/132
[#133]: https://github.com/henrycgbaker/llenergymeasure/pull/133
[#134]: https://github.com/henrycgbaker/llenergymeasure/pull/134
[#135]: https://github.com/henrycgbaker/llenergymeasure/pull/135
[#137]: https://github.com/henrycgbaker/llenergymeasure/pull/137
[#138]: https://github.com/henrycgbaker/llenergymeasure/pull/138
[#140]: https://github.com/henrycgbaker/llenergymeasure/pull/140
[#141]: https://github.com/henrycgbaker/llenergymeasure/pull/141
[#143]: https://github.com/henrycgbaker/llenergymeasure/pull/143
[#144]: https://github.com/henrycgbaker/llenergymeasure/pull/144
[#145]: https://github.com/henrycgbaker/llenergymeasure/pull/145
[#146]: https://github.com/henrycgbaker/llenergymeasure/pull/146
[#147]: https://github.com/henrycgbaker/llenergymeasure/pull/147
[#152]: https://github.com/henrycgbaker/llenergymeasure/pull/152
[#161]: https://github.com/henrycgbaker/llenergymeasure/pull/161
[#165]: https://github.com/henrycgbaker/llenergymeasure/pull/165
[#171]: https://github.com/henrycgbaker/llenergymeasure/pull/171
[#172]: https://github.com/henrycgbaker/llenergymeasure/pull/172
[#175]: https://github.com/henrycgbaker/llenergymeasure/pull/175
[#176]: https://github.com/henrycgbaker/llenergymeasure/pull/176
[#182]: https://github.com/henrycgbaker/llenergymeasure/pull/182
[#190]: https://github.com/henrycgbaker/llenergymeasure/pull/190
[#195]: https://github.com/henrycgbaker/llenergymeasure/pull/195
[#196]: https://github.com/henrycgbaker/llenergymeasure/pull/196
[#201]: https://github.com/henrycgbaker/llenergymeasure/pull/201
[#203]: https://github.com/henrycgbaker/llenergymeasure/pull/203
[#213]: https://github.com/henrycgbaker/llenergymeasure/pull/213
[#214]: https://github.com/henrycgbaker/llenergymeasure/pull/214
[#242]: https://github.com/henrycgbaker/llenergymeasure/pull/242
[#243]: https://github.com/henrycgbaker/llenergymeasure/pull/243
[#248]: https://github.com/henrycgbaker/llenergymeasure/pull/248
[#250]: https://github.com/henrycgbaker/llenergymeasure/pull/250
[#256]: https://github.com/henrycgbaker/llenergymeasure/pull/256
[#260]: https://github.com/henrycgbaker/llenergymeasure/pull/260
[#261]: https://github.com/henrycgbaker/llenergymeasure/pull/261
[#264]: https://github.com/henrycgbaker/llenergymeasure/pull/264
[#265]: https://github.com/henrycgbaker/llenergymeasure/pull/265
[#266]: https://github.com/henrycgbaker/llenergymeasure/pull/266
[#268]: https://github.com/henrycgbaker/llenergymeasure/pull/268
[#269]: https://github.com/henrycgbaker/llenergymeasure/pull/269
[#270]: https://github.com/henrycgbaker/llenergymeasure/pull/270
[#274]: https://github.com/henrycgbaker/llenergymeasure/pull/274
[#275]: https://github.com/henrycgbaker/llenergymeasure/pull/275
[#276]: https://github.com/henrycgbaker/llenergymeasure/pull/276
[#277]: https://github.com/henrycgbaker/llenergymeasure/pull/277
[#290]: https://github.com/henrycgbaker/llenergymeasure/pull/290
[#291]: https://github.com/henrycgbaker/llenergymeasure/pull/291
[#292]: https://github.com/henrycgbaker/llenergymeasure/pull/292
[#293]: https://github.com/henrycgbaker/llenergymeasure/pull/293
[#375]: https://github.com/henrycgbaker/llenergymeasure/pull/375
[#395]: https://github.com/henrycgbaker/llenergymeasure/pull/395
[#397]: https://github.com/henrycgbaker/llenergymeasure/pull/397
[#410]: https://github.com/henrycgbaker/llenergymeasure/pull/410
[#414]: https://github.com/henrycgbaker/llenergymeasure/pull/414
[#433]: https://github.com/henrycgbaker/llenergymeasure/pull/433
[#434]: https://github.com/henrycgbaker/llenergymeasure/pull/434
[#440]: https://github.com/henrycgbaker/llenergymeasure/pull/440
[#444]: https://github.com/henrycgbaker/llenergymeasure/pull/444
[#447]: https://github.com/henrycgbaker/llenergymeasure/pull/447
[#477]: https://github.com/henrycgbaker/llenergymeasure/pull/477
[#480]: https://github.com/henrycgbaker/llenergymeasure/pull/480
[#481]: https://github.com/henrycgbaker/llenergymeasure/pull/481
[#482]: https://github.com/henrycgbaker/llenergymeasure/pull/482
[#483]: https://github.com/henrycgbaker/llenergymeasure/pull/483
[#484]: https://github.com/henrycgbaker/llenergymeasure/pull/484
[#485]: https://github.com/henrycgbaker/llenergymeasure/pull/485
[#486]: https://github.com/henrycgbaker/llenergymeasure/pull/486
[#498]: https://github.com/henrycgbaker/llenergymeasure/pull/498
[#509]: https://github.com/henrycgbaker/llenergymeasure/pull/509
[#514]: https://github.com/henrycgbaker/llenergymeasure/pull/514
[#523]: https://github.com/henrycgbaker/llenergymeasure/pull/523
[#529]: https://github.com/henrycgbaker/llenergymeasure/pull/529
[#546]: https://github.com/henrycgbaker/llenergymeasure/pull/546
[#560]: https://github.com/henrycgbaker/llenergymeasure/pull/560
[#566]: https://github.com/henrycgbaker/llenergymeasure/pull/566
[#570]: https://github.com/henrycgbaker/llenergymeasure/pull/570
[#572]: https://github.com/henrycgbaker/llenergymeasure/pull/572
[#573]: https://github.com/henrycgbaker/llenergymeasure/pull/573
[#575]: https://github.com/henrycgbaker/llenergymeasure/pull/575
[#599]: https://github.com/henrycgbaker/llenergymeasure/pull/599
[#600]: https://github.com/henrycgbaker/llenergymeasure/pull/600
[#601]: https://github.com/henrycgbaker/llenergymeasure/pull/601
[#635]: https://github.com/henrycgbaker/llenergymeasure/pull/635
[#636]: https://github.com/henrycgbaker/llenergymeasure/pull/636
[#637]: https://github.com/henrycgbaker/llenergymeasure/pull/637
[#638]: https://github.com/henrycgbaker/llenergymeasure/pull/638
[#639]: https://github.com/henrycgbaker/llenergymeasure/pull/639
[#640]: https://github.com/henrycgbaker/llenergymeasure/pull/640
[#641]: https://github.com/henrycgbaker/llenergymeasure/pull/641
[#642]: https://github.com/henrycgbaker/llenergymeasure/pull/642
[#643]: https://github.com/henrycgbaker/llenergymeasure/pull/643
[#644]: https://github.com/henrycgbaker/llenergymeasure/pull/644
[#649]: https://github.com/henrycgbaker/llenergymeasure/pull/649
[#650]: https://github.com/henrycgbaker/llenergymeasure/pull/650
[#732]: https://github.com/henrycgbaker/llenergymeasure/pull/732
[#733]: https://github.com/henrycgbaker/llenergymeasure/pull/733
[#734]: https://github.com/henrycgbaker/llenergymeasure/pull/734
[#735]: https://github.com/henrycgbaker/llenergymeasure/pull/735
[#736]: https://github.com/henrycgbaker/llenergymeasure/pull/736
[#737]: https://github.com/henrycgbaker/llenergymeasure/pull/737
[#738]: https://github.com/henrycgbaker/llenergymeasure/pull/738
[#740]: https://github.com/henrycgbaker/llenergymeasure/pull/740
[#741]: https://github.com/henrycgbaker/llenergymeasure/pull/741
[#745]: https://github.com/henrycgbaker/llenergymeasure/pull/745
[#746]: https://github.com/henrycgbaker/llenergymeasure/pull/746
[#747]: https://github.com/henrycgbaker/llenergymeasure/pull/747
[#749]: https://github.com/henrycgbaker/llenergymeasure/pull/749
[#750]: https://github.com/henrycgbaker/llenergymeasure/pull/750
[#752]: https://github.com/henrycgbaker/llenergymeasure/pull/752
[#753]: https://github.com/henrycgbaker/llenergymeasure/pull/753
[#754]: https://github.com/henrycgbaker/llenergymeasure/pull/754
[#755]: https://github.com/henrycgbaker/llenergymeasure/pull/755
[#756]: https://github.com/henrycgbaker/llenergymeasure/pull/756
[#757]: https://github.com/henrycgbaker/llenergymeasure/pull/757
[#758]: https://github.com/henrycgbaker/llenergymeasure/pull/758
[#759]: https://github.com/henrycgbaker/llenergymeasure/pull/759
[#760]: https://github.com/henrycgbaker/llenergymeasure/pull/760
[#761]: https://github.com/henrycgbaker/llenergymeasure/pull/761
[#762]: https://github.com/henrycgbaker/llenergymeasure/pull/762
[#763]: https://github.com/henrycgbaker/llenergymeasure/pull/763
[#764]: https://github.com/henrycgbaker/llenergymeasure/pull/764
[#765]: https://github.com/henrycgbaker/llenergymeasure/pull/765
[#766]: https://github.com/henrycgbaker/llenergymeasure/pull/766
[#767]: https://github.com/henrycgbaker/llenergymeasure/pull/767
[#768]: https://github.com/henrycgbaker/llenergymeasure/pull/768
[#769]: https://github.com/henrycgbaker/llenergymeasure/pull/769
[#770]: https://github.com/henrycgbaker/llenergymeasure/pull/770
[#771]: https://github.com/henrycgbaker/llenergymeasure/pull/771
[#772]: https://github.com/henrycgbaker/llenergymeasure/pull/772
[#773]: https://github.com/henrycgbaker/llenergymeasure/pull/773
[#774]: https://github.com/henrycgbaker/llenergymeasure/pull/774
[#775]: https://github.com/henrycgbaker/llenergymeasure/pull/775
[#776]: https://github.com/henrycgbaker/llenergymeasure/pull/776
[#779]: https://github.com/henrycgbaker/llenergymeasure/pull/779
[#780]: https://github.com/henrycgbaker/llenergymeasure/pull/780
[#781]: https://github.com/henrycgbaker/llenergymeasure/pull/781
[#782]: https://github.com/henrycgbaker/llenergymeasure/pull/782
[#783]: https://github.com/henrycgbaker/llenergymeasure/pull/783
[#784]: https://github.com/henrycgbaker/llenergymeasure/pull/784
[#785]: https://github.com/henrycgbaker/llenergymeasure/pull/785
[#786]: https://github.com/henrycgbaker/llenergymeasure/pull/786
[#787]: https://github.com/henrycgbaker/llenergymeasure/pull/787
[#788]: https://github.com/henrycgbaker/llenergymeasure/pull/788
[#789]: https://github.com/henrycgbaker/llenergymeasure/pull/789
[#791]: https://github.com/henrycgbaker/llenergymeasure/pull/791
[#792]: https://github.com/henrycgbaker/llenergymeasure/pull/792
[#795]: https://github.com/henrycgbaker/llenergymeasure/pull/795
[#796]: https://github.com/henrycgbaker/llenergymeasure/pull/796
[#797]: https://github.com/henrycgbaker/llenergymeasure/pull/797
[#798]: https://github.com/henrycgbaker/llenergymeasure/pull/798
[#799]: https://github.com/henrycgbaker/llenergymeasure/pull/799
[#800]: https://github.com/henrycgbaker/llenergymeasure/pull/800
[#801]: https://github.com/henrycgbaker/llenergymeasure/pull/801
[#802]: https://github.com/henrycgbaker/llenergymeasure/pull/802
[#803]: https://github.com/henrycgbaker/llenergymeasure/pull/803
[#804]: https://github.com/henrycgbaker/llenergymeasure/pull/804
[#805]: https://github.com/henrycgbaker/llenergymeasure/pull/805
[#806]: https://github.com/henrycgbaker/llenergymeasure/pull/806
[#807]: https://github.com/henrycgbaker/llenergymeasure/pull/807
[#808]: https://github.com/henrycgbaker/llenergymeasure/pull/808
[#809]: https://github.com/henrycgbaker/llenergymeasure/pull/809
[#810]: https://github.com/henrycgbaker/llenergymeasure/pull/810
[#811]: https://github.com/henrycgbaker/llenergymeasure/pull/811
[#812]: https://github.com/henrycgbaker/llenergymeasure/pull/812
[#813]: https://github.com/henrycgbaker/llenergymeasure/pull/813
[#814]: https://github.com/henrycgbaker/llenergymeasure/pull/814
[#815]: https://github.com/henrycgbaker/llenergymeasure/pull/815
[#816]: https://github.com/henrycgbaker/llenergymeasure/pull/816
[#817]: https://github.com/henrycgbaker/llenergymeasure/pull/817
[#819]: https://github.com/henrycgbaker/llenergymeasure/pull/819
[#820]: https://github.com/henrycgbaker/llenergymeasure/pull/820
[#821]: https://github.com/henrycgbaker/llenergymeasure/pull/821
[#822]: https://github.com/henrycgbaker/llenergymeasure/pull/822
[#823]: https://github.com/henrycgbaker/llenergymeasure/pull/823
[#824]: https://github.com/henrycgbaker/llenergymeasure/pull/824
[#826]: https://github.com/henrycgbaker/llenergymeasure/pull/826
[#827]: https://github.com/henrycgbaker/llenergymeasure/pull/827
[#828]: https://github.com/henrycgbaker/llenergymeasure/pull/828
[#829]: https://github.com/henrycgbaker/llenergymeasure/pull/829
[#830]: https://github.com/henrycgbaker/llenergymeasure/pull/830
[#831]: https://github.com/henrycgbaker/llenergymeasure/pull/831
[#832]: https://github.com/henrycgbaker/llenergymeasure/pull/832
[#833]: https://github.com/henrycgbaker/llenergymeasure/pull/833
[#834]: https://github.com/henrycgbaker/llenergymeasure/pull/834
[#835]: https://github.com/henrycgbaker/llenergymeasure/pull/835
[#836]: https://github.com/henrycgbaker/llenergymeasure/pull/836
[#837]: https://github.com/henrycgbaker/llenergymeasure/pull/837
[#838]: https://github.com/henrycgbaker/llenergymeasure/pull/838
[#839]: https://github.com/henrycgbaker/llenergymeasure/pull/839
[#840]: https://github.com/henrycgbaker/llenergymeasure/pull/840
[#842]: https://github.com/henrycgbaker/llenergymeasure/pull/842
[#843]: https://github.com/henrycgbaker/llenergymeasure/pull/843
[#844]: https://github.com/henrycgbaker/llenergymeasure/pull/844
[#845]: https://github.com/henrycgbaker/llenergymeasure/pull/845
[#849]: https://github.com/henrycgbaker/llenergymeasure/pull/849
[#850]: https://github.com/henrycgbaker/llenergymeasure/pull/850
[#851]: https://github.com/henrycgbaker/llenergymeasure/pull/851
[#852]: https://github.com/henrycgbaker/llenergymeasure/pull/852
[#853]: https://github.com/henrycgbaker/llenergymeasure/pull/853
[#854]: https://github.com/henrycgbaker/llenergymeasure/pull/854
[#855]: https://github.com/henrycgbaker/llenergymeasure/pull/855
[#856]: https://github.com/henrycgbaker/llenergymeasure/pull/856
[#857]: https://github.com/henrycgbaker/llenergymeasure/pull/857
[#860]: https://github.com/henrycgbaker/llenergymeasure/pull/860
[#861]: https://github.com/henrycgbaker/llenergymeasure/pull/861
[#862]: https://github.com/henrycgbaker/llenergymeasure/pull/862
[#863]: https://github.com/henrycgbaker/llenergymeasure/pull/863
[#864]: https://github.com/henrycgbaker/llenergymeasure/pull/864
[#866]: https://github.com/henrycgbaker/llenergymeasure/pull/866
[#867]: https://github.com/henrycgbaker/llenergymeasure/pull/867
[#868]: https://github.com/henrycgbaker/llenergymeasure/pull/868
[#869]: https://github.com/henrycgbaker/llenergymeasure/pull/869
[#871]: https://github.com/henrycgbaker/llenergymeasure/pull/871
[#872]: https://github.com/henrycgbaker/llenergymeasure/pull/872
[#875]: https://github.com/henrycgbaker/llenergymeasure/pull/875
[#879]: https://github.com/henrycgbaker/llenergymeasure/pull/879
[#880]: https://github.com/henrycgbaker/llenergymeasure/pull/880
[#881]: https://github.com/henrycgbaker/llenergymeasure/pull/881
