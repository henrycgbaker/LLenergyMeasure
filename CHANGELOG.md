# CHANGELOG


## v1.2.0 (2025-12-20)

### Features

- Add energy backend plugin registry (Phase 9)
  ([`619682c`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/619682c229d5b23e8a33b2564a2fba4c86179b5b))

- Add base.py with EnergyBackend protocol re-export - Implement plugin registry (register_backend,
  get_backend, list_backends) - Auto-register CodeCarbonBackend on import - Add 8 unit tests for
  plugin registry - 296 total tests passing


## v1.1.0 (2025-12-20)

### Continuous Integration

- Add automated semantic versioning
  ([`d88c8fc`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/d88c8fc4a05b830051a0053356f200b4c34bd071))

- Add python-semantic-release for automated version bumps - Configure in pyproject.toml: - feat:
  commits → minor version bump - fix: commits → patch version bump - Add
  .github/workflows/release.yml for automated releases - Creates GitHub releases with tags on push
  to main

### Features

- Add core modules for LLM benchmarking (Phase 7)
  ([`ffbbbab`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ffbbbab31f12836685bcaafb68e1392fb98f8d2c))

Core module migration from legacy experiment_core_utils: - distributed.py: Accelerator setup, unique
  ID generation, barrier sync - model_loader.py: Model/tokenizer loading with BitsAndBytes
  quantization - prompts.py: Prompt filtering, sorting, tokenization, batching strategies -
  inference.py: Inference engine with batch processing - compute_metrics.py: FLOPs calculation,
  memory stats, utilization tracking - energy_backends/codecarbon.py: CodeCarbon energy tracking
  backend

Improvements over legacy code: - Uses domain models from llm_bench.domain - Protocol-based
  interfaces for extensibility - Loguru structured logging (replaced print statements) -
  Comprehensive type hints and docstrings - 63 new unit tests for core modules (228 total)

Also fixes pre-commit mypy to only check src/.

- Add FlopsEstimator with multi-strategy fallback (Phase 8)
  ([`238c1f9`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/238c1f95d05d67419afc3f2feab01eed35db91ce))

- Add FlopsResult model with provenance tracking (value, method, confidence, precision) - Implement
  FlopsEstimator with 3-strategy fallback chain: 1. calflops (high confidence) - direct measurement
  2. architecture (medium confidence) - uses model.config 3. parameter_estimate (low confidence) - 2
  * params * seq_len - Update collect_compute_metrics() to use FlopsEstimator - Handle BNB
  quantization correctly (always reports fp16 precision) - Add 29 unit tests for FlopsEstimator

- Add results aggregation and export (Phase 8)
  ([`fc417fa`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/fc417fa8f457fcadc21ded3274e87bf3d30927e6))

New results module functionality: - aggregation.py: Combine raw per-process results into aggregated
  metrics - Sum energy across processes, average throughput - Temporal overlap verification
  (concurrent execution check) - GPU attribution verification (no double-counting) - Calculate
  derived efficiency metrics - exporters.py: Export results to CSV and JSON formats -
  ResultsExporter class for unified export interface - Flatten Pydantic models to tabular format -
  Logical column ordering

Also includes: - mypy config update: disable strict untyped call checks for torch/transformers -
  pre-commit mypy: use pyproject.toml config for consistency - 31 new unit tests (259 total)


## v1.0.0 (2025-12-16)

### Bug Fixes

- Correct do_sample behavior
  ([`ddd4c77`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ddd4c773a208f1f77bfbcae91c07c93935396e65))

- Improve exit handling
  ([`343b359`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/343b35965e3d753add7c46ee247d701dd07a11f8))

- Resolve stability for large models
  ([`124f335`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/124f335ad60b4af32d968986a21914fba62b1064))

### Documentation

- Add code annotations
  ([`c673936`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c6739368deb3a02a232e0353e13ce7bdf082712e))

### Features

- Add check for failed experiments
  ([`d6df151`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/d6df15161300f23ce609aee7d8ad999bba117694))

- Add cycle tracking
  ([`a42ddcd`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/a42ddcdd71aa0ed8fa136bc342acebce4315244e))

- Add min output length handling
  ([`b855d67`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/b855d671eed5d23e6ba24c8e5c7ee06fa6205963))

### Refactoring

- Improve config process
  ([`78bb442`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/78bb442dbf3b7d3972c7336cbc90c01523d3bf37))

- Restructure directories
  ([`467d2d8`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/467d2d845b0c080f1dbbbc99cff3455b9bb8de44))

- Update config generation
  ([`9fb4e02`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/9fb4e02eea3cac42e22252d69856757d2324e5af))

- Update configs and scenario loading
  ([`b63fe19`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/b63fe19dc7df0f989755230372928031f127ef75))

### Testing

- Check project state
  ([`fd08c69`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/fd08c6914b8fe23309e8957682c388be4c5392a6))

- Run with 1b and 3b models
  ([`40a6763`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/40a67631767d835a8acc0408ba034800e1b0ae97))


## v0.9.0 (2025-04-13)

### Bug Fixes

- Improve quantized flops handling
  ([`39c2c6a`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/39c2c6a141a157f51fa53c5f8ef8e901c04392b8))

### Features

- Add additional plots
  ([`23964c0`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/23964c030753ec3667fb50129356b2de52e659b8))

- Add data wrangling
  ([`b024ffa`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/b024ffa059613e307020c6165721086fb3727f90))

- Add experiment suite csv naming
  ([`a6d836f`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/a6d836f3fb053f6428858a879eb9af4e99f2fa43))

- Add flops caching for models
  ([`19279dd`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/19279dd9c2e784d338e4e777e50e343f061afc63))

- Add further analysis
  ([`fd607ad`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/fd607ad2a775018d13ea1a4e8f989cf88bae91fd))

- Add plotting functionality
  ([`465cff8`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/465cff8b510789514afc1a6954bc8c5cb0f00c6a))

- Add support for multiple models
  ([`bd49c11`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/bd49c11879966a7538e08bf5cd1776ad46ad9084))

- Complete data cleaning and prep
  ([`3449f2d`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/3449f2d4ea3fde4fc41fa1d8c3e81d1b65533e89))

- Implement robust csv saving
  ([`ddfe549`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ddfe549d9618715d47fa4fe3474f2ac4853bd02c))

### Testing

- Validate with tinyllama
  ([`c777374`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c777374207ea27637543c0c71670138c8d5716cc))


## v0.8.0 (2025-04-06)

### Bug Fixes

- Resolve accelerate launcher issues
  ([`7228271`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/722827174be73b84758a82764b7bd85da96cc87e))

- Resolve csv saving issues
  ([`ba7bf21`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ba7bf21dd91ce6d944384c77de33a1b24b3b3099))

- Resolve flops and energy saving
  ([`9f6623e`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/9f6623e1819bf50fe82989051b75dae312fcef00))

- Resolve flops via accelerate launch
  ([`31409a2`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/31409a275e00cd3aabd735eb154674fb3e6c66d5))

### Features

- Add decoder options and dynamic gpu selection
  ([`6519df1`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/6519df1e554875ae3e3d964b8f1fb2c5a26ba5e2))

- Add latency and burstiness metrics
  ([`eaa2992`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/eaa2992f4e23c2d52ed0469c33510fd2430f1358))

- Add scenario configurations
  ([`1b7683f`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/1b7683f483832899f4d5bf70deff3ebe008517bc))

- Convert json results to csv
  ([`e53a0c6`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/e53a0c617de6d4ba8bec1cb3e9f8ebc6b2eb1ac9))

- Load single and scenario configs
  ([`9fbab19`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/9fbab19abb7be8423c17cb770a9d61b32853950f))

- Support launching from multiple scripts
  ([`aae1baa`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/aae1baa0312888faa8bb0f3006777351ffafb3f6))

- Support multi-scenario experiments
  ([`5608fa4`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5608fa4b6732bbb9a997754d5dcddf06f881bc88))


## v0.7.0 (2025-03-29)

### Bug Fixes

- Improve grid search stability
  ([`5e7ec83`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5e7ec83a45abea180917e1a491b1a658e291427e))

### Features

- Add grid search functionality
  ([`f13b20a`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/f13b20aad57987f03d0a8e296a9bc3e0afbbe124))

- Get single experiment runs working
  ([`f2c8b0f`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/f2c8b0f34b989ab82b0fc1b85ede5fe96b27bce6))

- Robustly implement single process mode
  ([`5e5ae03`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5e5ae036d48050617b948a79fb2d9d49970e5d4f))


## v0.6.0 (2025-03-28)

### Bug Fixes

- Add manual flops workaround
  ([`ce9568b`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ce9568b9ee95897f6a8da3cea0bcc13911290e1e))

- Finalize quantization patch
  ([`c7bd843`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c7bd843fc12887bd5ff8eb3e72b7342bfec34ee7))

- Improve safe_wait function
  ([`fdee455`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/fdee45564b2873683797f296c1fa5260f99c7d44))

- Resolve codecarbon distributed error
  ([`0cf0fe5`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/0cf0fe53054480f017b406d7fa86261f0a59c1b6))

- Resolve quantization flops calculation
  ([`c85cfa2`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c85cfa2cc60fbc021eb61b052db750a1ab3cd3b4))

### Features

- Add adaptive batching
  ([`61d5612`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/61d561285218c51d72e87fc91968fa75653bd0ab))

- Add batch processing
  ([`5090016`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5090016369d266edae99640eb75206c3a9a0a98c))

- Add experiment retries and warmup
  ([`96c473c`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/96c473c5d33cd04f6684c3c9e52ceff4cd614782))

- Add experiment runner script
  ([`230ae01`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/230ae01978b63966d7e7e579338e6504d803de71))

- Add json output for text generation
  ([`17e4c53`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/17e4c5388a9e26b2ef9fb5474f0898616f7b9a33))

- Add quantization support
  ([`ae8e0ee`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ae8e0ee3fe9a63ea7fe5c7d1bbf66543cbe66799))

- Complete quantization implementation
  ([`a60423e`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/a60423e145cf260a0c378c88cb22a986989ef89c))

- Complete working pipeline
  ([`c13c32c`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c13c32ced4714003e789576aa674a412e46c8dc3))

- Implement robust aggregation
  ([`29b0d22`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/29b0d22c7339e271f59ef0db10176f03dc8e5bbb))

- Pass quantization config through
  ([`92335a7`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/92335a748b9da4d8e29f619fda70f76ef8529d37))

- Save token ids to json
  ([`453ae21`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/453ae21228ee1d0cbf5304b08767cc4e0780326a))


## v0.5.0 (2025-03-22)

### Bug Fixes

- Ensure flops working correctly
  ([`e673f12`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/e673f12a3ee4ffe7b81a5cc8fc74f4f0b74d6c04))

- Resolve flops calculation
  ([`788c193`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/788c19373ed2f335cb1c8c0be6a2166581bb9d25))

### Features

- Add robust process cleanup
  ([`9359aa9`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/9359aa982cd0d0b8acab44c8f039423ee4d0b4e3))

- Complete measurement tool functionality
  ([`eb78321`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/eb7832175f23029a9784952a68691450e3512fe8))

- Solve distributed results aggregation
  ([`12ff610`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/12ff61010e3d1684e849f654baa16675c906bc3c))

### Refactoring

- Complete major restructuring
  ([`7874080`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/7874080e1ad6e1f7f5f8784ea387e9775011942a))

- Finalize code reorganization
  ([`0bfc51a`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/0bfc51aa9760f7de79adb5c8b8c10307df5cb5c0))

- Improve distributed stability
  ([`7a8007e`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/7a8007eaa77aa7259604edada660bf7487be902a))

- Integrate optimum benchmark
  ([`df99159`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/df991590029250603d4601b1a54ee0c7b2ea6071))

- Migrate to main script
  ([`c9a428c`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/c9a428ce9752fdfcd31fe7af009670c89b1191fb))


## v0.4.0 (2025-03-12)

### Documentation

- Update todo list
  ([`ddd7df8`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ddd7df8136c5b1d1135267e03b13312a5f04ebb9))

### Features

- Add decoder temp and query rate
  ([`5913ae4`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5913ae43fe45f0fb3739e1f51cfb20fc77a1f524))

- Add tunable variables for multi-gpu
  ([`94d7173`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/94d7173f644943a1d4297b2664707f2e308c2566))

### Refactoring

- Modularize code structure
  ([`69a4098`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/69a4098af6b392d239ab366ab507fc9b3b0dc79f))

- Restructure as package
  ([`8502581`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/85025815505be7a457aafff352453315dbc645ad))


## v0.3.0 (2025-03-05)

### Bug Fixes

- Get metrics working
  ([`f1aa0b5`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/f1aa0b57fce4eaa13ca8f232914d87454ac29699))

- Resolve individual process logging
  ([`83ad7b2`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/83ad7b2a95197f2b93183d5aaf9187ff485376ec))

- Troubleshoot metric aggregation
  ([`ec3c1e8`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ec3c1e8cc6e0fe7a82a8c74cd57940ae0c9ccbca))

### Features

- Add gpu utilization tracking
  ([`785697d`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/785697d3f83bd22790d194f4e972955ffc2fe866))

- Complete initial prototype
  ([`155e993`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/155e993bc2abb2bd041d3c363cba3ed46d58fd9d))


## v0.2.0 (2025-03-04)

### Features

- Add distributed metrics tracking
  ([`5908d8f`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/5908d8fe3f7f19af4a6659859f7c8cd42cde22b9))

- Add json logging
  ([`f0d2d69`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/f0d2d69dc376835ef3a3867dec2b22f26081a23a))

- Implement emission tracking
  ([`a657367`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/a657367da7c09010f9cce2b2b1dbdd6b94d99382))

### Refactoring

- Improve json logging
  ([`1b9096b`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/1b9096b0f0bb0c2492488df2d50282e22efa3efd))

### Testing

- Run initial experiments
  ([`d6276e3`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/d6276e30ced22eb1f678958b063409f31ba544be))


## v0.1.0 (2025-03-02)

### Documentation

- Add todo list
  ([`d599dbc`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/d599dbc5b3bdce3cd7b1e424552bc8a97c4f3bca))

### Features

- Add text generation script
  ([`ce86751`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/ce86751917c0327497e911aa4485e6248d484aec))

- Initial project setup
  ([`dff12a3`](https://github.com/henrycgbaker/llm-efficiency-measurement-tool/commit/dff12a3ab8264d71268dd5a0f5061ff31673a314))
