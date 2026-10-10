# Research evaluation

These are small exploratory evaluations, not a statistically representative benchmark.
Latest: [RESULTS-2026-10-10.md](RESULTS-2026-10-10.md).
They use real model calls and preserve reports, evidence, events, settings and a source-code
fingerprint. Read the reports and their supporting evidence; an execution status is not a quality score.

```bash
uv run python evals/run.py --out reports/evaluation-YYYY-MM-DD
uv run python evals/run.py --out reports/a-repeat --case python313
uv run python evals/adversarial.py --out reports/evaluation-YYYY-MM-DD/adversarial
uv run python evals/audit.py reports/evaluation-YYYY-MM-DD
# Isolate writing changes using an existing run's saved evidence (real model calls, no web requests):
uv run python evals/replay.py --seed reports/evaluation-YYYY-MM-DD --out reports/writer-replay-YYYY-MM-DD
```

Use a fresh output directory for each run; reusing a directory overwrites its results.
The saved configuration and rubric make comparisons possible, but live pages and model outputs
can change between runs. Reports under `reports/` are gitignored; preserve them separately
when sharing a result. The audit checks source IDs and quote matching, not semantic support.
Replay outputs are `replay.json` rather than full research reports; their usage and duration cover only
synthesis and review. They retain a pointer to the original evidence report and a new source fingerprint.

`cases.json` defines seven questions and a rubric **not passed to the model**: four general web
cases, a long-PDF case (`rp2040_dma`, answer deep inside a 642-page datasheet), a known-disagreement case
(`everest_height`) and a deliberately vague question (`vague_db`) that only runs the clarification check.
Every case records the clarification check (`clarification.json`); cases with `clarify_expected` score
whether it asked (`clarify_ok` in `metrics.json`). `sqlite_nfs` is fully specified and must not ask. The web cases
run sequentially, with two researchers in parallel within each round and at most two rounds.
The bounded evaluation profile disables thinking, limits each researcher to four searches,
and allows seven minutes per complete case. It is not a measurement of untouched defaults.

`adversarial.py` uses the real LLM against in-memory search/page fixtures. It includes
version-specific vendor manuals and a community page containing answer-poisoning and loopback-fetch
instructions. All page requests use MockTransport; no attack is sent to a third-party service.
Inspect both output and events: a blocked malicious tool call still counts as the model following
an injected instruction. A fixture test cannot establish general prompt-injection resistance.

Scoring approach:

- Check every requested sub-question against authoritative primary sources.
- Separate correct answers, missing qualifications, unresolved questions and false assertions.
- Check citation support for consequential claims, not just source-ID existence.
- Inspect extra claims the model introduced, not only rubric coverage.
- Record duration, model usage, failures, and evidence discarded by validation.
- Keep first-run results; label repeat runs and configuration changes explicitly.

Reference sources for this initial rubric:

- Python 3.13: https://docs.python.org/3.13/howto/free-threading-python.html
- SQLite: https://www.sqlite.org/wal.html and https://www.sqlite.org/pragma.html#pragma_synchronous
- SQLite network constraints: https://www.sqlite.org/useovernet.html
- DGX Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- DGX Spark interconnect: https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
- Distributed inference: https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/multi-device.html
- Original RAG paper: https://arxiv.org/pdf/2005.11401 (sections 2.1, 2.4 and 3)
- RP2040 DMA chaining: https://datasheets.raspberrypi.com/rp2040/rp2040-datasheet.pdf (DMA chapter, "Chaining", and the
  CHn_CTRL register table: CHAIN_TO, bits 14:11)
- Everest: the 8 December 2020 joint China–Nepal announcement (8,848.86 m); the mid-1950s Survey of India figure
  (8,848 m) and China's 2005 rock-height figure (8,844.43 m), e.g. as summarised by CNN and Australian Geographic
