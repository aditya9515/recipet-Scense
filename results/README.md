# Saved experiment evidence

This directory contains a portable snapshot of the measured project results: held-out predictions and scores, the fixed robustness probe, timing rows, alignment coverage, model training metadata, and verification records. The snapshot uses the public SROIE mirror and explicitly synthetic workflow examples.

[manifest.json](manifest.json) records the copied files and their SHA-256 hashes. Files preserve their recorded contents, including paths from the original execution machine. Those paths are provenance, not portable input locations.

The model metadata here is **not an installed model bundle**. Runtime weights, dataset images, OCR caches, uploaded receipts, and the review database remain outside Git. A fresh checkout runs the rules workflow after installation; follow [RUNNING.md](../RUNNING.md) to acquire assets and train a learned bundle.

Runtime commands continue writing under `artifacts/` and `data/processed/`. Run `python tools/snapshot_results.py` after completing those experiments to refresh this selected snapshot.

The [novelty extension](novelty/baseline_manifest.json) separately preserves the original baseline and publishes an explicit allowlist of aggregate validation, timing and verification evidence under `novelty/`. Its report generator never copies the private confirmation collection or transaction-level predictions. The existing snapshot above is retained as the original experiment record.
