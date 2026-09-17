# Production bundle delivery

The production release artifact must be delivered at
`bundle/seq2seq_attention/`. It is intentionally separate from Git because the
project policy excludes trained weights and run artifacts from commits.

Locked release:

- run ID: `seq2seq_attention_20260917_100234_95dc80`
- model version: `1.0.0`
- bundle schema: `2.0`
- bundle manifest canonical SHA-256:
  `3fa89932b9d60fd596a9e8329fb694078b7daadca1b51854f76c88572efc5460`
- `bundle_manifest.json` file SHA-256:
  `b77bf3d2a0ca6d89aabf038060eab406d6b6022b692374d6aee739a992bac699`

Do not reconstruct or substitute `model.pt`. Copy the complete released bundle
directory, then verify it before startup:

```bash
python -c "from pathlib import Path; from src.serving.bundle import verify_bundle; verify_bundle(Path('bundle/seq2seq_attention')); print('PASS')"
python scripts/run_stack.py --bundle bundle/seq2seq_attention
```

The demo payload is also a separately delivered local artifact. Generate it
from the verified Jena source with `scripts/generate_demo_payload.py`, or place
the supplied real validation payload under an arbitrary directory and set
`JENA_DEMO_PAYLOAD_DIR` to that directory. Configured samples must declare
`provenance.synthetic=false`.
