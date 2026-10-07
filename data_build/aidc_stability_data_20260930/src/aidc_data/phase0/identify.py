from pathlib import Path
import json
import pandas as pd

def identify(out:Path):
    files=list(out.glob('*_nvml.parquet'))
    if not files:
        (out/'phase0_identification.json').write_text(json.dumps({'status':'blocked_hardware','reason':'no paired command, power and performance trials; passive traces cannot identify control effects'},indent=2))
        return
    (out/'phase0_identification.json').write_text(json.dumps({'status':'requires_paired_trials','passive_collections':len(files),'reason':'commands and matched no-treatment performance trials needed'},indent=2))
