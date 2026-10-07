"""Probe the official benchmark with project-local generated code."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / '.grid_deps'))
import andes

ss = andes.load(andes.get_case('kundur/kundur_full.xlsx'), setup=False,
                no_output=True, no_undill=True,
                pycode_path=str(PROJECT / '.grid_pycode'))
if not ss._load_calls():
    ss.prepare(quick=True, nomp=True)
print('Models:', {k: v.n for k, v in ss.models.items() if v.n}, flush=True)
print('Loads:', ss.PQ.as_df().to_string(), flush=True)
ss.Toggle.u.v[:] = [0] * ss.Toggle.n
ss.setup()
print('Power flow:', ss.PFlow.run(), flush=True)
ss.TDS.config.tf = 2.0
ss.TDS.config.no_tqdm = 1
ss.TDS.config.fixt = 1
ss.TDS.config.tstep = .02
print('TDS:', ss.TDS.run(), flush=True)
print('Final time:', ss.dae.t, flush=True)
