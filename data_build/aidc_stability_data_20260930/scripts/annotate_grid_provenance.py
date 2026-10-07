from pathlib import Path
import json,sys
out=Path('outputs')/sys.argv[1]
p=out/'grid_model_provenance.json';d=json.loads(p.read_text())
d['source_license_status']='not established; no paper/model file redistributed'
d['parameter_origin']={'inertia_m_s':'published example','damping_pu':'published example','droop_r_pu':'published example','governor_tg_s':'published example','nominal_frequency_hz':'scenario nominal','base_mva_scenario':'scenario assumption','synchronized_copies_scenario':'scenario assumption','simulation_dt_s':'numerical choice','control_dt_s':'input bin period'}
p.write_text(json.dumps(d,indent=2))
