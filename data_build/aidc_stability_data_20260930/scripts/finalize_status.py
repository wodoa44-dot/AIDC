#!/usr/bin/env python3
import json,sys
from pathlib import Path
out=Path('outputs')/sys.argv[1];p=out/'build_status.json';s=json.loads(p.read_text())
nlr=json.loads((out/'nlr_raw/build_status.json').read_text()) if (out/'nlr_raw/build_status.json').exists() else {}
grid=json.loads((out/'grid_status.json').read_text()) if (out/'grid_status.json').exists() else {}
s['stages']={
 'D1_highres':{'status':'completed' if s.get('build_existing')=='completed' else 'running','implementation_complete':True,'data_generated':s.get('build_existing')=='completed'},
 'D2_highres':{'status':'completed' if s.get('validation')=='completed' else 'running','implementation_complete':True,'data_generated':s.get('validation')=='completed'},
 'D1_D2_nlr_raw':{'status':nlr.get('status','not_applicable'),'implementation_complete':True,'data_generated':nlr.get('status')=='completed'},
 'D3_measured_control':{'status':'blocked_hardware','implementation_complete':False,'collector_implemented':True,'protocol_prepared':True,'data_generated':False,'reason':'no authorized GPU UUID, instrumented workload, paired trials or service logs'},
 'D4_frequency_scenario':{'status':grid.get('status','blocked_model'),'implementation_complete':True,'data_generated':grid.get('status')=='completed'},
 'D4_voltage_network':{'status':'blocked_model','implementation_complete':False,'data_generated':False,'reason':'no validated dynamic network/voltage model'},
 'D5_facility':{'status':'blocked_source','implementation_complete':False,'data_generated':False,'reason':'no PCC/ESS/UPS meter logs'} }
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(s['stages'],indent=2))
