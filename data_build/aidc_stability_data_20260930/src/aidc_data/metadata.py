from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

PATTERN=re.compile(r'(ModelSize|ImageSize|BatchSize|SeqLen|SequenceLength|ZeroStage|ZeROStage)(\d+)',re.I)
def extract(rows):
    data=[]
    for r in rows.itertuples():
        raw=Path(r.source_path).stem
        tokens={k.lower():int(v) for k,v in PATTERN.findall(raw)}
        static={'workload_family':r.workload,'hardware_label':r.hardware,'gpu_count':8,'filename_raw':raw,
            'model_size_token':tokens.get('modelsize'),'image_size_token':tokens.get('imagesize'),
            'batch_size_token':tokens.get('batchsize'),'sequence_length_token':tokens.get('seqlen',tokens.get('sequencelength')),
            'zero_stage_token':tokens.get('zerostage')}
        for key,value in static.items():
            data.append({'run_id':r.run_id,'split_group_id':r.split_group_id,'field':key,'value':str(value) if pd.notna(value) else None,
                'unit':'token' if key.endswith('_token') else None,'source':str(r.source_path),
                'extraction_method':'filename_regex' if key.endswith('_token') or key=='filename_raw' else 'source_manifest',
                'confidence':'medium' if key.endswith('_token') else 'high','valid_from':None,'valid_to':None,
                'available_at':None,'availability_assumption':'static known before run only if job configuration was disclosed',
                'evidence_type':'derived','parent_evidence_type':'measured_existing'})
    return pd.DataFrame(data)
