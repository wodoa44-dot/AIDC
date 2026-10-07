#!/usr/bin/env python3
"""Read one built highres fold. No model is fit here."""
import argparse,json
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--build-id',default='main_20260929');p.add_argument('--fold',type=int,default=0);a=p.parse_args()
out=Path(__file__).resolve().parents[1]/'outputs'/a.build_id
manifest=json.loads((out/'dataset_manifest.json').read_text())
split=json.loads((out/'split_manifest.json').read_text())['outer_folds'][a.fold]
index=pd.read_parquet(manifest['files']['forecast_index']['path'])[['row_id','split_group_id']]
features=pd.read_parquet(manifest['files']['features_power_history']['path'])
targets=pd.read_parquet(manifest['files']['targets']['path'])[['row_id','target_power_1s_w']]
frame=index.merge(features,on='row_id',validate='one_to_one',suffixes=('','_feature')).merge(targets,on='row_id',validate='one_to_one')
feature_columns=['current_power_w']+[c for c in frame if c.startswith('hist_') and c.endswith(('_w','_w_per_s'))]
train=frame[frame.split_group_id.isin(split['train_groups'])]
test=frame[frame.split_group_id.isin(split['test_groups'])]
assert not set(train.split_group_id)&set(test.split_group_id)
print({'schema_version':manifest['schema_version'],'train_rows':len(train),'test_rows':len(test),
       'feature_columns':feature_columns,'target':'target_power_1s_w'})
