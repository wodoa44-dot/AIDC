"""Evidence-aware enrichment. Unknown fields remain null; source fields are retained."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .discovery import discover, _normalize_result_key, stream_hash
from .contract import digest, SCHEMA_VERSION
FIELDS = ['model_name','model_size','workload_type','gpu_model','gpu_count','node_count',
          'batch_size','batch_semantics','sequence_length','precision','parallelism','gradient_accumulation',
          'request_rate','concurrency','input_length','output_length','power_cap_w','clock_settings',
          'completion_status','repeat_id','original_run_id','original_experiment_group',
          'source_sensor_cadence_s','sensor_averaging_window_s']


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if value is None or (np.isscalar(value) and pd.isna(value)): return None
    if isinstance(value, np.generic): return value.item()
    return value


def recover(cfg, selected=None, limit=None):
    selected=selected or cfg['datasets']
    manifest = discover(cfg, selected, limit)
    if 'highres_single' in selected:
        files=sorted(p for p in Path(cfg['inputs']['highres']).rglob('*') if p.suffix.lower()=='.csv' and 'Single_Machine_Dataset' in str(p))
        if limit: files=files[:limit]
        additional=[]
        for p in files:
            hs=stream_hash(p)
            additional.append(dict(dataset='highres_single',source_path=str(p.resolve()),source_hash=hs,
                source_bytes=p.stat().st_size,run_id='highres_single-'+hs[:16],canonical_path=str(p.resolve()),
                is_duplicate=False,workload=p.parents[1].name,model=None,hardware=None,gpu_count=None,node_count=None,
                power_scope='single GPU board power; CPU excluded',metadata_join_status='not_applicable',
                completion_status='unknown',timestamp_column='Date + Time',source_units='W',scope_evidence='GPU Power [W] column; dataset README single-machine monitoring'))
        manifest=pd.concat([manifest,pd.DataFrame(additional)],ignore_index=True)
        for hs,g in manifest.groupby('source_hash'):
            canonical=min(g.source_path)
            manifest.loc[g.index,'canonical_path']=canonical
            manifest.loc[g.index,'is_duplicate']=g.source_path!=canonical
    tables = {}
    for p in Path(cfg['inputs']['nlr']).glob('01_aggregated_datasets/*/metadata.csv'):
        table = pd.read_csv(p, dtype={'id':str, 'slurmid':str})
        pathcol = 'path_run' if 'path_run' in table else 'path_save'
        tables[p.parent.name] = (p, table, pathcol)
    evidence, records = [], []
    for r in manifest.itertuples():
        p = Path(r.source_path)
        values = {k: None for k in FIELDS}
        statuses, sources = {}, {}
        def setv(field, value, source, status='source_declared'):
            value = clean(value)
            if value is not None:
                values[field], statuses[field], sources[field] = value, status, str(source)
        extra = {}
        if r.dataset in ['highres_node','highres_single']:
            s = str(p)
            def token(pattern):
                found = re.search(pattern, s)
                return found.group(1) if found else None
            setv('gpu_model', token(r'(H100|B200)'), p)
            setv('gpu_count', int(token(r'(\d+)GPU')) if token(r'(\d+)GPU') else None, p)
            setv('node_count', 1 if '_Node_' in p.name else None, p)
            llm = token(r'(?:Lamma|Llama)(\d+B)')
            setv('model_name', 'Llama' if llm else ('diffusion' if 'Diffusion' in s else None), p)
            setv('model_size', llm, p)
            extra['filename_model_size_token'] = token(r'ModelSize([\d.]+)')
            setv('workload_type', 'image_generation_training' if 'Diffusion' in s else ('llm_training' if llm else None), p)
            for field, pattern in [('batch_size',r'BatchSize(\d+)'),('sequence_length',r'SeqLength(\d+)'),('parallelism',r'Deepspeedds_(z\d+)')]:
                setv(field, token(pattern), p)
            extra['image_size'] = token(r'ImageSize(\d+)')
            setv('original_run_id', p.stem, p, 'verified')
            setv('original_experiment_group', str(p.parent.relative_to(Path(cfg['inputs']['highres']))), p, 'verified')
            interpolation = 'none introduced; upstream sensor averaging unknown'
            channels = [f'gpu{i}_power_W' for i in range(8)]
            rule = 'strict sum of eight finite nonnegative GPU readings; CPU excluded'
            group = 'content-'+r.source_hash
            causal = 'sensor update/averaging latency unknown; recorded-sample interpretation only'
            notes=list(p.parent.glob('*.txt'))
            if r.dataset=='highres_node':
                notes+=list((p.parents[1]/'Model_size_impact').glob('*.txt'))
                values_size=set()
                for note in notes:
                    for base,size in re.findall(r'Base channel (\d+).*?([\d.]+[MKB]) Learnable',note.read_text(errors='replace'),re.I):
                        if base==extra.get('filename_model_size_token'):values_size.add(size)
                if len(values_size)==1:setv('model_size',next(iter(values_size)), '; '.join(map(str,notes)))
                extra['model_size_note_ambiguous']=len(values_size)>1
            else:
                readme=next(Path(cfg['inputs']['highres']).rglob('README.md'))
                setv('gpu_model','NVIDIA GeForce RTX 3060',readme)
                setv('gpu_count',1,readme)
                setv('node_count',1,readme)
                values['model_name']=None
                statuses.pop('model_name',None)
                setv('workload_type',p.parents[1].name,p.parents[1],'source_declared')
                setv('batch_size',token(r'batchsize(\d+)'),p)
                setv('sequence_length',token(r'(?:SeqLen|Sequancelength)(\d+)'),p)
                extra['filename_tokens']=p.stem
                channels=['GPU Power [W]']
                rule='preserve measured GPU Power [W] board channel; measured CPU package channel retained but excluded from target'
            extra['source_model_notes']={str(note):note.read_text(errors='replace') for note in set(notes)}
        else:
            mp, table, pathcol = tables[p.parent.parent.name]
            matches = table[table[pathcol].map(_normalize_result_key) == _normalize_result_key(p)]
            extra = clean(matches.iloc[0].to_dict()) if len(matches)==1 else {}
            mapping = {'model_name':['model','model_id'], 'node_count':['nodes'], 'batch_size':['batch_size'],
                       'request_rate':['request_rate','request_rate_x'], 'concurrency':['max_concurrency'],
                       'output_length':['max_output_tokens','hf-output-len'], 'completion_status':['run_completed'],
                       'repeat_id':['repeat','_repeat'], 'original_run_id':['slurmid','id']}
            for field, names in mapping.items():
                for name in names:
                    if clean(extra.get(name)) is not None:
                        setv(field, extra[name], f'{mp}: {name}'); break
            setv('workload_type', 'training' if r.dataset=='nlr_training' else 'inference', p.parent.parent, 'verified')
            setv('original_experiment_group', p.parent.parent.name, p.parent.parent, 'verified')
            if r.dataset != 'nlr_training':
                setv('node_count', 1, p.parent.parent/'postprocess.py: single-node NVML/RAPL extraction', 'verified')
                # A directory name is evidence of a declared family, never an observed exact variant.
                if values['model_name'] is None:
                    setv('model_name', 'Llama3 70B (directory family; exact variant unknown)', p.parent.parent)
            rawcfg = Path(cfg['inputs']['nlr'])/'00_raw_datasets'/p.parent.parent.name/'config.json'
            if rawcfg.exists():
                setting = json.loads(rawcfg.read_text())
                if setting.get('model'): setv('model_name', setting['model'], str(rawcfg)+': model')
                slurm = setting.get('slurm',setting)
                if slurm.get('SLURM_GPUS_ON_NODE'): setv('gpu_count',int(slurm['SLURM_GPUS_ON_NODE']),str(rawcfg)+': SLURM_GPUS_ON_NODE')
                extra['acquisition_config_path'] = str(rawcfg)
                extra['acquisition_config_hash'] = stream_hash(rawcfg)
            interpolation = 'upstream polynomial order-1 interpolation on common grid; CPU/GPU sum; source coverage unknown'
            if r.dataset=='nlr_online_rate':
                interpolation += '; upstream >800 W per-channel cleaning with backward fill; 1 ms grid'
            channels = ['power[W]']
            rule = 'preserve supplied CPU+GPU aggregate; training includes participating nodes; RAPL package/core accounting inherited'
            # Inference extracts share underlying acquisition logs. Conservative grouping avoids uncertain overlap leakage.
            group = ('job-'+str(values['original_run_id'])) if r.dataset=='nlr_training' and values['original_run_id'] else 'acquisition-'+r.dataset
            causal = 'unsupported: upstream interpolation can use future observations'
        for field in FIELDS:
            evidence.append({'run_id':r.run_id, 'source_file_id':digest(str(p)), 'field':field,
                             'value':values[field], 'status':statuses.get(field,'unknown'),
                             'evidence':sources.get(field,'not established from inspected source')})
        record = dict(dataset_id=r.dataset, run_id=r.run_id, source_file_id=digest(str(p)),
                      source_hash=r.source_hash, source_path=str(p), schema_version=SCHEMA_VERSION,
                      group_id=group, power_scope=r.power_scope, target_unit='W',
                      included_channels=json.dumps(channels), aggregation_rule=rule,
                      interpolation_history=interpolation, realtime_interpretation=causal,
                      raw_source_coverage='unknown', source_metadata_json=json.dumps(extra, sort_keys=True),
                      metadata_status_json=json.dumps({k:statuses.get(k,'unknown') for k in FIELDS}), **values)
        records.append(record)
    metadata = pd.DataFrame(records)
    # Exact aliases always share the same leakage group, regardless of path-based metadata.
    groups = metadata.groupby('source_hash').group_id.first()
    metadata['group_id'] = metadata.source_hash.map(groups)
    for col, field in [('model','model_name'),('hardware','gpu_model'),('gpu_count','gpu_count'),('node_count','node_count')]:
        manifest[col] = metadata[field].to_numpy()
    manifest['group_id'] = metadata.group_id.to_numpy()
    manifest['schema_version'] = SCHEMA_VERSION
    return manifest, metadata, evidence
