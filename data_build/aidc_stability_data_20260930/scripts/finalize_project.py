#!/usr/bin/env python3
from pathlib import Path
import json,sys
import pandas as pd
from aidc_data.inventory import sha256
out=Path('outputs')/sys.argv[1]
nlr=out/'nlr_raw';q=pd.read_csv(nlr/'quality_audit.csv');v=json.loads((nlr/'validation.json').read_text())
if not v['passed']:raise RuntimeError('NLR validation failed')
summary=pd.read_csv(out/'resolution_export_summary.csv')
resolution={'evidence_type':'derived','parent_evidence_type':'measured_existing','sensor_bandwidth_verified':False,
            'summary_path':str(out/'resolution_export_summary.csv'),'summary_sha256':sha256(out/'resolution_export_summary.csv'),
            'partitions':{str(int(r.bin_seconds*1000))+'ms':str(out/f'bins_{int(r.bin_seconds*1000)}ms') for r in summary.itertuples()}}
(out/'resolution_export_manifest.json').write_text(json.dumps(resolution,indent=2))
mp=out/'dataset_manifest.json';m=json.loads(mp.read_text());m['additional_resolutions']=str(out/'resolution_export_manifest.json');m['additional_raw_track']=str(nlr/'dataset_manifest.json');mp.write_text(json.dumps(m,indent=2))
card=out/'dataset_card_ko.md';text=card.read_text().split('## NLR 원시 로그')[0]
section=f'''\n## NLR 원시 로그 별도 트랙\n\n- 원시 NVML node 로그 {len(q)}개, SHA 중복 0개, 보수적 job group {q.parent_group.nunique()}개. 실제 독립 session 검증은 없다.\n- 원시 {q.source_rows.sum():,}행과 주석 제외 데이터 행 수가 일치한다. GPU 4채널 mW를 W로 변환한 node GPU 합계이며 job/PCC 합계가 아니다.\n- 유효 200 ms bin {q.valid_200ms_bins.sum():,}개, 예측 행 {v['forecast_rows']:,}개, 전력 형태 이벤트 {v['event_count']:,}개. 주로 약 200 ms 원시 timestamp 간격이며 일부 약 100 ms 파일이 있다. 이는 센서 대역폭 확인이 아니다.\n- 공개 NLR 1 ms 집계·미래 보간을 입력으로 사용하지 않았다. timezone, 도착 지연, 동적 phase와 서비스 성능은 미확인이다.\n- 별도 manifest: `{nlr/'dataset_manifest.json'}`.\n'''
card.write_text(text+section)
report=out/'report_ko.md';text=report.read_text().split('## NLR 원시 로그 추가 결과')[0]
report.write_text(text+'\n## NLR 원시 로그 추가 결과\n'+section.split('## NLR 원시 로그 별도 트랙\n',1)[1])
(nlr/'dataset_card_ko.md').write_text('# NLR 원시 NVML 데이터 카드\n'+section.split('## NLR 원시 로그 별도 트랙\n',1)[1])
print('project manifests and dataset cards finalized')
