# 첨부 프롬프트 대비 남은 작업 (2026-09-28)

## 이번 실행

`results/nested_remaining_20260928/progress.json`이 +200/+400/+600/+800/+1200/+1400/+1600/+1800 ms grouped-inner 실험의 현재 상태다. 기존 +1000/+2000 ms 선택 실험 및 10-horizon 고정 설정 실험과 분리한다. 1개 CPU에서 순차 실행하며 각 cell의 source·split·code·예측 hash를 검증한다. 완료 전 수치를 성능 결과로 인용하지 않는다. 완료 후 `compare_200ms_nested.py`로 10개 horizon의 표와 그림을 만든다.

## 이미 실행된 범위

- 직렬화 모델 감사, 데이터 inventory·source timestamp 감사, 관측·제어 요구사항 문서화: `STATUS.md`와 `results/actual_20260926/`.
- +1 s 32-run 5×3 nested CV, +2/+5/+10/+30 s reused-holdout grouped-inner 선택, S2 ablation·permutation, 조건 밖 평가: 각 실험 폴더의 `report_ko.md`.
- 200–2000 ms 10-horizon 고정 설정 평가와 5/10/30 s 고정 설정 평가: `results/grid_fixed_20260926/`, `results/long_fixed_20260927/`.
- NLR training job-grouped 4-fold·3-seed 평가, NLR online-rate 단일 수집 그룹 파일럿, recovery detector 재측정, event onset 탐색 평가와 문턱값 선택.
- 시나리오 기반 이상화 제어 replay와 T1–T6/F1–F8 산출물. 실측 PCC/장치 성능은 아니다.

## 별도 결과와 연결 시 주의

`trace/predictability/outputs/main_20260914/report_ko.md`의 50/200/1000 ms 결과는 12개 선택 run의 *동일 run 내부 시간순 후반부*를 평가한다. 32개 run의 그룹 분할 metadata 평가와 표본·분할·목표가 달라 CRPS를 합산하거나 직접 비교하지 않는다. 이 결과는 50 ms 원본 관측 격자 실험의 선행 근거이나 100/500 ms, 전체 32 run, 독립 새 session 확인을 대체하지 않는다.

## 현재 자료로 확정할 수 없는 항목

- Highres 원본의 네 수집일과 GPU×workload 네 조합이 일대일로 결합된다. 새롭고 균형 잡힌 수집 session 없이는 날짜 효과와 조합 효과를 분리하거나 독립 session 일반화를 입증할 수 없다.
- NLR online-rate 200 run은 한 acquisition group이며, request-level arrival/latency와 logger clock 동기화가 검증되지 않았다. 1 ms 보간 grid는 native 1 ms 측정이 아니다. 원시 비동기 로그의 causal 재구성은 아직 없다.
- 실제 센서 availability, filler/cap/BESS·firmware 응답, 성능 간섭, PCC 계측·계통 모델, 실제 허용 ramp/SLA가 없다. 따라서 제어 replay는 가정별 GPU 전력 대리지표이고 계통 주파수·전압 안정성 결과가 아니다.
- 전체 highres 32 run에서 원시 50/100/500/1000 ms bin을 같은 물리적 target window·issuance 조건으로 맞춘 해상도 확장 평가와 배포 전 calibration 재확인은 아직 실행되지 않았다. 기존 predictability 결과는 50/1000 ms의 일부 근거로만 사용한다.

재개 명령과 출력 경로는 `results/nested_remaining_20260928/README.md`에 있다.
