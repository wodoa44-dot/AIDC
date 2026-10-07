"""Korean Markdown/HTML and plots, including an honest blocked-stage report."""
import json
import html
from pathlib import Path
import pandas as pd
from plotting import plot_results
from common import atomic_json
from config import load_config


def report(out: Path, results_ready: bool = False) -> None:
    """Generate Korean report from saved artifacts; no training or label generation."""
    (out / "figures").mkdir(exist_ok=True)
    status = json.loads((out / "completion_status.json").read_text())
    if results_ready:
        status["all_completed"] = True
    inventory = pd.read_csv(out / "run_inventory.csv")
    labels = pd.read_parquet(out / "labels.parquet")
    dates = (
        inventory.groupby("collection_day")
        .agg(
            runs=("run_id", "nunique"),
            training_rows=("training_rows", "sum"),
            evaluation_rows=("evaluation_rows", "sum"),
        )
        .reset_index()
    )
    for c in ["R0", "R1", "R2", "R3", "U"]:
        counts = labels[labels.label.eq(c)].groupby("collection_day").size()
        dates[c] = dates.collection_day.map(counts).fillna(0).astype(int)
    dates["U_fraction"] = dates.U / dates[["R0", "R1", "R2", "R3", "U"]].sum(axis=1)
    dates.to_csv(out / "date_class_summary.csv", index=False)
    plot_results(out, dates, status)
    plot_status = []
    if status["all_completed"]:
        metrics = pd.read_csv(out / "aggregate_metrics.csv")
        cl = pd.read_csv(out / "class_metrics.csv")
        tables = (
            "\n\n"
            + metrics.to_markdown(index=False)
            + "\n\n"
            + cl.to_markdown(index=False)
            + "\n\n"
            + pd.read_csv(out / "metadata_effect.csv").to_markdown(index=False)
        )
    else:
        tables = "성능 표는 미실행(N/A)이다. 본 학습은 아직 실행하지 않았다. smoke 실행 여부와 fit 수는 completion_status.json을 따른다. confusion/recall/run MAE/확률/실제-예측 수준 그림은 label 기준 확정 후에만 생성할 수 있다."
        plot_status = [
            {"plot": p, "status": "blocked_no_valid_labels_or_predictions"}
            for p in [
                "confusion_matrix",
                "class_recall",
                "run_ordinal_mae",
                "power_actual_predicted_response",
                "probability_by_true_class",
            ]
        ]
    atomic_json(out / "plot_status.json", plot_status)
    missing = pd.read_csv(out / "missing_configuration.csv")
    evidence = pd.read_csv(out / "source_evidence.csv")
    definitions = pd.DataFrame(
        [
            ["R0", "F0=True", "추가 대응 없이 충족"],
            ["R1", "F0=False, F1=True", "기본 자원으로 충족"],
            ["R2", "F0=False, F1=False, F2=True", "확대 자원으로 충족"],
            ["R3", "모든 F=False, U2 status=2", "확대 자원 infeasibility 확인"],
            ["U", "설정·자료·수치·solver 문제", "학습 제외; R3로 치환하지 않음"],
        ],
        columns=["등급", "조건", "의미"],
    )
    spec = load_config(out / "response_label_spec.yaml")
    config = spec["experiment"]
    answers = """1. 실제 수치 label 기준은 아직 없다. 설정 부족을 U로 기록했으며 임의 경계를 만들지 않았다.
2. 실제 계통 기준이 아니라 PCC proxy 설계를 준비하는 단계다.
3. R0–R3 구분 자료가 충분한지 판정하지 못했다. 미래 파형은 준비됐지만 자원·요구조건이 없다.
4. Baseline보다 좋은지 판단 불가. 학습·성능 평가를 실행하지 않았다.
5. 과소분류 run·변화 형태는 미평가다. 상승·하강·혼합은 수준과 독립된 보조 속성으로만 준비했다.
6. Metadata 효과는 미평가다.
7. 확대 검증 가치는 현재 판단할 수 없다. 먼저 요구조건·자원·복구 조건을 확정해야 한다. 다른 모델·대규모 실험은 자동 시작하지 않는다."""
    if status["all_completed"]:
        rb = metrics[metrics.aggregation.eq("run_balanced")].pivot(
            index="scope", columns="model", values="ordinal_mae"
        )
        severe = int(labels.label.isin(["R2", "R3"]).sum())
        answers = "1. 실제 수치 기준은 response_label_spec.yaml 및 source_evidence.csv에 고정했다.\n2. GPU 합산 전력에 대한 PCC proxy 실험이며 실제 주파수·전압 안정성 검증이 아니다.\n3. 심각 수준별 분포는 위 표를 참조한다."
        if severe < config["severe_minimum_anchors"]:
            answers += f' 심각 대응 수준 예측을 검증하지 못함: R2/R3가 {config["severe_minimum_anchors"]}개 미만이다(사전 pilot 보고 기준).'
        answers += (
            "\n4. Run-balanced ordinal MAE 비교:\n\n"
            + rb.to_markdown()
            + "\n\n5. run/상승/하강/혼합별 오분류는 direction_error_analysis.csv를 참조한다.\n6. M1−M2는 metadata_effect.csv에 run별 저장했다. 양수는 M2 개선이다.\n7. 독립 날짜·session 검증이 필요하며 다른 모델·대규모 실험은 자동 시작하지 않는다."
        )
    spec = load_config(out / "response_label_spec.yaml")
    config = spec["experiment"]
    assumption_note = (
        "가상 공학 pilot: 실측 GPU 전력에 명시적인 가상 자원을 적용했다. 자원 출력·용량·복구·허용오차는 장치 실측 또는 제조사 검증값이 아니다. 성능은 이 설정에 조건부이며 실제 PCC 또는 계통 안정성을 입증하지 않는다."
        if spec.get("status") == "hypothetical_engineering_pilot"
        else spec["interpretation"]
    )
    requirements = spec["requirements"]
    parameter_summary = f"P_base=최신 완료 200ms bin W; ramp 상향/하향={requirements['ramp_up_w_s']}/{requirements['ramp_down_w_s']} W/s; 복구={requirements['recovery_s']} s; W/J/SoC 허용오차={requirements['tolerance_w']}/{requirements['tolerance_j']}/{requirements['tolerance_fraction']}."
    resource_rows = []
    for level in ["U1", "U2_additional"]:
        for device in spec["resources"].get(level) or []:
            resource_rows.append(
                {
                    "level": level,
                    "id": device["id"],
                    "charge_w": device["charge_w"],
                    "discharge_w": device["discharge_w"],
                    "capacity_j": device["capacity_j"],
                    "throughput_budget_j": device["throughput_budget_j"],
                    "terminal_soc_tolerance": device["terminal_soc_tolerance"],
                    "category": device["category"],
                }
            )
    resource_table = (
        pd.DataFrame(resource_rows).to_markdown(index=False)
        if resource_rows
        else "자원 미확정"
    )
    text = f"""# 고정된 PCC proxy 요구조건에 대한 대응 수준 예측 — 소규모 pilot

{assumption_note}

완료 상태: **{'전체 완료' if status['all_completed'] else '부분 완료 — completion_status.json 참조'}**. 기존 회귀 실험은 보존했다. 이는 회귀 실험 완료 결과 또는 계통 안정성 입증이 아니다.

## label 기준과 적용 한계

{parameter_summary}

{resource_table}

{definitions.to_markdown(index=False)}

R0는 전력 변화가 없다는 뜻이 아니며 R3는 실제 정전·계통 붕괴를 뜻하지 않는다. R0–R3는 이번 연구의 정의이며 문헌 표준 등급이 아니다. 기본/확대 자원의 허용 집합을 확정하기 전에는 실제 R label을 부여하지 않았다. `labels.parquet`의 U는 설정·데이터·수치 실패를 뜻하며 물리 infeasibility로 치환하지 않는다. 미래 actual을 사용하는 oracle label은 causal controller 운전 가능성의 입증이 아니다.

실측은 GPU 8개 합산 W이며 CPU·냉각·기타 부하는 포함되지 않았다. 시설 PCC 실측과 주파수·전압 모형이 없다. GPU sum identity PCC proxy 채택 여부는 저장된 설정을 따른다. 200ms bin 평균 해상도이며 300ms 내부 세부 파형·sub-bin peak를 복원하지 않았다. 미래는 (t,t+0.2]부터 (t+0.8,t+1.0]까지 5개 연속 bin으로 평가한다. endpoint가 같아도 중간 변화는 별도로 평가한다.

## 문헌 확인과 수치 출처

{evidence.to_markdown(index=False)}

NERC는 현장별 운영 한계와 모델·장치 정보를 요구하는 참고 지침이다. 500 W/s를 실제 계통 기준으로 제공하지 않는다. 변동 억제율은 mitigation 실험 변수이며 등급 경계로 복사하지 않았다. NVIDIA의 GB300 에너지 저장 수치는 H100/B200 node의 대응 자원을 증명하지 않는다. Grid-mode MPC의 soft constraint 정상 종료는 요구조건 충족을 보장하지 않는다. 이번 판정기는 hard feasibility와 별도 물리 재계산을 구현했다.

## 미확정 필수 설정

{missing.to_markdown(index=False)}

U1/U2의 각 장치는 charge/discharge W, capacity J, initial/min/max SoC, 방향별 효율, delay bins, slew W/s, throughput budget J, terminal SoC tolerance와 출처·정확한 위치를 요구한다. U2는 U1 장치를 그대로 포함하고 추가 장치는 사용하지 않는 선택을 허용해야 한다. 종료 조건은 모든 장치의 초기 SoC 복귀 허용오차와 출력 0이며, 그 조건을 이번 실험에 채택할 근거가 필요하다. workload 동작은 아직 모델링하지 않았으므로 요청될 경우 구현 검증 후에만 진행한다.

## 데이터 준비·날짜·표본

{dates.to_markdown(index=False)}

표의 R/U 수는 train/evaluation anchor 합집합의 고유 sample key 기준이다. U 비율 {float(labels.label.eq('U').mean()):.1%}. U 사유는 labels.parquet의 status·unknown_reason으로 구분하며 학습에서 제외한다.

32개 중복 제거 highres_node run과 기존 manifest·prepared data·feature helper를 재사용했다. 기존 coverage는 count≥ceil(0.8×0.2/관측 dt), 관측 간격 5배 초과 gap의 양쪽 bin 무효화, 보간 없음이다. 개발은 09-16/17/20의 leave-day-out 3 folds, 보류는 09-22다. 날짜 불일치는 split_manifest.json에 저장했다. 수집 session의 독립성은 미확인이고 보류 날짜도 과거 실험에서 사용했으므로 독립 외부검증·새로운 test가 아니다.

각 연속 block에서 10초 history와 관측 미래를 확보한 첫 유효 anchor부터 최소 1초 간격으로 후보를 만들었다. run 후보 30개 미만 제외, training 최대 {config['train_cap']}개(10구간 비복원, seed {config['seed']}와 run SHA256), evaluation 최대 {config['evaluation_cap']}개(10구간 인덱스 등간격), cap 미만 그대로 사용한다. 전력·등급·오차를 보고 표본을 바꾸지 않는다. 구간은 겹치므로 독립 표본이 아니다.

복구 시간 {spec['requirements']['recovery_s']} s를 포함한 전체 관측 미래로 anchor를 계산한다. 미확정 설정에서는 1초 미래만 확보한 잠정 anchor이며 persistence로 미래를 채우지 않는다. Feature는 추출 전 원본 연속 시계열에서 계산했다. 현재 bin·lag·causal rolling mean/std·과거 차이만 사용하고 future/solver/날짜/run ID/파일명은 feature에서 제외한다. 저장된 개발 선택 residual_h5에 따라 history 5초를 재사용했고 보류 성능으로 재선택하지 않았다. Metadata는 실제 존재하는 알려진 정적 항목만 포함한다.

## 모델과 평가 계획·현재 성능

{tables}

LightGBM 계열 M1 history-only, M2 history+metadata 두 구성만 고정한다. 본 fit 최대 8회, smoke 최대 100행·5trees·1fit, seed17·100trees·15leaves·learning rate0.07·min child30·L1/L2=0·subsampling 없음·early stopping 없음·weight 추가 없음·최대 CPU4 threads·순차 학습이다. U 제외 후 N/(R×nr)로 동일 run 기여를 맞춘다. 범주 mapping은 train에서만 fit, unseen=-1(missing)이며 feature별 비율을 기록한다. 누락 클래스 확률=0, 단일 클래스는 상수 예측, 합성 표본을 학습에 사용하지 않는다. 예측은 argmax다.

Primary는 run-balanced ordinal MAE(R0=0…R3=3)이며 등간격은 평가 설계값이지 피해 선형성이 아니다. row pooled와 run-balanced를 구분한다. Confusion matrix는 run별 전체 행으로 정규화해 run 평균을 내며 class recall은 그 matrix의 각 실제 클래스 행합으로 다시 나눈다. support0 recall=N/A, Macro-F1은 해당 평가 cohort의 실제 support>0 클래스만 평균한다. 심각 수준→낮은 수준 비율은 전체 표본 분모와 실제 R2/R3 조건부 분모를 모두 저장한다. log loss는 0확률을 라이브러리의 수치 epsilon으로 clip한다.

## 최종 질문에 대한 현재 답

{answers}

## 실행·검증

completion_status.json, label_validation.json, preparation_validation.json, progress.json 참조. 합성 파형 판정기 테스트는 실제 label 검증이나 classifier smoke가 아니다. label 검증·smoke가 통과하기 전에는 nohup 학습을 시작하지 않는다. 실행 결과의 완료 여부는 completion_status.json과 exit_status.json으로 확인한다. report-only는 저장 산출물로 표·그림·한국어 보고서만 재생성한다.

![날짜별 U 감사](figures/date_class_ratio.png)

![날짜별 준비 anchor](figures/anchor_counts.png)
"""
    (out / "report_ko.md").write_text(text)
    try:
        import markdown

        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + html.escape(text) + "</pre>"
    (out / "report_ko.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>AIDC 대응 수준 pilot</title><style>body{font-family:sans-serif;max-width:1400px;margin:32px auto;line-height:1.6}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
        + body
        + "</html>"
    )
