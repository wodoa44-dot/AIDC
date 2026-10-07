from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_tables(out:Path, quantiles):
    parts=sorted((out/"prediction_parts").glob("*.parquet"))
    if not parts: return False
    from .evaluation import metrics,aggregate,paired_bootstrap
    predictions=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
    key=["experiment_id","dataset","group_id","run_id","segment","fold","seed","model","target","horizon","issuance_s","target_time_s"]
    if predictions.duplicated(key).any(): raise ValueError("duplicate prediction key")
    predictions.to_parquet(out/"per_target_predictions.parquet",index=False)
    run,tail=metrics(predictions,quantiles)
    run.to_csv(out/"per_run_metrics.csv",index=False)
    tail.to_csv(out/"tail_metrics.csv",index=False)
    aggregate(run).to_csv(out/"aggregate_metrics.csv",index=False)
    paired_bootstrap(run).to_csv(out/"paired_gain_ci.csv",index=False)
    return True


def make_figures(out:Path):
    p=out/"aggregate_metrics.csv"
    if not p.exists():return False
    a=pd.read_csv(p);a.to_csv(out/"figure_data_horizon_crps.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    for name,g in a.groupby("model"):
        ax.plot(g.horizon,g.crps,marker="o",label=name)
    ax.set(xlabel="Forecast horizon (s)",ylabel="CRPS approximation (W)")
    ax.legend();fig.tight_layout()
    fig.savefig(out/"horizon_crps.png",dpi=300);fig.savefig(out/"horizon_crps.pdf");plt.close(fig)
    return True


def report(out:Path):
    a=out/"aggregate_metrics.csv"; c=out/"control_metrics.csv"
    lines=["# AIDC 4단계 실행 보고", "", "## 증거 수준", "", "E0는 GPU 합산 전력 trace 예측입니다. E1은 가정한 actuator의 PCC proxy입니다. 검증된 E2/E3 계통 응답은 현재 없습니다.",""]
    if a.exists():
        x=pd.read_csv(a);lines.extend(["## 예측", "",x.to_markdown(index=False),""])
        if {"P2","P4"}.issubset(x.model.unique()):
            for h in [1.,2.]:
                y=x[np.isclose(x.horizon,h)].set_index("model")
                if {"P2","P4"}.issubset(y.index):
                    gain=100*(1-y.loc["P4","crps"]/y.loc["P2","crps"])
                    lines.append(f"{h:g} s P2 대비 P4 CRPS gain: {gain:.3f}% (고정 예측·run 균등 집계).")
    else:lines.append("예측 본 실험은 아직 완료되지 않았습니다.")
    if c.exists():lines.extend(["", "## E1 가정 시나리오", "",pd.read_csv(c).groupby("controller").mean(numeric_only=True).to_markdown(),""])
    lines.extend(["", "## 미식별 요소", "", "원본 trace는 GPU만 측정합니다. GPU cap 이후 소비·처리량/지연, ESS 반응, 시설 P/Q/V/f, 계통 동특성은 식별되지 않았습니다. 따라서 계통 안정화 결론은 낼 수 없습니다."])
    (out/"report_ko.md").write_text("\n".join(lines))
