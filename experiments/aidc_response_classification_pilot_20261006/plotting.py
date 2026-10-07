"""Plot saved experiment tables; do not fit models or change sampling."""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def savefig(out, name, frame):
    frame.to_csv(out / "figures" / f"{name}.csv", index=False)
    plt.savefig(out / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(out / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close()


def plot_results(out, dates: pd.DataFrame, status: dict) -> None:
    ratio = dates[["collection_day", "R0", "R1", "R2", "R3", "U"]].set_index(
        "collection_day"
    )
    ratio = ratio.div(ratio.sum(axis=1), axis=0)
    ratio.plot.bar(stacked=True)
    plt.ylabel("Fraction of unique union anchors")
    plt.title(
        "Unresolved configuration audit"
        if not status["all_completed"]
        else "Oracle label distribution"
    )
    savefig(out, "date_class_ratio", ratio.reset_index())
    dates.set_index("collection_day")[["training_rows", "evaluation_rows"]].plot.bar()
    plt.ylabel("Anchors")
    savefig(out, "anchor_counts", dates)
    if status["all_completed"]:
        cl = pd.read_csv(out / "class_metrics.csv")
        cm = pd.read_csv(out / "confusion_plotting.csv")
        per = pd.read_csv(out / "per_run_metrics.csv")
        pred = pd.read_parquet(out / "predictions.parquet")
        for (scope, model), g in cm[cm.aggregation.eq("run_balanced")].groupby(
            ["scope", "model"]
        ):
            plt.imshow(
                g.pivot(
                    index="actual", columns="predicted", values="weight"
                ).to_numpy(),
                vmin=0,
            )
            plt.colorbar()
            plt.xticks(range(4), ["R0", "R1", "R2", "R3"])
            plt.yticks(range(4), ["R0", "R1", "R2", "R3"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            savefig(out, f"confusion_{scope}_{model}", g)
        for scope, g in cl[cl.aggregation.eq("run_balanced")].groupby("scope"):
            g.pivot(index="class", columns="model", values="recall").plot.bar()
            plt.ylim(0, 1)
            savefig(out, f"recall_{scope}", g)
        for scope, g in per.groupby("scope"):
            g.pivot(index="run_id", columns="model", values="ordinal_mae").plot.bar(
                figsize=(12, 5)
            )
            savefig(out, f"run_mae_{scope}", g)
        for (scope, model), g in pred.groupby(["scope", "model"]):
            z = (
                g.assign(actual=g.label.str[1:].astype(int))
                .groupby("actual")[[f"p{i}" for i in range(4)]]
                .mean()
                .reset_index()
            )
            z.set_index("actual").plot.bar()
            plt.ylim(0, 1)
            savefig(out, f"probability_{scope}_{model}", z)
        # Select chronological first run only, independent of performance or label.
        series = pd.read_parquet(out / "continuous_features.parquet")
        for scope, g in pred[pred.model.isin(["M1", "M2"])].groupby("scope"):
            run = sorted(g.run_id.unique())[0]
            g = g[g.run_id.eq(run)]
            start = g.issuance_s.min()
            g = g[g.issuance_s.le(start + 30)]
            b = series[
                series.run_id.eq(run) & series.issuance_s.between(start - 2, start + 31)
            ]
            fig, (ax, bx) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
            for _, x in b.groupby("continuous_block"):
                ax.plot(x.issuance_s, x.current_w)
            ax.set_ylabel("GPU sum W")
            actual = g.drop_duplicates("sample_key")
            # Draw each issued label for just one 200ms interval: never bridge sparse anchors.
            for name, x in [("Actual", actual)] + [
                (name, x) for name, x in g.groupby("model")
            ]:
                color = {"Actual": "black", "M1": "tab:blue", "M2": "tab:orange"}[name]
                y = (
                    x.label.str[1:].astype(int)
                    if name == "Actual"
                    else x.predicted_class
                )
                for t, value in zip(x.issuance_s, y):
                    bx.step([t, t + 0.2], [value, value], where="post", color=color)
                bx.plot([], [], color=color, label=name)
            for _, r in actual.iterrows():
                ax.axvspan(r.issuance_s, r.future_end_s, alpha=0.05, color="gray")
            bx.legend()
            bx.set_yticks(range(4), ["R0", "R1", "R2", "R3"])
            bx.set_xlabel("Issuance t (s); shaded regions show label future")
            savefig(out, f"timeline_{scope}", g)
            b.to_csv(
                out / "figures" / f"timeline_{scope}_continuous_power.csv", index=False
            )
