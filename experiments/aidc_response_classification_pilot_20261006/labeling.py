"""Hard feasibility of a fixed storage-only PCC proxy; no controller-failure labels."""
import json
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

REQUIRED = [
    "assessment.proxy_mapping",
    "requirements.ramp_up_w_s",
    "requirements.ramp_down_w_s",
    "requirements.source",
    "requirements.source_location",
    "requirements.recovery_s",
    "requirements.recovery_ramp_enforced",
    "requirements.envelope_required",
    "requirements.tolerance_w",
    "requirements.tolerance_j",
    "requirements.tolerance_fraction",
    "resources.U1",
    "resources.U2_additional",
    "resources.expansion_cost_or_workload_constraints",
    "scenario.grid_conditions",
    "scenario.initial_pcc_rule",
]
DEVICE_NUMERIC = [
    "charge_w",
    "discharge_w",
    "capacity_j",
    "initial_soc",
    "soc_min",
    "soc_max",
    "eta_charge",
    "eta_discharge",
    "delay_bins",
    "slew_w_s",
    "throughput_budget_j",
    "terminal_soc_tolerance",
]


def get(spec, future_power_w):
    obj = spec
    for key in future_power_w.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def validate_spec(spec: dict) -> dict:
    missing = [x for x in REQUIRED if get(spec, x) is None]
    errors = []
    req = spec["requirements"]
    if req.get("envelope_required"):
        missing += [
            f"requirements.{k}"
            for k in ["pcc_min_w", "pcc_max_w"]
            if req.get(k) is None
        ]
    for level in ["U1", "U2_additional"]:
        devices = spec["resources"].get(level)
        if devices is None:
            continue
        if not isinstance(devices, list):
            errors.append(f"{level}: expected device list")
            continue
        if not devices:
            errors.append(f"{level}: requires explicit nonempty resource set")
        for i, d in enumerate(devices):
            for k in (
                ["id"] + DEVICE_NUMERIC + ["source", "source_location", "category"]
            ):
                if d.get(k) is None:
                    missing.append(f"resources.{level}[{i}].{k}")
            if any(d.get(k) is None for k in DEVICE_NUMERIC):
                continue
            if not all(
                isinstance(d[k], (float, int)) and np.isfinite(d[k])
                for k in DEVICE_NUMERIC
            ):
                errors.append(f"{level}[{i}]: finite numeric values required")
                continue
            if not 0 <= d["soc_min"] <= d["initial_soc"] <= d["soc_max"] <= 1:
                errors.append(f"{level}[{i}]: SOC bounds")
            if not 0 < d["eta_charge"] <= 1 or not 0 < d["eta_discharge"] <= 1:
                errors.append(f"{level}[{i}]: efficiency")
            if d["capacity_j"] <= 0 or any(
                d[k] < 0
                for k in [
                    "charge_w",
                    "discharge_w",
                    "delay_bins",
                    "slew_w_s",
                    "throughput_budget_j",
                    "terminal_soc_tolerance",
                ]
            ):
                errors.append(f"{level}[{i}]: resource bounds")
            if int(d["delay_bins"]) != d["delay_bins"]:
                errors.append(
                    "delay_bins must be integer; sub-bin response unsupported"
                )
            if (
                d.get("initial_net_power_w", 0) != 0
                or d.get("terminal_net_power_w", 0) != 0
            ):
                errors.append("only zero initial/terminal device net power implemented")
    if req.get("category") == "미확정":
        missing.append("requirements.category")
    if not missing:
        if get(spec, "assessment.proxy_mapping") != "identity_GPU_sum_W":
            errors.append(
                "only explicitly adopted identity GPU-sum proxy implemented; facility scaling requires separate implementation"
            )
        if get(spec, "scenario.initial_pcc_rule") != "current_completed_bin_W":
            errors.append("unsupported initial PCC rule")
        if spec["scenario"]["dt_s"] != 0.2 or spec["scenario"]["future_bins"] != 5:
            errors.append("expected five 200ms future bins")
        if req["kind"] != "discrete_PCC_proxy_ramp":
            errors.append("unsupported requirement kind")
        if req["evaluation_window_s"] != 1:
            errors.append("expected 1s evaluation window")
        for k in [
            "ramp_up_w_s",
            "ramp_down_w_s",
            "recovery_s",
            "tolerance_w",
            "tolerance_j",
            "tolerance_fraction",
        ]:
            if (
                not isinstance(req[k], (int, float))
                or not np.isfinite(req[k])
                or req[k] < 0
            ):
                errors.append(f"invalid {k}")
        if not np.isclose(req["recovery_s"] / 0.2, round(req["recovery_s"] / 0.2)):
            errors.append("recovery must use whole observed bins")
        if req["envelope_required"] and req["pcc_min_w"] > req["pcc_max_w"]:
            errors.append("inverted PCC envelope")
        devices = spec["resources"]["U1"] + spec["resources"]["U2_additional"]
        if len({d["id"] for d in devices}) != len(devices):
            errors.append("device IDs must be unique")
    return {
        "ready": not missing and not errors,
        "missing_required_fields": missing,
        "errors": errors,
        "nesting": "U0 zero action; U2 is U1 plus optional additional devices with same terminal restoration; zero action admissible",
    }


def physical_check(
    future_power_w: list | np.ndarray,
    base_power_w: float,
    devices: list[dict],
    charge: np.ndarray,
    discharge: np.ndarray,
    spec: dict,
) -> dict:
    """Recheck PCC W, energy J and SoC constraints independently of solver status."""
    interval_s = spec["scenario"]["dt_s"]
    req = spec["requirements"]
    n = len(future_power_w)
    pcc_power_w = (
        np.array(future_power_w, dtype=float)
        + charge.sum(axis=0)
        - discharge.sum(axis=0)
    )
    ramp_step_w = np.diff(np.r_[base_power_w, pcc_power_w])
    nr = n if req["recovery_ramp_enforced"] else spec["scenario"]["future_bins"]
    max_violation_w = max(
        0.0,
        float(np.max(ramp_step_w[:nr] - req["ramp_up_w_s"] * interval_s)),
        float(np.max(-ramp_step_w[:nr] - req["ramp_down_w_s"] * interval_s)),
    )
    max_violation_j = 0.0
    max_violation_fraction = 0.0
    if req["envelope_required"]:
        max_violation_w = max(
            max_violation_w,
            float(np.max(req["pcc_min_w"] - pcc_power_w)),
            float(np.max(pcc_power_w - req["pcc_max_w"])),
        )
    for i, d in enumerate(devices):
        charge_w = charge[i]
        discharge_w = discharge[i]
        net_power_w = discharge_w - charge_w
        energy_j = d["initial_soc"] * d["capacity_j"] + np.cumsum(
            interval_s * (d["eta_charge"] * charge_w - discharge_w / d["eta_discharge"])
        )
        max_violation_w = max(
            max_violation_w,
            float(np.max(charge_w - d["charge_w"])),
            float(np.max(discharge_w - d["discharge_w"])),
            float(np.max(-charge_w)),
            float(np.max(-discharge_w)),
            float(np.max(np.minimum(charge_w, discharge_w))),
            float(
                np.max(abs(np.diff(np.r_[0.0, net_power_w])))
                - d["slew_w_s"] * interval_s
            ),
            abs(float(net_power_w[-1])),
        )
        if d["delay_bins"]:
            max_violation_w = max(
                max_violation_w, float(np.max(abs(net_power_w[: int(d["delay_bins"])])))
            )
        max_violation_j = max(
            max_violation_j,
            float(np.max(d["soc_min"] * d["capacity_j"] - energy_j)),
            float(np.max(energy_j - d["soc_max"] * d["capacity_j"])),
            float(
                interval_s * np.sum(charge_w + discharge_w) - d["throughput_budget_j"]
            ),
        )
        max_violation_fraction = max(
            max_violation_fraction,
            abs(float(energy_j[-1] / d["capacity_j"] - d["initial_soc"]))
            - d["terminal_soc_tolerance"],
        )
    # A documented 32-ULP arithmetic guard avoids rejecting the same inclusive bound
    # solely because subtracting large powers changes its last floating digits.
    guard_w = 32 * float(
        np.spacing(max(1.0, abs(base_power_w), float(np.max(abs(pcc_power_w)))))
    )
    guard_j = 32 * float(np.spacing(max([1.0] + [d["capacity_j"] for d in devices])))
    guard_fraction = 32 * float(np.spacing(1.0))
    return {
        "max_violation_w": max(0.0, max_violation_w),
        "max_violation_j": max(0.0, max_violation_j),
        "max_violation_fraction": max(0.0, max_violation_fraction),
        "arithmetic_guard_w": guard_w,
        "arithmetic_guard_j": guard_j,
        "arithmetic_guard_fraction": guard_fraction,
        "feasible": max_violation_w <= req["tolerance_w"] + guard_w
        and max_violation_j <= req["tolerance_j"] + guard_j
        and max_violation_fraction <= req["tolerance_fraction"] + guard_fraction,
    }


def feasibility(
    future_power_w: list | np.ndarray,
    base_power_w: float,
    devices: list[dict],
    spec: dict,
) -> dict:
    """Return feasibility evidence; current/path are W and intervals are seconds."""
    interval_s = spec["scenario"]["dt_s"]
    power_w = np.array(future_power_w, dtype=float)
    n = len(power_w)
    m = len(devices)
    if not np.isfinite(power_w).all() or not np.isfinite(base_power_w):
        return {"F": None, "status": "nonfinite_input"}
    if not m:
        checked = physical_check(
            power_w, base_power_w, [], np.zeros((0, n)), np.zeros((0, n)), spec
        )
        return {"F": checked["feasible"], "status": "direct_U0_check", **checked}
    # Device/time variables: charge, discharge, binary mode. Energy is cumulative.
    nv = 3 * m * n
    lower = np.zeros(nv)
    upper = np.zeros(nv)
    integrality = np.zeros(nv, dtype=int)
    rows = []
    lo = []
    hi = []

    def add(row, l=-np.inf, h=np.inf):
        rows.append(row)
        lo.append(l)
        hi.append(h)

    def index(i, kind, t):
        return i * 3 * n + kind * n + t

    def row():
        return np.zeros(nv)

    for i, d in enumerate(devices):
        for t in range(n):
            charge_w = index(i, 0, t)
            discharge_w = index(i, 1, t)
            b = index(i, 2, t)
            upper[charge_w] = d["charge_w"] if t >= d["delay_bins"] else 0
            upper[discharge_w] = d["discharge_w"] if t >= d["delay_bins"] else 0
            upper[b] = 1
            integrality[b] = 1
            a = row()
            a[charge_w] = 1
            a[b] = -d["charge_w"]
            add(a, h=0)
            a = row()
            a[discharge_w] = 1
            a[b] = d["discharge_w"]
            add(a, h=d["discharge_w"])
            a = row()
            for k in range(t + 1):
                a[index(i, 0, k)] = interval_s * d["eta_charge"]
                a[index(i, 1, k)] = -interval_s / d["eta_discharge"]
            add(
                a,
                (d["soc_min"] - d["initial_soc"]) * d["capacity_j"],
                (d["soc_max"] - d["initial_soc"]) * d["capacity_j"],
            )
            if t == n - 1:
                add(
                    a,
                    -d["terminal_soc_tolerance"] * d["capacity_j"],
                    d["terminal_soc_tolerance"] * d["capacity_j"],
                )
            a = row()
            a[discharge_w] = 1
            a[charge_w] = -1
            if t > 0:
                a[index(i, 1, t - 1)] = -1
                a[index(i, 0, t - 1)] = 1
            add(a, -d["slew_w_s"] * interval_s, d["slew_w_s"] * interval_s)
        a = row()
        for t in range(n):
            a[index(i, 0, t)] = interval_s
            a[index(i, 1, t)] = interval_s
        add(a, h=d["throughput_budget_j"])
        a = row()
        a[index(i, 1, n - 1)] = 1
        a[index(i, 0, n - 1)] = -1
        add(a, 0, 0)
    req = spec["requirements"]
    nr = n if req["recovery_ramp_enforced"] else spec["scenario"]["future_bins"]
    for t in range(n):
        a = row()
        for i in range(m):
            a[index(i, 0, t)] = 1
            a[index(i, 1, t)] = -1
        if req["envelope_required"]:
            add(
                a,
                req["pcc_min_w"] - power_w[t] - req["tolerance_w"],
                req["pcc_max_w"] - power_w[t] + req["tolerance_w"],
            )
        if t < nr:
            if t > 0:
                for i in range(m):
                    a[index(i, 0, t - 1)] = -1
                    a[index(i, 1, t - 1)] = 1
            delta = power_w[t] - (power_w[t - 1] if t else base_power_w)
            add(
                a,
                -req["ramp_down_w_s"] * interval_s - delta - req["tolerance_w"],
                req["ramp_up_w_s"] * interval_s - delta + req["tolerance_w"],
            )
    try:
        result = milp(
            np.zeros(nv),
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(csr_matrix(np.array(rows)), lo, hi),
            options={
                "time_limit": spec["solver"]["time_limit_s"],
                "mip_rel_gap": spec["solver"]["mip_rel_gap"],
            },
        )
    except Exception as exc:
        return {"F": None, "status": "solver_exception", "message": str(exc)}
    status = {"solver_status": int(result.status), "message": str(result.message)}
    if result.status == 2:
        return {"F": False, "status": "certified_infeasible", **status}
    if result.status != 0 or result.x is None:
        return {"F": None, "status": "solver_failure_or_timeout", **status}
    charge = np.array([[result.x[index(i, 0, t)] for t in range(n)] for i in range(m)])
    discharge = np.array(
        [[result.x[index(i, 1, t)] for t in range(n)] for i in range(m)]
    )
    check = physical_check(power_w, base_power_w, devices, charge, discharge, spec)
    return {
        "F": True if check["feasible"] else None,
        "status": "feasible_rechecked"
        if check["feasible"]
        else "physical_recheck_failed",
        **status,
        **check,
        "charge_w": charge.tolist(),
        "discharge_w": discharge.tolist(),
    }


def calculate_base_power_w(current_power_w: float) -> float:
    """P_base is the latest completed bin in W; no future or facility scaling."""
    if not np.isfinite(current_power_w):
        raise ValueError("stage=labeling: P_base must be finite W")
    return float(current_power_w)


def judge(
    future_power_w: list | np.ndarray, base_power_w: float, spec: dict, config_hash: str
) -> dict:
    """Assign oracle R0..R3 or U using observed future W and nested resources."""
    ready = validate_spec(spec)
    if not ready["ready"]:
        return {
            "label": "U",
            "F0": None,
            "F1": None,
            "F2": None,
            "status": "missing_configuration",
            "unknown_reason": ";".join(
                ready["missing_required_fields"] + ready["errors"]
            ),
            "scenario_id": spec["scenario_id"],
            "config_hash": config_hash,
            "evidence_json": "[]",
        }
    if len(future_power_w) != spec["scenario"]["future_bins"] + round(
        spec["requirements"]["recovery_s"] / spec["scenario"]["dt_s"]
    ):
        return {
            "label": "U",
            "F0": None,
            "F1": None,
            "F2": None,
            "status": "missing_future",
            "unknown_reason": "full_observed_future_required",
            "scenario_id": spec["scenario_id"],
            "config_hash": config_hash,
            "evidence_json": "[]",
        }
    if not np.isfinite(base_power_w) or not np.isfinite(future_power_w).all():
        return {
            "label": "U",
            "F0": None,
            "F1": None,
            "F2": None,
            "status": "data_error",
            "unknown_reason": "nonfinite_power_w",
            "scenario_id": spec["scenario_id"],
            "config_hash": config_hash,
            "evidence_json": "[]",
        }
    base_power_w = calculate_base_power_w(base_power_w)
    u1 = spec["resources"]["U1"]
    u2 = u1 + spec["resources"]["U2_additional"]
    results = [feasibility(future_power_w, base_power_w, d, spec) for d in [[], u1, u2]]
    f = [r["F"] for r in results]
    contradiction = (
        (f[0] is True and f[1] is False)
        or (f[1] is True and f[2] is False)
        or (f[0] is True and f[2] is False)
    )
    label = "U"
    if not contradiction:
        if f[0] is True:
            label = "R0"
        elif f[0] is False and f[1] is True:
            label = "R1"
        elif f[0] is False and f[1] is False and f[2] is True:
            label = "R2"
        elif all(x is False for x in f):
            label = "R3"
    return {
        "label": label,
        **dict(zip(["F0", "F1", "F2"], f)),
        "status": "nesting_contradiction"
        if contradiction
        else "determined"
        if label != "U"
        else "undetermined",
        "unknown_reason": ""
        if label != "U"
        else ";".join(r["status"] for r in results),
        "scenario_id": spec["scenario_id"],
        "config_hash": config_hash,
        "evidence_json": json.dumps(results, ensure_ascii=False),
    }
