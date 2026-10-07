"""Validate meter exports without inventing absent facility measurements."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

UNITS = {'pcc_active_power_w': 'W', 'pcc_reactive_power_var': 'var',
         'voltage_v': 'V', 'frequency_hz': 'Hz', 'ess_soc': '1'}
REQUIRED = ['timestamp_utc', 'available_at_utc', 'meter_id', 'measurement_boundary',
            'quantity', 'value', 'unit', 'evidence_type', 'clock_offset_s',
            'clock_uncertainty_s', 'missing_reason']


def validate_meter_frame(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No measured rows supplied')
    result = frame.copy()
    if not result.evidence_type.isin(['measured_existing', 'measured_new']).all():
        raise ValueError('Only declared measured records may enter D5')
    for col in ['timestamp_utc', 'available_at_utc']:
        if result[col].isna().any():
            raise ValueError(f'{col} contains missing timestamps')
        parsed = result[col].map(pd.Timestamp)
        if any(t.tzinfo is None for t in parsed):
            raise ValueError('Explicit source timezone is required')
        result[col] = pd.to_datetime(result[col], utc=True)
    if (result.available_at_utc < result.timestamp_utc).any():
        raise ValueError('Availability precedes observation time')
    if not result.measurement_boundary.isin(['PCC', 'ESS']).all():
        raise ValueError('Facility channels require PCC or ESS measurement boundary')
    if not result.quantity.isin(UNITS).all():
        raise ValueError('Unsupported quantity; UPS mode requires a separate categorical log')
    expected_boundary = result.quantity.map(lambda q: 'ESS' if q == 'ess_soc' else 'PCC')
    if (result.measurement_boundary != expected_boundary).any():
        raise ValueError('Quantity and measurement boundary disagree')
    if not (result.unit == result.quantity.map(UNITS)).all():
        raise ValueError('Explicit canonical units required; implicit conversion is disabled')
    if result.meter_id.isna().any() or result.meter_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing meter identifier')
    if result.duplicated(['meter_id', 'timestamp_utc', 'quantity']).any():
        raise ValueError('Duplicate meter/time/quantity key')
    for col in ['value', 'clock_offset_s', 'clock_uncertainty_s']:
        result[col] = pd.to_numeric(result[col], errors='raise')
    if not np.isfinite(result[['clock_offset_s', 'clock_uncertainty_s']]).all().all():
        raise ValueError('Clock offset and uncertainty must be finite')
    if (result.clock_uncertainty_s < 0).any():
        raise ValueError('Clock uncertainty cannot be negative')
    if np.isinf(result.value).any():
        raise ValueError('Infinite measurement')
    absent = result.value.isna()
    if (absent & result.missing_reason.fillna('').astype(str).str.strip().eq('')).any():
        raise ValueError('Missing measurements require a reason')
    soc = result.loc[result.quantity == 'ess_soc', 'value'].dropna()
    if not soc.between(0, 1).all():
        raise ValueError('ESS SoC must be a fraction in [0, 1]')
    positive = result.quantity.isin(['voltage_v', 'frequency_hz']) & ~absent
    if (result.loc[positive, 'value'] <= 0).any():
        raise ValueError('Voltage and frequency must be positive')
    return result.sort_values(['meter_id', 'timestamp_utc', 'quantity']).reset_index(drop=True)


def import_meter_file(source: Path, out: Path):
    frame = pd.read_parquet(source) if source.suffix == '.parquet' else pd.read_csv(source)
    validated = validate_meter_frame(frame)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'facility_measurements.parquet'
    if target.exists():
        raise FileExistsError('Use a fresh output directory to preserve earlier measurements')
    validated['source_file'] = str(source.resolve())
    h = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    validated['source_sha256'] = h.hexdigest()
    validated.to_parquet(target, index=False)
    report = {'status': 'imported_schema_validated', 'rows': len(validated),
              'meters': int(validated.meter_id.nunique()),
              'source': str(source.resolve()), 'source_sha256': h.hexdigest(),
              'clock_offsets_applied': False, 'physical_provenance_verified': False,
              'limitations': ['Meter identity, calibration and provenance require external verification.',
                              'Clock metadata retained; no implicit synchronization or interpolation.',
                              'UPS categorical mode logs are not imported by this numeric channel importer.']}
    (out / 'facility_import_status.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
