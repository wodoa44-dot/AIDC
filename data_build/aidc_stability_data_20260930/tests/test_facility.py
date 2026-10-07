import pandas as pd
import pytest
from aidc_data.facility import validate_meter_frame


def fixture():
    return pd.DataFrame([{'timestamp_utc': '2026-09-29T12:00:00+00:00',
                          'available_at_utc': '2026-09-29T12:00:00.1+00:00',
                          'meter_id': 'test-only-meter', 'measurement_boundary': 'PCC',
                          'quantity': 'frequency_hz', 'value': 60., 'unit': 'Hz',
                          'evidence_type': 'measured_new', 'clock_offset_s': 0.,
                          'clock_uncertainty_s': .01, 'missing_reason': None}])


def test_valid_schema_and_no_interpolation():
    data = fixture()
    result = validate_meter_frame(data)
    assert len(result) == 1 and result.value.iloc[0] == 60.


@pytest.mark.parametrize('column,value', [('evidence_type', 'simulated'),
                                         ('measurement_boundary', 'GPU'),
                                         ('unit', 'kHz'), ('clock_uncertainty_s', -1.),
                                         ('timestamp_utc', '2026-09-29T12:00:00')])
def test_wrong_evidence_boundary_units_clock_rejected(column, value):
    data = fixture()
    data[column] = value
    with pytest.raises(ValueError):
        validate_meter_frame(data)


def test_missing_value_reason_and_duplicates():
    data = fixture()
    data['value'] = float('nan')
    with pytest.raises(ValueError):
        validate_meter_frame(data)
    data['missing_reason'] = 'sensor unavailable'
    assert validate_meter_frame(data).value.isna().all()
    with pytest.raises(ValueError):
        validate_meter_frame(pd.concat([data, data]))
