import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering import add_rul_column, cap_rul

def test_rul_counts_down_correctly():
    """RUL should decrease by exactly 1 each cycle, and hit 0 at the engine's last cycle."""
    # Build a tiny fake engine: 5 cycles, unit_number 1
    fake_data = pd.DataFrame({
        'unit_number': [1, 1, 1, 1, 1],
        'time_cycles': [1, 2, 3, 4, 5],
    })

    result = add_rul_column(fake_data)

    # Last cycle should have RUL = 0 (engine just failed)
    assert result.iloc[-1]['RUL'] == 0
    # First cycle should have RUL = 4 (5 total cycles - cycle 1 = 4 remaining)
    assert result.iloc[0]['RUL'] == 4
    # RUL should decrease by exactly 1 each row
    assert (result['RUL'].diff().dropna() == -1).all()

def test_rul_handles_multiple_engines_independently():
    """Each engine's RUL should be calculated from ITS OWN max cycle, not mixed with others."""
    fake_data = pd.DataFrame({
        'unit_number': [1, 1, 2, 2, 2],
        'time_cycles': [1, 2, 1, 2, 3],
    })

    result = add_rul_column(fake_data)

    # Engine 1 lived 2 cycles total, engine 2 lived 3 cycles total - they shouldn't interfere
    engine_1_rul = result[result['unit_number'] == 1]['RUL'].tolist()
    engine_2_rul = result[result['unit_number'] == 2]['RUL'].tolist()

    assert engine_1_rul == [1, 0]
    assert engine_2_rul == [2, 1, 0]

def test_cap_rul_caps_at_max_value():
    """Values above the cap should be clipped; values below should be untouched."""
    fake_data = pd.DataFrame({'RUL': [200, 100, 50, 10, 0]})

    result = cap_rul(fake_data, cap=125)

    assert result['RUL'].tolist() == [125, 100, 50, 10, 0]