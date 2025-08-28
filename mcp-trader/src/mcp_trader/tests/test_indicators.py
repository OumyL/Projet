import pandas as pd
import pandas_ta as ta
import pytest
from mcp_trader.indicators import macd, rsi  

@pytest.fixture
def sample_data():
    # Simule un DataFrame de prix sur 100 périodes
    data = {
        "close": pd.Series([100 + i + (i % 5) for i in range(100)]),
        "high": pd.Series([102 + i for i in range(100)]),
        "low": pd.Series([98 + i for i in range(100)])
    }
    return pd.DataFrame(data)

def test_rsi(sample_data):
    result = ta.rsi(sample_data["close"], length=14)
    assert isinstance(result, pd.Series)
    assert result.isna().sum() > 0  # Les premières valeurs sont NaN
    assert result.dropna().between(0, 100).all()

def test_macd(sample_data):
    macd_df = ta.macd(sample_data["close"])
    assert isinstance(macd_df, pd.DataFrame)
    assert "MACD_12_26_9" in macd_df.columns
    assert "MACDh_12_26_9" in macd_df.columns
    assert "MACDs_12_26_9" in macd_df.columns

def test_supertrend(sample_data):
    st = ta.supertrend(sample_data["high"], sample_data["low"], sample_data["close"])
    assert isinstance(st, pd.DataFrame)
    assert any("SUPERT" in col for col in st.columns)
    assert any("SUPERTd" in col for col in st.columns)

def test_sma(sample_data):
    sma = ta.sma(sample_data["close"], length=20)
    assert isinstance(sma, pd.Series)
    assert sma.name.startswith("SMA_")

def test_ema(sample_data):
    ema = ta.ema(sample_data["close"], length=20)
    assert isinstance(ema, pd.Series)
    assert ema.name.startswith("EMA_")
