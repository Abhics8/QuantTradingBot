import numpy as np
from quantbot.features.pipeline import build_feature_matrix, get_feature_columns
from quantbot.models.xgboost_signal import XGBoostSignalModel


def test_xgboost_fit_predict(ohlcv):
    fm = build_feature_matrix(ohlcv)
    cols = get_feature_columns()
    X, y = fm[cols], fm["target"]

    model = XGBoostSignalModel(n_estimators=10, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(int(p) for p in np.unique(preds)).issubset({0, 1})
    assert len(model.feature_importance()) == len(cols)


def test_lstm_fit_predict(ohlcv):
    """Also guards the de-duplicated _create_sequences helper used by fit()."""
    from quantbot.models.lstm_signal import LSTMSignalModel

    fm = build_feature_matrix(ohlcv)
    cols = get_feature_columns()
    X, y = fm[cols], fm["target"]

    model = LSTMSignalModel(seq_length=5, hidden_size=8, epochs=1, batch_size=16)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)  # padded back to the original length
    assert set(float(p) for p in np.unique(preds)).issubset({0.0, 1.0})
