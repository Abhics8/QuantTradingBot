from quantbot.features.pipeline import build_feature_matrix, get_feature_columns


def test_target_has_no_lookahead(ohlcv):
    """The label must be strictly the NEXT day's direction — never same-day or past."""
    fm = build_feature_matrix(ohlcv, include_target=True)
    close = ohlcv["Close"]
    for d in fm.index[:25]:
        pos = close.index.get_loc(d)
        expected = int(close.iloc[pos + 1] > close.iloc[pos])
        assert int(fm.loc[d, "target"]) == expected


def test_features_present_and_clean(ohlcv):
    fm = build_feature_matrix(ohlcv)
    cols = get_feature_columns()
    for c in cols:
        assert c in fm.columns
    assert not fm[cols].isnull().any().any()  # warmup rows dropped, no NaNs leak into features
