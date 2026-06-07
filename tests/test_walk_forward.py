import pandas as pd
from quantbot.models.walk_forward import WalkForwardValidator


def test_split_is_expanding_and_nonoverlapping():
    df = pd.DataFrame({"x": range(1000)})
    wf = WalkForwardValidator(train_period_days=252, test_period_days=63, min_train_size=252)
    splits = wf.split(df)

    assert len(splits) >= 2
    for train, test in splits:
        assert len(train) >= 252                       # respects min train size
        assert train.index[0] == 0                     # anchored: always starts at the beginning
        assert test.index[0] == train.index[-1] + 1    # test immediately follows train (no overlap, no gap)

    assert len(splits[1][0]) > len(splits[0][0])        # training window grows each fold


def test_small_dataset_falls_back_to_single_split():
    df = pd.DataFrame({"x": range(50)})
    splits = WalkForwardValidator().split(df)
    assert len(splits) == 1
