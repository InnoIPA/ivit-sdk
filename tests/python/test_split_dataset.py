"""split_dataset honours both ratios and leaves the global random state alone."""

import random

import pytest

dataset_module = pytest.importorskip("ivit.train.dataset")
split_dataset = dataset_module.split_dataset


class Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def legacy_split(n, train_ratio, seed):
    """The previous implementation, which ignored val_ratio."""
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    n_train = int(n * train_ratio)
    return indices[:n_train], indices[n_train:]


@pytest.mark.parametrize("n", [1, 7, 10, 101])
def test_default_split_is_unchanged_and_loses_no_sample(n):
    state = random.getstate()
    train, val = split_dataset(Sized(n))
    expected = legacy_split(n, 0.8, 42)
    random.setstate(state)
    assert (train, val) == expected
    assert sorted(train + val) == list(range(n))


def test_val_ratio_is_honoured_when_the_ratios_do_not_cover_everything():
    train, val = split_dataset(Sized(10), train_ratio=0.6, val_ratio=0.1)
    assert (len(train), len(val)) == (6, 1)
    assert not set(train) & set(val)


def test_the_global_random_state_is_untouched():
    random.seed(123)
    before = random.random()
    random.seed(123)
    split_dataset(Sized(50), seed=7)
    assert random.random() == before


def test_same_seed_same_split_and_different_seed_different_split():
    assert split_dataset(Sized(30), seed=1) == split_dataset(Sized(30), seed=1)
    assert split_dataset(Sized(30), seed=1) != split_dataset(Sized(30), seed=2)


@pytest.mark.parametrize("train, val", [(0, 0.2), (1.2, 0), (0.8, -0.1), (0.8, 0.3)])
def test_invalid_ratios_are_rejected(train, val):
    with pytest.raises(ValueError, match="split ratios"):
        split_dataset(Sized(10), train_ratio=train, val_ratio=val)
