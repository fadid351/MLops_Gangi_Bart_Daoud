import pytest

from src.config import COLUMNS
from src.data.load_data import load_raw, load_train_test, split_X_y


def test_load_train_has_expected_columns(raw_dir):
    df = load_raw("train", raw_dir)
    assert list(df.columns) == COLUMNS
    assert len(df) == 600


def test_test_labels_are_cleaned(raw_dir):
    """Le fichier de test écrit '>50K.' : le point final doit être retiré."""
    df = load_raw("test", raw_dir)
    assert set(df["income"].unique()) <= {">50K", "<=50K"}
    assert len(df) == 300  # la ligne de commentaire a bien été sautée


def test_question_marks_become_nan(raw_dir):
    train, _ = load_train_test(raw_dir)
    assert train["workclass"].isna().any()
    assert not (train["workclass"] == "?").any()


def test_split_X_y_is_binary(raw_dir):
    X, y = split_X_y(load_raw("train", raw_dir))
    assert "income" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw("train", tmp_path)


def test_bad_split_name(raw_dir):
    with pytest.raises(ValueError):
        load_raw("validation", raw_dir)
