import numpy as np
from src.tools.genre_vocab import genre_significant_labels, select_genre_vocab


def test_genre_significant_labels_threshold():
    labels = ["A", "B", "C", "D"]
    mean400 = np.array([0.5, 0.09, 0.30, 0.11], dtype=np.float32)
    assert genre_significant_labels(mean400, labels, prob_thresh=0.10) == ["A", "C", "D"]


def test_select_genre_vocab_min_support_and_ordering():
    counts = {"Goa": 900, "Psy": 900, "Rare": 302, "Trance": 305, "Ambient": 303}
    # >=303 kept; sorted by count desc then label asc (Goa/Psy tie -> alpha)
    assert select_genre_vocab(counts, min_support=303) == ["Goa", "Psy", "Trance", "Ambient"]
