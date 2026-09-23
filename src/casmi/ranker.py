"""A learned ranker that combines the three evidence channels per molecule.

WHY
    The fixed blend `fpnet + 0.5*analog + 0.25*frag` weighs the channels the
    same way for every molecule. But how much each channel deserves to be
    trusted changes from molecule to molecule:

      - a near-perfect analog match (class 1) should dominate everything
      - a weak best analog means the analog channel is mostly noise here,
        so fpnet and fragments should carry the decision
      - a sparse spectrum explains itself trivially, so fragments mean less

    A small neural network that sees both the channel scores AND this context
    can learn those "trust X when Y" rules from data instead of us guessing
    one weight that has to serve every case.

HOW IT IS TRAINED (scripts/train_ranker.py)
    Listwise: for each molecule, softmax over its candidates' scores and
    maximise the probability of the true one. That directly rewards putting
    the right answer FIRST, which is what MRR mostly measures.

    Trained only on molecules the fingerprint model never saw (docs/09), or it
    would learn to over-trust a channel that was memorising.

THE ONE RULE
    `ranker_features` is the ONLY place features are computed. Training and
    the Kaggle notebook both call it. This project's worst bugs were all the
    same bug: the same quantity computed two ways in two places.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Order matters: it is the column order of the feature matrix, and a saved
# model is only valid for the exact order it was trained with. It is stored in
# the checkpoint and checked on load.
FEATURES = [
    "z_fp", "z_an", "z_fr",        # the three channels, per-molecule z-scores
    "rk_fp", "rk_an", "rk_fr",     # the candidate's rank in each channel, 0 = top
    "an_raw", "fr_raw",            # absolute strength, which z-scoring discards
    "fp_gap",                      # fpnet log-likelihood minus the best one
    "an_conf",                     # how good the best spectral relative is
    "log_ncand",                   # how crowded the candidate list is
    "log_npk",                     # how rich the richest spectrum is
]


def _z(x: np.ndarray) -> np.ndarray:
    """Per-molecule z-score; all-equal inputs carry no ranking information."""
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Position of each candidate when sorted best-first, scaled to [0, 1).

    kind="stable" so exact ties always break the same way. The default sort
    is not stable, which would let the same inputs produce different features
    on different runs.
    """
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / len(x)


def ranker_features(s_fpnet, s_analog, s_frag, an_conf: float,
                    n_peaks: int) -> np.ndarray:
    """(n_candidates, len(FEATURES)) float32 matrix for ONE molecule.

    Every input is available at test time. In particular nothing here may use
    the true answer -- the validator computes one such quantity (Tanimoto of
    the TRUE structure to the analogs) for reporting, and passing it in would
    hand the ranker the label.
    """
    fp = np.asarray(s_fpnet, dtype=np.float64)
    an = np.asarray(s_analog, dtype=np.float64)
    fr = np.asarray(s_frag, dtype=np.float64)
    n = len(fp)
    full = lambda v: np.full(n, v, dtype=np.float64)   # a per-molecule constant
    cols = [
        _z(fp), _z(an), _z(fr),
        _rank01(fp), _rank01(an), _rank01(fr),
        an, fr,
        fp - fp.max(),
        full(an_conf),
        full(np.log(n)),
        full(np.log(max(n_peaks, 1))),
    ]
    assert len(cols) == len(FEATURES), "FEATURES and ranker_features disagree"
    return np.stack(cols, axis=1).astype(np.float32)


class RankNet(nn.Module):
    """Scores one candidate from its feature row. hidden=0 gives a linear model.

    Deliberately tiny (~1.5k parameters at hidden=32). There are only ~2k
    training molecules; a large network would memorise them.
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.cfg = dict(n_features=n_features, hidden=hidden)
        if hidden == 0:
            self.net = nn.Linear(n_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def expected_rr(scores, truth: int, k: int = 25) -> float:
    """Reciprocal rank of candidate `truth`, averaged over random tie-breaking.

    "1 + number of candidates scored strictly higher" puts the truth FIRST
    among everything it ties with. Ties are common (many candidates score
    exactly 0 on analog similarity), so that silently favours tie-heavy methods
    such as the fixed blend over continuous ones such as a learned ranker. On
    the leaderboard a tie's order is arbitrary, so average over every position
    the truth could land in among its ties.
    """
    s = np.asarray(scores, dtype=np.float64)
    higher = int((s > s[truth]).sum())
    ties = int((s == s[truth]).sum()) - 1        # other candidates tied with it
    ranks = np.arange(higher + 1, higher + ties + 2)
    return float(np.where(ranks <= k, 1.0 / ranks, 0.0).mean())


def load_ranker(path, device="cpu"):
    """Load a trained ranker plus the standardisation it was trained with."""
    ck = torch.load(path, map_location=device, weights_only=False)
    # A model trained on a different feature list would silently read the
    # wrong columns. Refuse instead.
    assert list(ck["features"]) == FEATURES, (
        f"ranker was trained on {list(ck['features'])}, code has {FEATURES}")
    net = RankNet(**ck["cfg"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def score_with_ranker(net, ck, feats: np.ndarray) -> np.ndarray:
    """Higher = better. Applies the SAME standardisation used in training."""
    x = (feats - ck["mean"]) / ck["std"]
    with torch.no_grad():
        return net(torch.as_tensor(x, dtype=torch.float32)).numpy()


def _self_check() -> None:
    rng = np.random.default_rng(0)
    fp, an, fr = rng.normal(size=20), rng.random(20), rng.random(20)
    f = ranker_features(fp, an, fr, an_conf=0.7, n_peaks=50)
    assert f.shape == (20, len(FEATURES))
    assert np.isfinite(f).all()

    # Shifting the fpnet scores by a constant must not change anything: the
    # log-likelihood's absolute level is arbitrary, only differences carry
    # information. This is also what lets training rebuild these features
    # from the dumped fp_gap column.
    g = ranker_features(fp + 123.0, an, fr, an_conf=0.7, n_peaks=50)
    assert np.allclose(f, g, atol=1e-5), "features depend on fpnet's offset"

    # Rank 0 must be the best candidate in each channel.
    assert f[np.argmax(fp), FEATURES.index("rk_fp")] == 0.0

    # A single candidate must not produce NaN (std = 0).
    one = ranker_features([1.0], [0.5], [0.2], an_conf=0.1, n_peaks=3)
    assert np.isfinite(one).all()

    print("ranker self-check passed")


if __name__ == "__main__":
    _self_check()
