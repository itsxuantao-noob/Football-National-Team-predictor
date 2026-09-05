"""International football match outcome predictor.

    from wcpredictor import Predictor
    Predictor.load().predict("Brazil", "France", neutral=True)

Training lives in `wcpredictor.train` and is not re-exported here, so that the
submodule name stays reachable as `wcpredictor.train`.
"""

from .config import CLASSES, FEATURE_COLUMNS, EloConfig
from .predict import MatchPrediction, Predictor, UnknownTeamError

__version__ = "1.0.0"

__all__ = [
    "CLASSES",
    "EloConfig",
    "FEATURE_COLUMNS",
    "MatchPrediction",
    "Predictor",
    "UnknownTeamError",
    "__version__",
]
