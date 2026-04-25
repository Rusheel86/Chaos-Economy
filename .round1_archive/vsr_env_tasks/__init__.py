"""Task implementations for VSR-Env.

This module provides the three graded tasks:
- DeltaHedgingTask (medium): Neutralize portfolio delta through market shock
- EarningsVolCrushTask (hard): Position for and recover from earnings vol crush
- GammaScalpingTask (expert): Profit from gamma scalping through spot oscillations
"""

from vsr_env.tasks.delta_hedging import DeltaHedgingTask, DeltaHedgingGrader
from vsr_env.tasks.earnings_vol_crush import EarningsVolCrushTask, EarningsVolCrushGrader
from vsr_env.tasks.gamma_scalping import GammaScalpingTask, GammaScalpingGrader
from vsr_env.tasks.vol_regime_detection import VolRegimeDetectionTask, VolRegimeDetectionGrader
from vsr_env.tasks.vega_gamma_stress import VegaGammaStressTask, VegaGammaStressGrader
from vsr_env.tasks.vertical_spread import VerticalSpreadTask, VerticalSpreadGrader
from vsr_env.tasks.straddle_trading import StraddleTradingTask, StraddleTradingGrader

__all__ = [
    "DeltaHedgingTask",
    "EarningsVolCrushTask",
    "GammaScalpingTask",
    "VolRegimeDetectionTask",
    "VegaGammaStressTask",
    "VerticalSpreadTask",
    "StraddleTradingTask",
]
