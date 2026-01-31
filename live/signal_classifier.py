# live/signal_classifier.py
"""
Modulo per classificare i segnali del Radar in classi:
- A_PLUS
- A
- B (per futuro, ora disabilitata)
- REJECT (nessun trade)

Usa:
- strength = |p_blend - 0.5|
- soglie da config_v3
- qualche flag di contesto (mtf_ok, context_ok, news_ok).
"""

from dataclasses import dataclass
from typing import Literal

from live.config_v3 import (
    STRENGTH_MIN,
    STRENGTH_B,
    STRENGTH_A,
    STRENGTH_A_PLUS,
    ENABLE_CLASS_B,
)

SignalClass = Literal["A_PLUS", "A", "B", "REJECT"]


@dataclass
class SignalClassification:
    signal_class: SignalClass
    strength: float         # |p_blend - 0.5|
    allowed: bool           # True se può proseguire verso il risk engine
    reason: str             # spiegazione umana
    size_multiplier_hint: float  # suggerimento di moltiplicatore size (0–1.5)


def classify_signal(
    p_blend: float,
    mtf_ok: bool = True,
    context_ok: bool = True,
    news_ok: bool = True,
) -> SignalClassification:
    """
    Classifica il segnale in base a:
    - p_blend (probabilità blended del radar),
    - strength = |p_blend - 0.5|,
    - MTF / contesto / news (flag booleani).

    Per ora:
    - Se uno dei flag è False -> REJECT diretto.
    - Usiamo SOLO classi A e A_PLUS (B è per il futuro).
    """
    strength = abs(p_blend - 0.5)

    # 1) Filtri "hard" di contesto
    if not mtf_ok:
        return SignalClassification(
            signal_class="REJECT",
            strength=strength,
            allowed=False,
            reason="MTF non allineato",
            size_multiplier_hint=0.0,
        )

    if not context_ok:
        return SignalClassification(
            signal_class="REJECT",
            strength=strength,
            allowed=False,
            reason="Contesto sfavorevole (trend/choppy/volatilità)",
            size_multiplier_hint=0.0,
        )

    if not news_ok:
        return SignalClassification(
            signal_class="REJECT",
            strength=strength,
            allowed=False,
            reason="News rosse imminenti / rischio macro elevato",
            size_multiplier_hint=0.0,
        )

    # 2) Sotto soglia minima di strength → rumore
    if strength < STRENGTH_MIN:
        return SignalClassification(
            signal_class="REJECT",
            strength=strength,
            allowed=False,
            reason=f"Strength {strength:.4f} < STRENGTH_MIN {STRENGTH_MIN:.4f}",
            size_multiplier_hint=0.0,
        )

    # 3) Classe A_PLUS
    if strength >= STRENGTH_A_PLUS:
        return SignalClassification(
            signal_class="A_PLUS",
            strength=strength,
            allowed=True,
            reason="Segnale élite (A+)",
            size_multiplier_hint=1.3,   # suggerimento, poi il risk engine applica i suoi cap
        )

    # 4) Classe A
    if strength >= STRENGTH_A:
        return SignalClassification(
            signal_class="A",
            strength=strength,
            allowed=True,
            reason="Segnale di alta qualità (A)",
            size_multiplier_hint=1.0,
        )

    # 5) Classe B (solo se abilitate in config; altrimenti reject)
    if ENABLE_CLASS_B and strength >= STRENGTH_B:
        return SignalClassification(
            signal_class="B",
            strength=strength,
            allowed=True,
            reason="Segnale discreto (B)",
            size_multiplier_hint=0.7,
        )

    # 6) Tutto il resto → scartato
    return SignalClassification(
        signal_class="REJECT",
        strength=strength,
        allowed=False,
        reason=f"Strength {strength:.4f} insufficiente per A/B",
        size_multiplier_hint=0.0,
    )
