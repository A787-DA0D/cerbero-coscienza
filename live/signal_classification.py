# live/signal_classification.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Literal

from live.position_sizing import SignalClass


Direction = Literal["LONG", "SHORT"]


class SignalQuality(Enum):
    """
    Quality generale del segnale, se in futuro vorrai distinguere più livelli.
    Per ora ci basta sapere se è un segnale valido oppure no.
    """
    NONE = auto()   # da ignorare
    A = auto()
    A_PLUS = auto()


@dataclass
class ClassifiedSignal:
    direction: Direction          # "LONG" o "SHORT"
    p_blend: float                # probabilità grezza del radar (0..1)
    strength: float               # |p_blend - 0.5|
    quality: SignalQuality        # NONE / A / A_PLUS
    signal_class: Optional[SignalClass]  # A / A_PLUS oppure None se da ignorare


# Soglie per la prima settimana (configurabili):
# - strength_min: sotto questo è rumore totale
# - A: soglia base per classe A
# - A_PLUS: soglia élite
STRENGTH_MIN = 0.02      # tutto sotto è rumore
STRENGTH_A = 0.10        # soglia minima per A
STRENGTH_A_PLUS = 0.12   # soglia per A+ (élite)


def classify_signal_from_p(p_blend: float) -> ClassifiedSignal:
    """
    Data la probabilità blend del radar (p_blend),
    determina:
      - direzione (LONG se p > 0.5, SHORT se p < 0.5)
      - strength = |p - 0.5|
      - qualità (NONE / A / A_PLUS)
      - SignalClass corrispondente (A / A_PLUS / None)

    Per la prima settimana:
      - accettiamo SOLO segnali A (>= 0.10) e A+ (>= 0.12)
      - ignoriamo tutto il resto (nessuna classe B o C).
    """

    # direction
    direction: Direction = "LONG" if p_blend >= 0.5 else "SHORT"

    # strength
    strength = abs(p_blend - 0.5)

    # rumore puro
    if strength < STRENGTH_MIN:
        return ClassifiedSignal(
            direction=direction,
            p_blend=p_blend,
            strength=strength,
            quality=SignalQuality.NONE,
            signal_class=None,
        )

    # segnali élite (A+)
    if strength >= STRENGTH_A_PLUS:
        quality = SignalQuality.A_PLUS
        sig_class = SignalClass.A_PLUS
    # segnali forti (A)
    elif strength >= STRENGTH_A:
        quality = SignalQuality.A
        sig_class = SignalClass.A
    else:
        # sotto A = scarta (niente B/C per ora)
        return ClassifiedSignal(
            direction=direction,
            p_blend=p_blend,
            strength=strength,
            quality=SignalQuality.NONE,
            signal_class=None,
        )

    return ClassifiedSignal(
        direction=direction,
        p_blend=p_blend,
        strength=strength,
        quality=quality,
        signal_class=sig_class,
    )
