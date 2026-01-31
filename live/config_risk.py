# live/config_risk.py
from dataclasses import dataclass
from enum import Enum, auto


class RiskProfile(Enum):
    SMALL = auto()        # equity < 300 USDC
    INTERMEDIATE = auto() # 300 <= equity < 600
    STANDARD = auto()     # equity >= 600


class Aggressiveness(Enum):
    NORMAL = auto()
    AGGRESSIVE = auto()


@dataclass
@dataclass
class RiskSettings:
    base_risk_fx: float          # rischio base per FX/indici (frazione, es. 0.0025 = 0.25%)
    base_risk_crypto: float      # rischio base per crypto (frazione)
    hard_cap_per_trade: float    # cap assoluto per trade (in frazione di equity)


# soglie di equity (parametriche, modificabili)
EQUITY_SMALL_MAX = 300.0          # sotto 300 USDC = Small Account Boost
EQUITY_INTERMEDIATE_MAX = 600.0   # tra 300 e 600 = Intermedio; sopra = Standard


def get_risk_profile(equity: float) -> RiskProfile:
    """
    Determina il profilo di rischio in base all'equity attuale.
    """
    if equity < EQUITY_SMALL_MAX:
        return RiskProfile.SMALL
    elif equity < EQUITY_INTERMEDIATE_MAX:
        return RiskProfile.INTERMEDIATE
    else:
        return RiskProfile.STANDARD


def get_risk_settings(equity: float) -> RiskSettings:
    """
    Restituisce le impostazioni di rischio (in frazione di equity per trade)
    in base al profilo calcolato dall'equity.
    """
    profile = get_risk_profile(equity)

    if profile == RiskProfile.SMALL:
        # Small Account Boost
        return RiskSettings(
            base_risk_fx=0.0075,       # 0.75% FX/indici
            base_risk_crypto=0.0100,   # 1.00% crypto
            hard_cap_per_trade=0.0200  # max 2.00% (consente AGGRESSIVE x2)
        )
    elif profile == RiskProfile.INTERMEDIATE:
        # account 300–600
        return RiskSettings(
            base_risk_fx=0.0040,       # 0.40%
            base_risk_crypto=0.0045,   # 0.45%
            hard_cap_per_trade=0.0100  # max 1.00%
        )
    else:
        # STANDARD ≥ 600
        return RiskSettings(
            base_risk_fx=0.0025,       # 0.25%
            base_risk_crypto=0.0035,   # 0.35%
            hard_cap_per_trade=0.0070  # max 0.70% (AGGRESSIVE x2 => 0.50% FX / 0.70% crypto cap)
        )



def compute_risk_per_trade(
    equity: float,
    is_crypto: bool,
    aggressiveness: Aggressiveness
) -> tuple[float, float]:
    """
    Calcola il rischio per trade (in frazione di equity) e l'hard cap effettivo
    in base a:
      - equity attuale (profilo SMALL / INTERMEDIATE / STANDARD)
      - tipo di asset (crypto vs FX/indici)
      - stato di aggressività (NORMAL / AGGRESSIVE)

    Ritorna:
      (risk_pct_effettivo, hard_cap_per_trade)

    dove risk_pct_effettivo è già moltiplicato per l'aggressività
    ma non supera mai l'hard_cap_per_trade.
    """
    base_settings = get_risk_settings(equity)

    # scegliamo il rischio base per asset class
    base_risk = base_settings.base_risk_crypto if is_crypto else base_settings.base_risk_fx

    # moltiplicatore per stato di aggressività
# moltiplicatore per stato di aggressività
    if aggressiveness == Aggressiveness.AGGRESSIVE:
        multiplier = 2.0   # spinge di più (x2)
    else:
        multiplier = 1.0   # NORMAL



    raw_risk = base_risk * multiplier
    # non superare mai l'hard cap definito per il profilo
    risk_pct = min(raw_risk, base_settings.hard_cap_per_trade)

    return risk_pct, base_settings.hard_cap_per_trade
