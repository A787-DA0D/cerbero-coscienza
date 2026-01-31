# live/signal_classifier_v3.py
"""
Classificatore segnali per Cerbero Coscienza v3 PRO.

Prende:
- symbol, tf: info di contesto
- p_blend: probabilità blend (0–1)
- strength: |p_blend - 0.5|
- thresholds: dict da thresholds.json (per symbol/tf)
- mtf_ok: True/False (bias multi-timeframe)
- context_status: output di compute_context_status(...)
- econ_block: output di check_economic_block(...)
- probs: dict con p_grafico / p_tecnico / p_contestuale (opzionale)
- context_ok: bool opzionale (se il caller lo calcola già)
- news_ok: bool opzionale (se il caller ha già fatto un filtro news extra)

Restituisce un dict con:
{
  "ok": bool,                # se il trade è permesso
  "class": "A" | "A+" | "B" | "NONE",
  "reason": str,             # motivo principale (per log / debug)
  "p_blend": float,
  "strength": float,
  "mtf_ok": bool,
  "context_ok": bool,
  "blocked_by_news": bool,
  "econ_block": {...},       # pass-through
  "extra": {...},            # dettagli aggiuntivi
}
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from live.config_v3 import (
    STRENGTH_MIN,
    STRENGTH_B,
    STRENGTH_A,
    STRENGTH_A_PLUS,
    ENABLE_CLASS_B,
)


def classify_signal(
    *,
    symbol: str,
    tf: str,
    p_blend: float,
    strength: float,
    thresholds: Dict[str, float],
    mtf_ok: bool,
    context_status: Dict[str, Any],
    econ_block: Dict[str, Any],
    probs: Optional[Dict[str, float]] = None,
    context_ok: Optional[bool] = None,
    news_ok: Optional[bool] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Classifica il segnale in A / A+ / B / NONE e decide se è operabile.
    Si usa SEMPRE con argomenti keyword (come da live_loop_v3).
    """

    # --- Derivazione context_ok se non passato esplicitamente ---
    if context_ok is None:
        ctx_ok = bool(context_status.get("ok", True))
    else:
        ctx_ok = bool(context_ok)

    # --- Stato news macro ---
    econ_blocked = bool(econ_block.get("blocked", False))

    # Se news_ok non è passato, inferiamo da econ_block:
    if news_ok is None:
        effective_news_ok = not econ_blocked
    else:
        effective_news_ok = bool(news_ok)

    blocked_by_news = econ_blocked or (not effective_news_ok)

    if blocked_by_news:
        return {
            "ok": False,
            "class": "NONE",
            "reason": "blocked_by_news",
            "p_blend": float(p_blend),
            "strength": float(strength),
            "mtf_ok": bool(mtf_ok),
            "context_ok": ctx_ok,
            "blocked_by_news": True,
            "econ_block": econ_block,
            "extra": {
                "probs": probs or {},
            },
        }

    # ---- 2) Strength minimo ----
    if strength < STRENGTH_MIN:
        return {
            "ok": False,
            "class": "NONE",
            "reason": "strength_below_min",
            "p_blend": float(p_blend),
            "strength": float(strength),
            "mtf_ok": bool(mtf_ok),
            "context_ok": ctx_ok,
            "blocked_by_news": False,
            "econ_block": econ_block,
            "extra": {
                "probs": probs or {},
            },
        }

    # ---- 3) Soglia di probabilità blend per quel symbol/tf ----
    thr_blend = float(thresholds.get("thr_blend", 0.58))

    if p_blend < thr_blend:
        return {
            "ok": False,
            "class": "NONE",
            "reason": "p_blend_below_threshold",
            "p_blend": float(p_blend),
            "strength": float(strength),
            "mtf_ok": bool(mtf_ok),
            "context_ok": ctx_ok,
            "blocked_by_news": False,
            "econ_block": econ_block,
            "extra": {
                "thr_blend": thr_blend,
                "probs": probs or {},
            },
        }

    # ---- 4) Multi-timeframe bias ----
    if not mtf_ok:
        return {
            "ok": False,
            "class": "NONE",
            "reason": "mtf_not_ok",
            "p_blend": float(p_blend),
            "strength": float(strength),
            "mtf_ok": False,
            "context_ok": ctx_ok,
            "blocked_by_news": False,
            "econ_block": econ_block,
            "extra": {
                "probs": probs or {},
            },
        }

    # ---- 5) (Per ora) il contesto non blocca, ma lo tracciamo ----
    # In futuro potremo far sì che se ctx_ok è False → ridurre classe o rischio.

    # ---- 6) Classi A / A+ / B in base a strength ----
    if strength >= STRENGTH_A_PLUS:
        cls = "A+"
    elif strength >= STRENGTH_A:
        cls = "A"
    elif ENABLE_CLASS_B and strength >= STRENGTH_B:
        cls = "B"
    else:
        cls = "NONE"

    if cls == "NONE":
        return {
            "ok": False,
            "class": "NONE",
            "reason": "strength_not_enough_for_class",
            "p_blend": float(p_blend),
            "strength": float(strength),
            "mtf_ok": bool(mtf_ok),
            "context_ok": ctx_ok,
            "blocked_by_news": False,
            "econ_block": econ_block,
            "extra": {
                "thr_blend": thr_blend,
                "probs": probs or {},
            },
        }

    # Se arriviamo qui → segnale valido
    return {
        "ok": True,
        "class": cls,
        "reason": "ok",
        "p_blend": float(p_blend),
        "strength": float(strength),
        "mtf_ok": bool(mtf_ok),
        "context_ok": ctx_ok,
        "blocked_by_news": False,
        "econ_block": econ_block,
        "extra": {
            "thr_blend": thr_blend,
            "symbol": symbol,
            "tf": tf,
            "probs": probs or {},
        },
    }
