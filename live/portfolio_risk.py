# live/portfolio_risk.py

from dataclasses import dataclass
from typing import Dict, List

from live.ranking import RankedTrade


# Limiti di portafoglio (parametrici)
MAX_TOTAL_RISK_PCT = 0.035        # 3.5% dell'equity come rischio TOTALE (aperto + nuovi trade)
MAX_TRADES_PER_SYMBOL = 2         # massimo numero di trade aperti per singolo simbolo
MAX_NEW_TRADES_PER_BATCH = 5      # massimo numero di NUOVI trade da aprire in un singolo giro di live loop


@dataclass
class PortfolioState:
    """
    Stato sintetico del portafoglio al momento del giro di decisione.
    """
    equity: float                          # equity attuale (USDC)
    open_risk_pct: float                   # rischio totale già aperto (frazione di equity, es. 0.015 = 1.5%)
    open_trades_per_symbol: Dict[str, int] # numero di trade aperti per simbolo (es. {"EURUSD": 1, ...})


@dataclass
class SelectedTrade:
    """
    Trade selezionato per l'esecuzione in questo batch.
    """
    ranked_trade: RankedTrade
    new_total_risk_pct: float  # rischio totale (aperto + nuovi) dopo aver aggiunto questo trade


def select_trades_to_execute(
    ranked_trades: List[RankedTrade],
    portfolio: PortfolioState,
) -> List[SelectedTrade]:
    """
    Dato un elenco di RankedTrade (ordinati dal migliore al peggiore)
    e lo stato attuale del portafoglio, decide quali nuovi trade aprire
    rispettando i limiti di rischio e di numerosità.

    Logica:
      - scorre i ranked_trades in ordine
      - per ciascuno controlla:
          * rischio totale dopo l'aggiunta <= MAX_TOTAL_RISK_PCT
          * numero di trade per simbolo <= MAX_TRADES_PER_SYMBOL
          * numero di NUOVI trade nel batch <= MAX_NEW_TRADES_PER_BATCH
      - se tutte le condizioni sono rispettate, il trade viene selezionato
      - altrimenti viene saltato
    """

    selected: List[SelectedTrade] = []

    current_total_risk_pct = portfolio.open_risk_pct
    trades_count_per_symbol = dict(portfolio.open_trades_per_symbol)
    new_trades_count = 0

    for rt in ranked_trades:
        if new_trades_count >= MAX_NEW_TRADES_PER_BATCH:
            # abbiamo già raggiunto il massimo di nuove aperture per questo giro
            break

        cand = rt.candidate
        sym = cand.symbol

        # numero di trade già aperti su questo simbolo
        sym_open_count = trades_count_per_symbol.get(sym, 0)

        if sym_open_count >= MAX_TRADES_PER_SYMBOL:
            # troppo esposto su questo simbolo
            continue

        # rischio aggiuntivo se apriamo questo trade
        additional_risk_pct = cand.risk_pct

        # rischio totale dopo l'apertura
        new_total_risk_pct = current_total_risk_pct + additional_risk_pct

        if new_total_risk_pct > MAX_TOTAL_RISK_PCT:
            # supereremmo il limite massimo di rischio totale
            continue

        # Se siamo qui, il trade rispetta tutti i limiti -> lo selezioniamo
        selected.append(
            SelectedTrade(
                ranked_trade=rt,
                new_total_risk_pct=new_total_risk_pct,
            )
        )

        # aggiorniamo lo stato "simulato"
        current_total_risk_pct = new_total_risk_pct
        trades_count_per_symbol[sym] = sym_open_count + 1
        new_trades_count += 1

    return selected
