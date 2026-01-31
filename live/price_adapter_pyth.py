# live/price_adapter_pyth.py
from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime, timezone

PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest"

# Mappa symbol Cerbero -> Pyth price ID
SYMBOL_TO_PYTH_ID: Dict[str, str] = {
    # FX majors & JPY crosses
    "EURUSD": "0xa995d00bb36a63cef7fd2c287dc105fc8f3d93779f062f09551b0af3e81ec30b",  # FX.EUR/USD
    "GBPUSD": "0x84c2dde9633d93d1bcad84e7dc41c9d56578b7ec52fabedc1f335d673df0a7c1",  # FX.GBP/USD
    "AUDUSD": "0x67a6f93030420c1c9e3fe37c1ab6b77966af82f995944a9fefce357a22854a80",  # FX.AUD/USD
    "USDJPY": "0xef2c98c804ba503c6a707e38be4dfbb16683775f195b091252bf24693042fd52",  # FX.USD/JPY
    "USDCHF": "0x0b1e3297e69f162877b577b0d6a47a0d63b2392bc8499e6540da4187a63e28f8",  # FX.USD/CHF
    "USDCAD": "0x3112b03a41c910ed446852aacf67118cb1bec67b2cd0b9a214c58cc0eaa2ecca",  # FX.USD/CAD

    "EURJPY": "0xd8c874fa511b9838d094109f996890642421e462c3b29501a2560cecf82c2eb4",  # FX.EUR/JPY
    "AUDJPY": "0x8dbbb66dff44114f0bfc34a1d19f0fe6fc3906dcc72f7668d3ea936e1d6544ce",  # FX.AUD/JPY
    "CADJPY": "0x9e19cbf0b363b3ce3fa8533e171f449f605a7ca5bb272a9b80df4264591c4cbb",  # FX.CAD/JPY
    "GBPJPY": "0xcfa65905787703c692c3cac2b8a009a1db51ce68b54f5b206ce6a55bfa2c3cd1",  # FX.GBP/JPY

    # Dollar Index
    "DOLLARIDXUSD": "0x710afe0041a07156bfd71971160c78a326bf8121403e0d4e140d06bea0353b7f",  # FX.USDXY

    # Metals
    "XAGUSD": "0xf2fb02c32b055c805e7238d628e5e9dadef274376114eb1f012337cabe93871e",  # Metallo.XAG/USD
    "XAUUSD": "0x765d2ba906dbc32ca17cc11f5310a89e9ee1f6420508c63861f2f8ba4ee34bb2",  # Metallo.XAU/USD

    # Oil (WTI spot)
    "LIGHTCMDUSD": "0x925ca92ff005ae943c158e3563f59698ce7e75c5a8c8dd43303a0a154887b3e6",  # Materie prime.USOILSPOT

    # Crypto
    "BTCUSD": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",  # Cripto.BTC/USD
    "ETHUSD": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",  # Cripto.ETH/USD
}


@dataclass
class PythPrice:
    price: float
    conf: float
    expo: int
    publish_time_utc: datetime


class PythAdapter:
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout

    def get_price(self, symbol: str) -> Optional[PythPrice]:
        """
        Ritorna il prezzo Pyth per il simbolo dato, oppure None se manca.
        """
        sym = symbol.upper()
        price_id = SYMBOL_TO_PYTH_ID.get(sym)
        if not price_id:
            return None

        try:
            resp = requests.get(
                PYTH_HERMES_URL,
                params={"ids[]": price_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[PYTH] Errore HTTP per {symbol}: {e}")
            return None

        try:
            parsed_list = data.get("parsed", [])
            if not parsed_list:
                return None

            p = parsed_list[0]["price"]

            # Conversione robusta: API può dare stringhe → castiamo
            raw_price = float(p["price"])
            raw_conf  = float(p["conf"])
            expo      = int(p["expo"])

            scale = 10 ** expo  # expo può essere negativo

            price = raw_price * scale
            conf  = raw_conf * scale
            publish_time = datetime.fromtimestamp(int(p["publish_time"]), tz=timezone.utc)

            return PythPrice(
                price=float(price),
                conf=float(conf),
                expo=expo,
                publish_time_utc=publish_time,
            )
        except Exception as e:
            print(f"[PYTH] Errore parse JSON per {symbol}: {e}")
            return None


# Istanza globale (comoda da importare)
pyth_adapter = PythAdapter()
