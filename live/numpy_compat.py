# live/numpy_compat.py
"""
Shim di compatibilità per i modelli Radar v3:

Alcuni modelli sono stati salvati aspettandosi il modulo
`numpy._core.numeric` (tipico di NumPy 2.x).

Su NumPy 1.26.x questo modulo NON esiste, ma esiste
`numpy.core.numeric`.

Qui creiamo un modulo finto `numpy._core.numeric` che
re-esporta tutto da `numpy.core.numeric`, così joblib
può importarlo senza errori.
"""

import sys
import types

try:
    # Se esiste già (es. in futuro con NumPy 2), non facciamo nulla.
    import numpy._core.numeric  # type: ignore
except ModuleNotFoundError:
    from numpy.core import numeric as _numeric  # type: ignore

    shim = types.ModuleType("numpy._core.numeric")
    shim.__dict__.update(_numeric.__dict__)

    # Registra il modulo finto in sys.modules
    sys.modules["numpy._core.numeric"] = shim
