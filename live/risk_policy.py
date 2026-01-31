# live/risk_policy.py
"""
Loader centrale della Risk Policy Cerbero.
La policy governa il rischio, NON decide il mercato.
"""

from pathlib import Path
import yaml


_POLICY_CACHE = None


def load_risk_policy() -> dict:
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE

    policy_path = Path(__file__).parent / "config_risk.yaml"
    if not policy_path.exists():
        raise FileNotFoundError(f"Risk policy not found: {policy_path}")

    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)

    if not policy or not policy.get("active", False):
        raise RuntimeError("Risk policy missing or inactive")

    _POLICY_CACHE = policy
    return policy
