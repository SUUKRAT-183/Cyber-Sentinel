"""
engines/auth.py
Password + TOTP (authenticator app) login for the dashboard.
This is a security FEATURE of the platform, not a machine-learning model.

Setup once with:  python setup_2fa.py
The password is stored as a salted PBKDF2 hash; the TOTP secret lives in auth_config.json
(listed in .gitignore - never commit it).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import pyotp

CONFIG_PATH = Path(__file__).resolve().parent.parent / "auth_config.json"
ISSUER = "Cyber Sentinel"
_ITERATIONS = 200_000


def _hash(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _ITERATIONS).hex()


def is_configured() -> bool:
    return CONFIG_PATH.exists()


def create_config(username: str, password: str) -> str:
    """Create/overwrite the login. Returns the new TOTP secret."""
    if len(password) < 8:
        raise ValueError("Use a password of at least 8 characters.")
    salt = os.urandom(16).hex()
    secret = pyotp.random_base32()
    CONFIG_PATH.write_text(json.dumps({
        "username": username, "salt": salt,
        "password_hash": _hash(password, salt), "totp_secret": secret,
    }, indent=2))
    return secret


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def provisioning_uri() -> str:
    cfg = _load()
    return pyotp.TOTP(cfg["totp_secret"]).provisioning_uri(
        name=cfg["username"], issuer_name=ISSUER)


def verify(username: str, password: str, code: str) -> tuple[bool, str]:
    """Check username, password and the 6-digit code. Returns (ok, message)."""
    if not is_configured():
        return False, "2FA is not set up. Run: python setup_2fa.py"
    cfg = _load()
    user_ok = hmac.compare_digest(username.strip(), cfg["username"])
    pass_ok = hmac.compare_digest(_hash(password, cfg["salt"]), cfg["password_hash"])
    code = (code or "").strip().replace(" ", "")
    totp_ok = code.isdigit() and len(code) == 6 and pyotp.TOTP(cfg["totp_secret"]).verify(code, valid_window=1)
    if user_ok and pass_ok and totp_ok:
        return True, "ok"
    return False, "Invalid username, password or 2FA code."
