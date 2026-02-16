"""
Darwin v4 Dashboard — Encryption Helper.

Fernet symmetric encryption for API credentials.
Key loaded from DASHBOARD_SECRET_KEY environment variable.

Usage:
    from dashboard.crypto_vault import CryptoVault
    vault = CryptoVault()
    encrypted = vault.encrypt("my-api-key")
    decrypted = vault.decrypt(encrypted)
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet


class CryptoVault:
    """Fernet encryption for API credentials. Never logs decrypted values."""

    def __init__(self, key: str | None = None):
        raw_key = key or os.environ.get("DASHBOARD_SECRET_KEY", "")
        if not raw_key:
            raise RuntimeError(
                "DASHBOARD_SECRET_KEY environment variable is required. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        # If the key is already a valid Fernet key (44 url-safe base64 chars), use it directly.
        # Otherwise, derive a Fernet-compatible key via SHA256.
        try:
            Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
            self._fernet = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
        except Exception:
            derived = base64.urlsafe_b64encode(
                hashlib.sha256(raw_key.encode()).digest()
            )
            self._fernet = Fernet(derived)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string. Returns base64-encoded ciphertext."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string. Returns plaintext."""
        return self._fernet.decrypt(ciphertext.encode()).decode()

    def mask(self, value: str) -> str:
        """Return masked version for display: first 4 + last 4 chars."""
        if len(value) <= 8:
            return "****"
        return value[:4] + "****" + value[-4:]
