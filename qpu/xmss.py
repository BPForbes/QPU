"""
xmss.py

Provides secure signing functions using Ed25519 from the cryptography library.
Implements key generation, signing, verification, and public key serialization.
"""

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization


def xmss_keygen():
    """
    Generate a secure Ed25519 key pair.

    Ensures:
         Returns (priv_key, pub_key) as key objects.
    """
    priv_key = Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()
    return priv_key, pub_key


def xmss_sign(message: bytes, priv_key):
    """
    Sign a message using the provided Ed25519 private key.

    Requires:
         message (bytes), priv_key (Ed25519PrivateKey).
    Ensures:
         Returns the signature as bytes.
    """
    signature = priv_key.sign(message)
    return signature


def xmss_verify(message: bytes, signature: bytes, pub_key) -> bool:
    """
    Verify an Ed25519 signature.

    Requires:
         message (bytes), signature (bytes), pub_key (Ed25519PublicKey).
    Ensures:
         Returns True if valid, else False.
    """
    try:
        pub_key.verify(signature, message)
        return True
    except Exception:
        return False


def serialize_pub_key(pub_key) -> bytes:
    """
    Serialize an Ed25519 public key to raw bytes.

    Ensures:
         Returns raw public key bytes.
    """
    return pub_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
