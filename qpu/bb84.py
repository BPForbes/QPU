"""
bb84.py

Simulates a secure BB84 key exchange and encryption using the Fernet module from the cryptography library.
"""

from cryptography.fernet import Fernet


def bb84_key_exchange():
    """
    Generate a shared symmetric key securely using Fernet.

    Ensures:
         Returns a Fernet key as bytes.
    """
    return Fernet.generate_key()


def bb84_encrypt(message: bytes, key: bytes):
    """
    Encrypt the message using the provided Fernet key.

    Requires:
         message (bytes), key (bytes).
    Ensures:
         Returns ciphertext as bytes.
    """
    f = Fernet(key)
    return f.encrypt(message)


def bb84_decrypt(ciphertext: bytes, key: bytes):
    """
    Decrypt the ciphertext using the provided Fernet key.

    Requires:
         ciphertext (bytes), key (bytes).
    Ensures:
         Returns the original plaintext as bytes.
    """
    f = Fernet(key)
    return f.decrypt(ciphertext)
