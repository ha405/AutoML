"""
Centralized Google Gemini client configuration.

All route modules should import from here instead of configuring their own client.
Usage:
    from routes.gemini_client import get_model, MODEL_NAME
"""

import os
import sys

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # loads from .env in project root and overrides existing
except ImportError:
    pass  # fall back to system environment variables

import google.generativeai as genai

MODEL_NAME = "gemini-2.0-flash"

_model = None
_configured = False


def _configure():
    """Configure the Gemini API client once."""
    global _model, _configured

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "ERROR: GOOGLE_API_KEY not found.\n"
            "Either create a .env file with GOOGLE_API_KEY=<your-key>\n"
            "or set the environment variable directly.",
            file=sys.stderr,
        )
        _configured = False
        return

    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(MODEL_NAME)
        _configured = True
        print(f"Gemini client configured (model: {MODEL_NAME}).")
    except Exception as e:
        print(f"ERROR: Failed to configure Gemini client: {e}", file=sys.stderr)
        _configured = False


# Configure on first import
_configure()


def get_model() -> genai.GenerativeModel | None:
    """Return the shared GenerativeModel instance, or None if not configured."""
    return _model


def is_configured() -> bool:
    """Return True if the Gemini client is ready to use."""
    return _configured
