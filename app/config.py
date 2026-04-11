"""
Application configuration.

Set GOOGLE_API_KEY as an environment variable before running the app:
    Windows:  set GOOGLE_API_KEY=<your-key>
    Linux:    export GOOGLE_API_KEY=<your-key>
"""

import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")