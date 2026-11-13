"""etfcorr: multi-horizon correlation, forecasting and hedging study of thematic US ETFs.

Research code, not investment advice. Importing this package has no side effects
(no network calls, no file writes, no global state).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("etfcorr")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
