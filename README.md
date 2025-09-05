# NeuroPlus-Market-Scanner

A real-time cryptocurrency market scanner with a cyberpunk-themed PyQt6 interface.  
It fetches live data from the **MEXC Exchange**, applies advanced statistical tests, and visualizes results in an interactive dashboard.

---

## Features
- Fetches live market data for **all MEXC trading pairs**.
- Runs statistical tests:
  - Hurst Exponent
  - Augmented Dickey-Fuller (ADF)
  - KPSS Test
  - Correlation with BTC
- Interactive **PyQt6 GUI** with:
  - Start/Stop scanning
  - Logging area
  - Real-time charts with Z-score overlays
  - Export results to CSV (all or favorites)
- Sci-fi themed **loading screen, sounds, animations**.

---

## Tech Stack
- **Python 3.10+**
- **PyQt6** (GUI)
- **PyQtGraph** (charting)
- **aiohttp + asyncio** (async API fetching)
- **pandas & numpy** (data handling)
- **statsmodels & hurst** (statistical analysis)
- **pygame** (sound effects)
