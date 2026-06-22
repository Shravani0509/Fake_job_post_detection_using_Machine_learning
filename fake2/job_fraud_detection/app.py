"""Compatibility entrypoint.

This repo originally used a top-level Flask app (`app.py`).
For production readiness and cleaner structure, the actual app now lives in
`src/app.py`.

Run:
  python app.py
or
  python -m src.app
"""

from job_fraud_detection.src.app import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

