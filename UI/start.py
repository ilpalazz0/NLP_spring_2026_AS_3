import os
import subprocess
import sys
import time

# ── Check models exist, train if not ─────────────────────────────────────────
if not os.path.exists('./backend/models/weights.pt') or \
   not os.path.exists('./backend/models/artifacts.pkl'):
    print("No trained models found. Running train.py first...")
    result = subprocess.run([sys.executable, 'train.py'], cwd='./backend')
    if result.returncode != 0:
        print("Training failed. Exiting.")
        sys.exit(1)
    print("Training complete.")
else:
    print("Pre-trained models found. Skipping training.")

# ── Start both servers ────────────────────────────────────────────────────────
flask = subprocess.Popen([sys.executable, 'app.py'],   cwd='./backend')
react = subprocess.Popen(['npm', 'start'],              cwd='./frontend')

print("Flask  → http://localhost:5000")
print("React  → http://localhost:3000")
print("Press Ctrl+C to stop both.\n")

# ── Keep alive, kill both on Ctrl+C ──────────────────────────────────────────
try:
    flask.wait()
    react.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    flask.terminate()
    react.terminate()