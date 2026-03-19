import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import subprocess

TRAINING_SCRIPTS = [
#     "training_scripts.rnn_count",
#     "training_scripts.rnn_tfidf",
#     "training_scripts.rnn_pmi",
#     "training_scripts.rnn_word2vec",
     "training_scripts.rnn_glove",
#     "training_scripts.birnn_count",
#    "training_scripts.birnn_tfidf",
#     "training_scripts.birnn_pmi",
    "training_scripts.birnn_word2vec",
    "training_scripts.birnn_glove",
#    "training_scripts.lstm_count",
#     "training_scripts.lstm_tfidf",
#     "training_scripts.lstm_pmi",
     "training_scripts.lstm_word2vec",
     "training_scripts.lstm_glove",
]

print("=" * 60)
print("TRAINING ALL MODELS")
print("=" * 60)

for script in TRAINING_SCRIPTS:
    print(f"\n>>> Training {script}...")
    result = subprocess.run(
        [sys.executable, "-m", script],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"[WARNING] {script} exited with code {result.returncode}")

print("\n" + "=" * 60)
print("ALL TRAINING COMPLETE")
print("=" * 60)