"""Run once:  python train.py   (creates data/ and models/ files)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from piq.training import train_and_save

if __name__ == "__main__":
    m = train_and_save(verbose=True)
    print(f"\nBest model: {m['best_model']}  |  accuracy {m['accuracy']:.1%}  |  "
          f"F1 {m['f1']:.1%}  |  ROC-AUC {m['roc_auc']:.2f}")
