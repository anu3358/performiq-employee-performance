"""Single place that describes the project: features, labels and file paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "employees.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
IMPORTANCE_PATH = ROOT / "models" / "importance.csv"

RANDOM_STATE = 42
TARGET = "performance"
CLASS_LABELS = ["Needs Improvement", "Meets Expectations", "Exceeds Expectations"]

# name -> (label, min, max, default, step)
NUM_FEATURES = {
    "years_at_company":   ("Years at company", 0, 30, 4, 1),
    "training_hours":     ("Training hours / year", 0, 120, 40, 1),
    "satisfaction":       ("Job satisfaction (1-5)", 1, 5, 3, 1),
    "work_life_balance":  ("Work-life balance (1-5)", 1, 5, 3, 1),
    "peer_rating":        ("Peer rating (1-5)", 1, 5, 3, 1),
    "projects_completed": ("Projects completed / year", 0, 20, 6, 1),
    "goal_completion":    ("Goals achieved (%)", 0, 100, 70, 1),
    "overtime_hours":     ("Overtime hours / month", 0, 60, 10, 1),
    "absences":           ("Absence days / year", 0, 30, 5, 1),
}

# name -> (label, options)
CAT_FEATURES = {
    "department": ("Department", ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]),
    "education":  ("Education", ["High School", "Bachelor's", "Master's", "PhD"]),
    "work_mode":  ("Work mode", ["On-site", "Hybrid", "Remote"]),
}

FEATURES = list(NUM_FEATURES) + list(CAT_FEATURES)
LABELS = {**{k: v[0] for k, v in NUM_FEATURES.items()},
          **{k: v[0] for k, v in CAT_FEATURES.items()}}

# Kept OUT of the model. Used only to audit fairness.
FAIRNESS_ATTRIBUTES = ["gender", "age_band"]
