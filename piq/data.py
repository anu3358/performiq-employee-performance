"""Creates (or loads) the employee dataset.

The bundled data is SYNTHETIC: generated with realistic rules so the project runs
anywhere. To use real data, put a CSV at data/employees.csv with the columns in
piq/config.py plus a 'performance' column (Needs Improvement / Meets Expectations /
Exceeds Expectations).
"""
import numpy as np
import pandas as pd

from piq import config


def generate_dataset(n: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    depts = config.CAT_FEATURES["department"][1]
    edus = config.CAT_FEATURES["education"][1]
    modes = config.CAT_FEATURES["work_mode"][1]

    age = np.clip(rng.normal(36, 9, n), 21, 62).round().astype(int)
    years = np.clip((age - 21) * rng.uniform(0.1, 0.6, n), 0, 30).round().astype(int)
    dept = rng.choice(depts, n, p=[.28, .20, .08, .12, .10, .22])
    edu = rng.choice(edus, n, p=[.15, .50, .28, .07])
    mode = rng.choice(modes, n, p=[.40, .40, .20])
    gender = rng.choice(["Female", "Male", "Non-binary"], n, p=[.47, .50, .03])

    training = np.clip(rng.normal(40, 20, n), 0, 120).round().astype(int)
    sat = np.clip(np.round(rng.normal(3.3, 1.0, n)), 1, 5).astype(int)
    wlb = np.clip(np.round(rng.normal(3.2, 1.0, n)), 1, 5).astype(int)
    peer = np.clip(np.round(rng.normal(3.4, 0.9, n)), 1, 5).astype(int)
    projects = np.clip(rng.poisson(6, n), 0, 20)
    goals = np.clip(rng.normal(68, 16, n), 0, 100).round().astype(int)
    overtime = np.clip(rng.gamma(2.0, 6.0, n), 0, 60).round().astype(int)
    absences = np.clip(rng.poisson(5, n), 0, 30)

    edu_idx = pd.Series(edu).map({e: i for i, e in enumerate(edus)}).to_numpy()
    dept_bonus = pd.Series(dept).map({"Engineering": .15, "Sales": .20, "HR": 0, "Marketing": .05,
                                      "Finance": .10, "Operations": -.10}).to_numpy()
    score = (0.030 * goals + 0.50 * (goals > 85) + 0.40 * peer + 0.22 * sat
             + 0.12 * np.sqrt(training) + 0.06 * projects
             - 0.07 * np.maximum(overtime - 18, 0) + 0.02 * np.minimum(overtime, 18)
             - 0.07 * absences + 0.10 * edu_idx + 0.08 * np.minimum(years, 10)
             + 0.30 * (sat * wlb) / 5 + 0.10 * (peer * sat) / 5
             + np.where(mode == "Hybrid", 0.1, 0) + dept_bonus + rng.normal(0, 0.45, n))
    q20, q75 = np.quantile(score, [0.20, 0.75])
    idx = np.where(score < q20, 0, np.where(score < q75, 1, 2))

    return pd.DataFrame({
        "employee_id": [f"E{i + 1:04d}" for i in range(n)],
        "department": dept, "education": edu, "work_mode": mode,
        "years_at_company": years, "training_hours": training, "satisfaction": sat,
        "work_life_balance": wlb, "peer_rating": peer, "projects_completed": projects,
        "goal_completion": goals, "overtime_hours": overtime, "absences": absences,
        "gender": gender,
        "age_band": pd.cut(age, [0, 29, 39, 49, 200], labels=["20s", "30s", "40s", "50+"]).astype(str),
        config.TARGET: [config.CLASS_LABELS[i] for i in idx],
    })


def load_dataset() -> pd.DataFrame:
    if config.DATA_PATH.exists():
        return pd.read_csv(config.DATA_PATH)
    df = generate_dataset()
    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PATH, index=False)
    return df
