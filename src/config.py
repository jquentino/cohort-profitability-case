from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DECISION_TIME_DAYS = 400  # Decision time t in days after cohort creation
TIME_HORIZON_DAYS = 600  # Time horizon H in days for target variable
DATABASE_PATH = str(PROJ_ROOT / "database.db")
