
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

df = pd.read_csv(
    DATA_DIR / "census_demographics.csv",
    dtype={"GEOID": str}
)

print("DEMOGRAPHIC DATA VALIDATION")
print("=" * 50)

# Recalculate vulnerability using available indicators
df["vulnerability"] = df[
    ["uninsured", "elderly"]
].mean(axis=1, skipna=True)

missing = df[df["vulnerability"].isna()]

print("Total neighborhoods:", len(df))
print("Zero population neighborhoods:",
      (df["population"] == 0).sum())

print("Missing vulnerability:", len(missing))

print("Missing vulnerability with zero population:",
      (missing["population"] == 0).sum())

print("Missing vulnerability with nonzero population:",
      (missing["population"] > 0).sum())

print("\nMISSING VULNERABILITY RECORDS")
print(missing.to_string(index=False))

print("\nAVERAGE VULNERABILITY")
print(df["vulnerability"].mean())

print("\nGEOID VERIFICATION")
print("Unique GEOIDs:", df["GEOID"].nunique())
print("Duplicate GEOIDs:", df["GEOID"].duplicated().sum())

df.to_csv(
    DATA_DIR / "census_demographics_validated.csv",
    index=False
)

print("\nValidated demographic file saved.")
