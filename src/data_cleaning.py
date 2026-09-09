import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/training_set.csv")

print("========== ORIGINAL DATA ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ==========================================
# 2. CHECK TIMESTAMP
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

print("\n========== TIMESTAMP CHECK ==========")

print("Invalid timestamps:",
      df["timestamp"].isna().sum())

print("Start date:",
      df["timestamp"].min())

print("End date:",
      df["timestamp"].max())


# ==========================================
# 3. CHECK DUPLICATE TIMESTAMPS
# ==========================================

print("\n========== TIMESTAMP DUPLICATES ==========")

print(
    "Duplicate timestamps:",
    df["timestamp"].duplicated().sum()
)


# ==========================================
# 4. SORT CHRONOLOGICALLY
# ==========================================

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

print("\nData sorted chronologically.")


# ==========================================
# 5. DEFINE FEATURES AND TARGET
# ==========================================

feature_columns = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6"
]

target_column = "target"


# ==========================================
# 6. CONVERT FEATURES TO NUMERIC
# ==========================================

print("\n========== NUMERIC CHECK ==========")

for column in feature_columns + [target_column]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print(
        column,
        "missing after conversion:",
        df[column].isna().sum()
    )


# ==========================================
# 7. HANDLE MISSING VALUES
# ==========================================

print("\n========== MISSING VALUE HANDLING ==========")

# Interpolate missing feature values
df[feature_columns] = (
    df[feature_columns]
    .interpolate(method="linear")
    .ffill()
    .bfill()
)

# Target values are not artificially created
df = df.dropna(
    subset=[target_column]
).reset_index(drop=True)

print("Missing values after cleaning:")
print(df.isna().sum())


# ==========================================
# 8. DETECT OUTLIERS
# ==========================================

print("\n========== OUTLIER CHECK ==========")

for column in feature_columns + [target_column]:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outlier_count = (
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ).sum()

    print(
        column,
        "outliers:",
        outlier_count
    )


# ==========================================
# 9. FINAL VALIDATION
# ==========================================

print("\n========== FINAL VALIDATION ==========")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print(
    "Missing values:",
    df.isna().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Duplicate timestamps:",
    df["timestamp"].duplicated().sum()
)


# ==========================================
# 10. SAVE CLEANED DATASET
# ==========================================

output_path = "data/cleaned_training_set.csv"

df.to_csv(
    output_path,
    index=False
)

print("\n========== CLEANING COMPLETE ==========")
print("Cleaned dataset saved to:")
print(output_path)