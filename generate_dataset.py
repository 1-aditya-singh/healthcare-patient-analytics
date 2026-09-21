import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

np.random.seed(42)

TOTAL_PATIENTS = 10000


# --------------------------------------------------
# 2. Patient IDs
# --------------------------------------------------

patient_ids = np.arange(
    100001,
    100001 + TOTAL_PATIENTS
)


# --------------------------------------------------
# 3. Patient Demographics
# --------------------------------------------------

age = np.random.randint(
    18,
    86,
    TOTAL_PATIENTS
)

gender = np.random.choice(
    ["Male", "Female", "Other"],
    TOTAL_PATIENTS,
    p=[0.49, 0.49, 0.02]
)


# --------------------------------------------------
# 4. Vital Signs
# --------------------------------------------------

blood_pressure = np.clip(
    np.random.normal(128, 18, TOTAL_PATIENTS),
    90,
    190
).round().astype(int)

heart_rate = np.clip(
    np.random.normal(78, 12, TOTAL_PATIENTS),
    45,
    130
).round().astype(int)

bmi = np.clip(
    np.random.normal(26.2, 5.0, TOTAL_PATIENTS),
    15,
    45
).round(1)


# --------------------------------------------------
# 5. Lifestyle
# --------------------------------------------------

smoking_status = np.random.choice(
    ["Never", "Former", "Current"],
    TOTAL_PATIENTS,
    p=[0.58, 0.25, 0.17]
)


# --------------------------------------------------
# 6. Chronic Conditions
# --------------------------------------------------

age_factor = (age - 18) / 67
bmi_factor = np.clip(
    (bmi - 25) / 15,
    0,
    1
)

diabetes_probability = np.clip(
    0.06
    + 0.22 * age_factor
    + 0.12 * bmi_factor,
    0.03,
    0.42
)

hypertension_probability = np.clip(
    0.10
    + 0.34 * age_factor
    + 0.10 * bmi_factor,
    0.05,
    0.55
)

diabetes = np.where(
    np.random.random(TOTAL_PATIENTS)
    < diabetes_probability,
    "Yes",
    "No"
)

hypertension = np.where(
    np.random.random(TOTAL_PATIENTS)
    < hypertension_probability,
    "Yes",
    "No"
)


# --------------------------------------------------
# 7. Diseases
# --------------------------------------------------

disease_names = [
    "Healthy/Preventive",
    "Diabetes",
    "Hypertension",
    "Heart Disease",
    "Respiratory Disease",
    "Kidney Disease",
    "Cancer",
    "Other"
]

disease_probabilities = [
    0.18,
    0.16,
    0.19,
    0.11,
    0.11,
    0.07,
    0.06,
    0.12
]

disease = np.random.choice(
    disease_names,
    TOTAL_PATIENTS,
    p=disease_probabilities
)


# --------------------------------------------------
# 8. Admission Type
# --------------------------------------------------

admission_type = np.random.choice(
    ["Emergency", "Elective", "Urgent"],
    TOTAL_PATIENTS,
    p=[0.32, 0.43, 0.25]
)


# --------------------------------------------------
# 9. Length of Stay
# --------------------------------------------------

base_length_of_stay = np.select(
    [
        disease == "Cancer",
        disease == "Heart Disease",
        disease == "Kidney Disease",
        disease == "Respiratory Disease",
        disease == "Diabetes",
        disease == "Hypertension",
        disease == "Other"
    ],
    [
        9.0,
        7.0,
        6.5,
        5.5,
        4.0,
        3.5,
        4.5
    ],
    default=2.0
)

admission_effect = np.where(
    admission_type == "Emergency",
    2.0,
    np.where(
        admission_type == "Urgent",
        1.0,
        0.0
    )
)

length_of_stay = np.clip(
    np.random.poisson(
        np.maximum(
            base_length_of_stay + admission_effect,
            1.2
        )
    ) + 1,
    1,
    30
)


# --------------------------------------------------
# 10. Medication Count
# --------------------------------------------------

condition_count = (
    (diabetes == "Yes").astype(int)
    + (hypertension == "Yes").astype(int)
    + (bmi >= 30).astype(int)
    + (smoking_status == "Current").astype(int)
)

medication_count = np.clip(
    np.random.poisson(
        1.8
        + 0.75 * condition_count
        + 0.025 * np.maximum(age - 40, 0)
    ),
    0,
    15
)


# --------------------------------------------------
# 11. Treatment Cost
# --------------------------------------------------

disease_cost = {
    "Healthy/Preventive": 1500,
    "Diabetes": 6500,
    "Hypertension": 5000,
    "Heart Disease": 18000,
    "Respiratory Disease": 11000,
    "Kidney Disease": 15000,
    "Cancer": 28000,
    "Other": 8000
}

base_cost = np.array(
    [disease_cost[item] for item in disease],
    dtype=float
)

treatment_cost = (
    base_cost
    + length_of_stay
    * np.random.uniform(
        1800,
        3200,
        TOTAL_PATIENTS
    )
    + medication_count
    * np.random.uniform(
        450,
        900,
        TOTAL_PATIENTS
    )
    + np.where(
        admission_type == "Emergency",
        8000,
        0
    )
    + np.random.normal(
        0,
        4500,
        TOTAL_PATIENTS
    )
)

treatment_cost = np.clip(
    treatment_cost,
    800,
    None
).round().astype(int)


# --------------------------------------------------
# 12. Risk Score
# --------------------------------------------------

risk_score = (
    12
    + age_factor * 28
    + (diabetes == "Yes") * 12
    + (hypertension == "Yes") * 10
    + np.where(bmi >= 30, 7, 0)
    + np.where(
        smoking_status == "Current",
        7,
        np.where(
            smoking_status == "Former",
            3,
            0
        )
    )
    + np.where(
        disease == "Heart Disease",
        18,
        0
    )
    + np.where(
        disease == "Cancer",
        20,
        0
    )
    + np.where(
        disease == "Kidney Disease",
        15,
        0
    )
    + np.where(
        admission_type == "Emergency",
        8,
        0
    )
    + np.maximum(
        length_of_stay - 3,
        0
    ) * 1.3
    + np.random.normal(
        0,
        7,
        TOTAL_PATIENTS
    )
)

risk_score = np.clip(
    risk_score,
    1,
    100
).round(1)


# --------------------------------------------------
# 13. Readmission
# --------------------------------------------------

readmission_probability = np.clip(
    0.05
    + risk_score / 400
    + np.maximum(
        length_of_stay - 5,
        0
    ) * 0.012
    + medication_count * 0.008,
    0.03,
    0.55
)

readmission = np.where(
    np.random.random(TOTAL_PATIENTS)
    < readmission_probability,
    "Yes",
    "No"
)


# --------------------------------------------------
# 14. Create DataFrame
# --------------------------------------------------

df = pd.DataFrame({
    "patient_id": patient_ids,
    "age": age,
    "gender": gender,
    "blood_pressure": blood_pressure,
    "heart_rate": heart_rate,
    "bmi": bmi,
    "smoking_status": smoking_status,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "disease": disease,
    "admission_type": admission_type,
    "length_of_stay": length_of_stay,
    "medication_count": medication_count,
    "treatment_cost": treatment_cost,
    "readmission": readmission,
    "risk_score": risk_score
})


# --------------------------------------------------
# 15. Add Missing Values
# --------------------------------------------------

missing_rates = {
    "blood_pressure": 0.015,
    "heart_rate": 0.012,
    "bmi": 0.018,
    "smoking_status": 0.010,
    "medication_count": 0.012,
    "treatment_cost": 0.008
}

for column, rate in missing_rates.items():

    number_of_missing = int(
        TOTAL_PATIENTS * rate
    )

    missing_indices = np.random.choice(
        df.index,
        size=number_of_missing,
        replace=False
    )

    df.loc[
        missing_indices,
        column
    ] = np.nan


# --------------------------------------------------
# 16. Add Duplicate Rows
# --------------------------------------------------

duplicates = df.sample(
    20,
    random_state=7
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)


# --------------------------------------------------
# 17. Shuffle Dataset
# --------------------------------------------------

df = df.sample(
    frac=1,
    random_state=123
).reset_index(drop=True)


# --------------------------------------------------
# 18. Save Dataset
# --------------------------------------------------

output_path = Path(
    "data/raw/healthcare_patients.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 19. Print Dataset Information
# --------------------------------------------------

print("=" * 60)
print("HEALTHCARE PATIENT DATASET GENERATED")
print("=" * 60)

print(f"\nDataset location:")
print(output_path)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")

print("\nMissing values:")
print(df.isnull().sum())

print(
    f"\nDuplicate rows: "
    f"{df.duplicated().sum()}"
)

print("\nDataset preview:")
print(df.head())

print("\nDataset generated successfully!")