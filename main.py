import pandas as pd


DATA_PATH = "data/processed/cleaned_healthcare_patients.csv"


def load_data(file_path):
    """Load the cleaned healthcare dataset."""
    return pd.read_csv(file_path)


def display_dataset_summary(df):
    """Display a basic summary of the healthcare dataset."""

    print("\n" + "=" * 60)
    print("HEALTHCARE PATIENT ANALYTICS SYSTEM")
    print("=" * 60)

    print(f"\nTotal Patients: {len(df)}")
    print(f"Total Features: {df.shape[1]}")

    print(f"\nAverage Age: {df['age'].mean():.2f}")
    print(f"Average Risk Score: {df['risk_score'].mean():.2f}")
    print(f"Average Treatment Cost: ₹{df['treatment_cost'].mean():,.2f}")
    print(f"Average Length of Stay: {df['length_of_stay'].mean():.2f} days")

    readmission_rate = df["readmission"].eq("Yes").mean() * 100
    print(f"Readmission Rate: {readmission_rate:.2f}%")

    print("\nTop Disease Categories:")
    disease_counts = df["disease"].value_counts().head(5)

    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")

    print("\nAdmission Type Distribution:")
    admission_counts = df["admission_type"].value_counts()

    for admission_type, count in admission_counts.items():
        print(f"  {admission_type}: {count}")

    print("\n" + "=" * 60)


def main():
    """Run the healthcare analytics summary."""

    df = load_data(DATA_PATH)

    display_dataset_summary(df)


if __name__ == "__main__":
    main()