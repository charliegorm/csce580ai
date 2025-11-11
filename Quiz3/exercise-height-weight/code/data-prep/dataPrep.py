import pandas as pd
import os

def clean_data(input_file, output_file):
    # Load CSV (skip first row, blank)
    df = pd.read_csv(input_file, skiprows=1)

    # Strip whitespace from headers
    df.columns = df.columns.str.strip()

    # Rename columns
    df = df.rename(columns={
        "Height (cm)": "Height",
        "Weight (kg)": "Weight"
    })

    # Convert to numeric and drop invalid rows
    df["Height"] = pd.to_numeric(df["Height"], errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Height", "Weight"])
    df = df[(df["Height"] > 0) & (df["Weight"] > 0)]

    print("Cleaned dataset shape:", df.shape)

    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "../../data/weight_height_raw.csv")
    output_path = os.path.join(BASE_DIR, "../../data/cleaned_data.csv")
    print("Looking for input at:", os.path.abspath(input_path))
    print("Will save cleaned file to:", os.path.abspath(output_path))
    clean_data(input_path, output_path)
    


