import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def clean_data(df):
    text_cols = ["student_name", "father_name", "mother_name", "institution_name"]
    for col in text_cols:
        df[col] = df[col].str.lower().str.strip()

    df["is_legitimate"] = df["is_legitimate"].astype(str).str.lower().map({
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
    "yes": 1,
    "no": 0
})


    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")

    return df

def create_features(df):

    df["student_father_same"] = (df["student_name"] == df["father_name"]).astype(int)
    df["student_mother_same"] = (df["student_name"] == df["mother_name"]).astype(int)
    df["father_mother_same"] = (df["father_name"] == df["mother_name"]).astype(int)

    institute_map = {
        "bir": "birla institute of technology",
        "xav": "xavier institute of social service",
        "cen": "central university of jharkhand",
        "ran": "ranchi university"
    }

    df["certificate_prefix"] = df["certificate_id"].str[:3].str.lower()
    df["expected_institution"] = df["certificate_prefix"].map(institute_map)

    df["institution_match"] = (
        df["expected_institution"] == df["institution_name"]
    ).astype(int)

    df["year_in_certificate"] = df["certificate_id"].str[3:7].astype(int)
    df["year_in_date"] = df["issue_date"].dt.year

    df["year_match"] = (
        df["year_in_certificate"] == df["year_in_date"]
    ).astype(int)

    df["name_repetition_score"] = (
    (df["student_name"] == df["father_name"]).astype(int) +
    (df["student_name"] == df["mother_name"]).astype(int) +
    (df["father_name"] == df["mother_name"]).astype(int)
)

    df["certificate_id_length"] = df["certificate_id"].str.len()
    df["valid_id_length"] = (df["certificate_id_length"] == 13).astype(int)

    df["issue_year"] = df["issue_date"].dt.year
    df["valid_year_range"] = (
        (df["issue_year"] >= 2015) & (df["issue_year"] <= 2025)
    ).astype(int)


    return df


def get_model_data(df):
    feature_cols = [
    "student_father_same",
    "student_mother_same",
    "father_mother_same",
    "name_repetition_score",
    "institution_match",
    "year_match",
    "valid_id_length",
    "valid_year_range"
]


    X = df[feature_cols]
    y = df["is_legitimate"]

    return X, y


def preprocess_pipeline(csv_path):
    df = load_data(csv_path)
    df = clean_data(df)
    df = create_features(df)

    X, y = get_model_data(df)

    return X, y, df

if __name__ == "__main__":
    X, y, df = preprocess_pipeline("data/generated_certificates.csv")

    print("Preprocessing completed successfully ✅")
    print("Feature matrix shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())
