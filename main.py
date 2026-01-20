from src.components.data_ingestion import DataIngestion
from src.components.data_cleaner import DataCleaner

EXPECTED_COLUMNS = [
    "Tempo",
    "Dynamics Range",
    "Vocal Presence",
    "Percussion Strength",
    "String Instrument Detection",
    "Electronic Element Presence",
    "Rhythm Complexity",
    "Drums Influence",
    "Distorted Guitar",
    "Metal Frequencies",
    "Ambient Sound Influence",
    "Instrumental Overlaps",
    "Genre"
]

def main():
    ingestion = DataIngestion(
        raw_data_path=r"data\raw\music_dataset_mod.csv",
        processed_data_path=r"data/processed/processed_music_dataset.csv"
    )

    # Load data
    df = ingestion.load_data()
    # Validate data
    ingestion.validate_data(df, EXPECTED_COLUMNS)
    # Save processed data
    ingestion.save_processed_data(df)

    cleaner = DataCleaner()
    # Standardize column names
    df = cleaner.standardize_column_names(df)
    # Remove duplicates
    df = cleaner.remove_duplicates(df)
    # Handle missing values
    df = cleaner.handle_missing_values(df, strategy='drop')


if __name__ == "__main__":
    main()