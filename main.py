from src.components.data_ingestion import DataIngestion
from src.components.data_cleaner import DataCleaner
from src.components.data_transformation import DataTransformation
from src.components.target_label_encoder import TargetLabelEncoder
from src.components.pca_handler import PCAHandler
from src.components.model_trainer import ModelTrainer

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

    # Split features / target
    X = df.drop(columns=['genre'])
    y = df['genre']

    transformer = DataTransformation()
    # Fit and transform features
    X_scaled,_ = transformer.fit_transform(X)

    # Encoding target variable
    label_encoder = TargetLabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply PCA
    pca_handler = PCAHandler(n_components=0.85)
    X_pca = pca_handler.fit_transform(X_scaled)

    # Train models
    trainer = ModelTrainer(X_pca, y_encoded)
    models, X_test, y_test = trainer.train_models()

    # Model evaluation


if __name__ == "__main__":
    main()