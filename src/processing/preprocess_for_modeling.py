import pandas as pd
import os
import logging
from typing import List
import re
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data_collection.real_data_processor import clean_and_prepare_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_participant_id_column(df: pd.DataFrame) -> str:
    """Finds the participant ID column in a DataFrame, checking for common variations."""
    for col in df.columns:
        if 'participant id' in col.lower() or 'participant_id' in col.lower():
            logging.info(f"Found participant ID column: '{col}'")
            return col
    return None

def standardize_participant_id(pid: str) -> str:
    """
    Cleans and standardizes a participant ID to the format PXXX.
    - Removes whitespace.
    - Extracts digits.
    - Formats to a 3-digit number.
    - Prepends 'P'.
    """
    if not isinstance(pid, str):
        return None
    # Extract all digits from the string
    digits = re.findall(r'\d+', pid)
    if not digits:
        return None
    # Join digits and convert to integer
    num = int("".join(digits))
    # Format to 3 digits with leading zeros and prepend 'P'
    return f'P{num:03d}'

def preprocess_for_modeling(experiment_path: str, survey_path: str, output_path: str, survey_raw_path: str):
    """
    Cleans, merges, and prepares experiment and survey data for ML modeling.

    Args:
        experiment_path (str): Path to the raw experiment results CSV file.
        survey_path (str): Path to the raw survey data CSV file.
        output_path (str): Path to save the final processed CSV file.
        survey_raw_path (str): Path to the raw survey data to get original participant ID.
    """
    logging.info("Starting preprocessing for modeling...")

    # --- Load Data ---
    try:
        df_exp = pd.read_csv(experiment_path)
        logging.info(f"Successfully loaded experiment data from {experiment_path}")
        # Use the raw survey path just to get the original Participant ID column,
        # as the clean_and_prepare_data function might alter or drop it.
        survey_raw_df = pd.read_csv(survey_raw_path)
        logging.info(f"Successfully loaded raw survey data from {survey_raw_path}")

    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")
        return

    # --- Process Survey Data ---
    # Find the original participant ID column name from the raw data
    participant_id_col_name = find_participant_id_column(survey_raw_df)
    if not participant_id_col_name:
        logging.error("Could not find the participant ID column in the raw survey data. Aborting.")
        return

    # Process the survey data, making sure to keep the original participant ID column
    df_survey_processed = clean_and_prepare_data(survey_path, columns_to_keep_extra=[participant_id_col_name])
    if df_survey_processed.empty:
        logging.error("Processing survey data resulted in an empty DataFrame. Aborting.")
        return
    logging.info("Successfully cleaned and processed survey data.")


    # --- Unify Participant ID Columns for Merging ---
    exp_id_col = find_participant_id_column(df_exp)
    survey_id_col = find_participant_id_column(df_survey_processed)

    if not exp_id_col or not survey_id_col:
        logging.error(f"Participant ID column missing. Found in experiment data: '{exp_id_col}'. Found in survey data: '{survey_id_col}'. Aborting.")
        return

    # Rename to a standard 'participant_id' for a clean merge
    df_exp.rename(columns={exp_id_col: 'participant_id'}, inplace=True)
    df_survey_processed.rename(columns={survey_id_col: 'participant_id'}, inplace=True)
    logging.info("Standardized participant ID columns to 'participant_id'.")
    
    # Ensure participant_id is of a common type to avoid merge issues
    df_exp['participant_id'] = df_exp['participant_id'].astype(str)
    df_survey_processed['participant_id'] = df_survey_processed['participant_id'].astype(str)

    # --- FIX: Robustly standardize ID format before merging ---
    df_survey_processed['participant_id'] = df_survey_processed['participant_id'].apply(standardize_participant_id)
    df_exp['participant_id'] = df_exp['participant_id'].apply(standardize_participant_id)
    logging.info("Robustly standardized all participant IDs to PXXX format.")

    # Drop any rows where the ID could not be standardized
    df_survey_processed.dropna(subset=['participant_id'], inplace=True)
    df_exp.dropna(subset=['participant_id'], inplace=True)

    # --- Merge DataFrames ---
    logging.info(f"Merging survey data (shape: {df_survey_processed.shape}) and experiment data (shape: {df_exp.shape}) on 'participant_id'.")
    
    # Select only necessary columns from experiment data
    df_exp_to_merge = df_exp[['participant_id', 'participant_action']]

    df_merged = pd.merge(df_survey_processed, df_exp_to_merge, on='participant_id', how='inner')
    logging.info(f"Successfully merged data. New shape: {df_merged.shape}")

    if df_merged.empty:
        logging.warning("The merged DataFrame is empty. This might happen if participant IDs do not match between the files.")
        logging.info(f"Experiment IDs: {df_exp['participant_id'].unique()}")
        logging.info(f"Survey IDs: {df_survey_processed['participant_id'].unique()}")
        return

    # --- Final Processing ---
    # Convert 'participant_action' to binary 'vulnerable' column
    df_merged['vulnerable'] = (df_merged['participant_action'] == 'Clicked Link').astype(int)
    df_merged.drop(columns=['participant_action'], inplace=True)
    logging.info("Created binary 'vulnerable' target column.")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- Save Processed Data ---
    df_merged.to_csv(output_path, index=False)
    logging.info(f"Preprocessing complete. Final dataset saved to {output_path}")

def main():
    """Main function to define file paths and run the preprocessing."""
    # Define paths
    # It's safer to use relative paths from the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    experiment_file = os.path.join(base_dir, 'data', 'synthetic', 'experiment_results.csv')
    
    # The raw file is needed to reliably get the participant ID column before it's cleaned
    survey_raw_file = os.path.join(base_dir, 'data', 'raw', 'Cyber Security Study Participation Form (Responses) - Form responses 1.csv')
    
    # The processed file is used for the actual data cleaning
    survey_processed_file = os.path.join(base_dir, 'data', 'raw', 'Cyber Security Study Participation Form (Responses) - Form responses 1.csv')

    output_file = os.path.join(base_dir, 'data', 'processed', 'model_training_data.csv')

    preprocess_for_modeling(experiment_file, survey_processed_file, output_file, survey_raw_file)

if __name__ == "__main__":
    main()
