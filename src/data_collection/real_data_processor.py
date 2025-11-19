"""
Real Data Processor for Sailaja's Phishing Study
Processes and analyzes actual survey responses collected from participants
"""

import pandas as pd
from typing import List, Dict
import numpy as np
import logging
import os

from src.data_collection.survey_constants import (
    QUESTION_MAP, DEMOGRAPHIC_COLUMNS, COGNITIVE_BIAS_COLUMNS,
    STRESS_COLUMNS, MULTITASKING_COLUMNS, DIGITAL_LITERACY_COLUMNS,
    SECURITY_TRAINING_COLUMNS, PHISHING_AWARENESS_COLUMNS,
    DEVICE_USAGE_COLUMNS, WORK_ENVIRONMENT_COLUMNS, ALL_QUESTION_COLUMNS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_and_prepare_data(file_path: str, columns_to_keep_extra: List[str] = None) -> pd.DataFrame:
    """
    Cleans raw survey data, converts text answers to numerical scores,
    and calculates composite scores for various behavioral factors.

    Args:
        file_path (str): The path to the raw survey CSV file.
        columns_to_keep_extra (List[str]): A list of extra columns to retain in the final DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame with numerical scores.
    """
    logging.info(f"Starting data cleaning and preparation for {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return pd.DataFrame()

    # Create a mapping from full question text to its short identifier
    q_text_to_id = {v['text']: k for k, v in QUESTION_MAP.items()}

    # Standardize column names by mapping the full question text to the short ID
    original_columns = df.columns
    df.rename(columns=q_text_to_id, inplace=True)
    
    renamed_columns = df.columns
    renamed_count = sum(1 for oc, nc in zip(original_columns, renamed_columns) if oc != nc)
    logging.info(f"Renamed {renamed_count} columns to their short IDs.")


    # --- Convert Categorical Answers to Numerical Scores ---
    for col_id in df.columns:
        if col_id in QUESTION_MAP and 'options' in QUESTION_MAP[col_id]:
            # Check if the column is already numeric (e.g. from Google Forms scale)
            if pd.api.types.is_numeric_dtype(df[col_id]):
                logging.info(f"Column {col_id} is already numeric. Skipping mapping.")
                continue

            # Create a mapping from the option text to its score
            options_map = QUESTION_MAP[col_id]['options']
            # Apply the mapping. Unmapped values become NaN.
            df[col_id] = df[col_id].map(options_map)

    logging.info("Converted categorical text answers to numerical scores.")
    
    # --- Define columns to keep ---
    # We want to keep all columns that we can score plus demographics
    # --- Define columns to keep ---
    # We want to keep all columns that we can score plus demographics
    # Use dict.fromkeys to remove duplicates while preserving order
    columns_to_keep = list(dict.fromkeys(DEMOGRAPHIC_COLUMNS + ALL_QUESTION_COLUMNS))
    
    # Add any extra columns specified by the user
    if columns_to_keep_extra:
        for col in columns_to_keep_extra:
            if col in original_columns:
                columns_to_keep.append(col)
                # If the extra column was renamed, use its new name
                if col in q_text_to_id:
                     columns_to_keep.append(q_text_to_id[col])


    # Filter the DataFrame to only include the columns we intend to use or keep
    # We need to handle the case where a column from the constants might not be in the df
    final_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[final_columns]
    
    logging.info(f"Filtered DataFrame to {len(df.columns)} relevant columns.")


    # --- Calculate Composite Scores ---
    score_mapping = {
        'cognitive_bias': COGNITIVE_BIAS_COLUMNS,
        'stress_level': STRESS_COLUMNS,
        'multitasking_habits': MULTITASKING_COLUMNS,
        'digital_literacy': DIGITAL_LITERACY_COLUMNS,
        'security_training_effectiveness': SECURITY_TRAINING_COLUMNS,
        'phishing_awareness': PHISHING_AWARENESS_COLUMNS,
        'device_usage_patterns': DEVICE_USAGE_COLUMNS,
        'work_environment': WORK_ENVIRONMENT_COLUMNS
    }

    for score_name, cols in score_mapping.items():
        # Ensure we only use columns that are actually present in the DataFrame
        existing_cols = [col for col in cols if col in df.columns]
        if existing_cols:
            df[f'{score_name}_score'] = df[existing_cols].mean(axis=1)

    logging.info("Calculated composite scores for behavioral factors.")


    # --- Calculate Final Vulnerability Score ---
    # Define weights for each component. These can be adjusted based on domain knowledge.
    weights = {
        'cognitive_bias_score': 0.20,
        'stress_level_score': 0.15,
        'multitasking_habits_score': 0.10,
        'digital_literacy_score': -0.20,  # Negative weight as higher literacy decreases vulnerability
        'security_training_effectiveness_score': -0.15,
        'phishing_awareness_score': -0.15,
        'device_usage_patterns_score': 0.05,
        'work_environment_score': 0.10
    }

    df['vulnerability_score'] = 0
    for component, weight in weights.items():
        if component in df.columns:
            df['vulnerability_score'] += df[component] * weight

    # Normalize the score to be between 0 and 1 for easier interpretation
    min_score = df['vulnerability_score'].min()
    max_score = df['vulnerability_score'].max()
    if max_score > min_score:
        df['vulnerability_score'] = (df['vulnerability_score'] - min_score) / (max_score - min_score)
    else:
        df['vulnerability_score'] = 0.5 # If all values are the same, assign a neutral score
        
    logging.info("Calculated and normalized the final vulnerability score.")

    return df

def main():
    """
    Main function to demonstrate cleaning a sample data file.
    """
    # This assumes the script is run from the project's root directory
    input_file = 'data/raw/Cyber Security Study Participation Form (Responses) - Form responses 1.csv'
    output_file = 'data/processed/real_survey_data_cleaned.csv'
    
    # Example of how to keep an extra column (e.g., the original Participant ID)
    # Find the actual column name from the raw file first.
    try:
        raw_df = pd.read_csv(input_file)
        participant_id_col = next((col for col in raw_df.columns if 'Participant ID' in col), None)
        
        cleaned_df = clean_and_prepare_data(input_file, columns_to_keep_extra=[participant_id_col])
        
        if not cleaned_df.empty:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cleaned_df.to_csv(output_file, index=False)
            logging.info(f"Cleaned data successfully saved to {output_file}")
            print("\n--- Sample of Cleaned Data ---")
            print(cleaned_df.head())
            print("\n--- Data Summary ---")
            print(cleaned_df.describe())

    except FileNotFoundError:
        logging.error(f"The input file was not found: {input_file}")

if __name__ == '__main__':
    main()
