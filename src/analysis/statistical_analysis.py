import pandas as pd
import json
import os
import logging
from scipy.stats import ttest_ind
import pingouin as pg

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_statistical_analysis(data_path: str, output_path: str):
    """
    Performs a comprehensive statistical analysis based on the project's research questions.
    """
    logging.info(f"Starting statistical analysis on {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}. Aborting analysis.")
        return

    report = {
        "source_file": data_path,
        "num_participants": len(df),
        "research_question_1": {},
        "research_question_2": {},
        "research_question_3": {},
        "research_question_4": {},
    }

    # RQ1: How do cognitive biases affect vulnerability? (Correlation)
    logging.info("Analyzing Research Question 1: Cognitive Biases vs. Vulnerability")
    corr_bias = df[['cognitive_bias_score', 'vulnerability_score']].corr()
    report['research_question_1'] = {
        "analysis_type": "Pearson Correlation",
        "description": "Correlation between cognitive bias score and the final vulnerability score.",
        "correlation_matrix": corr_bias.to_dict(),
        "interpretation": f"The correlation between cognitive bias and vulnerability is {corr_bias.iloc[0,1]:.3f}."
    }

    # RQ2: To what extent does stress and multitasking make a person more vulnerable? (Correlation)
    logging.info("Analyzing Research Question 2: Stress/Multitasking vs. Vulnerability")
    corr_stress = df[['stress_level_score', 'multitasking_habits_score', 'vulnerability_score']].corr()
    report['research_question_2'] = {
        "analysis_type": "Pearson Correlation",
        "description": "Correlation between stress/multitasking scores and the final vulnerability score.",
        "correlation_matrix": corr_stress.to_dict(),
        "interpretation": f"Stress correlation: {corr_stress.loc['stress_level_score', 'vulnerability_score']:.3f}. Multitasking correlation: {corr_stress.loc['multitasking_habits_score', 'vulnerability_score']:.3f}."
    }

    # RQ3: Does previous cybersecurity training lower vulnerability? (T-test)
    logging.info("Analyzing Research Question 3: Effect of Prior Training")
    trained_group = df[df['T_PRIOR_TRAINING'] == 1]['vulnerability_score']
    untrained_group = df[df['T_PRIOR_TRAINING'] == 0]['vulnerability_score']
    if not trained_group.empty and not untrained_group.empty:
        t_stat, p_value = ttest_ind(trained_group, untrained_group, nan_policy='omit')
        report['research_question_3'] = {
            "analysis_type": "Independent Samples T-test",
            "description": "Comparing vulnerability scores between participants with and without prior training.",
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_vulnerability_trained": trained_group.mean(),
            "mean_vulnerability_untrained": untrained_group.mean(),
            "interpretation": f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} (p={p_value:.4f})."
        }

    # RQ4: To what extent do demographics correlate with phishing vulnerability? (ANOVA for job role)
    logging.info("Analyzing Research Question 4: Demographics vs. Vulnerability")
    
    job_role_counts = df['D_JOB_ROLE'].value_counts()
    valid_groups = job_role_counts[job_role_counts > 1]
    
    interpretation = "Not enough data for ANOVA. Requires at least two job roles with more than one participant each."
    anova_results_dict = {}

    if len(valid_groups) >= 2:
        df_for_anova = df[df['D_JOB_ROLE'].isin(valid_groups.index)]
        anova_result = pg.anova(data=df_for_anova, dv='vulnerability_score', between='D_JOB_ROLE', detailed=True)
        anova_results_dict = anova_result.to_dict('records')
        
        if 'p-unc' in anova_result.columns and not pd.isna(anova_result.loc[0, 'p-unc']):
            p_value = anova_result.loc[0, 'p-unc']
            interpretation = f"ANOVA test for job role significance resulted in p-value of {p_value:.4f}."
        else:
            interpretation = "ANOVA could not be computed, likely due to insufficient variance within groups."
            logging.warning(interpretation)
    else:
        logging.warning(interpretation)

    report['research_question_4'] = {
        "analysis_type": "ANOVA",
        "description": "Comparing mean vulnerability scores across different job roles.",
        "anova_results": anova_results_dict,
        "interpretation": interpretation
    }

    # --- Save Report ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logging.info(f"Statistical analysis complete. Report saved to {output_path}")

def main():
    """Main function to run the analysis."""
    # Use relative paths from the assumed project root
    input_file = 'data/processed/model_training_data.csv'
    output_file = 'results/statistical_analysis_report.json'
    
    run_statistical_analysis(input_file, output_file)

if __name__ == "__main__":
    main()
