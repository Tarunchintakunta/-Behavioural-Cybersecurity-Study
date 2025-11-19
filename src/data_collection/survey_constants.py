"""
This file acts as the single source of truth for the survey structure,
including question IDs, full text, and the scoring for their options.
"""

# A dictionary mapping the shorthand question ID to its properties
QUESTION_MAP = {
    # --- Consent and ID ---
    'TIMESTAMP': {'text': 'Timestamp'},
    'PARTICIPANT_ID': {'text': 'Participant ID\n(use S0001, S0002, etc)'},
    'CONSENT': {'text': 'I have read the participant information and consent to take part in this study. I understand I can withdraw at any time', 'options': {'I Consent': 1, 'I Do Not Consent': 0}},

    # --- Demographics (D) ---
    'D_AGE': {'text': 'Age bracket', 'options': {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55+': 5, '18–25': 1, '26 - 35': 2}},
    'D_EXPERIENCE': {'text': 'Years of professional computer use', 'options': {'0-2 years': 1, '3-5 years': 2, '6-10 years': 3, '11+ years': 4, '0–1': 1, '1 - 3': 1, '3 - 5': 2, '5+': 3}},
    'D_JOB_ROLE': {'text': 'Job role / Department \ne.g., Software Engineer, Finance, Admin'},

    # --- Digital Literacy & Training (T) ---
    'T_PRIOR_TRAINING': {'text': 'Prior cybersecurity training?', 'options': {'Yes': 1, 'No': 0}},
    'T_CONFIDENCE': {'text': 'How confident are you in identifying a phishing email?', 'options': {'Not confident': 1, 'Slightly confident': 2, 'Moderately confident': 3, 'Very confident': 4, 'Extremely confident': 5}},
    'T_DIGITAL_LITERACY': {'text': 'How would you rate your overall digital/computer literacy?', 'options': {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Expert': 4}},

    # --- Behavioral Factors (B) ---
    'B_STRESS': {'text': 'I feel stressed at work most days.', 'options': {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}},
    'B_AUTHORITY': {'text': 'I tend to follow requests that appear to come from managers.', 'options': {'Strongly disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly agree': 5}},
    'B_MULTITASKING': {'text': 'Multitasking frequency', 'options': {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Always': 5}},
    'B_PRIVACY': {'text': 'On a scale of 1-5, how important is data privacy to you in your professional role?', 'options': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}},
    
    # --- Optional ---
    'COMMENTS': {'text': 'Optional comments'},
    'WITHDRAWAL': {'text': 'Would you like to be contacted to withdraw your data if you later request it?'}
}


# --- Column Groupings for Analysis ---

DEMOGRAPHIC_COLUMNS = ['D_AGE', 'D_EXPERIENCE', 'D_JOB_ROLE']
COGNITIVE_BIAS_COLUMNS = ['B_AUTHORITY'] # Add other bias questions here if they exist
STRESS_COLUMNS = ['B_STRESS']
MULTITASKING_COLUMNS = ['B_MULTITASKING']
DIGITAL_LITERACY_COLUMNS = ['T_DIGITAL_LITERACY']
SECURITY_TRAINING_COLUMNS = ['T_PRIOR_TRAINING', 'T_CONFIDENCE']
PHISHING_AWARENESS_COLUMNS = ['T_CONFIDENCE'] # Can be the same as training or a separate group
DEVICE_USAGE_COLUMNS = [] # Add columns related to device usage if they exist
WORK_ENVIRONMENT_COLUMNS = ['B_PRIVACY'] # Example, can be refined

# A list of all question IDs that have numerical scores
ALL_QUESTION_COLUMNS = [
    'D_AGE', 'D_EXPERIENCE', 'T_PRIOR_TRAINING', 'T_CONFIDENCE', 
    'T_DIGITAL_LITERACY', 'B_STRESS', 'B_AUTHORITY', 'B_MULTITASKING', 'B_PRIVACY'
]
