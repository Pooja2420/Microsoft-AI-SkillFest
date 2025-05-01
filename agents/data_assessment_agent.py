# agents/data_assessment_agent.py

def assess_data(known_info):
    """
    Check what critical data points are missing and generate questions.
    """
    questions = []

    if 'db_size_gb' not in known_info:
        questions.append("Can you provide the approximate size of your database (in GB)?")
    if 'api_traffic' not in known_info:
        questions.append("What is the average expected traffic/load for your APIs (e.g., requests per hour)?")
    if 'storage_type' not in known_info:
        questions.append("Are you using blob storage, file storage, or only databases?")
    if 'peak_users' not in known_info:
        questions.append("How many concurrent users are expected at peak time?")
    if 'criticality' not in known_info:
        questions.append("Is this application mission-critical (downtime tolerance)?")

    return questions
