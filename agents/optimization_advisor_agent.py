# agents/optimization_advisor_agent.py

def suggest_optimizations(known_info):
    """
    Recommend ways to optimize Azure costs.
    """
    recommendations = []

    if known_info.get('peak_users', 0) < 500:
        recommendations.append("Consider using Azure App Service scaling rules (AutoScale) to save costs during off-peak hours.")
    if known_info.get('criticality', 'low') == "low":
        recommendations.append("Consider Azure Reserved Instances for database and compute to save up to 60% costs.")
    recommendations.append("Review using Azure Functions (serverless) for infrequent API endpoints to minimize hosting costs.")
    
    return recommendations
