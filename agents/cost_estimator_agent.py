# agents/cost_estimator_agent.py

def estimate_cost(known_info):
    """
    Rough cost estimation based on known parameters.
    """
    cost_estimate = {}

    # Base assumptions
    base_sql_cost = 100  # USD/month for SQL Server small
    base_app_service_cost = 75  # USD/month basic instance
    base_api_mgmt_cost = 50  # USD/month

    db_size = known_info.get('db_size_gb', 100)  # default 100GB
    api_traffic = known_info.get('api_traffic', 1000)  # default 1000 requests/hour

    # Database cost
    db_cost = base_sql_cost + (db_size * 0.5)
    app_cost = base_app_service_cost
    api_cost = base_api_mgmt_cost + (api_traffic / 1000) * 20  # assume $20 per 1000 req/h load

    estimated_total = db_cost + app_cost + api_cost

    cost_estimate['database'] = round(db_cost, 2)
    cost_estimate['app_service'] = round(app_cost, 2)
    cost_estimate['api_management'] = round(api_cost, 2)
    cost_estimate['estimated_total'] = round(estimated_total, 2)

    return cost_estimate
