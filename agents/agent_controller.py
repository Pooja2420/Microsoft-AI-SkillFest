# agents/agent_controller.py

from agents.migration_analyzer_agent import analyze_project
from agents.data_assessment_agent import assess_data
from agents.service_mapping_agent import map_services
from agents.cost_estimator_agent import estimate_cost
from agents.optimization_advisor_agent import suggest_optimizations

def multi_agent_system(user_input, known_info={}):
    """
    Full end-to-end orchestration using multiple agents.
    """
    result = {}

    # 1. Migration analysis
    tech_stack = analyze_project(user_input)
    result['tech_stack'] = tech_stack

    # 2. Ask if important data missing
    followup_questions = assess_data(known_info)
    result['followup_questions'] = followup_questions

    if followup_questions:
        result['status'] = "Need more info"
        return result

    # 3. Map to Azure services
    services = map_services(tech_stack, known_info)
    result['azure_services'] = services

    # 4. Estimate costs
    costs = estimate_cost(known_info)
    result['cost_estimate'] = costs

    # 5. Recommend optimizations
    optimizations = suggest_optimizations(known_info)
    result['optimization_recommendations'] = optimizations

    result['status'] = "Complete Plan"

    return result
