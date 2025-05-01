# agents/migration_analyzer_agent.py

def analyze_project(user_input):
    """
    Analyze the user's tech stack based on input query.
    """
    tech_stack = []
    user_input = user_input.lower()

    if "java" in user_input:
        tech_stack.append("Java application backend")
    if "python" in user_input:
        tech_stack.append("Python backend (Flask/Django)")
    if "node" in user_input or "node.js" in user_input:
        tech_stack.append("Node.js backend")
    if "api" in user_input:
        tech_stack.append("API layer")
    if "sql server" in user_input or "sql" in user_input:
        tech_stack.append("SQL Server database")
    if "mysql" in user_input:
        tech_stack.append("MySQL database")
    if "postgres" in user_input:
        tech_stack.append("PostgreSQL database")
    if "mongodb" in user_input:
        tech_stack.append("MongoDB database")

    return tech_stack
