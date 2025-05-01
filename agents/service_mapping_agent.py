# agents/service_mapping_agent.py

def map_services(tech_stack, known_info):
    """
    Map the tech stack to appropriate Azure services.
    """
    services = []

    for tech in tech_stack:
        if "Java application backend" in tech:
            services.append("Azure App Service (Java)")
        if "Python backend" in tech:
            services.append("Azure App Service (Python)")
        if "Node.js backend" in tech:
            services.append("Azure App Service (Node.js)")
        if "API layer" in tech:
            services.append("Azure API Management")
        if "SQL Server database" in tech:
            services.append("Azure SQL Managed Instance")
        if "MySQL database" in tech:
            services.append("Azure Database for MySQL")
        if "PostgreSQL database" in tech:
            services.append("Azure Database for PostgreSQL")
        if "MongoDB database" in tech:
            services.append("Azure Cosmos DB (Mongo API)")

    if known_info.get('storage_type') == "blob":
        services.append("Azure Blob Storage")
    elif known_info.get('storage_type') == "file":
        services.append("Azure Files Storage")

    return services
