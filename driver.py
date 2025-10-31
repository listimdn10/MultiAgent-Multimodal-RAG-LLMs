# test_neo4j_connection.py
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load biến môi trường
load_dotenv()

# Lấy thông tin từ file .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("🔍 Testing Neo4j connection...")
print(f"URI: {NEO4J_URI}")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN '✅ Neo4j connection successful!' AS message")
        print(result.single()["message"])
    driver.close()

except Exception as e:
    print("❌ Connection failed!")
    print("Error details:", e)