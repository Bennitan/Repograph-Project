from langchain_community.graphs import Neo4jGraph
from parser import get_functions_and_calls

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# --- CONNECT ---
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USER, 
    password=NEO4J_PASSWORD
)

# --- PARSE ---
print("Reading Source Code...")
# We use your test file (or any file you want!)
funcs, calls = get_functions_and_calls("test_code.py")

# --- CLEAR OLD DATA ---
print("Cleaning old data...")
graph.query("MATCH (n) DETACH DELETE n")

# --- BUILD NODES (WITH CODE!) ---
print(f"Creating {len(funcs)} functions with source code...")
for f in funcs:
    # We escape quotes to prevent database errors
    safe_code = f['code'].replace("'", '"')
    
    # Cypher query to create node with a "code" property
    query = f"""
    CREATE (:Function {{
        name: '{f['name']}', 
        code: '{safe_code}'
    }})
    """
    graph.query(query)

# --- BUILD RELATIONSHIPS ---
print(f"Creating {len(calls)} connections...")
for caller, callee in calls:
    query = f"""
    MATCH (a:Function {{name: '{caller}'}}), (b:Function {{name: '{callee}'}})
    CREATE (a)-[:CALLS]->(b)
    """
    graph.query(query)

print("Success! Data uploaded to Neo4j.")