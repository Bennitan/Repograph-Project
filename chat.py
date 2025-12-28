import os
# --- FIXED IMPORTS START ---
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
# We pull the chain directly from the community package to avoid the error
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
# --- FIXED IMPORTS END ---

# --- CONFIGURATION ---
# 1. PASTE YOUR KEY HERE (Inside the quotes!)
GOOGLE_API_KEY = "AIzaSyBFssu0wBm5-8zJ0wq5S-_lMOOC2rIy1cw"

# 2. Neo4j Connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# --- THE LOGIC ---
print("Connecting to the Graph Brain...")

try:
    # Connect to Neo4j
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USER, 
        password=NEO4J_PASSWORD
    )

    # Connect to Google Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # Create the "Chain" (Translator)
    chain = GraphCypherQAChain.from_llm(
        llm=llm, 
        graph=graph, 
        verbose=True,
        allow_dangerous_requests=True
    )

    print("\n--- REPOGRAPH AI IS READY ---")
    print("Ask questions like: 'Who calls the authenticate function?'")
    print("Type 'exit' to stop.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            break
        
        response = chain.invoke(user_query)
        print(f"AI: {response['result']}\n")

except Exception as e:
    print(f"Error: {e}")