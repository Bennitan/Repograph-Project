import streamlit as st
import os
import tempfile
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from parser import get_functions_and_calls
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# --- NEW IMPORTS FOR THE "BRAIN" ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="RepoGraph AI", page_icon="🧠", layout="wide")

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #1F1F1F; font-family: 'SF Pro Display', sans-serif; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; font-weight: 600; background-color: #F5F5F7; border-radius: 15px; padding: 12px 25px; }
    .stTabs [aria-selected="true"] { background-color: #007AFF !important; color: white !important; }
    div[data-testid="metric-container"] { background-color: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 20px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    div[data-testid="metric-container"]:hover { transform: scale(1.05); border-color: #007AFF; }
    div[data-testid="stMetricValue"] { font-size: 42px; font-weight: 800; color: #1D1D1F; }
</style>
""", unsafe_allow_html=True)

# --- 3. SECRETS ---
import streamlit as st
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Initialize Embeddings Engine
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# --- 4. HELPER FUNCTIONS ---
def clean_code(text): return text.replace("```python", "").replace("```", "").strip()

def get_stats():
    """Fetches counts from Neo4j to update the top dashboard"""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            f = s.run("MATCH (n:Function) RETURN count(n) as c").single()["c"]
            r = s.run("MATCH (n:Risk) RETURN count(n) as c").single()
            risk = r["c"] if r else 0
            e = s.run("MATCH (n:Endpoint) RETURN count(n) as c").single()
            end = e["c"] if e else 0
    except:
        return 0, 0, 0
    return f, risk, end

def get_function_data(func_name):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run(f"MATCH (f:Function {{name: '{func_name}'}}) RETURN f.code as code, f.complexity as score")
        record = result.single()
        return record["code"], record["score"]

def generate_code_upgrade(func_name, task_type):
    code, score = get_function_data(func_name)
    prompt = ""
    
    if task_type == "Unit Tests":
        prompt = f"Write a 'pytest' suite for this function:\n{code}\nReturn ONLY code."
    elif "Refactor" in task_type:
        prompt = f"Optimize this Python code from O(n^2) to O(n) or O(log n). Return ONLY code:\n{code}"
    elif task_type == "Docstring":
        prompt = f"Add Google-style docstrings to:\n{code}\nReturn ONLY code."
    
    if not prompt: return "Error: Unknown Task"

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list): content = content[0]['text']
    return clean_code(str(content))

# --- NEW: VECTOR BRAIN FUNCTIONS ---
def create_vector_index(file_path):
    with open(file_path, "r") as f: text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def semantic_search(query, vector_store):
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🧠 RepoGraph v9.0")
    uploaded_file = st.file_uploader("Upload Source Code", type="py")
    
    if uploaded_file and st.button("🚀 Audit System"):
        with st.spinner("Building Knowledge Graph & Semantic Brain..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                # 1. Graph Processing
                funcs, calls = get_functions_and_calls(tmp_path)
                graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
                graph.query("MATCH (n) DETACH DELETE n")
                for f in funcs:
                    safe_code = f['code'].replace("'", '"')
                    graph.query(f"CREATE (:Function {{name: '{f['name']}', code: '{safe_code}', complexity: {f['complexity']}}})")
                    if f['api_route']:
                        graph.query(f"MERGE (e:Endpoint {{name: 'API: {f['api_route']}'}})")
                        graph.query(f"MATCH (f:Function {{name: '{f['name']}'}}), (e:Endpoint {{name: 'API: {f['api_route']}'}}) MERGE (e)-[:TRIGGERS]->(f)")
                    for risk in f['risks']:
                        graph.query(f"MERGE (r:Risk {{name: '{risk}'}})")
                        graph.query(f"MATCH (f:Function {{name: '{f['name']}'}}), (r:Risk {{name: '{risk}'}}) MERGE (f)-[:HAS_VULNERABILITY]->(r)")
                for a, b in calls:
                    graph.query(f"MATCH (a:Function {{name: '{a}'}}), (b:Function {{name: '{b}'}}) CREATE (a)-[:CALLS]->(b)")
                
                # 2. Vector Processing (The Brain)
                st.session_state['vector_store'] = create_vector_index(tmp_path)
                
                st.success(f"Indexed {len(funcs)} functions & Built Semantic Brain")
                os.remove(tmp_path)
            except Exception as e: st.error(f"Error: {e}")

# --- 6. DASHBOARD ---
f_total, risk_total, route_total = get_stats()
c1, c2, c3 = st.columns(3)
with c1: st.metric("Functions", f_total)
with c2: st.metric("Microservices", route_total)
with c3: st.metric("Risks", risk_total)
st.divider()

# --- 7. TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Architect", "🗺️ Graph", "🛡️ Security", "⚡ God Mode"])

with tab1:
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages: st.chat_message(m["role"]).write(m["content"])
    
    if prompt := st.chat_input("Ask about logic (e.g., 'How do we handle money?')..."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Consulting Semantic Brain..."):
            try:
                # 1. Search the Brain
                context = "No context."
                if 'vector_store' in st.session_state:
                    context = semantic_search(prompt, st.session_state['vector_store'])
                
                # 2. Generate Answer
                llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GOOGLE_API_KEY)
                final_prompt = f"User Question: {prompt}\n\nCode Context:\n{context}\n\nAnswer as a Senior Architect:"
                res = llm.invoke(final_prompt).content
                if isinstance(res, list): res = res[0]['text']
            except Exception as e: res = f"Error: {e}"
            
        st.chat_message("assistant").write(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

with tab2:
    st.subheader("🗺️ System Architecture & Blast Radius")
    
    # 1. Fetch function names from Neo4j for the selector
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        all_funcs = [r["n"] for r in s.run("MATCH (n:Function) RETURN n.name as n")]
    
    # 2. Add the simulator controls in a nice layout
    col1, col2 = st.columns([1, 3])
    with col1:
        target = st.selectbox("Simulate Failure in:", ["None"] + all_funcs)
        run_sim = st.button("🔥 Simulate Failure")
    
    # 3. Logic to calculate the "Impact"
    if run_sim and target != "None":
        with driver.session() as s:
            # Cypher query to find ALL functions downstream from the failure
            impact_query = f"""
            MATCH (start:Function {{name: '{target}'}})-[*]->(downstream) 
            RETURN collect(DISTINCT downstream.name) as affected
            """
            affected = s.run(impact_query).single()["affected"]
            
            # Prepare Nodes with "Heat Map" coloring (Failure = Red, Impact = Orange)
            nodes = []
            full_list = s.run("MATCH (n:Function) RETURN n.name as n")
            for r in full_list:
                node_name = r["n"]
                color = "#4B4B4B" # Default Grey
                if node_name == target: color = "#FF0000" # RED for failure
                elif node_name in affected: color = "#FFA500" # ORANGE for impact
                
                nodes.append(Node(id=node_name, label=node_name, size=25 if node_name == target else 20, color=color))
            
            edges = [Edge(source=r["a"], target=r["b"]) for r in s.run("MATCH (a)-[:CALLS]->(b) RETURN a.name as a, b.name as b")]
            
        st.error(f"⚠️ FAILURE ALERT: A crash in **{target}** would impact **{len(affected)}** connected services!")
        agraph(nodes, edges, Config(width=900, height=500, directed=True))
    else:
        # Show regular graph if no simulation is running
        if st.button("Refresh Standard View"):
            with driver.session() as s:
                nodes = [Node(id=r["n"], label=r["n"], size=20, color="#007AFF") for r in s.run("MATCH (n:Function) RETURN n.name as n")]
                edges = [Edge(source=r["a"], target=r["b"]) for r in s.run("MATCH (a)-[:CALLS]->(b) RETURN a.name as a, b.name as b")]
            agraph(nodes, edges, Config(width=900, height=500, directed=True))

with tab3:
    st.subheader("🛡️ Vulnerability Report")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        risks = list(s.run("MATCH (f:Function)-[:HAS_VULNERABILITY]->(r:Risk) RETURN f.name as func, r.name as risk, f.code as code"))
    if not risks: st.success("✅ System Secure")
    else:
        for r in risks:
            with st.expander(f"⚠️ {r['risk']} in {r['func']}", expanded=True):
                st.code(r['code'], language='python')

with tab4:
    st.subheader("⚡ Senior Engineer Tools")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            func_list = [r["name"] for r in s.run("MATCH (n:Function) RETURN n.name as name")]
    except: func_list = []

    if func_list:
        c1, c2 = st.columns([1, 2])
        with c1:
            sel_func = st.selectbox("Function:", func_list)
            action = st.radio("Task:", ["Refactor (Big-O)", "Unit Tests", "Docstring"])
            if st.button("✨ Execute Agent", use_container_width=True):
                with st.spinner(f"Running {action}..."):
                    res = generate_code_upgrade(sel_func, action)
                    st.session_state['gen_code'] = res
        with c2:
            if 'gen_code' in st.session_state:
                st.code(st.session_state['gen_code'], language='python')