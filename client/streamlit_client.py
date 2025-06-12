import os
import asyncio
import nest_asyncio
import logging
import inspect
import re
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import tempfile
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_extras.stylable_container import stylable_container
import json
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.prompt import PROMPTS

# Apply nest_asyncio to allow running async code in Streamlit
nest_asyncio.apply()

# Configure logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Set Streamlit page configuration
st.set_page_config(
    page_title="LightRAG Game Master",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom prompt template
custom_prompt = f"""
---Role---

You are an expert rule engine for a game with knowledge about the game
world, relationships and rules. You will retrieve the knowledge about the
game world, relationships and rules through only the Knowledge Base provided below.

---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base. Verify if the user's input action is valid based on Conversation History. Decide the appropriate consequence of the given action through the provided context containing location information, inventory, surrounding details and observations about the game world.  

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{chat_history}


---Response Rules---

- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."""

# Enhanced prompt with stronger instruction
enhanced_prompt = custom_prompt 

# Define working directory
WORKING_DIR = os.path.join(".", "build")
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# Initialize session state for RAG instance and chat history
if 'rag_instance' not in st.session_state:
    st.session_state.rag_instance = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_genre' not in st.session_state:
    st.session_state.current_genre = None
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'available_genres' not in st.session_state:
    st.session_state.available_genres = []
if 'available_sessions' not in st.session_state:
    st.session_state.available_sessions = {}
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Define async functions
async def load_genre_rules(genre):
    """Load rule text for the specified genre."""
    genre_rules_file = os.path.join(WORKING_DIR, "genres", genre, f"{genre}_rules.txt")
    if not os.path.exists(genre_rules_file):
        raise FileNotFoundError(f"Rules file not found for genre '{genre}' at {genre_rules_file}")

    with open(genre_rules_file, "r") as file:
        rules_text = file.read()

    return rules_text

async def initialize_rag(working_dir):
    """Initialize the RAG instance with the specified working directory."""
    logging.info(f"Initializing LightRAG with working directory: {working_dir}")
    rag_instance = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name="deepseek-r1:14b",
        llm_model_max_async=2,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 10240},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda embeds: ollama_embed(embeds, embed_model="nomic-embed-text")
        )
    )

    # Initialize storages and pipeline status
    await asyncio.gather(
        rag_instance.initialize_storages(),
        initialize_pipeline_status()
    )
    
    return rag_instance

async def detect_existing_sessions(genre):
    """Detect all existing sessions for a given genre."""
    genre_dir = os.path.join(WORKING_DIR, genre)
    if not os.path.exists(genre_dir):
        return []

    # List all directories in the genre folder as potential sessions
    return [d for d in os.listdir(genre_dir) if os.path.isdir(os.path.join(genre_dir, d))]

async def detect_available_genres():
    """Detect all available genres."""
    if not os.path.exists(os.path.join(WORKING_DIR, "genres")):
        return []
    
    return [d for d in os.listdir(os.path.join(WORKING_DIR, "genres")) 
            if os.path.isdir(os.path.join(WORKING_DIR, "genres", d))]

async def initialize_session(genre, session):
    """Ensure the session folder for a genre has the required files or generate them."""
    # Ensure the genre has a working directory
    genre_dir = os.path.join(WORKING_DIR, genre)
    session_dir = os.path.join(genre_dir, session)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    # Initialize the RAG instance with the session directory
    if st.session_state.rag_instance is None:
        st.session_state.rag_instance = await initialize_rag(working_dir=session_dir)

    # Check if the session already exists in LightRAG storage
    if os.path.exists(session_dir) and not any(fname.endswith('.graphml') for fname in os.listdir(session_dir)):
        # Load the rules for the genre
        rules_text = await load_genre_rules(genre)
        logging.info(f"Initializing session '{session}' for genre '{genre}'...")
        
        insert_result = st.session_state.rag_instance.insert(rules_text)
        if inspect.isawaitable(insert_result):
            await insert_result  # Ensure the insertion is awaited if it's a coroutine
            
        logging.info(f"Session '{session}' for genre '{genre}' initialized.")
    else:
        logging.info(f"Session '{session}' for genre '{genre}' already exists.")

async def process_chat_input(user_input):
    """Process user chat input and generate response."""
    # Save original prompts
    original_rag_response = PROMPTS["rag_response"]
    original_naive_response = PROMPTS["naive_rag_response"]
    original_mix_response = PROMPTS["mix_rag_response"]
    
    # Replace with enhanced prompts
    PROMPTS["rag_response"] = enhanced_prompt
    PROMPTS["naive_rag_response"] = enhanced_prompt
    PROMPTS["mix_rag_response"] = enhanced_prompt
    
    # Query the LightRAG instance
    resp = st.session_state.rag_instance.query(
        user_input,
        param=QueryParam(mode="local", stream=True),
        system_prompt=enhanced_prompt
    )
    
    # Restore the original prompts
    PROMPTS["rag_response"] = original_rag_response
    PROMPTS["naive_rag_response"] = original_naive_response
    PROMPTS["mix_rag_response"] = original_mix_response

    response_text = ""
    if inspect.isasyncgen(resp):
        # Accumulate all chunks from the async generator
        async for chunk in resp:
            response_text += chunk
            # Update the placeholder with the latest chunk
            if 'response_placeholder' in st.session_state:
                st.session_state.response_placeholder.markdown(response_text)
    else:
        response_text = resp[0]

    # Extract text after </think> tags
    result_texts = re.findall(r'(?<=</think>).*', response_text)
    if result_texts:
        # Combine all extracted results into a single string
        extracted_text = " ".join(result_texts)
        logging.info(f"Inserting extracted text into the knowledge graph...")
        
        insert_result = st.session_state.rag_instance.insert(response_text)
        if inspect.isawaitable(insert_result):
            await insert_result  # Ensure the insertion is awaited if it's a coroutine
            
        logging.info(f"Insertion complete.")
    else:
        logging.error(f"No text found after </think> tags for insertion.")
    
    return response_text

# Extract knowledge graph from text content
def extract_entities_from_text(text):
    """Extract entities and relationships from text using simple patterns."""
    entities = []
    relations = []
    entity_map = {}
    
    # Patterns to extract entities - more comprehensive patterns to catch different ways entities are mentioned
    # Match patterns like: "There exists/is a <type> called/named <name>"
    entity_patterns = [
        r"(?:There exists|There is|I see|I found|you see|we have) an? (NPC|character|item|location|monster|object|weapon|place|person|enemy) (?:called|named) ([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+) is an? (NPC|character|item|location|monster|object|weapon|place|person|enemy)",
        r"an? (NPC|character|item|location|monster|object|weapon|place|person|enemy) named ([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+), (?:the|an?) (NPC|character|item|location|monster|object|weapon|place|person|enemy)"
    ]
    
    # Process each pattern to find entities
    for pattern in entity_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # The patterns might have different group orders
            if match.group(1).lower() in ["npc", "character", "item", "location", "monster", "object", "weapon", "place", "person", "enemy"]:
                entity_type = match.group(1).lower()
                entity_name = match.group(2)
            else:
                entity_name = match.group(1)
                entity_type = match.group(2).lower()
            
            entity_id = f"{entity_name.lower().replace(' ', '_')}"
            
            if entity_id not in entity_map:
                entity_map[entity_id] = {
                    "id": entity_id,
                    "type": entity_type,
                    "label": entity_name
                }
    
    # Relationship patterns - expanded to catch more relationship types
    relation_patterns = [
        r"([A-Za-z0-9_]+) (has|owns|contains|is|attacks|talks to|gives|receives|knows|likes|hates|loves|fears) ([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+) is (?:a|the) (friend|enemy|ally|foe|rival|companion|master|servant|owner) of ([A-Za-z0-9_]+)",
        r"([A-Za-z0-9_]+) (lives in|works at|guards|protects|visits|comes from) ([A-Za-z0-9_]+)"
    ]
    
    # Process each relationship pattern
    for pattern in relation_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            source = match.group(1).lower().replace(' ', '_')
            relation_type = match.group(2).lower()
            target = match.group(3).lower().replace(' ', '_')
            
            # Add source and target if they don't exist yet (with default type)
            if source not in entity_map:
                entity_map[source] = {
                    "id": source,
                    "type": "entity",
                    "label": source.replace('_', ' ').title()
                }
            
            if target not in entity_map:
                entity_map[target] = {
                    "id": target,
                    "type": "entity",
                    "label": target.replace('_', ' ').title()
                }
            
            # Add the relationship
            relations.append({
                "source": source,
                "target": target,
                "type": relation_type
            })
    
    # Process each user message specifically looking for entity creation statements
    for message in text.split('\n\n'):
        if "there exists" in message.lower() or "is a" in message.lower():
            # Find specific entity creation commands
            names = re.findall(r'\b([A-Z][a-z]+)\b', message)  # Find capitalized names
            for name in names:
                if len(name) > 2:  # Avoid catching short capitalized words
                    entity_id = name.lower()
                    if entity_id not in entity_map:
                        # Try to determine entity type from context
                        entity_type = "npc"  # Default type
                        if "item" in message.lower() or "object" in message.lower():
                            entity_type = "item"
                        elif "location" in message.lower() or "place" in message.lower():
                            entity_type = "location"
                        
                        entity_map[entity_id] = {
                            "id": entity_id,
                            "type": entity_type,
                            "label": name
                        }
    
    return list(entity_map.values()), relations

# Function to extract the knowledge graph from LightRAG
def extract_knowledge_graph():
    """Extract knowledge graph from the LightRAG instance for visualization."""
    if st.session_state.rag_instance is None:
        return None, None, None
    
    try:
        # Step 1: Try to extract entities and relationships from the entire chat history
        all_text = ""
        for role, message in st.session_state.chat_history:
            # Focus on AI responses as they contain most of the world knowledge
            if role == "assistant":
                # Remove the "hacked" prefix
                cleaned_message = re.sub(r'^hacked\s*', '', message, flags=re.IGNORECASE)
                # Remove think tags content
                cleaned_message = re.sub(r'<think>.*?</think>', '', cleaned_message, flags=re.DOTALL)
                all_text += cleaned_message + "\n\n"
        
        # Extract entities and relations from the text
        entities, relationships = extract_entities_from_text(all_text)
        
        # If we found entities from text processing, create a graph
        if entities:
            G = nx.DiGraph()
            
            # Add nodes
            for entity in entities:
                G.add_node(entity["id"], label=entity["label"], type=entity["type"])
            
            # Add edges
            for relation in relationships:
                if relation["source"] in G.nodes and relation["target"] in G.nodes:
                    G.add_edge(
                        relation["source"], 
                        relation["target"], 
                        type=relation["type"],
                        label=relation["type"]
                    )
            
            return G, entities, relationships
        
        # Step 2: Try to access the graph using LightRAG's API methods or direct access
        # If we couldn't extract anything from text, create a fallback graph from chat messages
        entities_map = {}
        relations_list = []
        
        # Create a mapping of message types to more descriptive labels
        message_type_mapping = {
            "user": "Player Action",
            "assistant": "Game Response"
        }
        
        for i, (role, message) in enumerate(st.session_state.chat_history):
            # Create a clean message without <think> tags for display
            if role == "assistant":
                # Remove think tags content
                clean_message = re.sub(r'<think>.*?</think>', '', message, flags=re.DOTALL)
                # Remove the "hacked" prefix
                clean_message = re.sub(r'^hacked\s*', '', clean_message, flags=re.IGNORECASE)
            else:
                clean_message = message
                
            # Truncate for display
            display_text = clean_message[:50] + "..." if len(clean_message) > 50 else clean_message
            
            # Add a game entity for this message
            message_id = f"message_{i}"
            entities_map[message_id] = {
                "id": message_id,
                "type": role,  # user or assistant
                "label": display_text
            }
            
            # If this message has a previous message, create a relationship
            if i > 0:
                prev_message_id = f"message_{i-1}"
                relation_type = "responds_to" if role == "assistant" else "follows"
                
                relations_list.append({
                    "source": message_id,
                    "target": prev_message_id,
                    "type": relation_type
                })
        
        # Convert to lists and create the graph
        entities = list(entities_map.values())
        relationships = relations_list
        
        # Create the graph
        G = nx.DiGraph()
        
        # Add nodes and format labels based on type
        for entity in entities:
            node_type = entity["type"]
            display_type = message_type_mapping.get(node_type, node_type.capitalize())
            display_label = f"{display_type}: {entity['label']}"
            
            G.add_node(entity["id"], label=display_label, type=node_type)
        
        # Add edges
        for relation in relationships:
            G.add_edge(
                relation["source"], 
                relation["target"], 
                type=relation["type"],
                label=relation["type"]
            )
        
        if len(G.nodes) > 0:
            return G, entities, relationships
                
        # If we reach here, no valid graph was found
        return None, None, None
    except Exception as e:
        st.error(f"Error extracting knowledge graph: {str(e)}")
        logging.error(f"Error in extract_knowledge_graph: {str(e)}")
        return None, None, None

# Function to create a PyVis network visualization
def create_pyvis_network(G):
    """Create a PyVis network visualization from a NetworkX graph."""
    # Create a PyVis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Set physics layout
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250, spring_strength=0.001, damping=0.09)
    
    # Add nodes with different colors based on type
    node_colors = {
        "entity": "#4287f5",  # Blue
        "concept": "#f54242",  # Red
        "action": "#42f54e",   # Green
        "location": "#f5a442", # Orange
        "item": "#f542f2",     # Pink
        "rule": "#f5e642",     # Yellow
        "unknown": "#808080"   # Gray
    }

    # Add nodes to the network
    for node_id in G.nodes():
        node_type = G.nodes[node_id].get('type', 'unknown')
        # Get the label if available, otherwise use a formatted version of the ID
        label = G.nodes[node_id].get('label', node_id)
        if isinstance(label, str):
            # Format the label - add the node type for non-conversation nodes
            if node_type not in ["user", "assistant"]:
                display_label = f"{label} ({node_type})"
            else:
                # For conversation nodes, truncate the label
                message_text = label[:30] + "..." if len(label) > 30 else label
                display_label = f"{node_type.capitalize()}: {message_text}"
        else:
            # If the label is not a string, use the node ID
            display_label = str(node_id)
            
        net.add_node(
            node_id, 
            label=display_label, 
            title=f"Type: {node_type}\nID: {node_id}", 
            color=node_colors.get(node_type, node_colors['unknown'])
        )
    
    # Add edges to the network
    for source, target, edge_attrs in G.edges(data=True):
        relation = edge_attrs.get('label', 'related_to')
        net.add_edge(source, target, title=relation, label=relation)
    
    # Generate HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_path = temp_file.name
        net.save_graph(temp_path)
    
    return temp_path

# Function to create StreamlitAgraph visualization
def create_agraph_visualization(entities, relationships):
    """Create a StreamlitAgraph visualization from entities and relationships."""
    # Define node colors based on type
    node_colors = {
        "entity": "#4287f5",  # Blue
        "concept": "#f54242",  # Red
        "action": "#42f54e",   # Green
        "location": "#f5a442", # Orange
        "item": "#f542f2",     # Pink
        "rule": "#f5e642",     # Yellow
        "unknown": "#808080"   # Gray
    }
    
    # Create nodes
    nodes = []
    for entity in entities:
        node_type = entity.get("type", "unknown")
        # Use the 'label' field or fall back to a formatted version of the ID
        display_label = entity.get("label", entity["id"].replace('_', ' ').title())
        
        # Add node type to the label if it's not a conversation message
        if node_type not in ["user", "assistant"]:
            display_label = f"{display_label} ({node_type})"
            
        nodes.append(
            Node(
                id=entity["id"],
                label=display_label,
                size=25,
                color=node_colors.get(node_type, node_colors['unknown'])
            )
        )
    
    # Create edges
    edges = []
    for relation in relationships:
        edges.append(
            Edge(
                source=relation["source"],
                target=relation["target"],
                label=relation["type"]
            )
        )
    
    # Configure the graph
    config = Config(
        width=1000,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True
    )
    
    return nodes, edges, config

# Streamlit app layout
st.title("🎮 LightRAG Game Master")
st.markdown("An AI game master that responds based on game genre rules and maintains history.")

# Create tabs for chat and knowledge graph visualization
chat_tab, graph_tab = st.tabs(["Chat", "Knowledge Graph"])

# Sidebar for configuration
st.sidebar.title("Game Settings")

# Initialize the app if needed
async def init_app():
    if not st.session_state.initialized:
        # Detect available genres
        st.session_state.available_genres = await detect_available_genres()
        for genre in st.session_state.available_genres:
            sessions = await detect_existing_sessions(genre)
            st.session_state.available_sessions[genre] = sessions
        st.session_state.initialized = True

# Run initialization in a non-blocking way
if not st.session_state.initialized:
    asyncio.run(init_app())

# Genre selection
if st.session_state.available_genres:
    genre = st.sidebar.selectbox(
        "Select Game Genre:",
        st.session_state.available_genres,
        index=0 if st.session_state.current_genre is None else st.session_state.available_genres.index(st.session_state.current_genre)
    )
    
    # Update sessions when genre changes
    if genre != st.session_state.current_genre:
        st.session_state.current_genre = genre
        if genre not in st.session_state.available_sessions:
            st.session_state.available_sessions[genre] = asyncio.run(detect_existing_sessions(genre))
        st.rerun()
    
    # Session selection or creation
    if st.session_state.available_sessions.get(genre, []):
        session_options = st.session_state.available_sessions[genre] + ["Create New Session"]
        session_selection = st.sidebar.selectbox(
            "Select or Create Session:",
            session_options,
            index=0 if st.session_state.current_session is None else 
                  session_options.index(st.session_state.current_session) if st.session_state.current_session in session_options else 0
        )
        
        if session_selection == "Create New Session":
            new_session_name = st.sidebar.text_input("Enter new session name:", value="session" + str(len(st.session_state.available_sessions[genre]) + 1))
            create_pressed = st.sidebar.button("Create Session")
            
            if create_pressed and new_session_name:
                if new_session_name not in st.session_state.available_sessions[genre]:
                    asyncio.run(initialize_session(genre, new_session_name))
                    st.session_state.available_sessions[genre].append(new_session_name)
                    st.session_state.current_session = new_session_name
                    st.session_state.chat_history = []
                    st.success(f"New session '{new_session_name}' created!")
                    st.rerun()
                else:
                    st.error(f"Session '{new_session_name}' already exists!")
        else:
            if session_selection != st.session_state.current_session:
                st.session_state.current_session = session_selection
                asyncio.run(initialize_session(genre, session_selection))
                st.session_state.chat_history = []
                st.rerun()
    else:
        new_session_name = st.sidebar.text_input("Enter session name:", value="session1")
        create_pressed = st.sidebar.button("Create First Session")
        
        if create_pressed and new_session_name:
            asyncio.run(initialize_session(genre, new_session_name))
            st.session_state.available_sessions[genre] = [new_session_name]
            st.session_state.current_session = new_session_name
            st.session_state.chat_history = []
            st.success(f"Session '{new_session_name}' created for genre '{genre}'!")
            st.rerun()
else:
    st.sidebar.error("No game genres found. Please create a genre folder with rules.")

# Display current genre rules if a genre is selected
if st.session_state.current_genre:
    with st.sidebar.expander("View Genre Rules"):
        try:
            rules_text = asyncio.run(load_genre_rules(st.session_state.current_genre))
            st.markdown(rules_text)
        except FileNotFoundError as e:
            st.error(str(e))

# Main interface with tabs
if st.session_state.current_genre and st.session_state.current_session:
    st.markdown(f"### 🎲 Playing: {st.session_state.current_genre.upper()} - Session: {st.session_state.current_session}")
    
    # Chat tab content
    with chat_tab:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f"**You:** {message}")
                else:
                    # Format the AI response
                    message_clean = message
                    st.markdown(f"**Game Master:** {message_clean}")
        
        # Create a form for user input
        with st.form(key="chat_form", clear_on_submit=True):
            user_message = st.text_area("Your action:", height=100)
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_message:
                # Add user message to chat history
                st.session_state.chat_history.append(("user", user_message))
                
                # Create a placeholder for streaming response outside of the form
                st.session_state.response_placeholder = st.empty()
                
                # Set flag to process after form submission
                st.session_state.process_input = user_message
                
        # Process input after form submission (outside the form)
        if 'process_input' in st.session_state and st.session_state.process_input:
            with st.spinner("Game Master is thinking..."):
                response = asyncio.run(process_chat_input(st.session_state.process_input))
            
            # Add AI response to chat history
            st.session_state.chat_history.append(("assistant", response))
            
            # Clear the processing flag
            st.session_state.process_input = None
            
            # Rerun to refresh the chat display
            st.rerun()
    
    # Knowledge Graph tab content
    with graph_tab:
        st.markdown("### 🌐 Knowledge Graph Visualization")
        st.markdown("This visualization shows the entities and relationships in the game world.")
        
        # Add refresh button
        if st.button("🔄 Refresh Knowledge Graph"):
            st.rerun()
        
        # Extract knowledge graph
        G, entities, relationships = extract_knowledge_graph()
        
        if G is None or len(G.nodes()) == 0:
            st.info("No knowledge graph data available yet. Start chatting to build the knowledge graph!")
        else:
            # Display graph statistics
            st.markdown(f"**Graph Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entities", len(G.nodes()))
            with col2:
                st.metric("Relationships", len(G.edges()))
            with col3:
                # Count different entity types
                entity_types = {}
                for node_id, attrs in G.nodes(data=True):
                    node_type = attrs.get('type', 'unknown')
                    entity_types[node_type] = entity_types.get(node_type, 0) + 1
                st.metric("Entity Types", len(entity_types))
            
            # Display visualization options
            viz_option = st.radio(
                "Select Visualization",
                ["Interactive Graph", "Network Graph"],
                horizontal=True
            )
            
            if viz_option == "Interactive Graph":
                # Create StreamlitAgraph visualization
                nodes, edges, config = create_agraph_visualization(entities, relationships)
                
                # Display the graph
                with st.container():
                    st.markdown("#### Interactive Knowledge Graph")
                    agraph(nodes=nodes, edges=edges, config=config)
                    
            else:  # Network Graph
                # Create PyVis network visualization
                html_path = create_pyvis_network(G)
                
                # Display the graph in an iframe
                with st.container():
                    st.markdown("#### Network Knowledge Graph")
                    st.components.v1.html(open(html_path, 'r', encoding='utf-8').read(), height=600)
            
            # Display entity and relationship tables
            with st.expander("View Entities"):
                entity_df = pd.DataFrame(entities)
                st.dataframe(entity_df, use_container_width=True)
                
            with st.expander("View Relationships"):
                rel_df = pd.DataFrame(relationships)
                st.dataframe(rel_df, use_container_width=True)
else:
    st.info("Please select a genre and session from the sidebar to start chatting.")

# Footer
st.markdown("---")
st.caption("Built with LightRAG - An AI-powered game master")
