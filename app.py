import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer, util
import time

# Set page config for a welcoming animation
st.set_page_config(page_title="Daraz Laptop Search", page_icon="💻", layout="centered")

# Load the joblib file containing precomputed data and embeddings
data = joblib.load('product_data_embeddings.joblib')

# Extract preloaded data from joblib file
titles = data['titles']
prices = data['prices']
ratings = data['ratings']
colors = data['colors']
links = data['links']
embeddings = data['embeddings']

# Load the SentenceTransformer model (LaBSE)
model = SentenceTransformer('LaBSE')

# Define the search function
def search_product(query, threshold=0.3):
    # Encode the user's query using the LaBSE model
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Perform semantic search to find the closest embeddings
    hits = util.semantic_search(query_embedding, embeddings)[0]
    
    # Sort hits by similarity score (descending order)
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    
    # Filter results based on a threshold
    top_hits = [hit for hit in hits if hit['score'] >= threshold]
    
    return top_hits

# Streamlit App Layout with animation
st.title("💻 Semantic Search of Daraz Laptop")
st.markdown("<h3 style='color: #4CAF50;'>Search for the best laptops with AI-powered search!</h3>", unsafe_allow_html=True)

# Brief welcome animation using a placeholder
with st.spinner('Loading AI-powered search engine...'):
    time.sleep(1)

# Centered input box and button layout using columns
col1, col2, col3 = st.columns([1, 2, 1])  # Creates three columns; col2 is wider for centering

with col2:
    query = st.text_input("Enter a product search query", "")
    search_button = st.button("Search", use_container_width=True)

# Search button logic
if search_button:
    start_time = time.time()  # Track search start time
    
    # Progress bar animation while performing search
    progress_bar = st.progress(0)
    
    # Simulate progress
    for percent_complete in range(1, 101, 20):
        time.sleep(0.1)
        progress_bar.progress(percent_complete)
    
    # Perform the search
    results = search_product(query)
    
    end_time = time.time()  # Track search end time
    elapsed_time = end_time - start_time
    
    # Display search time with a success message
    st.success(f"Search completed in {elapsed_time:.3f} seconds!")
    
    # Display the top 5 results
    if results:
        st.markdown(f"<h4 style='color: #2196F3;'>Top matches for: '{query}'</h4>", unsafe_allow_html=True)
        
        for hit in results[:5]:  # Limit to top 5 results
            index = hit['corpus_id']
            st.markdown(f"### **🔹 Product title**: {titles[index]}")
            st.markdown(f"<h5 style='color: #FFC107;'>Price: NRP {prices[index]}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h6 style='color: #4CAF50;'>Rating: {ratings[index]}</h6>", unsafe_allow_html=True)
            st.markdown(f"**Color**: {colors[index]}")
            st.markdown(f"[Link to Product]({links[index]})", unsafe_allow_html=True)
            st.markdown(f"<i>Relevance Score</i>: {hit['score']:.3f}", unsafe_allow_html=True)
            st.markdown("---")  # Separator for readability
    else:
        st.error("No matching product found.")