import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Music Recommender", layout="wide")

st.title("🎵 Guts of Darkness Recommender")

# Sidebar for status
with st.sidebar:
    st.header("Status")
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                st.success(f"API: {data.get('status')}")
                st.info(f"Milvus: {data.get('milvus')}")
                st.info(f"Model: {data.get('model')}")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

# Main Search
st.header("Search")
query = st.text_input("Describe what you want to listen to:", placeholder="Ex: dark ambient industrial, atmospheric black metal...")
top_k = st.slider("Number of results", 1, 50, 10)

if st.button("Search", key="btn_search"):
    if query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(f"{API_URL}/search", params={"query": query, "top_k": top_k})
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        for album in results:
                            with st.expander(f"{album['title']} - {album['artist']} (Score: {album['distance']:.2f})"):
                                st.write(f"**ID:** {album['id']}")
                                # Add more fields here if available in 'extra'
                    else:
                        st.warning("No results found.")
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("Please enter a query.")
