import streamlit as st
import requests
import os
import uuid

if os.getenv("DOCKER_ENV") == "true":
    BACKEND_URL = "http://backend:8000"
else:
    BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Documentation RAG Agent",
    layout="centered"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("About this System")
    st.markdown("""
This Agent uses a RAG architecture to answer questions regarding some code documentation
**How to**
1. Input the url of some code documentation
2. Ask the AI any question regarding this documentation
""")
    
st.caption(f"Session-ID: {st.session_state.session_id}")
st.caption("Data will be deleted after 2 hours of inactivity")
st.markdown("---")

st.title("DOCUMENTATION RAG AGENT")
st.subheader("Upload documentation url")

col1, col2 = st.columns([3,1])

with col1:
    target_url = st.text_input("Documentation Url:", placeholder="https://...",label_visibility="collapsed")
with col2:
    scrape_button = st.button("Upload Documentation")

if scrape_button:
    if target_url:
        with st.spinner():
            payload = {
                "url": target_url,
                "session_id": st.session_state.session_id
            }

            try:
                response = requests.post(f"{BACKEND_URL}/scrape", json=payload)
                result = response.json()

                if response.status_code == 200 and "Error" not in result:
                    st.success("Successfully loaded documentation to database.")
                else:
                    st.error(f"Something went wrong when uploading documentation to database: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to backend: {str(e)}")
    else:
        st.warning("Please input an Url first")
st.markdown("---")

st.header("Ask a question")
user_question = st.text_area("What do you want to know", placeholder="Tell me how to ...")

if st.button("Answer Question", type="primary"):
    if user_question:
        with st.spinner():
            payload = {
                "question": user_question,
                "session_id": st.session_state.session_id
            }
        try:
            response = requests.post(f"{BACKEND_URL}/ask", json=payload)
            result = response.json()

            if response.status_code == 200 and "Error" not in result:
                st.write("### Response:")
                st.info(result.get("Response"))
                st.caption(f"Used {result.get('Chunks')} for the response")
            else:
                st.error(f"Something went wrong when trying to answer your question: {result.get('Response', result.get('Error', 'unknown error'))}")
        except Exception as e:
            st.error(f"Could not connect to backend: {str(e)}")
    else:
        st.warning("Please ask a question first")
