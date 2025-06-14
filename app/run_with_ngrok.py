import os
import streamlit.web.bootstrap as bootstrap
from pyngrok import ngrok, conf

def run_streamlit_with_ngrok():
    # Configure ngrok
    ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN', ''))  # Set your auth token if you have one
    
    # Start ngrok tunnel to port 8501 (default Streamlit port)
    public_url = ngrok.connect(8501).public_url
    print(f"\n👉 Streamlit app URL: {public_url}")

    # Run Streamlit app
    bootstrap.run("test_app.py", "", [], [])

if __name__ == "__main__":
    run_streamlit_with_ngrok()
