import os
import streamlit as st
from pyngrok import ngrok
import logging

def run_streamlit_with_ngrok():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Get Ngrok auth token from environment variable or input
        ngrok_token = os.getenv('NGROK_AUTH_TOKEN')
        if not ngrok_token:
            logger.info("No Ngrok auth token found in environment variables.")
            logger.info("Get your auth token from https://dashboard.ngrok.com/auth")
            ngrok_token = input("Please enter your ngrok auth token: ").strip()
        
        # Configure ngrok
        ngrok.set_auth_token(ngrok_token)
        
        # Start ngrok tunnel to port 8501 (default Streamlit port)
        public_url = ngrok.connect(8501).public_url
        logger.info(f"Ngrok tunnel created! Your Streamlit app is available at: {public_url}")
        
        # Update Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        # Return the public URL
        return public_url
        
    except Exception as e:
        logger.error(f"Error setting up ngrok: {str(e)}")
        raise

if __name__ == "__main__":
    run_streamlit_with_ngrok()
