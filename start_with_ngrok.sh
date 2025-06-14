#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Streamlit app with Ngrok...${NC}"

# Install requirements if not already installed
pip install -r requirements.txt

# Run the Streamlit app with Ngrok in the background
python app/share_with_ngrok.py &
NGROK_PID=$!

# Start the Streamlit app
streamlit run app/test_app.py

# Clean up when the script exits
trap "kill $NGROK_PID" EXIT
