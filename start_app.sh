#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Install requirements if needed
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt

# Start the Streamlit app with ngrok
echo -e "${GREEN}Starting Streamlit app with ngrok...${NC}"
python app/run_with_ngrok.py
