#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  ./deploy.sh [platform]"
    echo -e "${BLUE}Platforms:${NC}"
    echo "  docker    - Build and run using Docker"
    echo "  render    - Deploy to Render.com"
    echo "  heroku    - Deploy to Heroku"
    echo "  railway   - Deploy to Railway.app"
}

# Function to deploy using Docker
deploy_docker() {
    echo -e "${GREEN}Building Docker image...${NC}"
    docker build -t forcastapp .
    
    echo -e "${GREEN}Running container...${NC}"
    docker run -p 8501:8501 forcastapp
}

# Function to deploy to Render
deploy_render() {
    echo -e "${GREEN}Deploying to Render...${NC}"
    echo "1. Create a new Web Service on render.com"
    echo "2. Connect your GitHub repository"
    echo "3. Use the following settings:"
    echo "   - Environment: Python"
    echo "   - Build Command: pip install -r requirements.txt"
    echo "   - Start Command: streamlit run app/test_app.py --server.address 0.0.0.0 --server.port \$PORT"
    echo "4. Click 'Create Web Service'"
}

# Function to deploy to Heroku
deploy_heroku() {
    echo -e "${GREEN}Deploying to Heroku...${NC}"
    if ! command -v heroku &> /dev/null; then
        echo "Please install the Heroku CLI first"
        exit 1
    fi
    
    heroku create forcastapp || true
    git push heroku master
}

# Function to deploy to Railway
deploy_railway() {
    echo -e "${GREEN}Deploying to Railway...${NC}"
    echo "1. Visit railway.app and create a new project"
    echo "2. Connect your GitHub repository"
    echo "3. Add the following environment variables:"
    echo "   PORT=8501"
    echo "4. Railway will automatically deploy your app"
}

# Main script logic
case "$1" in
    "docker")
        deploy_docker
        ;;
    "render")
        deploy_render
        ;;
    "heroku")
        deploy_heroku
        ;;
    "railway")
        deploy_railway
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
