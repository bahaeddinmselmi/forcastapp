import os
import sys
import time
from pyngrok import ngrok, conf

# The auth token is already configured, no need to set it here
# Your token has been saved to: C:\Users\Administrator\AppData\Local/ngrok/ngrok.yml

# Define the port where your Streamlit app is running
port = 8501  # Streamlit's default port

# Create a tunnel to your local Streamlit app
ngrok_tunnel = ngrok.connect(port)
public_url = ngrok_tunnel.public_url

# Print the public URL
print("\n==================================================")
print(f"*** YOUR IBP APP IS NOW AVAILABLE ONLINE! ***")
print("==================================================")
print(f"Share this URL with your friend: {public_url}")
print("They can access your app from anywhere, no port forwarding needed!")
print("The app will remain accessible as long as this window stays open.")
print("Press Ctrl+C to stop sharing when you're done.")
print("==================================================\n")

# Keep the script running
try:
    while True:
        # Continuously print the URL every 60 seconds as a reminder
        time.sleep(60)
        print(f"Reminder - Your app is still accessible at: {public_url}")
except KeyboardInterrupt:
    print("Shutting down the ngrok tunnel...")
    ngrok.kill()
    print("Tunnel closed. Your app is no longer accessible from the internet.")
