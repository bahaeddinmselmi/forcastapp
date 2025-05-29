import os
import streamlit as st
from pyngrok import ngrok, conf

# The free authtoken allows usage without an account
# This is a temporary token just for demonstration
conf.get_default().auth_token = "2b16UmVyQYSFB4cNR7x2x0KxZA2_7mnTNx6TnPBjy4PaHpYXR" 

# Set up a tunnel for the streamlit port (8508)
ngrok_tunnel = ngrok.connect(8508)

# Print the public URL
public_url = ngrok_tunnel.public_url
print(f"\n\n==================================================")
print(f"🌐 PUBLIC URL FOR YOUR FRIEND: {public_url}")
print(f"==================================================\n")
print(f"Share this URL with your friend. They can access your app from anywhere!")
print(f"The app will remain accessible as long as this window stays open.")
print(f"Press Ctrl+C to stop sharing.\n")

# Keep the script running
try:
    # Block until the user presses Ctrl+C
    ngrok_process = ngrok.get_ngrok_process()
    ngrok_process.proc.wait()
except KeyboardInterrupt:
    print("Shutting down the ngrok tunnel...")
    ngrok.kill()
