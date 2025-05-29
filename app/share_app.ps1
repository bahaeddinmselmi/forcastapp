# Activate the virtual environment
& "C:\Users\Public\Downloads\ibp\dd\venv310\Scripts\Activate.ps1"

# Start Streamlit server in the background
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m streamlit run main.py --server.port=8507 --server.address=0.0.0.0" -WorkingDirectory "C:\Users\Public\Downloads\ibp\dd\app"

# Give Streamlit a moment to start
Start-Sleep -Seconds 5

# Use localhost.run to create a tunnel (this service requires no account)
Write-Host "Creating public tunnel with localhost.run..."
Write-Host "Your friend will be able to access your app using the URL shown below."
Write-Host "When you want to stop sharing, press Ctrl+C in this window"

# Forward the local port to a public URL using SSH
ssh -R 80:localhost:8507 localhost.run
