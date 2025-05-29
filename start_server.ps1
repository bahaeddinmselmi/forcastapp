Write-Host "===========================================" -ForegroundColor Green
Write-Host "STARTING IBP SERVER FOR EXTERNAL ACCESS" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Activate virtual environment
& "$PSScriptRoot\venv310\Scripts\Activate.ps1"

# Show public IP information
Write-Host ""
Write-Host "Public IP Information:" -ForegroundColor Cyan
$publicIP = Invoke-RestMethod -Uri "https://api.ipify.org?format=json" | Select-Object -ExpandProperty ip
Write-Host "Your public IP address: $publicIP" -ForegroundColor Yellow
Write-Host "Port: 8510" -ForegroundColor Yellow
Write-Host ""
Write-Host "SHARE THIS URL WITH YOUR FRIEND:" -ForegroundColor Green
Write-Host "http://$publicIP:8510" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: For this to work, you need to set up port forwarding on your router" -ForegroundColor Red
Write-Host "Forward external port 8510 to internal port 8510 (TCP)" -ForegroundColor Red
Write-Host ""
Write-Host "Press Ctrl+C to stop the server when done." -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Green

# Start the Streamlit server
cd "$PSScriptRoot\app"
python -m streamlit run main.py
