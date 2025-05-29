# This script starts the IBP server and checks if the port is accessible

# Go to the IBP directory
cd "C:\Users\Public\Downloads\ibp\dd"

# Activate the virtual environment
.\venv310\Scripts\Activate.ps1

# Get public IP address
$publicIP = (Invoke-RestMethod -Uri "https://api.ipify.org?format=json").ip
Write-Host "Your public IP: $publicIP" -ForegroundColor Green

# Check if port 8510 is open
Write-Host "Checking if port 8510 is already in use..." -ForegroundColor Yellow
$portCheck = Test-NetConnection -ComputerName localhost -Port 8510 -InformationLevel Quiet -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
if ($portCheck) {
    Write-Host "WARNING: Port 8510 is already in use. Using an alternative port." -ForegroundColor Red
    $port = 8520
} else {
    $port = 8510
}

# Show sharing information
Write-Host "`n===================================" -ForegroundColor Cyan
Write-Host "IBP SERVER INFORMATION" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Local URL: http://localhost:$port" -ForegroundColor White
Write-Host "Network URL: http://192.168.1.12:$port" -ForegroundColor White
Write-Host "`nSHARE WITH YOUR FRIEND:"
Write-Host "http://$($publicIP):$port" -ForegroundColor Green
Write-Host "`nREMINDER: You need to set up port forwarding on your router!" -ForegroundColor Yellow
Write-Host "Forward external port $port to internal IP 192.168.1.12 port $port" -ForegroundColor Yellow
Write-Host "===================================" -ForegroundColor Cyan

# Start Streamlit server
cd "C:\Users\Public\Downloads\ibp\dd\app"
python -m streamlit run main.py --server.port=$port --server.address=0.0.0.0
