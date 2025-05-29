# Simple sharing script that uses serveo.net (no authentication needed)

# Activate virtual environment
& "$PSScriptRoot\..\venv310\Scripts\Activate.ps1"

# Show information banner
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "CREATING PUBLIC URL FOR YOUR IBP APPLICATION" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will create a public URL that your friend can access." -ForegroundColor Yellow
Write-Host "When prompted about host keys, type 'yes' and press Enter." -ForegroundColor Yellow
Write-Host ""
Write-Host "The public URL will be displayed after connecting." -ForegroundColor Green
Write-Host "Share that URL with your friend." -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C when you're done sharing." -ForegroundColor Red
Write-Host "======================================================" -ForegroundColor Cyan

# Use PowerShell to create a reverse SSH tunnel with serveo.net
ssh -R ibp-app:80:localhost:8510 serveo.net
