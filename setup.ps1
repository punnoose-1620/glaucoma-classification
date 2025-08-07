# PowerShell setup script for glaucoma classification project

Write-Host "Setting up glaucoma classification project environment..." -ForegroundColor Green

# Remove existing venv if it exists
if (Test-Path "venv") {
    Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create virtual environment with Python 3.10 or available Python version
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv venv
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error creating virtual environment. Please ensure Python is installed and in PATH." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create necessary directories if they don't exist
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
if (!(Test-Path "visualizations")) { New-Item -ItemType Directory -Name "visualizations" }
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Name "logs" }
if (!(Test-Path "models")) { New-Item -ItemType Directory -Name "models" }

Write-Host "Directories created successfully!" -ForegroundColor Green

# Set up Kaggle credentials
Write-Host "`nKaggle API Setup Instructions:" -ForegroundColor Cyan
Write-Host "1. Download your Kaggle API credentials (kaggle.json) from https://www.kaggle.com/settings/account" -ForegroundColor White
Write-Host "2. Place the kaggle.json file in your user directory: $env:USERPROFILE\.kaggle\" -ForegroundColor White
Write-Host "3. Set proper permissions: icacls `"$env:USERPROFILE\.kaggle\kaggle.json`" /inheritance:r /grant:r `"$env:USERNAME`:F`"" -ForegroundColor White

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the virtual environment in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan 