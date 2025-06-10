@echo off
echo 🌱 CropAssist Application Launcher
echo =====================================
echo.
echo Starting the CropAssist Flask Application...
echo This includes:
echo   - Fertilizer Prediction System
echo   - Crop Recommendation Engine  
echo   - Plant Disease Detection
echo   - User Management System
echo.
echo 🌐 Server will be available at: http://127.0.0.1:5000
echo ⏳ Please wait while models are loading...
echo.
echo =====================================
echo.

cd /d "%~dp0"
python app.py

echo.
echo 👋 Thank you for using CropAssist!
pause
