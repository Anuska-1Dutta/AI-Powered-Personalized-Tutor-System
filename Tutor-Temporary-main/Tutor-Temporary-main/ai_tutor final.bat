@echo off
echo ===============================================
echo            AI TUTOR APPLICATION
echo ===============================================
echo.
echo This application provides personalized tutoring
echo across multiple subjects using AI technology.
echo.

:: Check Python installation
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing required packages...
pip install streamlit
pip install scikit-learn
pip install numpy
pip install pandas
pip install scipy

echo.
echo Starting AI Tutor Application...
echo.

:: Try to run streamlit directly
python -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo Failed to run streamlit using normal method
    echo Trying alternative method...
    echo.
    
    :: Try to find where streamlit is installed
    for /f "tokens=*" %%i in ('where python') do set PYTHON_PATH=%%i
    set STREAMLIT_PATH=%PYTHON_PATH:python.exe=Scripts\streamlit.exe%
    
    echo Found Python at: %PYTHON_PATH%
    echo Looking for streamlit at: %STREAMLIT_PATH%
    
    if exist "%STREAMLIT_PATH%" (
        echo Found streamlit, running application...
        "%STREAMLIT_PATH%" run app.py
    ) else (
        echo.
        echo ERROR: Could not find streamlit executable.
        echo Please try running this command manually:
        echo python -m streamlit run app.py
    )
)

echo.
echo Application closed.
pause 