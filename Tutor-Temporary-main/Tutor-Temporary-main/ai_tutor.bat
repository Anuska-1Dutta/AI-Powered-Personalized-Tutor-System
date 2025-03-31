@echo off
setlocal EnableDelayedExpansion

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
    echo Python is not installed. Please install Python 3.10 from:
    echo https://www.python.org/downloads/release/python-3109/
    pause
    exit /b 1
)

:: Set up virtual environment
if not exist .venv (
    echo Creating Python virtual environment...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

:: Activate venv
echo Activating virtual environment...
call .venv\Scripts\activate.bat

:: Create folders
echo Creating necessary directories...
mkdir model 2>nul
mkdir data 2>nul
mkdir data\users 2>nul
mkdir logs 2>nul

:: Install requirements
echo Checking and installing required dependencies...
pip install --only-binary :all: streamlit==1.32.0
pip install --only-binary :all: numpy==1.24.3
pip install --only-binary :all: pandas==2.0.3
pip install --only-binary :all: scipy==1.10.1
pip install --only-binary :all: scikit-learn==1.3.0

:: Check for model file
if not exist model\ai_tutor_model.pkl (
    echo AI Tutor basic model not found.
    
    echo:
    echo Do you want to create the basic model (B) or the enhanced large model (L)?
    echo B = Basic model (faster but less comprehensive)
    echo L = Large model (more comprehensive but takes longer to create)
    echo:
    set /p model_choice="Enter your choice (B/L) [default: L]: "
    
    if /i "!model_choice!"=="B" (
        echo Creating basic AI Tutor model...
        python download_dataset.py
        if !errorlevel! neq 0 (
            echo Failed to create basic model.
            pause
            exit /b 1
        )
    ) else (
        echo Creating enhanced large AI Tutor model...
        python download_large_dataset.py
        if !errorlevel! neq 0 (
            echo Failed to create enhanced large model.
            pause
            exit /b 1
        )
    )
) else (
    echo AI Tutor model found.
    
    echo:
    echo Do you want to recreate all subject datasets?
    echo This will enhance the AI Tutor's knowledge of all subjects.
    echo:
    set /p enhance_choice="Recreate all subject datasets? (Y/N) [default: N]: "
    
    if /i "!enhance_choice!"=="Y" (
        echo Enhancing all subject datasets...
        echo Creating enhanced math dataset...
        python create_math_dataset.py
        echo Creating enhanced science dataset...
        python create_science_dataset.py
        echo Creating enhanced history dataset...
        python create_history_dataset.py
        echo Creating enhanced programming dataset...
        python create_programming_dataset.py
        echo All subject datasets have been enhanced.
    )
)

:: Set up log file
set log_file=logs\ai_tutor_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set log_file=%log_file: =0%

echo:
echo How do you want to run the application?
echo 1 = Normal mode (opens browser automatically)
echo 2 = Headless mode with logs (runs in background, logs to file)
echo:
set /p run_mode="Enter your choice (1/2) [default: 1]: "

if "!run_mode!"=="2" (
    echo Starting AI Tutor in headless mode with logging...
    echo Log file: %log_file%
    
    :: Run in background
    start /B cmd /c streamlit run app.py --server.headless=true > %log_file% 2>&1
    
    :: Wait for startup
    echo Waiting for server to start...
    ping 127.0.0.1 -n 6 > nul
    
    :: Open browser
    echo Opening browser...
    start http://localhost:8501
    
    echo:
    echo AI Tutor is running in the background.
    echo Press any key to stop the application.
    pause
    
    :: Cleanup
    taskkill /f /im streamlit.exe > nul 2>&1
    taskkill /f /im python.exe > nul 2>&1
    
    echo:
    echo AI Tutor has stopped.
    echo Logs saved to: %log_file%
) else (
    echo:
    echo Starting AI Tutor in normal mode...
    echo:
    echo The application will open in your browser automatically.
    echo Sign up with any name and a password (8+ characters)
    echo:
    echo (Press Ctrl+C to stop the application when done)
    echo:
    streamlit run app.py
)

:: Clean up
call .venv\Scripts\deactivate.bat

echo:
echo Thank you for using AI Tutor!
pause
endlocal 