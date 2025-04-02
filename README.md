<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Tutor App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #2c3e50;
        }
        p {
            color: #333;
        }
        .section {
            background: #fff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .highlight {
            font-weight: bold;
            color: #e74c3c;
        }
        .code-block {
            background: #2d2d2d;
            color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            overflow-x: auto;
        }
        .link {
            color: #3498db;
            font-weight: bold;
            text-decoration: none;
        }
    </style>
</head>
<body>

    <h1>AI Tutor App</h1>
    <p>A simple AI-powered learning platform built with <span class="highlight">Streamlit</span>, providing personalized tutoring across multiple subjects using our self-trained AI model.</p>

    <div class="section">
        <h2>Features</h2>
        <ul>
            <li><span class="highlight">AI-Powered Learning:</span> Interact with an AI tutor trained on educational content</li>
            <li><span class="highlight">Multiple Subjects:</span> Access tutoring in Mathematics, Science, History, and Programming</li>
            <li><span class="highlight">Progress Tracking:</span> View your learning progress across different subjects</li>
            <li><span class="highlight">User Authentication:</span> Simple login system to track individual progress</li>
        </ul>
    </div>

    <div class="section">
        <h2>Tech Stack</h2>
        <ul>
            <li><span class="highlight">Frontend & Backend:</span> Streamlit</li>
            <li><span class="highlight">AI Model:</span> Scikit-learn (TF-IDF + Random Forest classifier)</li>
            <li><span class="highlight">Styling:</span> Tailwind CSS (via CDN)</li>
            <li><span class="highlight">Data Storage:</span> Local file storage (pickle files)</li>
        </ul>
    </div>

    <div class="section">
        <h2>Getting Started</h2>
        <h3>Prerequisites</h3>
        <ul>
            <li>Python 3.8 or later</li>
            <li>pip (Python package manager)</li>
        </ul>

        <h3>Installation</h3>
        <p>1. Clone the repository</p>
        <div class="code-block">
            git clone https://github.com/yourusername/ai-tutor.git<br>
            cd ai-tutor
        </div>

        <p>2. Install dependencies</p>
        <div class="code-block">
            pip install -r requirements.txt
        </div>

        <p>3. Train the AI model</p>
        <div class="code-block">
            python train_model.py
        </div>

        <p>4. Run the application</p>
        <div class="code-block">
            streamlit run app.py
        </div>

        <p>5. Open your browser and go to <a class="link" href="http://localhost:8501" target="_blank">http://localhost:8501</a></p>

        <p>6. <span class="highlight">Project Work Explanation:</span> Watch the detailed project explanation video <a class="link" href="https://youtu.be/oAjCUJGwMhE" target="_blank">here</a>.</p>
    </div>

    <div class="section">
        <h2>Usage</h2>
        <ol>
            <li><span class="highlight">Login:</span> Enter any name, Password should be at least 8 characters</li>
            <li><span class="highlight">Educational Level:</span> Select your educational level from the dropdown menu</li>
            <li><span class="highlight">Select a Subject:</span> Choose from Mathematics, Science, History, or Programming</li>
            <li><span class="highlight">Ask Questions:</span> Type your question in the text area and click "Submit"</li>
            <li><span class="highlight">Track Progress:</span> Click "View Progress" to see your learning statistics</li>
        </ol>
    </div>

    <div class="section">
        <h2>Project Structure</h2>
        <pre class="code-block">
ai-tutor/
├── app.py                      # Main Streamlit application
├── aimodel.py                  # AI tutor model implementation
├── train_model.py              # Script to train the basic AI model
├── download_dataset.py         # Script to download basic datasets
├── download_large_dataset.py   # Script to download enhanced datasets
├── create_math_dataset.py      # Script to create enhanced math dataset
├── create_science_dataset.py   # Script to create enhanced science dataset
├── create_history_dataset.py   # Script to create enhanced history dataset
├── create_programming_dataset.py # Script to create enhanced programming dataset
├── ai_tutor.bat                # All-in-one script to setup and run the application
├── model/                      # Directory for storing the trained models
├── data/                       # Directory for user data and progress
│   └── users/                  # User-specific data storage
├── logs/                       # Application logs
└── requirements.txt            # Python dependencies
        </pre>
    </div>

    <div class="section">
        <h2>Customization</h2>
        <ul>
            <li><span class="highlight">Add More Training Data:</span> Extend the training data in the dataset creation scripts to improve the AI tutor's capabilities</li>
            <li><span class="highlight">Add New Subjects:</span> Update the subjects list in app.py and add corresponding training data</li>
        </ul>
    </div>

    <div class="section">
        <h2>Further Development Ideas</h2>
        <ol>
            <li>Implement more sophisticated AI models (transformers, etc.)</li>
            <li>Add support for file uploads (e.g., for math problems)</li>
            <li>Implement a proper database for user data</li>
            <li>Add visualization tools for learning analytics</li>
            <li>Support for multiple languages</li>
        </ol>
    </div>

</body>
</html>
