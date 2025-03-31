AI Tutor App

A simple AI-powered learning platform built with Streamlit, providing personalized tutoring across multiple subjects using our self-trained AI model.

Features

- AI-Powered Learning: Interact with an AI tutor trained on educational content
- Multiple Subjects: Access tutoring in Mathematics, Science, History, and Programming
- Progress Tracking: View your learning progress across different subjects
- User Authentication: Simple login system to track individual progress

Tech Stack

- Frontend & Backend: Streamlit
- AI Model: Scikit-learn (TF-IDF + Random Forest classifier)
- Styling: Tailwind CSS (via CDN)
- Data Storage: Local file storage (pickle files)

Getting Started

Prerequisites

- Python 3.8 or later
- pip (Python package manager)

Installation

1. Clone the repository
```
git clone https://github.com/yourusername/ai-tutor.git
cd ai-tutor
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Train the AI model
```
python train_model.py
```

4. Run the application
```
streamlit run app.py
```

5. Open your browser and go to http://localhost:8501

Usage

1. Login:
   - Enter any name
   - Password should be at least 8 characters
   
2. Educational Level:
   - Select your educational level from the dropdown menu

3. Select a Subject: 
   - Choose from Mathematics, Science, History, or Programming

4. Ask Questions: 
   - Type your question in the text area and click "Submit"

5. Track Progress: 
   - Click "View Progress" to see your learning statistics

Project Structure

```
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
```

Customization

- Add More Training Data: Extend the training data in the dataset creation scripts to improve the AI tutor's capabilities
- Add New Subjects: Update the subjects list in app.py and add corresponding training data

Further Development Ideas

1. Implement more sophisticated AI models (transformers, etc.)
2. Add support for file uploads (e.g., for math problems)
3. Implement a proper database for user data
4. Add visualization tools for learning analytics
5. Support for multiple languages 
