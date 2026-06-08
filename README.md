                           📧 Email Urgency Classifier

An AI-powered Email Urgency Classification system that analyzes email content and predicts its urgency level. The application helps users quickly identify high-priority emails, improving productivity and email management.

                            🚀 Live Demo

🌐 Application: https://emailurgencyclassifier-i3uahzszlxsphaes3rctkm.streamlit.app/

📌 Features

- Detects urgency level of emails automatically.
- User-friendly Streamlit web interface.
- Real-time predictions.
- NLP-based text preprocessing.
- Machine Learning powered classification.
- Fast and lightweight deployment.
- Supports custom email content input.

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- NLTK
- Joblib

## 📂 Project Structure

Email-Urgency-Classifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── model_training.ipynb
│
└── assets/
    └── screenshots

## ⚙️ Installation

### Clone the Repository

git clone https://github.com/yourusername/Email-Urgency-Classifier.git

cd Email-Urgency-Classifier

### Create a Virtual Environment

python -m venv venv

### Activate Environment

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

## ▶️ Running the Application

streamlit run app.py

The application will be available at:

http://localhost:8501

## 🧠 How It Works

1. User enters email content.
2. Text is cleaned and preprocessed.
3. The trained machine learning model converts text into numerical features.
4. The classifier predicts the urgency level.
5. Results are displayed instantly on the dashboard.

## 📊 Urgency Categories

### High Urgency
Immediate attention required.

### Medium Urgency
Important but not critical.

### Low Urgency
Can be addressed later.

## 📈 Machine Learning Pipeline

Email Text
↓
Text Cleaning
↓
Tokenization
↓
TF-IDF Vectorization
↓
Machine Learning Model
↓
Urgency Prediction

## 🎯 Use Cases

- Corporate Email Management
- Customer Support Prioritization
- Help Desk Systems
- Personal Productivity
- Automated Email Sorting

## 📸 Screenshots

Add screenshots of:
- Home Page
- Prediction Results
- Model Performance (optional)

## 🔮 Future Improvements

- Multi-language support
- Gmail Integration
- Outlook Integration
- Email Summarization
- Priority Score Visualization
- Deep Learning Models (BERT, RoBERTa)
- Email Notification System

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to your branch.
5. Create a Pull Request.

## 📜 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Himanshu Jaiswal



## Project Overview

Email overload is a common challenge faced by professionals, students, and organizations. Not every email requires immediate attention, yet manually sorting emails based on urgency can be time-consuming and inefficient. The Email Urgency Classifier addresses this problem by using Natural Language Processing (NLP) and Machine Learning techniques to automatically classify emails into urgency levels such as High, Medium, and Low.

The system preprocesses email text, extracts meaningful features using TF-IDF vectorization, and applies a trained machine learning model to predict urgency. The application is deployed using Streamlit, providing a simple and interactive user interface that allows users to paste email content and receive instant predictions.

This project demonstrates practical applications of NLP, text classification, machine learning model deployment, and web application development. It can be extended further by integrating with email platforms such as Gmail and Outlook to automate email prioritization in real-world environments.

⭐ If you found this project useful, consider giving it a star on GitHub.
