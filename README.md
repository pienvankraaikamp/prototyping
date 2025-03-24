Hotel Cancellation Risk & Management Tool

This Streamlit app predicts the likelihood of hotel booking cancellations using a trained logistic regression model. 
It also offers tailored advice to hotel managers: either based on predefined business rules or generated dynamically using a Large Language Model (LLM).

Files:
streamlit.py – Main app logic
model.py – ML model training and saving
requirements.txt – Python dependencies
hotel_bookings.csv – Sample dataset
hotel_cancelation_model.joblib – Trained logistic regression model
scaler_info.joblib – Scaler used during model training

How to Run the App:
- Clone the repo or download the folder
- (Optional) Create and activate a virtual environment
- Install dependencies:
pip install -r requirements.txt
- Run the app:
streamlit run streamlit.py

API Key Required (for LLM features):
To enable the AI-generated summaries, you'll need an OpenAI API key:
- Create a .env file in the root folder
- Add the following line:
OPENAI_API_KEY=your-api-key-here

Notes:
- The model must be used together with the exact same scaler used during training (scaler_info.joblib)
- LLM features are optional: users can toggle between rule-based or AI-generated summaries
- The PDF export reflects the selected advice type for each prediction

Author: Pien van Kraaikamp
Prototyping with AI — ESADE 2025
