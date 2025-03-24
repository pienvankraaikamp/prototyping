# Loading necessary libraries
import streamlit as st
import pandas as pd
import joblib
import os
from fpdf import FPDF
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import unicodedata

# Load the .env file and initialize the OpenAI client
load_dotenv()
client = OpenAI()

# Set page layout to 'wide' otherwise title does not fit
st.set_page_config(layout="wide")

# Changing the progress bar to the risk level
def get_progress_color(probability):
    if probability < 0.2:
        return "#4CAF50"  # green for very low risk
    elif 0.2 <= probability < 0.4:
        return "#2196F3"  # blue for low risk
    elif 0.4 <= probability < 0.6:
        return "#FFC107"  # yellow for medium risk
    else:
        return "#F44336"  #red for high and very high rik

# Loading model and scaling information from model.py
model = joblib.load("hotel_cancelation_model.joblib")
scaling_info = joblib.load("scaling_info.joblib")

# Create a file to store past predictions
history_file = "cancellation_history.csv"

# Check that the file exists
if not os.path.exists(history_file):
    pd.DataFrame(columns = ["probability"]).to_csv(history_file, index = False)

# Load the previous predictions
history_df = pd.read_csv(history_file)

# Model is trained with scaled features so function to scale user inputs
def scale_features(input_data):
    scaled_data = {}
    for feature, value in input_data.items():
        mean = scaling_info[feature]["mean"]
        std = scaling_info[feature]["std"]
        scaled_data[feature] = (value - mean) / std
    return scaled_data

# Function to generate the PDF report
def generate_pdf_report(input_data, probability, advice, avg_cancellation_rate = None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)

    # Add a title to the PDF
    pdf.cell(200, 10, txt = "Hotel Booking Cancellation Risk Report", ln = True, align = 'C')
    pdf.ln(10)

    # Add the input data to the PDF
    pdf.set_font("Arial", style = 'B', size = 12)
    pdf.cell(100, 10, txt = "Booking Details:", ln = True)
    pdf.set_font("Arial", size = 12)
    for key, value in input_data.items():
        label = key.replace("_", " ").title()
        pdf.cell(100, 10, txt = f"{label}: {value}", ln = True)

    # Add the prediction to the PDF
    pdf.ln(5)
    pdf.set_font("Arial", style = 'B', size = 12)
    pdf.cell(100, 10, txt = "Cancellation Prediction:", ln = True)
    pdf.set_font("Arial", size = 12)
    pdf.cell(100, 10, txt = f"Predicted Cancellation Probability: {probability:.1%}", ln = True)

    if avg_cancellation_rate is not None:
        pdf.cell(100, 10, txt = f"Historical Average Cancellation Rate: {avg_cancellation_rate:.1%}", ln = True)

    # Add the advice to the PDF
    pdf.ln(5)
    pdf.set_font("Arial", style = 'B', size = 12)
    pdf.cell(100, 10, txt = "Advice:", ln = True)
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(180, 10, txt = advice)

    pdf_bytes = pdf.output(dest = 'S').encode('latin1')
    return BytesIO(pdf_bytes)

# Function to sanitize text for OpenAI API
def sanitize_text(text):
    return unicodedata.normalize("NFKD", text).encode("latin1", "ignore").decode("latin1")

# Function to generate a summary using the LLM
def generate_llm_summary(input_data, probability):
    # Define the prompt for the LLM
    prompt = f"""
You are a helpful assistant for hotel managers. A booking has been analyzed for its risk of cancellation.
Here are the details:
- Lead time: {input_data['lead_time']} days
- Previous cancellations: {input_data['previous_cancellations']}
- Total nights: {input_data['total_nights']}
- Special requests: {input_data['total_of_special_requests']}
- Predicted cancellation probability: {probability:.1%}

Based on this information, write a short summary (3–5 sentences) explaining the situation and suggesting a smart next step to reduce cancellation risk or improve customer experience.
Use clear, professional language.
"""
    # Generate the summary using the LLM
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
        return sanitize_text(summary)

    # For any exceptions, return an error message
    except Exception as e:
        return f"⚠️ Could not generate summary:\n\n{e}"

# Streamlit title and text
st.title("Hotel Cancellation Risk & Management Tool")
st.markdown("*Fill in the booking details to predict the risk of cancellation, receive tailored advice, and compare with your hotel's historical cancellation trends.*")
st.divider()
col1, col2 = st.columns([1, 1]) # creating two columns for more visually appealing layout

with col1:
    st.subheader("Enter booking details")
    # User input fields in streamlit
    lead_time = st.number_input("Lead time (days until check-in):", min_value = 0, step = 1, value = 14)
    previous_cancellations = st.number_input("Previous cancellations by customer:", min_value = 0, step = 1, value = 0)
    total_nights = st.number_input("Total nights of the booking:", min_value = 1, step = 1, value = 2)
    total_of_special_requests = st.number_input("Amount of special requests:", min_value = 0, step = 1, value = 0)

    # Creating a user input o enter their known average cancellation rate for comparison
    user_avg_cancellation_rate = st.text_input("Enter your average cancellation rate (optional):")

    # Change toggle to radio button for better user experience
    advice_option = st.radio(
    "Select advice type:",
    ["Rule-based advice", "AI-generated summary"],
    index = 0
    )

    # Show advice based on the selected option
    show_advice = advice_option == "Rule-based advice"
    show_ai_summary = advice_option == "AI-generated summary"

    # Result of clicking predict button on streamlit
    if st.button("**Predict cancellation!**"):
        # Storing user input
        user_input = {
            "lead_time": lead_time,
            "previous_cancellations": previous_cancellations,
            "total_nights": total_nights,
            "total_of_special_requests": total_of_special_requests
        }

        # Input needs to be scaled using function
        scaled_input = scale_features(user_input)
        input_df = pd.DataFrame([scaled_input])

        # Prediction is made using model 
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]    

        # Generate LLM summary
        llm_summary = generate_llm_summary(user_input, probability)

        # Select advice based on risk (for both display and PDF export)
        if probability < 0.2:
            advice_text = "This is a great booking! Consider offering an upsell (room upgrade, early check-in)."
        elif 0.2 <= probability < 0.4:
            advice_text = "No immediate action needed. Send a friendly reminder email & ask if they have special requests."
        elif 0.4 <= probability < 0.6:
            advice_text = "Offer a flexible check-in option to reduce cancellation risk."
        elif 0.6 <= probability < 0.8:
            advice_text = "Send a discount offer to encourage them to keep the booking."
        else:
            advice_text = "Call the guest & send a last-minute discount offer or bonus (e.g. free breakfast) to keep their reservation."

        # Prediction is saved to history file
        new_data = pd.DataFrame({"probability": [probability]})
        history_df = pd.concat([history_df, new_data], ignore_index = True)
        history_df.to_csv(history_file, index = False)

        # Calculating hotel's historical average cancellation risk
        if len(history_df) > 10: # only is there is more than 10 lines of historical data available
            avg_cancellation_rate = history_df["probability"].mean()
        elif user_avg_cancellation_rate: # otherwise we use the user input
            try:
                avg_cancellation_rate = float(user_avg_cancellation_rate)
            except ValueError: # or None if user did not give an input
                avg_cancellation_rate = None
        else:
            avg_cancellation_rate = None

        # Default fallback
        comparison_advice_text = ""

        # Comparison-based advice (only if there's historical avg)
        if avg_cancellation_rate is not None:
            if probability > avg_cancellation_rate:
                comparison_advice_text = "This booking has a **higher cancellation risk** than your hotel's historical average. Consider offering discounted services."
            elif probability < avg_cancellation_rate:
                comparison_advice_text = "This booking has a **lower cancellation risk** than your hotel's historical average. Consider upselling services to maximize revenue."
            else:
                comparison_advice_text = "This booking's cancellation risk is **in line** with your hotel's historical average. No immediate action needed, keep monitoring trends over time."

        with col2:
            # Display the prediction result
            st.subheader("**Cancellation Probability**")
            
            # Using HTML and CSS to change the color of the progress bar
            progress_color = get_progress_color(probability)
            progress_html = f"""
                <div style = "width: 100%; background-color: #e0e0e0; border-radius: 10px;">
                    <div style = "width: {probability * 100}%; background-color: {progress_color}; 
                                padding: 8px; border-radius: 10px; text-align: center; 
                                font-weight: bold; color: white;">
                        {probability:.1%}
                    </div>
                </div>
            """
            # display the progress bar with the probability
            st.markdown(progress_html, unsafe_allow_html = True)

            # Show the risk classification and (if toggle is on) show advice on how to proceed with the risk level
            if probability < 0.2:
                st.success(f"Very low risk: Only **{probability:.1%}** chance of cancellation.")
                if show_advice:
                    st.write(f"💡 **Advice:** {advice_text}")
            elif 0.2 <= probability < 0.4:
                st.info(f"Low risk: **{probability:.1%}** chance of cancellation.")
                if show_advice:
                    st.write(f"💡 **Advice:** {advice_text}")
            elif 0.4 <= probability < 0.6:
                st.warning(f"Medium risk: **{probability:.1%}** chance of cancellation.")
                if show_advice:
                    st.write(f"💡 **Advice:** {advice_text}")
            elif 0.6 <= probability < 0.8:
                st.error(f"High risk: **{probability:.1%}** chance of cancellation.")
                if show_advice:
                    st.write(f"💡 **Advice:** {advice_text}")
            else:
                st.error(f"Very high risk: **{probability:.1%}** chance of cancellation. Immediate action needed!")
                if show_advice:
                    st.write(f"💡 **Advice:** {advice_text}")

            # Show comparison with hotel's historical average inside an expander
            with st.expander("**View Historical Comparison**"):
                st.metric(label = "Predicted cancellation rate", value = f"{probability:.1%}")
                if avg_cancellation_rate is not None:
                    st.metric(label = "Historical average cancellation rate", value = f"{avg_cancellation_rate:.1%}")
                    
                    # Insights based on probability range
                    if show_advice and comparison_advice_text:
                        if probability > avg_cancellation_rate:
                            st.warning("This booking has a **higher cancellation risk** than your hotel's historical average.")
                        elif probability < avg_cancellation_rate:
                            st.success("This booking has a **lower cancellation risk** than your hotel's historical average!")
                        else:
                            st.info("This booking's cancellation risk is **in line** with your hotel's historical average.")

                        st.write(f"**Historical comparison based advice:** {comparison_advice_text}")

                else:
                    st.warning("⚠️ Not enough historical data to provide a comparison.")

                if show_ai_summary:
                    st.markdown("### 🤖 AI-Generated Summary")
                    st.write(llm_summary)

            # Generate PDF file and sanitize whole text only if toggle is on
            report_sections = []

            if show_advice:
                report_sections.append("Advice:\n" + advice_text)
            if show_advice and comparison_advice_text:
                report_sections.append("Historical Comparison:\n" + comparison_advice_text)
            if show_ai_summary:
                report_sections.append("AI-Generated Summary:\n" + llm_summary)

            full_advice = sanitize_text("\n\n".join(report_sections))
            pdf_file = generate_pdf_report(user_input, probability, full_advice, avg_cancellation_rate)

            # Download button
            st.download_button(
                label="📄 Download Report as PDF",
                data=pdf_file,
                file_name="cancellation_report.pdf",
                mime="application/pdf"
            )
