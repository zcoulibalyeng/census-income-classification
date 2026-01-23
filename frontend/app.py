"""
Streamlit Frontend for Census Income Classification API.

A professional web interface for predicting income based on census data.
"""

import streamlit as st
import requests
import json
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# Update this with your Render API URL
API_URL = "https://census-income-classifier.onrender.com/"

# Page configuration
st.set_page_config(
    page_title="Census Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-high {
        background-color: #C8E6C9;
        color: #2E7D32;
        border: 2px solid #2E7D32;
    }
    .prediction-low {
        background-color: #FFECB3;
        color: #F57F17;
        border: 2px solid #F57F17;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card ul {
        color: #333333;
        margin: 0;
        padding-left: 1.2rem;
    }
    .metric-card li {
        color: #555555;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/money-bag.png", width=80)
    st.title("About")
    st.markdown("""
    This application predicts whether a person's annual income 
    exceeds **$50,000** based on census data.

    ---

    **Model Information:**
    - Algorithm: Random Forest
    - Test Precision: 79.72%
    - Test Recall: 54.04%
    - Test F1 Score: 64.42%

    ---

    **Data Source:**  
    UCI Census Income Dataset (1994)

    ---

    **Built with:**
    - 🐍 Python
    - ⚡ FastAPI
    - 🎈 Streamlit
    - 🤖 Scikit-learn
    """)

    st.markdown("---")

    # API Health Check
    st.subheader("🔌 API Status")
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        if response.status_code == 200:
            st.success("✅ API is online")
        else:
            st.warning(f"⚠️ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<p class="main-header">💰 Census Income Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether income exceeds $50K based on demographic data</p>',
            unsafe_allow_html=True)

# Divider
st.markdown("---")

# =============================================================================
# INPUT FORM
# =============================================================================

st.subheader("📝 Enter Personal Information")

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")

    age = st.slider("Age", min_value=17, max_value=90, value=35, help="Your age in years")

    sex = st.selectbox("Sex", options=["Male", "Female"])

    race = st.selectbox("Race", options=[
        "White", "Black", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other"
    ])

    native_country = st.selectbox("Native Country", options=[
        "United-States", "Mexico", "Philippines", "Germany", "Canada",
        "Puerto-Rico", "El-Salvador", "India", "Cuba", "England",
        "Jamaica", "South", "China", "Italy", "Dominican-Republic",
        "Vietnam", "Guatemala", "Japan", "Poland", "Columbia",
        "Taiwan", "Haiti", "Iran", "Portugal", "Nicaragua",
        "Peru", "Greece", "Ecuador", "France", "Ireland",
        "Hong", "Cambodia", "Trinadad&Tobago", "Laos", "Thailand",
        "Yugoslavia", "Honduras", "Hungary", "Scotland", "?"
    ], index=0)

with col2:
    st.markdown("**Education & Work**")

    education = st.selectbox("Education Level", options=[
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
    ])

    education_num = st.slider("Years of Education", min_value=1, max_value=16, value=10,
                              help="Number of years of education completed")

    workclass = st.selectbox("Work Class", options=[
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"
    ])

    occupation = st.selectbox("Occupation", options=[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces", "?"
    ])

with col3:
    st.markdown("**Financial & Family**")

    marital_status = st.selectbox("Marital Status", options=[
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])

    relationship = st.selectbox("Relationship", options=[
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ])

    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40,
                               help="Average hours worked per week")

    capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=99999, value=0,
                                   help="Income from investment sources")

    capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=99999, value=0,
                                   help="Losses from investment sources")

# Hidden field (fnlgt is a census weighting factor, not user-provided)
fnlgt = 200000

# =============================================================================
# PREDICTION
# =============================================================================

st.markdown("---")

# Center the predict button
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    predict_button = st.button("🔮 Predict Income", use_container_width=True)

if predict_button:
    # Prepare the data
    input_data = {
        "age": age,
        "workclass": workclass,
        "fnlgt": fnlgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    # Show loading spinner
    with st.spinner("Analyzing data..."):
        try:
            # Make API request
            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]

                # Display result
                st.markdown("---")
                st.subheader("📊 Prediction Result")

                col_result_left, col_result_center, col_result_right = st.columns([1, 2, 1])

                with col_result_center:
                    if prediction == ">50K":
                        st.markdown("""
                        <div class="prediction-box prediction-high">
                            💰 Income: Greater than $50,000
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown("""
                        <div class="prediction-box prediction-low">
                            📊 Income: Less than or equal to $50,000
                        </div>
                        """, unsafe_allow_html=True)

                # Show input summary
                with st.expander("📋 View Input Summary"):
                    # Create a nice summary table
                    summary_data = {
                        "Feature": [
                            "Age", "Sex", "Race", "Native Country",
                            "Education", "Years of Education", "Work Class", "Occupation",
                            "Marital Status", "Relationship", "Hours/Week",
                            "Capital Gain", "Capital Loss"
                        ],
                        "Value": [
                            age, sex, race, native_country,
                            education, education_num, workclass, occupation,
                            marital_status, relationship, hours_per_week,
                            f"${capital_gain:,}", f"${capital_loss:,}"
                        ]
                    }
                    st.table(pd.DataFrame(summary_data))

            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the API. Please check if the server is running.")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# =============================================================================
# SAMPLE PROFILES
# =============================================================================

st.markdown("---")
st.subheader("🎯 Try Sample Profiles")

sample_col1, sample_col2 = st.columns(2)

with sample_col1:
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2E7D32; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="color: #2E7D32; margin-bottom: 1rem;">👨‍💼 High Earner Profile</h4>
        <ul style="color: #333; padding-left: 1.2rem; margin: 0;">
            <li style="margin-bottom: 0.5rem;"><strong>Age:</strong> 45</li>
            <li style="margin-bottom: 0.5rem;"><strong>Education:</strong> Doctorate</li>
            <li style="margin-bottom: 0.5rem;"><strong>Occupation:</strong> Exec-managerial</li>
            <li style="margin-bottom: 0.5rem;"><strong>Hours/Week:</strong> 55</li>
            <li style="margin-bottom: 0.5rem;"><strong>Capital Gain:</strong> $15,000</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with sample_col2:
    st.markdown("""
    <div style="background-color: #fff8e1; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #F57F17; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h4 style="color: #F57F17; margin-bottom: 1rem;">👩‍🎓 Entry Level Profile</h4>
        <ul style="color: #333; padding-left: 1.2rem; margin: 0;">
            <li style="margin-bottom: 0.5rem;"><strong>Age:</strong> 23</li>
            <li style="margin-bottom: 0.5rem;"><strong>Education:</strong> HS-grad</li>
            <li style="margin-bottom: 0.5rem;"><strong>Occupation:</strong> Sales</li>
            <li style="margin-bottom: 0.5rem;"><strong>Hours/Week:</strong> 35</li>
            <li style="margin-bottom: 0.5rem;"><strong>Capital Gain:</strong> $0</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p> FastAPI & Streamlit | Census Income Classification Project</p>
    <p style="font-size: 0.8rem;">Data Source: UCI Machine Learning Repository | Model: Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)