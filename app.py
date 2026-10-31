
import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)


model = joblib.load("beans_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("selected_features.pkl")
method = joblib.load("preprocessing_method.pkl")


if method == "Clip + Power":
    lower, upper = joblib.load("outlier_bounds.pkl")


st.markdown("""
<style>
.stApp {
    background-color: #080d09;
    background-image:
        linear-gradient(145deg, rgba(4, 9, 6, 0.92), rgba(13, 23, 12, 0.84)),
        repeating-linear-gradient(168deg, transparent 0 42px, rgba(113, 157, 69, 0.2) 44px 47px, rgba(9, 19, 10, 0.9) 52px 83px),
        linear-gradient(180deg, #182519, #080d09);
    background-size: 100% 100%, 100% 220%, 100% 100%;
    background-attachment: fixed;
    animation: crop-field-drift 32s linear infinite;
    color: #ffffff;
}
/* Hide Enter-to-submit instructions */
[data-testid="InputInstructions"] {
    display: none !important;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

h1 {
    display: table;
    margin: 0 auto !important;
    padding: 0.35em 0.8em;
    background: #050805;
    border-radius: 6px;
    color: #77dc62 !important;
    text-align: center;
    text-shadow: none !important;
}

h2, h3 {
    color: #a7df82 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(174, 218, 142, 0.35);
    border-radius: 8px;
    background: rgba(8, 17, 10, 0.76);
}

.stButton > button {
    background: #397c35;
    color: white;
    border-radius: 6px;
    border: none;
    font-size: 16px;
    font-weight: bold;
    padding: 10px 12px;
    width: 100%;
}

.stButton > button:hover {
    background: #4a9843;
    color: white;
}

[data-testid="stNumberInput"] input {
    background-color: rgba(7, 13, 8, 0.98);
    color: #ffffff !important;
}

[data-testid="stNumberInput"] input::placeholder {
    color: #e4e9e4;
    opacity: 1;
}

@keyframes crop-field-drift {
    from { background-position: center, 0 0, center; }
    to { background-position: center, 0 220%, center; }
}
</style>
""", unsafe_allow_html=True)

# Heading
st.title("🫘 Dry Bean Classification")
# st.markdown(
#     "<p style='text-align:center;color:#A8BBD0;'>"
#     "Multiclass Machine Learning Prediction System"
#     "</p>",
#     unsafe_allow_html=True
# )

st.divider()

# values = {}

# with st.container(border=True):
#     st.subheader("Enter Bean Measurements")
#     st.write("Enter the physical measurements of the bean.")

#     for row_start in range(0, len(features), 3):
#         columns = st.columns(3, gap="medium")
#         for column, feature in zip(columns, features[row_start:row_start + 3]):
#             with column:
#                 values[feature] = st.number_input(
#                     feature,
#                     min_value=0.0,
#                     value=None,
#                     format="%.6f",
#                     placeholder=f"Enter {feature}"
#                 )

# with st.container(border=True):
#     st.subheader("Prediction Result")
#     result = st.empty()

#     if st.button("Predict Bean Type", use_container_width=True):
#         if any(value is None for value in values.values()):
#             st.warning("Please enter all measurements.")
#         else:
#             input_df = pd.DataFrame([values], columns=features)

#             if method == "Clip + Power":
#                 input_df = input_df.clip(lower, upper, axis=1)

#             input_scaled = scaler.transform(input_df)
#             prediction = model.predict(input_scaled)[0]
#             result.success(f"Predicted Bean Type: {prediction}")
#             st.balloons()

values = {}

with st.form("bean_prediction_form"):

    with st.container(border=True):
        st.subheader("Enter Bean Measurements")
        st.write("Enter the physical measurements of the bean.")

        for row_start in range(0, len(features), 3):
            columns = st.columns(3, gap="medium")

            for column, feature in zip(
                columns, features[row_start:row_start + 3]
            ):
                with column:
                    values[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        value=None,
                        format="%.6f",
                        placeholder=f"Enter {feature}"
                    )

    with st.container(border=True):
        st.subheader("Prediction Result")

        predict = st.form_submit_button(
            "Predict Bean Type",
            use_container_width=True
        )

if predict:

    if any(value is None for value in values.values()):
        st.warning("Please enter all measurements.")

    else:
        input_df = pd.DataFrame([values], columns=features)

        if method == "Clip + Power":
            input_df = input_df.clip(lower, upper, axis=1)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        st.success(f"Predicted Bean Type: {prediction}")
        st.balloons()


st.divider()

st.markdown(
    "<p style='text-align:center;color:#8496AD;'>"
    "Dry Bean Classification | Supervised Machine Learning"
    "</p>",
    unsafe_allow_html=True
)
