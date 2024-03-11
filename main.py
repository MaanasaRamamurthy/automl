import streamlit as st
import pandas as pd
import os
import ydata_profiling as pdp
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model,load_model, predict_model
import subprocess


current_dir_path = os.getcwd()
def main(uploaded_dataset):
    # Title for the app
    st.title("Make a Prediction")

    # Check if a dataset is uploaded
    if uploaded_dataset is None:
        st.warning("Please upload a dataset first.")
        return

    # Display the uploaded dataset
    st.subheader("Uploaded Dataset:")
    st.write(uploaded_dataset)

    # Create input fields based on the uploaded dataset
    user_input = create_input_fields(uploaded_dataset)

    # Create a DataFrame with user input
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Button to trigger prediction
    if st.button("Predict"):
        st.subheader("Prediction Results:")
        prediction = predict_model(pipeline, data=user_input_df)
        print(prediction)
        predicted_class = prediction["prediction_label"][0]
        st.write(f"Predicted Class: {predicted_class}")

def create_input_fields(df):
    input_fields = {}
    st.subheader("Please enter your information:")
    for column in df.columns:
        if column != st.session_state.target_column_val:
            # Check data type of the column and create appropriate input field
            if pd.api.types.is_string_dtype(df[column]):
                input_fields[column] = st.text_input(f"{column.capitalize()}")
            else:
                input_fields[column] = st.number_input(f"{column.capitalize()}")

    return input_fields

with st.sidebar:
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download", "Test Model"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Profiling and PyCaret.")


if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)


if choice == "Upload":
    st.title("Upload your data for Modelling !")
    file = st.file_uploader("Upload your dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    # profile_report = df.profile_report()
    # st_profile_report(profile_report)
    profile = pdp.ProfileReport(df)
    st_profile_report(profile)

if choice == "ML":
    st.title("Machine Learning Model")
    target = st.selectbox("Select your target", df.columns)
    print(type(target))

    if st.button("Train Model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        st.write(best_model)
        save_model(best_model, 'best_model')
    print("hi")
    st.session_state.target_column_val = target
    print(st.session_state.target_column_val)

if choice == "Download":
    if st.button("Download the model"):
        subprocess.run([
            "python", "test.py", "best_model.pkl", os.path.join(current_dir_path, "trained_model.pkl")
        ])
        st.info("The model has been downloaded to your project directory.")

if choice == "Test Model":
    print("Testing")
    print(st.session_state.target_column_val)
    pipeline = load_model("trained_model")
    uploaded_dataset = pd.read_csv("sourcedata.csv")
    main(uploaded_dataset)
