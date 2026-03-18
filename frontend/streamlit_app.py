import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
import os

# Define the URL of our running FastAPI Gateway
GATEWAY_URL = "http://localhost:8000"

# --- STREAMLIT CACHING ENGINE ---
@st.cache_data(show_spinner=False)
def upload_dataset(file_name, file_bytes, file_type):
    files = {"file": (file_name, file_bytes, file_type)}
    response = requests.post(f"{GATEWAY_URL}/upload-dataset", files=files)
    return response.json() if response.status_code == 200 else None

@st.cache_data(show_spinner=False)
def analyze_dataset(dataset_id, filename):
    response = requests.get(f"{GATEWAY_URL}/analyze-dataset/{dataset_id}", params={"filename": filename})
    return response.json() if response.status_code == 200 else None

@st.cache_data(show_spinner=False)
def determine_problem(dataset_id, filename, target_col):
    response = requests.get(f"{GATEWAY_URL}/determine-problem/{dataset_id}", params={"filename": filename, "target_column": target_col})
    return response.json() if response.status_code == 200 else None

@st.cache_data(show_spinner=False)
def generate_eda(dataset_id, filename, target_col):
    response = requests.get(f"{GATEWAY_URL}/eda-report/{dataset_id}", params={"filename": filename, "target_column": target_col})
    return response.json() if response.status_code == 200 else None

st.set_page_config(page_title="No-Code ML Platform", layout="wide")
st.title("🤖 Agentic No-Code ML Platform")
st.markdown("Upload your dataset and let the AI build, train, and evaluate your machine learning models automatically.")
st.divider()

# Section 1: Dataset Upload System
st.header("1. Dataset Upload & Validation")
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    
    # Reset all memory flags if a new file is uploaded
    if st.session_state.get('current_file_name') != uploaded_file.name:
        st.session_state['current_file_name'] = uploaded_file.name
        st.session_state['data_ready_for_training'] = False
        st.session_state['model_trained'] = False

    with st.spinner("Syncing with AI Gateway..."):
        upload_data = upload_dataset(uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        if not upload_data:
            st.error("Upload Error: Failed to communicate with the Gateway.")
            st.stop()
            
    metadata = upload_data["metadata"]
    dataset_id = metadata["dataset_id"]
    filename = metadata["filename"]

    # --- PHASE 1: Overview ---
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", metadata["num_rows"])
    col2.metric("Total Columns", metadata["num_columns"])
    col3.metric("File Size (KB)", round(metadata["file_size_bytes"] / 1024, 2))
    col4.metric("File Type", filename.split('.')[-1].upper())
    
    # --- PHASE 2: Intelligence Profiling ---
    st.divider()
    st.header("2. Dataset Intelligence Analysis")
    
    analysis_data = analyze_dataset(dataset_id, filename)
    if analysis_data:
        profile = analysis_data["profile"]
        if profile["duplicate_rows"] > 0:
            st.warning(f"⚠️ **Duplicate Rows Detected:** {profile['duplicate_rows']} duplicate rows found.")
        else:
            st.success("✅ **No Duplicate Rows Found!**")
            
        st.markdown("### 🔍 Column Profiling")
        profile_data = [
            {"Column Name": col, "Data Type": dtype, "Missing Data (%)": f"{profile['missing_values'].get(col, 0.0)}%"}
            for col, dtype in profile["column_types"].items()
        ]
        st.dataframe(pd.DataFrame(profile_data), width="stretch")
        
        # --- PHASE 2.5: Target Selection ---
        st.divider()
        st.header("🎯 Target Selection")
        target_col = st.selectbox("Select Target Variable", options=metadata["columns"])
        
        # Target State Reset
        if st.session_state.get('current_target') != target_col:
            st.session_state['current_target'] = target_col
            st.session_state['data_ready_for_training'] = False
            st.session_state['model_trained'] = False
            
        if target_col:
            prob_data = determine_problem(dataset_id, filename, target_col)
            
            # --- PHASE 3: Automated EDA ---
            st.divider()
            st.header("3. Automated Exploratory Data Analysis (EDA)")
            
            with st.expander("📊 View Detailed Data Visualizations & Statistics", expanded=False):
                with st.spinner("Generating statistical plots..."):
                    eda_data = generate_eda(dataset_id, filename, target_col)
                    if eda_data:
                        tab1, tab2, tab3, tab4 = st.tabs(["🔢 Descriptive Stats", "🔥 Correlation Heatmap", "🗂️ Categorical Distribution", "🎯 Target Analysis"])
                        
                        with tab1:
                            stats_df = pd.DataFrame(eda_data["descriptive_statistics"])
                            if not stats_df.empty: st.dataframe(stats_df.T, width="stretch")
                                
                        with tab2:
                            corr_dict = eda_data["correlation_matrix"]
                            if corr_dict:
                                fig = px.imshow(pd.DataFrame(corr_dict), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                                st.plotly_chart(fig, use_container_width=True)
                                
                        with tab3:
                            cat_dict = eda_data["categorical_summary"]
                            if cat_dict:
                                for col_name, counts in cat_dict.items():
                                    if counts: 
                                        fig = px.bar(pd.DataFrame(list(counts.items()), columns=["Category", "Count"]), x="Category", y="Count", title=f"Distribution of {col_name}")
                                        st.plotly_chart(fig, use_container_width=True)

                        with tab4:
                            target_stats = eda_data.get("target_analysis", {})
                            if target_stats.get("type") == "categorical":
                                fig_pie = px.pie(pd.DataFrame(list(target_stats["class_distribution"].items()), columns=["Class", "Freq"]), names="Class", values="Freq", title="Target Class Distribution")
                                st.plotly_chart(fig_pie, use_container_width=True)
                                grouped_df = pd.DataFrame.from_dict(target_stats["grouped_means"], orient="index")
                                if not grouped_df.empty: st.dataframe(grouped_df, width="stretch")
                            elif target_stats.get("type") == "numeric":
                                correlations = target_stats.get("feature_correlations", {})
                                if correlations:
                                    fig_bar = px.bar(pd.DataFrame(list(correlations.items()), columns=["Feature", "Score"]), x="Feature", y="Score", title="Correlation with Target", color="Score", color_continuous_scale="Viridis")
                                    st.plotly_chart(fig_bar, use_container_width=True)

            # --- PHASE 4: Feature Engineering ---
            st.divider()
            st.header("4. Automated Feature Engineering")
            
            if st.button("⚙️ Run Auto-Feature Engineering"):
                with st.spinner("Transforming data into ML-ready format..."):
                    eng_response = requests.post(f"{GATEWAY_URL}/engineer-features/{dataset_id}", params={"filename": filename, "target_column": target_col})
                    
                    if eng_response.status_code == 200:
                        eng_data = eng_response.json()
                        st.success("✅ " + eng_data["message"])
                        st.session_state['data_ready_for_training'] = True
                    else:
                        st.error(f"Engineering Error: {eng_response.text}")

            # --- PHASE 5: AutoML Training ---
            if st.session_state.get('data_ready_for_training', False):
                st.divider()
                st.header("5. AutoML Model Training & Evaluation")
                
                if st.button("🚀 Train Machine Learning Models"):
                    with st.spinner("Training models... This might take a moment!"):
                        train_response = requests.post(f"{GATEWAY_URL}/train-models/{dataset_id}", params={"target_column": target_col, "problem_type": prob_data['problem_type']})
                        
                        if train_response.status_code == 200:
                            st.session_state['train_data'] = train_response.json()
                            st.session_state['model_trained'] = True
                            st.balloons()
                        else:
                            st.error(f"Training Error: {train_response.text}")

                # --- PHASE 6 & 7: Model Inference & Export ---
                if st.session_state.get('model_trained', False):
                    train_data = st.session_state['train_data']
                    st.success(f"🎉 Training Complete! The best model is **{train_data['best_model']}**!")
                    
                    st.markdown("### 🏆 Model Leaderboard")
                    st.dataframe(pd.DataFrame(train_data["leaderboard"]), width="stretch")
                    st.info(f"💾 The winning model is saved at `{train_data['model_path']}` and is ready for deployment!")
                    
                    st.divider()
                    st.header("6. Use the Model (Live Inference)")
                    
                    # 6.1 RANDOM SAMPLE INFERENCE
                    st.markdown("#### 🎲 Auto-Test (5 Random Rows)")
                    if st.button("🔮 Generate Sample Predictions"):
                        with st.spinner("Waking up the AI model and predicting..."):
                            model_path = train_data['model_path']
                            loaded_model = joblib.load(model_path)
                            
                            processed_file_path = os.path.join("temp_datasets", f"{dataset_id}_processed.csv")
                            processed_df = pd.read_csv(processed_file_path)
                            
                            sample_df = processed_df.sample(5)
                            X_test_sample = sample_df.drop(columns=[target_col])
                            actual_answers = sample_df[target_col].tolist()
                            
                            predictions = loaded_model.predict(X_test_sample)
                            
                            if prob_data['problem_type'] == "Regression":
                                predictions = [round(float(p), 2) for p in predictions]
                                actual_answers = [round(float(a), 2) for a in actual_answers]
                                
                            results_df = pd.DataFrame({
                                "Actual True Value": actual_answers,
                                "AI Prediction": predictions
                            })
                            
                            st.success("Predictions generated successfully!")
                            st.dataframe(results_df, width="stretch")
                            
                    # 6.2 MANUAL DATA ENTRY FORM (DYNAMIC)
                    st.markdown("#### ✍️ Manual Data Entry")
                    st.markdown("Type in custom values below to get a live prediction from your AI!")
                    
                    with st.form("prediction_form"):
                        input_data = {}
                        
                        for col in metadata["columns"]:
                            if col == target_col:
                                continue 
                                
                            col_type = profile["column_types"].get(col, "Categorical/Text")
                            
                            if col_type == "Numeric":
                                input_data[col] = st.number_input(f"{col} (Numeric)", value=0.0)
                            else:
                                input_data[col] = st.text_input(f"{col} (Text)", value="")
                                
                        submit_button = st.form_submit_button("🧠 Predict My Data!")
                        
                        if submit_button:
                            with st.spinner("Sending data to Inference Service..."):
                                predict_response = requests.post(
                                    f"{GATEWAY_URL}/predict/{dataset_id}",
                                    json=input_data
                                )
                                
                                if predict_response.status_code == 200:
                                    pred_result = predict_response.json()
                                    prediction_val = pred_result["prediction_label"]
                                    
                                    if prob_data['problem_type'] == "Regression":
                                        prediction_val = round(float(prediction_val), 2)
                                        
                                    st.success(f"**AI Prediction:** {prediction_val}")
                                else:
                                    st.error(f"Prediction Error: {predict_response.text}")

                    # --- PHASE 8: Export Production Code ---
                    st.divider()
                    st.header("7. Export Production Code")
                    st.markdown("Don't get locked into our platform! Generate the exact Python code needed to run this AI model in your own custom applications (like a mobile app backend).")
                    
                    if st.button("💻 Generate Deployment Script"):
                        code_snippet = f"""import pandas as pd
import joblib

def predict_custom_data(input_dict):
    # 1. Load the frozen model and pipeline
    model = joblib.load("models/{dataset_id}_best_model.joblib")
    pipeline = joblib.load("models/{dataset_id}_pipeline.joblib")
    
    # 2. Convert input dictionary to Pandas DataFrame
    df = pd.DataFrame([input_dict])
    
    # 3. Apply Missing Value Imputers
    if pipeline["num_imputer"] and pipeline["numeric_cols"]:
        cols = [c for c in pipeline["numeric_cols"] if c in df.columns]
        if cols: df[cols] = pipeline["num_imputer"].transform(df[cols])
        
    if pipeline["cat_imputer"] and pipeline["categorical_cols"]:
        cols = [c for c in pipeline["categorical_cols"] if c in df.columns]
        if cols: df[cols] = pipeline["cat_imputer"].transform(df[cols])
        
    # 4. Apply One-Hot Encoding & Fix Column Alignment
    if pipeline["categorical_cols"]:
        df = pd.get_dummies(df, columns=[c for c in pipeline["categorical_cols"] if c in df.columns])
    df = df.reindex(columns=pipeline["final_columns"], fill_value=0)
    
    # 5. Apply Scaling
    if pipeline["scaler"] and pipeline["numeric_cols"]:
        df[pipeline["numeric_cols"]] = pipeline["scaler"].transform(df[pipeline["numeric_cols"]])
        
    # 6. Generate Prediction
    pred = model.predict(df)[0]
    
    # 7. Map back to original text label (if classification)
    if pipeline.get("target_mapping"):
        mapping = {{str(k): v for k, v in pipeline["target_mapping"].items()}}
        return mapping.get(str(pred), str(pred))
    return float(pred)

# --- Example Usage ---
# Pass in your data from your website or mobile app!
my_new_data = {{
    # Add your column names and values here
    # "Age": 25, 
    # "Sex": "male"
}}

print("AI Prediction:", predict_custom_data(my_new_data))
"""
                        st.info("Copy this code into your own Python project. Make sure to move the `.joblib` files from the `models/` folder into your new project as well!")
                        st.code(code_snippet, language="python")