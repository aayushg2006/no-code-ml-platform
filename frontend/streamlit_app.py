import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Define the URL of our running FastAPI Gateway
GATEWAY_URL = "http://localhost:8000"

st.set_page_config(page_title="No-Code ML Platform", layout="wide")
st.title("🤖 Agentic No-Code ML Platform")
st.markdown("Upload your dataset and let the AI build, train, and evaluate your machine learning models automatically.")
st.divider()

# Section 1: Dataset Upload System
st.header("1. Dataset Upload & Validation")
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    
    # --- FIXED STATE MANAGEMENT: Upload strictly ONCE per file ---
    if 'current_file' not in st.session_state or st.session_state['current_file'] != uploaded_file.name:
        st.session_state['current_file'] = uploaded_file.name
        st.session_state['data_ready_for_training'] = False
        
        with st.spinner("Uploading to server..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{GATEWAY_URL}/upload-dataset", files=files)
            
            if response.status_code == 200:
                # Save the metadata and the stable dataset_id into memory
                st.session_state['metadata'] = response.json()["metadata"]
            else:
                st.error(f"Upload Error: {response.text}")
                st.stop() # Halt execution if upload fails

    # Retrieve the stable ID that will not change when you click buttons
    metadata = st.session_state['metadata']
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
    
    analysis_response = requests.get(
        f"{GATEWAY_URL}/analyze-dataset/{dataset_id}", 
        params={"filename": filename}
    )
    
    if analysis_response.status_code == 200:
        profile = analysis_response.json()["profile"]
        
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
        st.markdown("Select the column you want the AI to predict.")
        
        target_col = st.selectbox("Select Target Variable", options=metadata["columns"])
        
        if target_col:
            problem_response = requests.get(
                f"{GATEWAY_URL}/determine-problem/{dataset_id}",
                params={"filename": filename, "target_column": target_col}
            )
            
            if problem_response.status_code == 200:
                prob_data = problem_response.json()
                st.info(f"🧠 **Agent Conclusion:** Based on the {prob_data['unique_values']} unique values in `{target_col}`, the AI determined this is a **{prob_data['problem_type']}** task.")
            else:
                st.error("Failed to determine problem type.")

            # --- PHASE 3: Automated EDA ---
            st.divider()
            st.header("3. Automated Exploratory Data Analysis (EDA)")
            
            with st.expander("📊 View Detailed Data Visualizations & Statistics", expanded=False):
                with st.spinner("Generating statistical plots..."):
                    eda_response = requests.get(
                        f"{GATEWAY_URL}/eda-report/{dataset_id}",
                        params={"filename": filename, "target_column": target_col}
                    )
                    
                    if eda_response.status_code == 200:
                        eda_data = eda_response.json()
                        tab1, tab2, tab3, tab4 = st.tabs(["🔢 Stats", "🔥 Correlations", "🗂️ Categories", "🎯 Target Analysis"])
                        
                        with tab1:
                            stats_df = pd.DataFrame(eda_data["descriptive_statistics"])
                            if not stats_df.empty: st.dataframe(stats_df.T, width="stretch")
                                
                        with tab2:
                            corr_dict = eda_data["correlation_matrix"]
                            if corr_dict:
                                fig = px.imshow(pd.DataFrame(corr_dict), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
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
                            elif target_stats.get("type") == "numeric":
                                correlations = target_stats.get("feature_correlations", {})
                                if correlations:
                                    fig_bar = px.bar(pd.DataFrame(list(correlations.items()), columns=["Feature", "Score"]), x="Feature", y="Score", title="Correlation with Target")
                                    st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.error("Failed to generate EDA report.")

            # --- PHASE 4: Feature Engineering ---
            st.divider()
            st.header("4. Automated Feature Engineering")
            
            if st.button("⚙️ Run Auto-Feature Engineering"):
                with st.spinner("Transforming data into ML-ready format..."):
                    eng_response = requests.post(
                        f"{GATEWAY_URL}/engineer-features/{dataset_id}",
                        params={"filename": filename, "target_column": target_col}
                    )
                    
                    if eng_response.status_code == 200:
                        eng_data = eng_response.json()
                        st.success("✅ " + eng_data["message"])
                        
                        e_col1, e_col2, e_col3 = st.columns(3)
                        e_col1.metric("Missing Values Handled", eng_data["missing_values_handled"])
                        e_col2.metric("Original Columns", eng_data["original_shape"][1])
                        e_col3.metric("New Columns (After Encoding)", eng_data["new_shape"][1])
                        
                        st.session_state['data_ready_for_training'] = True
                    else:
                        st.error(f"Engineering Error: {eng_response.text}")

            # --- PHASE 5: AutoML Training ---
            if st.session_state.get('data_ready_for_training', False):
                st.divider()
                st.header("5. AutoML Model Training & Evaluation")
                
                if st.button("🚀 Train Machine Learning Models"):
                    with st.spinner("Training models... This might take a moment!"):
                        train_response = requests.post(
                            f"{GATEWAY_URL}/train-models/{dataset_id}",
                            params={"target_column": target_col, "problem_type": prob_data['problem_type']}
                        )
                        
                        if train_response.status_code == 200:
                            train_data = train_response.json()
                            st.balloons()
                            st.success(f"🎉 Training Complete! The best model is **{train_data['best_model']}**!")
                            
                            st.markdown("### 🏆 Model Leaderboard")
                            st.dataframe(pd.DataFrame(train_data["leaderboard"]), width="stretch")
                        else:
                            st.error(f"Training Error: {train_response.text}")
    else:
        st.error(f"Analysis Error: {analysis_response.text}")