import streamlit as st
import os
import re
import pandas as pd
import glob
import plotly.express as px

# Setup paths
# The results directory is expected to be in the parent directory of webapp
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def parse_log_file(filepath):
    # Regex to match lines like:
    # Pre-train Epoch: 1 [38400/40200 (95%)]	Loss: 2.829866	Acc: 12.76%
    # Train Epoch: 1 [38400/40200 (95%)] Loss: ... Acc: ...
    # Keep the last recorded loss/acc for each epoch
    pattern = re.compile(r"Epoch:\s*(\d+).*?Loss:\s*([\d\.]+).*?Acc:\s*([\d\.]+)%")
    
    epochs_data = {}
    current_phase_offset = 0
    last_epoch = 0
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                acc = float(match.group(3))
                
                if epoch < last_epoch:
                    current_phase_offset += last_epoch
                    
                continuous_epoch = epoch + current_phase_offset
                # Update with the latest value for the given epoch
                epochs_data[continuous_epoch] = {'Loss': loss, 'Accuracy': acc}
                last_epoch = epoch
                
    if not epochs_data:
        return pd.DataFrame()
        
    df = pd.DataFrame.from_dict(epochs_data, orient='index')
    df.index.name = 'Epoch'
    df.sort_index(inplace=True)
    return df

st.set_page_config(page_title="Training Logs Dashboard", layout="wide")
st.title("Model Training Dashboard")

if not os.path.exists(RESULTS_DIR):
    st.error(f"Results directory not found at: {RESULTS_DIR}")
else:
    log_files = glob.glob(os.path.join(RESULTS_DIR, "*.log"))
    if not log_files:
        st.write("No log files found in the results directory.")
    else:
        log_file_names = sorted([os.path.basename(f) for f in log_files])
        selected_files = st.multiselect("Select Log Files to Compare", log_file_names, default=log_file_names[:1] if log_file_names else None)
        
        if selected_files:
            # Prepare dataframes
            all_loss = pd.DataFrame()
            all_acc = pd.DataFrame()
            
            for file in selected_files:
                filepath = os.path.join(RESULTS_DIR, file)
                df = parse_log_file(filepath)
                if not df.empty:
                    all_loss = pd.concat([all_loss, df['Loss'].rename(file)], axis=1)
                    all_acc = pd.concat([all_acc, df['Accuracy'].rename(file)], axis=1)
            
            if not all_loss.empty:
                st.subheader("Loss Curve")
                fig_loss = px.line(all_loss, labels={'value': 'Loss', 'index': 'Epoch', 'variable': 'Model'})
                st.plotly_chart(fig_loss, use_container_width=True)
                
                st.subheader("Accuracy Curve (%)")
                fig_acc = px.line(all_acc, labels={'value': 'Accuracy (%)', 'index': 'Epoch', 'variable': 'Model'})
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                st.warning("No valid training data found in the selected files.")
