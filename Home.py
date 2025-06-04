import streamlit as st

# Set page configuration (should be at the top of the file)
st.set_page_config(page_title="EEG CNN Model Evaluation", page_icon="🧠", layout="wide")

# Initialize session state for page navigation if not already set
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.selectbox(
        "Choose a page",
        ["Home", "Raw CNN Model", "Processed CNN Model"],
        key="page_select"  # Unique string key for the selectbox widget
    )

# Update session state with selected page
st.session_state.page = page

# Display content for the Home page
if page == "Home":
    # Title and header
    st.title("🧠 EEG CNN Model Evaluation")
    st.markdown("""
    Welcome to the **EEG CNN Model Evaluation App**! This application allows you to evaluate two convolutional neural network (CNN) models for EEG data analysis:
    
    - **Raw CNN Model**: Analyzes raw EEG data without preprocessing.
    - **Processed CNN Model**: Evaluates preprocessed EEG data with customizable parameters.
    
    Use the sidebar to navigate to the model evaluation pages, where you can upload EEG data and labels to perform 3-fold cross-validation and view detailed metrics.
    """)
   
    # Info box for additional guidance
    st.info("📢 Upload the appropriate EEG data and labels files on each page to perform 3-fold cross-validation and view evaluation metrics.")

# Placeholder for other pages (handled by separate files in pages/ directory)
else:
    st.write(f"You are on the **{st.session_state.page}** page.")
    st.info("This content is a placeholder. The actual page content is in the respective page file.")
    st.markdown("[GitHub Repository](https://github.com/bellapd/MNISTMindBigData)")