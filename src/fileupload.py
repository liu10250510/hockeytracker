import streamlit as st
import tempfile
import os

def upload_file():
    """
    Upload a file using Streamlit's file uploader.
    
    Returns:
        str: Path to the uploaded file.
    """
    uploaded_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        st.success(f"File uploaded successfully: {temp_file_path}")
        return temp_file_path
    
    return None

def display_uploaded_file(file_path: str):
    """
    Display the uploaded file in Streamlit.
    
    Args:
        file_path (str): Path to the uploaded file.
    """
    if file_path:
        st.video(file_path)
    else:
        st.warning("No file uploaded yet.")

if __name__ == "__main__":
    st.title("File Upload Example")
    
    # Upload file
    uploaded_file_path = upload_file()
    
    # Display the uploaded file
    display_uploaded_file(uploaded_file_path)
    


