"""
File export utilities for Hockey Tracker.

This module provides utilities for saving output files to a temporary folder
and creating a zip download functionality for the Streamlit interface.
"""

import os
import tempfile
import shutil
import zipfile
import io
from typing import List, Dict, Union, Optional, BinaryIO, Tuple
import streamlit as st
from pathlib import Path


def create_temp_output_folder() -> str:
    """
    Create a temporary folder for storing output files.
    
    Returns:
        str: Path to the created temporary folder
    """
    temp_dir = tempfile.mkdtemp(prefix="hockey_tracker_output_")
    return temp_dir

def create_zip_from_folder(output_folder: str, zip_filename: Optional[str] = None) -> Tuple[str, bytes]:
    """
    Create a zip file from all files in the output folder.
    
    Args:
        output_folder: Path to the folder containing files to zip
        zip_filename: Optional name for the zip file (default: uses folder name)
        
    Returns:
        Tuple[str, bytes]: Tuple containing zip filename and zip file content as bytes
        
    Raises:
        FileNotFoundError: If the output folder doesn't exist
        IOError: If there is an error creating the zip file
    """
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"Output folder not found: {output_folder}")
    
    # Determine zip filename if not provided
    if zip_filename is None:
        folder_name = os.path.basename(output_folder)
        zip_filename = f"{folder_name}.zip"
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Get relative path for the archive
                    arcname = os.path.relpath(file_path, output_folder)
                    zip_file.write(file_path, arcname=arcname)
        
        return zip_filename, zip_buffer.getvalue()
    except Exception as e:
        raise IOError(f"Error creating zip file from output folder: {str(e)}")


def create_zip_download_button(
    output_folder: str,
    button_text: str = "Download All Files",
    zip_filename: Optional[str] = None
) -> None:
    """
    Create a download button for a zip file containing all files in the output folder.
    
    Args:
        output_folder: Path to the folder containing files to zip
        button_text: Text to display on the download button
        zip_filename: Optional name for the zip file (default: uses folder name)
        
    Raises:
        FileNotFoundError: If the output folder doesn't exist
        IOError: If there is an error creating the zip file
    """
    # Get zip filename and content
    if zip_filename is None:
        folder_name = os.path.basename(output_folder)
        zip_filename = f"{folder_name}.zip"
    
    try:
        zip_filename, zip_content = create_zip_from_folder(output_folder, zip_filename)
        
        # Create download button for zip file
        st.download_button(
            label=button_text,
            data=zip_content,
            file_name=zip_filename,
            mime="application/zip"
        )
    except Exception as e:
        st.error(f"Error creating zip download: {str(e)}")

def cleanup_temp_folder(folder_path: str) -> None:
    """
    Clean up a temporary folder and its contents.
    
    Args:
        folder_path: Path to the temporary folder to clean up
    """
    try:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
    except Exception as e:
        # Just log the error, don't raise
        print(f"Warning: Failed to clean up temporary folder {folder_path}: {str(e)}")
