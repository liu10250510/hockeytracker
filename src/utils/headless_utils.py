"""
Utility functions to handle OpenCV in headless environments.

This module provides functions to detect and adapt to headless environments
where GUI functions of OpenCV are not available.
"""

import os
import sys
from typing import Callable, Any, Optional


def is_headless_environment() -> bool:
    """
    Detect if we're running in a headless environment.
    
    Returns:
        bool: True if we're in a headless environment, False otherwise
    """
    # Check for environment variables that indicate we're in Streamlit Cloud
    if os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == 'true':
        return True
        
    # Check if we're running in a Docker container or CI environment
    if os.path.exists('/.dockerenv') or os.environ.get('CI', '') == 'true':
        return True
        
    # Check for DISPLAY environment variable on Linux/Unix
    if sys.platform != 'win32' and not os.environ.get('DISPLAY', ''):
        return True
        
    return False


def gui_fallback(func: Callable) -> Callable:
    """
    Decorator to provide a fallback for GUI functions in headless environments.
    
    Args:
        func: The function to decorate
        
    Returns:
        A wrapped function that skips GUI operations in headless environments
    """
    def wrapper(*args, **kwargs):
        if is_headless_environment():
            return None
        return func(*args, **kwargs)
    return wrapper


class HeadlessSafeVideoProcessor:
    """
    Helper class to make OpenCV video operations safe in headless environments.
    """
    
    @staticmethod
    def is_headless_environment() -> bool:
        """
        Detect if we're running in a headless environment.
        
        Returns:
            bool: True if we're in a headless environment, False otherwise
        """
        # Check for environment variables that indicate we're in Streamlit Cloud
        if os.environ.get('STREAMLIT_SHARING', '') == 'true' or os.environ.get('STREAMLIT_SERVER_HEADLESS', '') == 'true':
            return True
            
        # Check if we're running in a Docker container or CI environment
        if os.path.exists('/.dockerenv') or os.environ.get('CI', '') == 'true':
            return True
            
        # Check for DISPLAY environment variable on Linux/Unix
        if sys.platform != 'win32' and not os.environ.get('DISPLAY', ''):
            return True
            
        return False
    
    @staticmethod
    def destroy_all_windows() -> None:
        """
        Safely call cv2.destroyAllWindows() if in a GUI environment.
        """
        if not HeadlessSafeVideoProcessor.is_headless_environment():
            import cv2
            cv2.destroyAllWindows()
    
    @staticmethod
    def show_image(window_name: str, image: Any) -> None:
        """
        Safely call cv2.imshow() if in a GUI environment.
        
        Args:
            window_name: Name of the window
            image: Image to display
        """
        if not is_headless_environment():
            import cv2
            cv2.imshow(window_name, image)
    
    @staticmethod
    def wait_key(delay: int = 1) -> Optional[int]:
        """
        Safely call cv2.waitKey() if in a GUI environment.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            Key code or None if in headless environment
        """
        if not is_headless_environment():
            import cv2
            return cv2.waitKey(delay) & 0xFF
        return None
    
    @staticmethod
    def create_window(window_name: str) -> None:
        """
        Safely call cv2.namedWindow() if in a GUI environment.
        
        Args:
            window_name: Name of the window
        """
        if not is_headless_environment():
            import cv2
            cv2.namedWindow(window_name)
    
    @staticmethod
    def set_mouse_callback(window_name: str, callback: Callable) -> None:
        """
        Safely call cv2.setMouseCallback() if in a GUI environment.
        
        Args:
            window_name: Name of the window
            callback: Mouse callback function
        """
        if not is_headless_environment():
            import cv2
            cv2.setMouseCallback(window_name, callback)
