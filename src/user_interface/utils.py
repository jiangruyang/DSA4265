import asyncio
import streamlit as st
from typing import Callable, Any, Awaitable
import threading
import sys
import os


def patch_streamlit_file_watcher():
    """
    Patches Streamlit's file watcher to safely handle PyTorch modules.
    This prevents the "torch::class_" error when Streamlit tries to watch PyTorch modules.
    """
    try:
        # Only apply if torch is installed
        if 'torch' in sys.modules:
            # Set environment variable to exclude torch from module watching
            # Only set if not already set (so app.py setting takes precedence)
            if "STREAMLIT_SERVER_WATCH_EXCLUDE_PATTERNS" not in os.environ:
                os.environ["STREAMLIT_SERVER_WATCH_EXCLUDE_PATTERNS"] = "torch.*,torchvision.*"
            
            # Try to patch Streamlit's watcher to prevent errors when paths are accessed
            if 'streamlit.watcher.local_sources_watcher' in sys.modules:
                watcher_module = sys.modules['streamlit.watcher.local_sources_watcher']
                
                # Only patch if these attributes actually exist
                if hasattr(watcher_module, 'extract_paths') and hasattr(watcher_module, 'get_module_paths'):
                    original_extract_paths = watcher_module.extract_paths
                    original_get_module_paths = watcher_module.get_module_paths
                    
                    # Create a safer version of extract_paths that handles torch modules
                    def safe_extract_paths(module):
                        # Skip path extraction for torch modules
                        if hasattr(module, "__name__") and (
                            module.__name__.startswith("torch.") or 
                            module.__name__ == "torch"
                        ):
                            return []
                        
                        # Safely handle modules with custom path attributes
                        try:
                            return original_extract_paths(module)
                        except (AttributeError, RuntimeError):
                            # If we hit any errors with __path__, return empty list
                            return []
                    
                    # Create a safer version of get_module_paths
                    def safe_get_module_paths(module_name):
                        # Skip torch modules completely
                        if module_name.startswith("torch.") or module_name == "torch":
                            return []
                        
                        try:
                            return original_get_module_paths(module_name)
                        except (RuntimeError, AttributeError) as e:
                            # If we hit the specific error about torch classes or event loop,
                            # log it but don't crash
                            if "Tried to instantiate class '__path__._path'" in str(e) or "no running event loop" in str(e):
                                return []
                            # Re-raise other errors
                            raise
                    
                    # Apply the patches
                    watcher_module.extract_paths = safe_extract_paths
                    watcher_module.get_module_paths = safe_get_module_paths
    except Exception as e:
        # Don't crash if patching fails - just continue without patching
        print(f"Warning: Failed to patch Streamlit file watcher: {e}")


def initialize_app_event_loop():
    """
    Initializes the application-level event loop in session state.
    This should be called once when the app starts.
    """
    # First patch the file watcher to prevent torch errors
    patch_streamlit_file_watcher()
    
    if "app_loop" not in st.session_state:
        # Create a new event loop
        loop = asyncio.new_event_loop()
        
        # Create a thread to run the event loop
        def run_event_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()
        
        thread = threading.Thread(target=run_event_loop, args=(loop,), daemon=True)
        thread.start()
        
        # Store in session state
        st.session_state.app_loop = loop
        st.session_state.app_loop_thread = thread


def run_async(func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
    """
    Run an async function using the application's shared event loop.
    This preserves async context across multiple calls and allows proper tool execution.
    
    Args:
        func: The async function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the async function
    """
    # Ensure the app event loop exists
    initialize_app_event_loop()
    
    # Create a future to get the result
    future = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), st.session_state.app_loop)
    
    # Wait for the result with a timeout
    try:
        return future.result(timeout=60.0)  # 60 second timeout
    except asyncio.TimeoutError:
        future.cancel()
        raise TimeoutError(f"Async operation timed out after 60 seconds: {func.__name__}")
    except Exception as e:
        raise e 