import json
import platform
from typing import List, Dict
from LEPAUTE import get_collected_data

def load_data(file_path: str = None) -> List[Dict]:
    """
    Load collected data from memory or file.
    
    Args:
        file_path (str, optional): Path to JSON file containing data (non-Pyodide environments).
    
    Returns:
        List[Dict]: List of dictionaries containing pipeline data (images, lie_params, output, etc.).
    """
    try:
        # Try loading from memory first
        data = get_collected_data()
        if data:
            return data
        
        # If file_path provided and not in Pyodide, try loading from file
        if file_path and platform.system() != "Emscripten":
            with open(file_path, "r") as f:
                return json.load(f)
        
        return []
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
