"""
SYNERGY ANALYSIS FOR HCC PREDICTION TOOL
Simple version - just reads pre-computed results from JSON file
"""

import json
import os

def add_synergy_to_tool():
    """Returns pre-computed synergy pairs for the web tool"""
    try:
        # Look for the cache file
        cache_path = 'synergy_cache.json'
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                synergies = json.load(f)
            return synergies
        else:
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

# This runs when you execute the script directly
if __name__ == "__main__":
    print("Synergy analysis module loaded")
    print(f"Found {len(add_synergy_to_tool())} synergistic pairs in cache")