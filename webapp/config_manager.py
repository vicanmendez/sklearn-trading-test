import os
import re
import importlib
import sys

# Add root to path if not already (app.py does this, but for standalone safety)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.py')

def get_config_values():
    """
    Reads config.py and extracts API keys using regex to avoid executing code.
    Returns a dict with found keys.
    """
    if not os.path.exists(CONFIG_PATH):
        return {}
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
        
    keys = {}
    
    # Regex for assignments like KEY = "VALUE" or KEY = 'VALUE'
    patterns = {
        'BINANCE_API_KEY': r'BINANCE_API_KEY\s*=\s*[\'"]([^\'"]*)[\'"]',
        'BINANCE_API_SECRET': r'BINANCE_API_SECRET\s*=\s*[\'"]([^\'"]*)[\'"]',
        'BREVO_API_KEY': r'BREVO_API_KEY\s*=\s*[\'"]([^\'"]*)[\'"]'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            keys[key] = match.group(1)
            
    return keys

def update_config_values(new_values):
    """
    Updates config.py with new values using regex replacement.
    new_values: dict of key -> new_value
    """
    if not os.path.exists(CONFIG_PATH):
        return False, "config.py not found"
        
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for key, value in new_values.items():
            if value is None: continue 
            
            # Pattern to find the existing assignment
            pattern = fr'({key}\s*=\s*)([\'"])([^\'"]*)([\'"])'
            
            if re.search(pattern, content):
                # Replace group 3 (the value)
                content = re.sub(pattern, fr'\g<1>\g<2>{value}\g<4>', content)
            else:
                # Append if not found? No, strictly update for parity safety.
                pass
                
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Reload config module if loaded
        if 'config' in sys.modules:
            import config
            importlib.reload(config)
            
        return True, "Config updated successfully"
        
    except Exception as e:
        return False, str(e)
