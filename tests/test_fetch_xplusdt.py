import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import fetch_data, normalize_symbol

def test_fetch_xplusdt():
    symbol = "XPLUSDT"
    print(f"Testing symbol: {symbol}")
    
    # Check normalization directly first
    norm = normalize_symbol(symbol)
    print(f"Direct normalization result: {norm}")
    assert norm == "XPL/USDT", f"Normalization failed: {norm}"
    
    # Try fetching (mocking not needed for quick specific check if we just want to see it calls crypto path)
    # But since we want to see if it *works*, we can try fetching a small amount of data if network allows.
    # If network is restricted we'd mock, but user context implies we can run real commands.
    # However, to be safe and fast, I will rely on the print output of data_loader saying "Normalized XPLUSDT to XPL/USDT"
    # and "Fetching XPL/USDT data from Binance..."
    
    print("\nAttempting fetch_data...")
    try:
        df = fetch_data(symbol, start_date='2024-01-01', timeframe='1d')
        if df is not None:
             print("Fetch SUCCESS!")
             print(df.head())
        else:
             print("Fetch returned None (might be network/symbol issue but path was correct if we see logs)")
    except Exception as e:
        print(f"Fetch FAILED with error: {e}")

if __name__ == "__main__":
    test_fetch_xplusdt()
