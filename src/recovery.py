import csv
import os
import pandas as pd
import logging
import ccxt

logger = logging.getLogger(__name__)

def get_last_state(csv_file):
    """
    Reads the CSV log file backwards to find the last known state.
    Returns:
        position (dict or None): The last open position, or None if closed.
        last_balance (float or None): The last recorded balance.
    """
    if not os.path.exists(csv_file):
        return None, None

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return None, None
            
        # Ensure column names are standardized (handling different casing if needed)
        df.columns = [c.strip() for c in df.columns]
        
        # Check for required columns
        required_cols = ['Action', 'Price']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"CSV missing required column: {col}")
                return None, None

        # Iterate backwards to find the last trade action
        # We need to find if the last action was an OPEN/BUY or CLOSE/SELL
        # And if it was OPEN/BUY, valid that it wasn't closed later (which shouldn't happen if we read backwards and find the last action)
        
        # But wait, we need to know if the *cycle* is complete.
        # Simple logic: Find the last row with an Action.
        # If Action is BUY/OPEN -> We are in a position.
        # If Action is SELL/CLOSE -> We are NOT in a position.
        
        last_row = df.iloc[-1]
        action = str(last_row['Action']).upper()
        
        last_balance = None
        if 'Total_Portfolio' in df.columns:
             last_balance = last_row['Total_Portfolio']
        elif 'Balance' in df.columns:
             last_balance = last_row['Balance']
             
        position = None
        
        if 'BUY' in action or 'OPEN' in action or action == 'LONG' or action == 'SHORT':
            # We are currently in a position
            entry_price = float(last_row['Price'])
            quantity = 0
            if 'Quantity' in df.columns:
                quantity = float(last_row['Quantity'])
            
            # Determine type
            pos_type = 'LONG'
            if 'SHORT' in action:
                pos_type = 'SHORT'
            
            position = {
                'type': pos_type,
                'entry_price': entry_price,
                'quantity': quantity,
                'start_time': last_row['Timestamp'] if 'Timestamp' in df.columns else None
            }
            logger.info(f"Recovered open position from CSV: {position}")
            
        elif 'SELL' in action or 'CLOSE' in action:
            logger.info("Last action was CLOSE/SELL. No open position.")
            position = None
            
        else:
            logger.warning(f"Unknown last action in CSV: {action}")
            
        return position, last_balance

    except Exception as e:
        logger.error(f"Error reading recovery CSV: {e}")
        return None, None

def sync_with_exchange(exchange, symbol, local_position, mode='SPOT'):
    """
    Verifies local state against Binance.
    Returns the corrected position (exchange state wins) and a status message.
    """
    try:
        logger.info(f"Syncing {symbol} with Binance ({mode})...")
        
        if mode == 'SPOT':
            # Check Balance of Base Asset
            base_asset = symbol.split('/')[0]
            balance = exchange.fetch_balance()
            free_amt = balance.get(base_asset, {}).get('free', 0)
            
            # Threshold to consider "in position" (avoid dust)
            # Assuming min trade is around $5-10, if we have > $5 worth, we are in position?
            # Better: compare with local_quantity
            
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            value = free_amt * price
            
            is_in_position_exchange = value > 10 # Min $10 worth
            
            if is_in_position_exchange:
                if local_position is None:
                    logger.warning(f"⚠️ Mismatch! Local says NO position, but Binance has {free_amt} {base_asset} (${value:.2f}).")
                    # We found a position we didn't know about.
                    # Recover it? Or just log it?
                    # For safety, we should probably update local to match, but we lack Entry Price.
                    # We can estimate Entry Price or just use current price to reset stops?
                    # Better to return it.
                    return {
                        'type': 'LONG',
                        'entry_price': price, # Unknown, effectively
                        'quantity': free_amt,
                        'recovered': True
                    }, "Recovered from Exchange"
                
                else:
                    # Both say YES. Check quantity mismatch?
                    diff = abs(free_amt - local_position['quantity'])
                    if diff / free_amt > 0.05: # >5% difference
                        logger.warning(f"⚠️ Quantity Mismatch: Local {local_position['quantity']}, Binance {free_amt}")
                        local_position['quantity'] = free_amt
                        return local_position, "Quantity Synced"
                    else:
                        return local_position, "Synced"
            
            else:
                if local_position is not None:
                     logger.warning(f"⚠️ Mismatch! Local says IN position, but Binance has {free_amt} {base_asset} (Dust).")
                     return None, "Position Start Cleared (Not found on Exchange)"
                else:
                    return None, "Synced (No Position)"

        elif mode == 'FUTURES':
             # For Futures, fetch_positions
             positions = exchange.fetch_positions([symbol])
             target_pos = None
             for p in positions:
                 if p['symbol'] == symbol:
                     target_pos = p
                     break
             
             if target_pos and float(target_pos['contracts']) > 0:
                 side = target_pos['side'].upper() # long/short
                 amt = float(target_pos['contracts'])
                 entry = float(target_pos['entryPrice'])
                 
                 if local_position is None:
                     return {
                         'type': side,
                         'entry_price': entry,
                         'quantity': amt,
                         'recovered': True
                     }, "Recovered Future from Exchange"
                 else:
                     # Compare
                     return {
                         'type': side,
                         'entry_price': entry,
                         'quantity': amt
                     }, "Synced Future"
             else:
                 if local_position is not None:
                     return None, "Future Closed on Exchange"
                 return None, "Synced (No Position)"
                 
    except Exception as e:
        logger.error(f"Sync Logic Failed: {e}")
        # If sync fails, trust local? Or abort?
        # Safe default: Trust local but warn
        return local_position, f"Sync Failed: {e}"
