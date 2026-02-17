import unittest
import os
import csv
import sys
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import recovery

class TestRecovery(unittest.TestCase):
    def setUp(self):
        self.test_csv = "test_recovery.csv"
        # CLEANUP
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)

    def write_csv(self, rows):
        with open(self.test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Price', 'PnL', 'Quantity', 'Balance', 'Info'])
            writer.writerows(rows)

    def test_recover_open_position(self):
        # Scenario: Buy happened, no sell
        rows = [
            ['2023-01-01 10:00:00', 'BUY', '1000', '0', '0.5', '500', 'Entry']
        ]
        self.write_csv(rows)
        
        pos, bal = recovery.get_last_state(self.test_csv)
        self.assertIsNotNone(pos)
        self.assertEqual(pos['type'], 'LONG')
        self.assertEqual(pos['entry_price'], 1000.0)
        self.assertEqual(pos['quantity'], 0.5)
        self.assertEqual(bal, 500.0) # Should be string or float depending on implementation, let's see

    def test_recover_closed_position(self):
        # Scenario: Buy then Sell
        rows = [
            ['2023-01-01 10:00:00', 'BUY', '1000', '0', '0.5', '500', 'Entry'],
            ['2023-01-01 11:00:00', 'SELL', '1100', '0.1', '0.5', '550', 'Exit']
        ]
        self.write_csv(rows)
        
        pos, bal = recovery.get_last_state(self.test_csv)
        self.assertIsNone(pos)
        self.assertEqual(float(bal), 550.0)

    def test_recover_long_action(self):
        # Scenario: Action is just 'LONG'. 
        # Helper schema: Timestamp, Action, Price, PnL, Quantity, Balance, Info
        rows = [
            ['2023-01-01 10:00:00', 'LONG', '2000', '0.0', '0.18', '1000', 'OPEN']
        ]
        self.write_csv(rows)
        
        pos, bal = recovery.get_last_state(self.test_csv)
        self.assertIsNotNone(pos)
        self.assertEqual(pos['type'], 'LONG')
        self.assertEqual(pos['entry_price'], 2000.0)
        self.assertEqual(pos['quantity'], 0.18)
        self.assertEqual(float(bal), 1000.0)

    def test_recover_simulator_schema(self):
        # Simulator uses: Timestamp,Action,Price,Quantity,Amount/PnL,Info,Total_Portfolio
        with open(self.test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Price', 'Quantity', 'Amount/PnL', 'Info', 'Total_Portfolio'])
            writer.writerow(['2026-02-11 12:00:00', 'LONG', '5119.37', '0.1855', '950.0', 'OPEN', '1000'])

        pos, bal = recovery.get_last_state(self.test_csv)
        self.assertIsNotNone(pos)
        self.assertEqual(pos['type'], 'LONG')
        self.assertEqual(pos['quantity'], 0.1855)
        self.assertEqual(float(bal), 1000.0)

    def test_recover_partial_columns(self):
        # Old CSV format might lack Quantity/Balance
        with open(self.test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Action', 'Price', 'PnL', 'Info']) # No Qty/Bal
            writer.writerow(['2023-01-01 10:00:00', 'BUY', '100', '0', 'Init'])
            
        pos, bal = recovery.get_last_state(self.test_csv)
        self.assertIsNotNone(pos)
        self.assertEqual(pos['entry_price'], 100.0)
        self.assertEqual(pos['quantity'], 0) # Defaulted
        self.assertIsNone(bal)

if __name__ == '__main__':
    unittest.main()
