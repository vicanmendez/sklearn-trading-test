"""
Tests para el mecanismo de cooldown post-stop-loss del RealTimeSimulator.

Verifica que:
- Tras un stop loss NO se abre posición inmediatamente.
- El bot espera COOLDOWN_CANDLES_AFTER_SL ciclos antes de considerar re-entrada.
- Se requieren REENTRY_SIGNAL_CONFIRMATION señales BUY consecutivas para abrir posición.
- Un take profit NO activa el cooldown (sólo el stop loss lo hace).
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockConfig:
    SIMULATION_CAPITAL = 1000
    LEVERAGE = 1
    BUY_THRESHOLD = 0.6
    SELL_THRESHOLD = 0.4
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.03
    COOLDOWN_CANDLES_AFTER_SL = 3
    REENTRY_SIGNAL_CONFIRMATION = 2
    EMAIL_ENABLED = False
    TIMEFRAME = '1h'
    CHECK_INTERVAL_SECONDS = 60


def make_simulator():
    """Create a RealTimeSimulator with mocked dependencies."""
    with patch('simulator.recovery') as mock_recovery, \
         patch('simulator.TradingStrategy'), \
         patch('simulator.RealTimeSimulator._initialize_csv'), \
         patch('simulator.RealTimeSimulator._recover_state'):

        mock_recovery.get_last_state.return_value = (None, None)

        from simulator import RealTimeSimulator

        model = MagicMock()
        config = MockConfig()
        sim = RealTimeSimulator(model, 'BTC/USDT', config)

        # Reset cooldown state explicitly (bypassed __init__ via patch)
        sim.stop_loss_cooldown = 0
        sim.consecutive_buy_signals = 0
        sim.position = None
        sim.trades = []
        sim.capital = 1000
        sim.initial_capital = 1000
        return sim


class TestStopLossCooldown(unittest.TestCase):

    def test_no_immediate_reentry_after_stop_loss(self):
        """Después de un SL, el bot NO debe abrir una nueva posición en el mismo ciclo."""
        sim = make_simulator()

        # Simular que justo se cerró una posición por SL
        sim.position = None
        sim.trades = [{'type': 'LONG', 'entry': 100, 'exit': 98, 'pnl': -20, 'pnl_pct': -0.02, 'reason': 'STOP_LOSS'}]
        sim.stop_loss_cooldown = 3  # Cooldown activo
        sim.consecutive_buy_signals = 0

        # Intentar entrar con señal BUY durante cooldown
        sim._manage_positions(signal=1, price=100, timestamp='2024-01-01')

        # El cooldown debe haberse decrementado pero NO debe haber posición
        self.assertIsNone(sim.position, "El bot NO debe abrir posición durante el cooldown.")
        self.assertEqual(sim.stop_loss_cooldown, 2, "El cooldown debe decrementar en 1.")

    def test_cooldown_expires_correctly(self):
        """El cooldown debe expirar correctamente después de N ciclos."""
        sim = make_simulator()
        sim.stop_loss_cooldown = 1  # Último ciclo de cooldown

        # Ciclo que consume el último ciclo de cooldown
        sim._manage_positions(signal=1, price=100, timestamp='2024-01-01')
        self.assertIsNone(sim.position, "Todavía dentro del cooldown, no debe abrir posición.")
        self.assertEqual(sim.stop_loss_cooldown, 0, "El cooldown debe llegar a 0.")

        # Ahora el cooldown terminó, pero se necesita confirmación de señal
        sim._manage_positions(signal=1, price=100, timestamp='2024-01-01')
        self.assertIsNone(sim.position, "Primera señal BUY post-cooldown no es suficiente (requiere 2 consecutivas).")
        self.assertEqual(sim.consecutive_buy_signals, 1, "Debe contar 1 señal BUY.")

        # Segunda señal consecutiva → debe abrir posición
        with patch.object(sim, '_open_position') as mock_open:
            sim._manage_positions(signal=1, price=100, timestamp='2024-01-01')
            mock_open.assert_called_once_with('LONG', 100, '2024-01-01')

    def test_signal_confirmation_required(self):
        """Se requieren N señales BUY consecutivas para abrir posición."""
        sim = make_simulator()
        sim.stop_loss_cooldown = 0  # Sin cooldown

        # Primera señal BUY — no suficiente
        sim._manage_positions(signal=1, price=100, timestamp='t1')
        self.assertIsNone(sim.position)
        self.assertEqual(sim.consecutive_buy_signals, 1)

        # Señal HOLD — resetea el contador
        sim._manage_positions(signal=0, price=100, timestamp='t2')
        self.assertEqual(sim.consecutive_buy_signals, 0, "Un HOLD debe resetear el contador de confirmaciones.")

        # Dos BUY consecutivas ahora sí deben abrir posición
        sim._manage_positions(signal=1, price=100, timestamp='t3')
        self.assertEqual(sim.consecutive_buy_signals, 1)
        with patch.object(sim, '_open_position') as mock_open:
            sim._manage_positions(signal=1, price=100, timestamp='t4')
            mock_open.assert_called_once()

    def test_take_profit_does_not_activate_cooldown(self):
        """Un take profit NO debe activar el cooldown — sólo el stop loss lo hace."""
        sim = make_simulator()

        # Simular cierre por take profit
        sim.trades = [{'type': 'LONG', 'entry': 100, 'exit': 103, 'pnl': 30, 'pnl_pct': 0.03, 'reason': 'TAKE_PROFIT'}]
        sim.position = None
        sim.stop_loss_cooldown = 0  # El TP no debe activar cooldown

        # Con señal BUY y sin cooldown, el bot puede acumular confirmaciones normalmente
        sim._manage_positions(signal=1, price=103, timestamp='t1')
        self.assertEqual(sim.stop_loss_cooldown, 0, "Take profit NO debe activar el cooldown.")
        self.assertEqual(sim.consecutive_buy_signals, 1)

    def test_cooldown_resets_buy_signal_counter(self):
        """Durante el cooldown, el contador de señales BUY debe mantenerse en 0."""
        sim = make_simulator()
        sim.stop_loss_cooldown = 2
        sim.consecutive_buy_signals = 0

        # Señales BUY durante el cooldown no deben acumularse
        sim._manage_positions(signal=1, price=100, timestamp='t1')
        self.assertEqual(sim.consecutive_buy_signals, 0, "No se acumularán señales BUY durante el cooldown.")

        sim._manage_positions(signal=1, price=100, timestamp='t2')
        self.assertEqual(sim.consecutive_buy_signals, 0)
        self.assertEqual(sim.stop_loss_cooldown, 0, "El cooldown debe haber expirado.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
