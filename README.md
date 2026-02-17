# ML Trading Bot Project

This project contains a suite of tools for Machine Learning-based crypto trading, backtesting, and simulation.

## ⚠️ Important Notes


**`copy_config.py`**: This file is a template for your API keys and should be copied to `config.py`. (or rename this file and update with your keys).

**DISCLAIMER**: This is a high-risk project. I am not responsible for any financial losses incurred while using this bot. Use at your own risk. **I am not a financial advisor, I'm just a coder who likes trading.** I would use the **main.py** excecutable for sideways market and maybe when BTC/ETH are growing and PAXG is not growing that much. I didn't tested **test_high_volatile_trading.py** enough time to say when could be used, but the goal is to reach good performance in high volatility markets.
**In bearish markets (like Q4 2025) I would prefer holding BTC/ETH and don't use any bot. Just buying and hodl until better times** 

## How to run

**First, install the requirements**
pip install -r requirements.txt

**Then, run the bot**
python main.py

**If you want to run the bot in simulation mode**
python main.py --mode simulation

**If you want to run the bot in real mode**
python main.py --mode real

**If you want to run the bot in simulation mode with email alerts**
python main.py --mode simulation --email

**If you want to run the bot in real mode with email alerts**
python main.py --mode real --email

## 🚀 Key Executables

### 1. `main.py` (Standard Bot)
**Best for:** Low Volatility / Stable Trends (e.g., PAXG/USDT, XPLA/USDT).
-   **Strategy:** Predicts market direction using ML models (Random Forest, Gradient Boosting).
-   **Features:**
    -   Train new models.
    -   Run standard backtests.
    -   **Real & Simulation Mode**: Supports crash recovery.
-   **Note:** Works well with Shorting on assets like XPLA.

### 2. `test_high_volatile_trading.py` (High Volatility Bot)
**Best for:** High Volatility Assets (e.g., ETH/USDT, Meme coins).
-   **Strategy:** "Barrier Strategy" (Scalping).
    -   Enters on high probability.
    -   Exits via fixed **Take Profit (1.5%)** or **Stop Loss (1.0%)**.
-   **Resilience:** Logs every move to CSV; recovers position automatically on restart.

### 3. `test_massive_backtests.py`
**Tool:** Standard Validator.
-   Runs hundreds of random simulations to verify the **Standard Bot** (`main.py`) strategy.
-   Reports: Win Rate vs Buy & Hold, Alpha, Average PnL.

### 4. `test_massive_high_volatile_backtests.py`
**Tool:** Barrier Volatility Validator.
-   Runs hundreds of random simulations using the **Barrier Strategy** logic.
-   Useful for tuning TP/SL settings for specific volatile coins.

---

## ⚙️ Configuration

-   **`config.py`**:
    -   **Binance Keys**: Set `BINANCE_API_KEY` and `SECRET` for real trading.
    -   **Email**: Configure Brevo/SMTP for alerts.
    -   **Risk**: Adjust `SIMULATION_CAPITAL` and risk percentages.

---

## 📂 Important Source Files (`src/`)

-   **`src/models.py`**: Logic for training, saving, and loading ML models (`.pkl`).
-   **`src/features.py`**: Technical indicator generation (RSI, MACD, Bollinger Bands).
-   **`src/recovery.py`**: **Critical**. Handles crash recovery by reading CSV logs and syncing with Binance.
-   **`src/real_trading.py`**: Implementation of the Binance connector for the Standard Bot.
-   **`src/simulator.py`**: Virtual trading logic for paper trading.

---

## 💡 Performance Tips

Based on recent testing:

| Asset Type | Recommended Script | Why? |
| :--- | :--- | :--- |
| **Stable / Low Volatility**<br>(e.g., PAXG, XPLA) | `main.py` | The standard ML models capture these trends well. Shorting works effectively here. |
| **High Volatility**<br>(e.g., ETH, SOL) | `test_high_volatile_trading.py` | Volatile assets hit fixed targets (TP/SL) more reliably than standard trend following. |
| **Unpredictable** | **Buy & Hold** | For some assets, no strategy beats simply holding. Use the Massive Backtesters to confirm this before deploying capital. |
