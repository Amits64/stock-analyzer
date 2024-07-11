import pandas as pd
import numpy as np
import logging
import yfinance as yf
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_data(coin_ids, vs_currency='INR', days='max', interval='1d'):
    """
    Fetch historical market data for cryptocurrencies from Yahoo Finance (yfinance).

    Parameters:
    - coin_ids: List of IDs or tickers of cryptocurrencies on Yahoo Finance.
    - vs_currency: The currency in which the prices are represented (default: 'INR').
    - days: The number of days of data to fetch ('max' for all available data).
    - interval: The interval of data points ('1d' for daily, '1h' for hourly, etc.).

    Returns:
    - data_dict: A dictionary where keys are coin_ids and values are pandas DataFrames with columns ['timestamp', 'price'] indexed by timestamp.
    """
    data_dict = {}
    for coin_id in coin_ids:
        try:
            ticker = yf.Ticker(coin_id)
            coin_info = ticker.info
            coin_name = coin_info['longName'] if 'longName' in coin_info else coin_id  # Use full name if available, else use ticker

            historical_data = ticker.history(period=days, interval=interval.lower())

            if not historical_data.empty:
                df = historical_data[['Close']].rename(columns={'Close': 'price'})
                df.index.name = 'timestamp'
                data_dict[coin_name] = df

                # Save to CSV in dataset directory with underscores instead of spaces
                dataset_dir = 'dataset'
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                csv_path = os.path.join(dataset_dir, f'{coin_name.replace(" ", "_")}.csv')
                df.to_csv(csv_path)
                logging.info(f"Saved {coin_name} data to: {csv_path}")

            else:
                logger.warning(f"No price data found for {coin_id} on Yahoo Finance.")
                data_dict[coin_name] = pd.DataFrame()

        except yf.TickerError as e:
            logger.error(f"Error fetching data for {coin_id}: {e}")
            data_dict[coin_name] = pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error fetching data for {coin_id}: {e}")
            data_dict[coin_name] = pd.DataFrame()

    return data_dict


def load_crypto_data(coin_id):
    """
    Load cryptocurrency data from a CSV file.

    Parameters:
    - coin_id: ID or ticker of the cryptocurrency.

    Returns:
    - df: A pandas DataFrame containing historical data for the specified cryptocurrency.
          None if the data file doesn't exist or cannot be loaded.
    """
    csv_file = f'dataset/{coin_id.replace(" ", "_")}_data.csv'  # Adjusted to handle underscores instead of spaces
    if not os.path.exists(csv_file):
        logging.error(f"CSV file not found: {csv_file}")
        return None  # Return None if file not found

    try:
        # Attempt to load CSV data
        df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        logging.info(f"Loaded data from {csv_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {csv_file}: {str(e)}")
        return None


def preprocess_data(df):
    """
    Preprocess the fetched data.

    Parameters:
    - df: A pandas DataFrame with columns ['timestamp', 'price'] indexed by timestamp.

    Returns:
    - df_processed: Preprocessed DataFrame with additional calculated columns.
    """
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log(1 + df['returns'])
    return df


# Function to read coin_ids from symbols.txt
def read_coin_ids_from_file(file_path):
    coin_ids = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            coin_ids = [line.strip() for line in file if line.strip()]  # Read non-empty lines
    return coin_ids


# Example usage:
if __name__ == "__main__":
    # Example fetching data for Bitcoin and Ethereum with correct ticker symbols
    coin_ids = [
        'ZRX-USD', '1INCH-USD', 'AAVE-USD', 'ELF-USD', 'AERO-USD', 'ATH-USD', 'AEVO-USD', 'AIOZ-USD', 'AKT-USD',
        'ALGO-USD',
        'ALT-USD', 'AMP-USD', 'ANKR-USD', 'APE-USD', 'NFT-USD', 'API3-USD', 'APT-USD', 'ANT-USD', 'ARB-USD', 'ABT-USD',
        'ARKM-USD', 'AR-USD', 'ASTR-USD', 'AVAX-USD', 'AXL-USD', 'AXS-USD', 'BRETT-USD', 'BAT-USD', 'BEAM-USD',
        'BDX-USD',
        'SAVAX-USD', 'BICO-USD', 'BNB-USD', 'BTC-USD', 'BTCB-USD', 'BCH-USD', 'BSV-USD', 'BTG-USD', 'BGB-USD',
        'TAO-USD',
        'BTT-USD', 'BLAST-USD', 'STX-USD', 'CDT-USD', 'BLUR-USD', 'BONK-USD', 'BOME-USD', 'ADA-USD', 'CSPR-USD',
        'MEW-USD',
        'TIA-USD', 'CELO-USD', 'CFG-USD', 'LINK-USD', 'XCH-USD', 'CHZ-USD', 'CHR-USD', 'CBETH-USD', 'CETH-USD',
        'COMP-USD',
        'CWBTC-USD', 'CFX-USD', 'PEOPLE-USD', 'CVX-USD', 'CORE-USD', 'CORGIAI-USD', 'ATOM-USD', 'CRO-USD', 'CRV-USD',
        'DAI-USD', 'DASH-USD', 'MANA-USD', 'DCR-USD', 'DESO-USD', 'DEXE-USD', 'DOG-USD', 'DOGE-USD', 'WIF-USD',
        'ETHDYDX-USD',
        'DYDX-USD', 'DYM-USD', 'XEC-USD', 'PRIME-USD', 'ECOIN-USD', 'EGLD-USD', 'ENJ-USD', 'EOS-USD', 'ENA-USD',
        'USDE-USD',
        'ETHFI-USD', 'EETH-USD', 'ETH-USD', 'ETC-USD', 'ENS-USD', 'ETHW-USD', 'FTM-USD', 'FTN-USD', 'FET-USD',
        'FIL-USD',
        'FDUSD-USD', 'FLR-USD', 'FLOKI-USD', 'FLOW-USD', 'FRAX-USD', 'FRXETH-USD', 'FXS-USD', 'GALA-USD', 'GAS-USD',
        'GT-USD',
        'GMX-USD', 'GNO-USD', 'GLM-USD', 'H2O-USD', 'SNX-USD', 'HBAR-USD', 'HNT-USD', 'HOT-USD', 'ILV-USD', 'IMX-USD',
        'INJ-USD', 'ICP-USD', 'IO-USD', 'IOTA-USD', 'IOTX-USD', 'JASMY-USD', 'JTO-USD', 'JUP-USD', 'JST-USD', 'KAS-USD',
        'KAVA-USD', 'RSETH-USD', 'KLAY-USD', 'KCS-USD', 'KSM-USD', 'ZRO-USD', 'LEO-USD', 'LDO-USD', 'STSOL-USD',
        'LSETH-USD',
        'LTC-USD', 'LPT-USD', 'LRC-USD', 'MMX-USD', 'TRUMP-USD', 'MKR-USD', 'MANTA-USD', 'MNT-USD', 'METH-USD',
        'OM-USD',
        'MASK-USD', 'MATIC-USD', 'MEME-USD', 'METIS-USD', 'MINA-USD', 'MOG-USD', 'XMR-USD', 'MSOL-USD', 'MX-USD',
        'MYTH-USD',
        'NEAR-USD', 'NEO-USD', 'CKB-USD', 'NMT-USD', 'NEXO-USD', 'NOS-USD', 'NOT-USD', 'NXM-USD', 'ROSE-USD',
        'OCEAN-USD',
        'OKB-USD', 'OHM-USD', 'ONDO-USD', 'OP-USD', 'ORDI-USD', 'TRAC-USD', 'OSAK-USD', 'OSMO-USD', 'PAAL-USD',
        'CAKE-USD',
        'PAXG-USD', 'PYUSD-USD', 'PENDLE-USD', 'PEPE-USD', 'PEPECOIN-USD', 'DOT-USD', 'POLYX-USD', 'PONKE-USD',
        'POPCAT-USD',
        'GAL-USD', 'PYTH-USD', 'QTUM-USD', 'QNT-USD', 'XRD-USD', 'RVN-USD', 'RAY-USD', 'RNDR-USD', 'EZETH-USD',
        'RSR-USD',
        'XRP-USD', 'RPL-USD', 'RETH-USD', 'RLB-USD', 'RON-USD', 'SAFE-USD', 'SFP-USD', 'SATS-USD', 'SEI-USD',
        'SHIB-USD',
        'SC-USD', 'AGIX-USD', 'SKL-USD', 'SOL-USD', 'SSV-USD', 'ETHX-USD', 'STETH-USD', 'SFRXETH-USD', 'STRK-USD',
        'XLM-USD',
        'GMT-USD', 'SUI-USD', 'SUPER-USD', 'SWETH-USD', 'TRB-USD', 'LUNC-USD', 'LUNA-USD', 'USDT-USD', 'XAUT-USD',
        'XTZ-USD',
        'GRT-USD', 'TON-USD', 'SAND-USD', 'TFUEL-USD', 'THETA-USD', 'RUNE-USD', 'T-USD', 'TKX-USD', 'TRIBE-USD',
        'TRX-USD',
        'TUSD-USD', 'TWT-USD', 'TURBO-USD', 'UNI-USD', 'USDC-USD', 'USDB-USD', 'USDD-USD', 'VET-USD', 'VENOM-USD',
        'VTHO-USD',
        'WEMIX-USD', 'WBT-USD', 'WOO-USD', 'WLD-USD', 'W-USD', 'WBETH-USD', 'WBTC-USD', 'WCFG-USD', 'WEETH-USD',
        'XDC-USD',
        'YFI-USD', 'ZEC-USD', 'ZBC-USD', 'FLUX-USD', 'ZETA-USD', 'ZIL-USD', 'ZK-USD'
    ]
    data = fetch_data(coin_ids)
    for coin_name, df in data.items():
        if not df.empty:
            print(f"Fetched {coin_name} data successfully:")
            print(df.head())

            # Example preprocessing
            df_processed = preprocess_data(df)
            print("\nProcessed Data:")
            print(df_processed.head())
        else:
            print(f"Failed to fetch data for {coin_name}.")
