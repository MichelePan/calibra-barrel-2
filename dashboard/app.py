import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ================================
# INIZIALIZZAZIONE DASH APP
# ================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Lista titoli
TICKERS = {
    "ALPHABET INC": "GOOGL",
    "AMAZON": "AMZN",
    "ATOSSA THERAPEUTICS INC": "ATOS",
    "ALIBABA GROUP HOLDING": "BABA",
    "AMERICA AIRLINES": "AAL",
    "BANK OF AMERICA CORP": "BAC",
    "BEYOND MEAT": "BYND",
    "CERENCE": "CRNC",
    "COMCAST CORPORATION": "CMCSA",
    "COTERRA ENERGY INC": "CTRA",
    "CRONOS GROUP INC": "CRON",
    "DELTA AIRLINES": "DAL",
    "DEVON ENERGY CORPORATION": "DVN",
    "FISERV": "FISV",
    "FORD MOTOR CO": "F",
    "HASBRO": "HAS",
    "HP INC": "HPQ",
    "HUNTINGTON BANCSHARES INC": "HBAN",
    "INCANNEX HEALTHCARE INC": "IXHL",
    "INTEL": "INTC",
    "LYFT INC": "LYFT",
    "PAYPAL HOLDINGS INC": "PYPL",
    "PINTEREST INC": "PINS",
    "RIVIAN AUTOMOTIVE INC": "RIVN",
    "SNAP INC": "SNAP",
    "THE COCA-COLA COMPANY": "KO",
    "TESLA": "TSLA",
    "TILRAY BRANDS INC": "TLRY",
    "UNIQURE NV": "QURE"
}

# ================================
# FUNZIONE ESTRAZIONE CLOSE
# ================================
def extract_close_column(df):

    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            ser = df['Close']
            if isinstance(ser, pd.DataFrame):
                ser = ser.iloc[:, 0]
            return ser.to_frame(name='Close')
        else:
            df.columns = ['_'.join(map(str, col)) for col in df.columns]
            close_cols = [c for c in df.columns if 'close' in c.lower()]
            if close_cols:
                return df[[close_cols[0]]].rename(columns={close_cols[0]: 'Close'})
            raise ValueError("Colonna 'Close' non trovata (MultiIndex).")

    if 'Close' in df.columns:
        return df[['Close']].copy()

    if df.shape[1] == 1:
        return df.rename(columns={df.columns[0]: 'Close'})

    for c in df.columns:
        if 'close' in str(c).lower():
            return df[[c]].rename(columns={c: 'Close'})

    raise ValueError("Impossibile identificare la colonna 'Close'")

# ================================
# LAYOUT
# ================================
app.layout = html.Div([

    html.H1(
        "BARREL #2 Portfolio - Stocks Screener",
        style={'textAlign': 'center', 'marginBottom': '30px'}
    ),

    html.Div([

        # ----------------------------
        # COLONNA SINISTRA - CONTROLLI
        # ----------------------------
        html.Div([

            html.H3("Parametri", style={'textAlign': 'center'}),

            html.Label("Numero valori storici da scaricare", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='historical-selector',
                options=[
                    {'label': '120', 'value': 120},
                    {'label': '360', 'value': 360},
                    {'label': '720', 'value': 720}
                ],
                value=120,
                style={'marginBottom': '15px'}
            ),

            html.Label("Periodo di previsione futura (giorni)", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='forecast-selector',
                options=[
                    {'label': '30 giorni', 'value': 30},
                    {'label': '60 giorni', 'value': 60},
                    {'label': '120 giorni', 'value': 120}
                ],
                value=30,
                style={'marginBottom': '20px'}
            ),

            html.Button(
                'Applica',
                id='apply-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '10px',
                    'backgroundColor': '#007BFF',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px'
                }
            )

        ], style={
            'width': '20%',
            'padding': '20px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'margin': '0 10px'
        }),

        # ----------------------------
        # COLONNA DESTRA - TABELLA
        # ----------------------------
        html.Div([

            dash_table.DataTable(
            id='results-table',
            columns=[
                {"name": "TICKER", "id": "ticker"},
                {"name": "ON MKT", "id": "on_mkt", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "MIN", "id": "minimo", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "AVG", "id": "media", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "MAX", "id": "massimo", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "FORECAST MIN", "id": "forecast_min", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "FORECAST VALUE", "id": "forecast_value", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "FORECAST MAX", "id": "forecast_max", "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Δ % FORECAST", "id": "delta_pct", "type": "numeric", "format": {"specifier": ".2f"}}
            ],
            data=[],
            sort_action="native",
            page_action="none",

            style_table={'overflowX': 'auto'},
            style_cell={'padding': '10px', 'fontFamily': 'Arial', 'fontSize': '14px'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#F5F5F5'},

            # >>> QUI: forecast_value in rosso
            style_data_conditional=[
                # ON MKT - GREEN
                {
                    "if": {
                        "filter_query": "{on_mkt} > -100",
                        "column_id": "on_mkt"
                    },
                    "color": "green",
                    "fontWeight": "bold"
                },
                
                # FORECAST MIN < 0 - ORANGE
                {
                    "if": {
                        "filter_query": "{forecast_min} < 0",
                        "column_id": "forecast_min"
                    },
                    "color": "orange",
                    "fontWeight": "bold"
                },
                
                # FORECAST VALUE > ON MKT → VERDE
                {
                    "if": {
                        "filter_query": "{forecast_value} > {on_mkt}",
                        "column_id": "forecast_value"
                    },
                    "color": "blue",
                    "fontWeight": "bold"
                },
            
                # FORECAST VALUE < ON MKT → ROSSO
                {
                    "if": {
                        "filter_query": "{forecast_value} < {on_mkt}",
                        "column_id": "forecast_value"
                    },
                    "color": "red",
                    "fontWeight": "bold"
                },
                # DELTA GREEN/FUCHSIA
                {
                    "if": {
                        "filter_query": "{delta_pct} > 20",
                        "column_id": "delta_pct"
                    },
                    "color": "green",
                    "fontWeight": "bold"
                },
                {
                    "if": {
                        "filter_query": "{delta_pct} < 0",
                        "column_id": "delta_pct"
                    },
                    "color": "#FF00FF",
                    "fontWeight": "bold"
                }
            ],
        )

        ], style={'width': '80%', 'padding': '20px'})

    ], style={'display': 'flex'})

])

# ================================
# CALLBACK
# ================================
@app.callback(
    Output('results-table', 'data'),
    Input('apply-button', 'n_clicks'),
    [
        State('historical-selector', 'value'),
        State('forecast-selector', 'value')
    ]
)
def update_table(n_clicks, historical_period, forecast_period):

    if n_clicks == 0:
        return []

    rows = []

    for _, ticker in TICKERS.items():
        try:
            df_raw = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            on_mkt_price = float(df_raw['Close'].dropna().iloc[-1])
            df_close = extract_close_column(df_raw).dropna()
            df_close = df_close.tail(historical_period).reset_index()

            if 'Date' not in df_close.columns:
                df_close = df_close.rename(columns={df_close.columns[0]: 'Date'})

            df_close['Date'] = pd.to_datetime(df_close['Date'])
            df_close['Close'] = pd.to_numeric(df_close['Close'], errors='coerce')
            df_close = df_close.dropna().reset_index(drop=True)

            if len(df_close) < 10:
                raise ValueError("Dati insufficienti per stimare ARMA(2,2).")

            model = ARIMA(df_close['Close'], order=(2, 0, 2)).fit()
            forecast = model.forecast(steps=forecast_period)
            conf_int = model.get_forecast(steps=forecast_period).conf_int()
            
            delta_pct = ((forecast.iloc[-1] - on_mkt_price) / on_mkt_price) * 100
            
            rows.append({
                "ticker": ticker,
                "on_mkt": on_mkt_price,
                "minimo": float(df_close['Close'].min()),
                "media": float(df_close['Close'].mean()),
                "massimo": float(df_close['Close'].max()),
                "forecast_min": float(conf_int.iloc[-1, 0]),
                "forecast_value": float(forecast.iloc[-1]),
                "forecast_max": float(conf_int.iloc[-1, 1]),
                "delta_pct": float(delta_pct)
            })

        except Exception:
            rows.append({
                "ticker": ticker,
                "on_mkt": on_mkt_price,
                "minimo": np.nan,
                "media": np.nan,
                "massimo": np.nan,
                "forecast_min": np.nan,
                "forecast_value": np.nan,
                "forecast_max": np.nan,
                "delta_pct": np.nan
            })

    return rows

# ================================
# SERVER PER PRODUZIONE
# ================================
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)