from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import ta
import os
from symtable import Symbol
import time
import gym
import dash
import config
import hashlib
import pandas_datareader.data as web
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as  dcc
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
from gym_mtsim import MtEnv, MtSimulator, OrderType, Timeframe, FOREX_DATA_PATH
from MetaTrader5 import SymbolInfo 
from random import randint
from stablebaselines3_model import DRLAgent as DRLAgent_sb3,MODEL_KWARGS


env = gym.make('mixed-hedge-v0')
class MyMtEnv(MtEnv):
    def _get_prices(self, keys: List[str] = ['Open','High','Low','Close','Volume','volume_adi','volume_obv','volume_cmf','volume_fi','volume_mfi','volume_em','volume_sma_em','volume_vpt','volume_nvi',
    'volume_vwap','volatility_atr','volatility_bbm','volatility_bbh','volatility_bbl','volatility_bbw','volatility_bbp','volatility_bbhi','volatility_bbli',
    'volatility_kcc','volatility_kch','volatility_kcl','volatility_kcw','volatility_kcp','volatility_kchi','volatility_kcli','volatility_dcl','volatility_dch',
    'volatility_dcm','volatility_dcw','volatility_dcp','volatility_ui','trend_macd','trend_macd_signal','trend_macd_diff','trend_sma_fast','trend_sma_slow',
    'trend_ema_fast','trend_ema_slow','trend_adx','trend_adx_pos','trend_adx_neg','trend_vortex_ind_pos','trend_vortex_ind_neg','trend_vortex_ind_diff',
    'trend_trix','trend_mass_index','trend_cci','trend_dpo','trend_kst','trend_kst_sig','trend_kst_diff','trend_ichimoku_conv','trend_ichimoku_base',
    'trend_ichimoku_a','trend_ichimoku_b','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_aroon_up','trend_aroon_down','trend_aroon_ind',
    'trend_psar_up','trend_psar_down','trend_psar_up_indicator','trend_psar_down_indicator','trend_stc','momentum_rsi','momentum_stoch_rsi','momentum_stoch_rsi_k',
    'momentum_stoch_rsi_d','momentum_tsi','momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao','momentum_kama','momentum_roc',
    'momentum_ppo','momentum_ppo_signal','momentum_ppo_hist','others_dr','others_dlr','others_cr']) -> Dict[str, np.ndarray]:
        return super()._get_prices(keys)


sim = MtSimulator()
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "13rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "11rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}

TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

MODELS_LIST = ["a2c", "ddpg", "td3", "sac", "ppo"]

def get_symbol_names():
    # connect to MetaTrader5 platform
    mt5.initialize()

    # get symbols
    symbols = mt5.symbols_get()
    symbols_df = pd.DataFrame(symbols, columns=symbols[0]._asdict().keys())

    symbol_names = symbols_df['name'].tolist()
    # mt5.shutdown()
    return symbol_names

def make_map():
    global sim
    sim = MtSimulator(
        unit='USD',
        balance=10000.,
        leverage=100.,
        stop_out_level=0.2,
        hedge=True,
        symbols_filename=config.DATA_SAVE_DIR+'/all_M1.pkl'
    )
    total =0
    labels = []
    parents= []
    dfObj = pd.DataFrame(columns=['Symbol', 'Map', 'Close','Volume'])
    figTRee = go.Treemap()
    for symbol in sim.symbols_data:
        
        if len(sim.symbols_data[symbol]['Close']) != 0:
            labels.append(symbol)
            parents.append(str(sim.symbols_info[symbol]).split('/')[0])
            dfObj = dfObj.append({'Symbol': symbol, 'Map': str(sim.symbols_info[symbol]).split('/')[0], 
            'Close': sim.symbols_data[symbol]['Close'][0],'Volume': sim.symbols_data[symbol]['Volume'][0]}, ignore_index=True)


    fig = px.treemap(dfObj, path=['Map','Symbol'],values='Volume', 

    width=750, height=500
    )
    colors = []

    for i in range(len(dfObj['Map'].unique())):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    fig.update_layout(
    treemapcolorway = colors, #defines the colors in the treemap
    margin = dict(t=50, l=25, r=25, b=25))
    return fig


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


sidebar = html.Div(
    [
        html.Br(),html.Br(),
        html.H2("Trade", className="display-5"),
        html.Hr(),
        html.P(
            "stock,crypto,forex...", className="lead"
        ),
        dbc.Nav(
            [
                # dbc.NavLink("Train", href="/", active="exact"),
                # dbc.NavLink("Test", href="/page-1", active="exact"),
                # dbc.NavLink("Validate", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=False,
        ),
    ],
    style=SIDEBAR_STYLE,
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Img(src='assets/logo1.gif', height="50px"),
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        # dbc.Col(html.Img(src='assets/logo2.gif', height="50px")),
                        dbc.Col(dbc.NavbarBrand("Time 2", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="./",
                style={"textDecoration": "none"},
            ),
        ]
    ),
    color="offwhite",
    dark=False,
)

# Dash App Layout
symbol_dropdown = html.Div([
    html.P('Symbol:'),
    dcc.Dropdown(
        id='symbol-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in get_symbol_names()],
        value='Volatility 300 (1s) Index',
        multi=True
    )
])

# Dash App Layout
indicator_dropdown = html.Div([
    html.P('Technical indicators:'),
    dcc.Dropdown(
        id='indicator-dropdown',
        options=[

# {'label': 'Volume','value':'()'},
# {'label': '---------------','value':'()'},
{'label': 'Money Flow Index (MFI)','value':'(MFI)'},
{'label': 'Accumulation/Distribution Index (ADI)','value':'(ADI)'},
{'label': 'On-Balance Volume (OBV)','value':'(OBV)'},
{'label': 'Chaikin Money Flow (CMF)','value':'(CMF)'},
{'label': 'Force Index (FI)','value':'(FI)'},
{'label': 'Ease of Movement (EoM, EMV)','value':'(EoM, EMV)'},
{'label': 'Volume-price Trend (VPT)','value':'(VPT)'},
{'label': 'Negative Volume Index (NVI)','value':'(NVI)'},
{'label': 'Volume Weighted Average Price (VWAP)','value':'(VWAP)'},

# {'label': 'Volatility','value':'()'},
{'label': 'Average True Range (ATR)','value':'(ATR)'},
{'label': 'Bollinger Bands (BB)','value':'(BB)'},
{'label': 'Keltner Channel (KC)','value':'(KC)'},
{'label': 'Donchian Channel (DC)','value':'(DC)'},
{'label': 'Ulcer Index (UI)','value':'(UI)'},
# {'label': '---------------','value':'()'},
# {'label': 'Trend','value':'()'},

{'label': 'Simple Moving Average (SMA)','value':'(SMA)'},
{'label': 'Exponential Moving Average (EMA)','value':'(EMA)'},
{'label': 'Weighted Moving Average (WMA)','value':'(WMA)'},
{'label': 'Moving Average Convergence Divergence (MACD)','value':'(MACD)'},
{'label': 'Average Directional Movement Index (ADX)','value':'(ADX)'},
{'label': 'Vortex Indicator (VI)','value':'(VI)'},
{'label': 'Trix (TRIX)','value':'(TRIX)'},
{'label': 'Mass Index (MI)','value':'(MI)'},
{'label': 'Commodity Channel Index (CCI)','value':'(CCI)'},
{'label': 'Detrended Price Oscillator (DPO)','value':'(DPO)'},
{'label': 'KST Oscillator (KST)','value':'(KST)'},
{'label': 'Ichimoku Kinkō Hyō (Ichimoku)','value':'(Ichimoku)'},
{'label': 'Parabolic Stop And Reverse (Parabolic SAR)','value':'(Parabolic SAR)'},
{'label': 'Schaff Trend Cycle (STC)','value':'(STC)'},
# {'label': '---------------','value':'()'},
# {'label': 'Momentum','value':'()'},
{'label': 'Relative Strength Index (RSI)','value':'(RSI)'},
{'label': 'Stochastic RSI (SRSI)','value':'(SRSI)'},
{'label': 'True strength index (TSI)','value':'(TSI)'},
{'label': 'Ultimate Oscillator (UO)','value':'(UO)'},
{'label': 'Stochastic Oscillator (SR)','value':'(SR)'},
{'label': 'Williams %R (WR)','value':'()'},
{'label': 'Awesome Oscillator (AO)','value':'(AO)'},
{'label': 'Kaufman`s Adaptive Moving Average (KAMA)','value':'(KAMA)'},
{'label': 'Rate of Change (ROC)','value':'(ROC)'},
{'label': 'Percentage Price Oscillator (PPO)','value':'(PPO)'},
{'label': 'Percentage Volume Oscillator (PVO)','value':'(PVO)'},
# {'label': '---------------','value':'()'},
# {'label': 'Others','value':'()'},
{'label': 'Daily Return (DR)','value':'(DR)'},
{'label': 'Daily Log Return (DLR)','value':'(DLR)'},
{'label': 'Cumulative Return (CR)','value':'(CR)'}
],
        value='(Ichimoku)',
        multi=True
    )
])

timeframe_dropdown = html.Div([
    html.P('Timeframe:'),
    dcc.Dropdown(
        id='timeframe-dropdown',
        options=[{'label': timeframe, 'value': timeframe} for timeframe in TIMEFRAMES],
        value='M1'
    )
])

model_dropdown = html.Div([
    html.P('Model:'),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model, 'value': model} for model in MODELS_LIST],
        value='a2c'
    )
])

num_bars_input = html.Div([
    html.P('Number of Candles'),
    dbc.Input(id='num-bar-input', type='number', value='15')
])

num_timestemps_input = html.Div([
    html.P('Timestemps'),
    dbc.Input(id='num-timestemps-input', type='number', value='200')
])

num_learn_rate_input = html.Div([
    html.P('Learning rate'),
    dbc.Input(id='num-lr-input', type='number', value='0.0001')
])

symbol_controls = dbc.Card(
    [
        dbc.Row(dbc.Col(symbol_dropdown)),
        dbc.Row([dbc.Col(timeframe_dropdown),
        dbc.Col(num_bars_input)]),
    ],
    body=True,
)

model_controls = dbc.Card(
    [
        dbc.Row(dbc.Col(indicator_dropdown)),
        dbc.Row([dbc.Col(model_dropdown),
        dbc.Col(num_timestemps_input),
        dbc.Col(num_learn_rate_input)]),
    ],
    body=True,
)


content_page_1_row = html.Div(
    [
        dbc.Row([
        dbc.Col(symbol_controls),
        dbc.Col(model_controls)
        ]),html.Hr(),
        dbc.Row([
        dbc.Col( [
        dbc.Row(dbc.Button(
            "Market Map",
            color="info",
            id="btn_map",
            className="me-1",
        ),className="d-grid gap-2 col-6 mx-auto"),
        dbc.Row(html.Div([html.P(id="msg_map_id", children=["Map button not clicked"])]))]
            
        ),
        dbc.Col( [
        dbc.Row(dbc.Button(
            "Market Charts",
            color="secondary",
            id="btn_chart",
            className="me-1",
        ),className="d-grid gap-2 col-6 mx-auto"),
        dbc.Row(html.Div([html.P(id="msg_chart_id", children=["Chart button not clicked"])]))]
            
        ),        
        dbc.Col( [
        dbc.Row(dbc.Button(
            "Start Training",
            color="danger",
            id="btn_train",
            className="me-1",
        ),className="d-grid gap-2 col-6 mx-auto"),
        dbc.Row(html.Div([html.P(id="msg_train_id", children=["Train button not clicked"])]))]
            
        )]),
        dbc.Tabs(
            [
                dbc.Tab(label="SymbolMap", tab_id="symbolmap_tid"),
                dbc.Tab(label="SymbolChart", tab_id="symbolchart_tid"),
            ],
            id="tabs",
            active_tab="symbolmap_tid"),
            html.Div(id="tab-content", className="p-4")
    ]
)

map_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("MAP", className="card-text"),
            html.Div([
                html.Div([
                    dcc.Graph(id='treemap-graph',
                    figure = make_map(),
                    )
                ])
            ]),
        ]
    ),
    className="mt-3",
)

chart_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("CHART", className="card-text"),
            html.Div([
                html.Div([
                    dcc.Graph(id='chart-graph')
                ])
            ]),
        ]
    ),
    className="mt-3",
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

# app.layout = html.Div([dcc.Location(id="url"), sidebar, navbar,content])
app.layout = dbc.Container([
    dcc.Store(id="time-2-trade-stock-crypto-forex"),
    dcc.Location(id="url"), 
    sidebar, 
    navbar,
    content
])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return content_page_1_row
    elif pathname == "/":
        return html.P("Page 1.")
    elif pathname == "/":
        return html.P("Page 2")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Alert(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


@app.callback(
    Output("msg_train_id", "children"),
    Input("btn_train", "n_clicks"),
    State('symbol-dropdown', 'value'),
    State('indicator-dropdown', 'value'), 
    State('model-dropdown', 'value'), 
    State('num-timestemps-input', 'value'), 
    State('num-lr-input', 'value'),  
    running=[
        (Output("btn_train", "disabled"), True, False),
    ],
)
def callback(n_clicks,symbols,indicator,model_name,timestemps,lr):
    filename = hashlib.sha224(bytes(' '.join(map(str, symbols)).replace(' ',''), encoding="raw_unicode_escape")).hexdigest()
    print(f"env information:{filename}")
    if n_clicks != None:
        for symbol in env.prices:
            print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

            print("> signal_features.shape:", env.signal_features.shape)
            print("> features_shape:", env.features_shape)
            agent_params = MODEL_KWARGS[model_name]
            agent = DRLAgent_sb3(env = env)

            model = agent.get_model(model_name, model_kwargs = agent_params)
            trained_model = agent.train_model(model=model, 
                                    tb_log_name=model_name,
                                    total_timesteps=int(timestemps))
            print('Training finished!')
            trained_model.save(config.TRAINED_MODEL_DIR+'/'+model_name+filename)
            print('Trained model saved in ' + str(model_name+filename))

    return [f"Trained {n_clicks} times"]

@app.callback(
    Output("msg_map_id", "children"),
    Input("btn_map", "n_clicks"),
    State('symbol-dropdown', 'value'), 
    State('timeframe-dropdown', 'value'), 
    State('num-bar-input', 'value'),
    running=[
        (Output("btn_map", "disabled"), True, False),
    ],
)
def callback(n_clicks,symbols,timeframe,barcount):
    # print(symbols)
    if n_clicks == None:
        global sim
        sim = MtSimulator(
            unit='USD',
            balance=10000.,
            leverage=100.,
            stop_out_level=0.2,
            hedge=False,
        )
        symli = get_symbol_names()

        sim.download_data(
            symbols=symli,
            time_range=(
                datetime.now() - timedelta(minutes=1) ,
                datetime.now()
            ),
            timeframe=Timeframe.M1
        )

        sim.save_symbols(config.DATA_SAVE_DIR+'/all_M1.pkl')        
    else:
        filename = hashlib.sha224(bytes(' '.join(map(str, symbols)).replace(' ',''), encoding="raw_unicode_escape")).hexdigest()
        if not type(symbols) in (tuple,list):
            symbols =[symbols]


        sim = MtSimulator(
            unit='USD',
            balance=10000.,
            leverage=0.3,
            stop_out_level=0.5,
            hedge=True,
        )  

        sim.download_data(
            symbols=symbols,
            time_range=(
                datetime.now() - timedelta(minutes=len(barcount)*1440),
                datetime.now()
            ),
            timeframe=Timeframe[timeframe]
        )

        sim.save_symbols(config.DATA_SAVE_DIR+'/'+filename+'_'+timeframe+'_Training.pkl')        
    return [f"Got {symbols} {n_clicks} times"]

@app.callback(
    Output("msg_chart_id", "children"), 
    Output('tabs','value'),
    Output('chart-graph','figure'),
    Input("btn_chart", "n_clicks"),
    State('symbol-dropdown', 'value'), 
    State('timeframe-dropdown', 'value'), 
    State('num-bar-input', 'value'),
    State('indicator-dropdown', 'value'), 
    State('model-dropdown', 'value'), 
    State('num-timestemps-input', 'value'), 
    State('num-lr-input', 'value'),  
    running=[
        (Output("btn_chart", "disabled"), True, False),
    ],       
)
def callback(n_clicks,symbols,timeframe,barcount,indicators,model_name,timestamp,learnrate):#,symbols,timeframe,barcount,indicators,model_name,timestamp,learnrate):

    filename = hashlib.sha224(bytes(' '.join(map(str, symbols)).replace(' ',''), encoding="raw_unicode_escape")).hexdigest()
    single_symbol=False
    if not type(symbols) in (tuple,list):
        symbols =[symbols]  
        single_symbol=True


    # print('----------------------------------------------------------------')
    # print(n_clicks,symbols,timeframe,barcount,model_name,timestamp,learnrate)
    sim = MtSimulator(
        unit='USD',
        balance=10000.,
        leverage=100.,
        stop_out_level=0.2,
        hedge=True,
        symbols_filename=config.DATA_SAVE_DIR+'/'+filename+'_'+timeframe+'_Training.pkl'
    )  
    # fees ={} 

    for symbol in symbols:
        df = ta.add_all_ta_features(sim.symbols_data[symbol],'Open','High','Low','Close','Volume',True)
        # print(df)
        sim.symbols_data[symbol] = df
        # a = mt5.symbol_info(symbol)
        # symbol_info_dict = mt5.symbol_info(symbol)._asdict()
        # for prop in symbol_info_dict:
        #     print("  {}={}".format(prop, symbol_info_dict[prop]))
        
    print(int(barcount))
    global env
    env = MyMtEnv(
        original_simulator=sim,
        trading_symbols=symbols,
        window_size=int(barcount),
        # time_points=[desired time points ...],
        hold_threshold=0.5,
        close_threshold=0.5,
        fee=lambda symbol: {
            symbol: max(0., np.random.normal(0.0007, 0.00005)),
        }[symbol],
        symbol_max_orders=2,
        multiprocessing_processes=2
    ) 

    print(env.signal_features)
    print("env information:")

    for symbol in env.prices:
        print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)

    print("> signal_features.shape:", env.signal_features.shape)
    print("> features_shape:", env.features_shape)   
    env.reset()    
    env.action_space.sample()
    fig = env.render('advanced_figure', time_format="%m-%d %H:%M:%S",return_figure=True,figsize=(900,600)) 
    return [f"Clicked {n_clicks} times","symbolchart_tid",fig]

@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "symbolmap_tid":
        return map_content
    elif at == "symbolchart_tid":
        return chart_content
    return html.P("This shouldn't ever be displayed...")


if __name__ == "__main__":
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)       
    app.run_server(port=8888,debug=True)