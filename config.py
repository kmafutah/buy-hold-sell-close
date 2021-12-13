import os
import datetime
MT_DATA_PATH = '/Volumes/Users/Jesa/AppData/Roaming/MetaQuotes/Terminal/6AB79ED795024EC1B7F61552A87628BC/MQL5/Files/DWX/'

# DASH_APP_PRIVACY = 'public'
# PATH_BASED_ROUTING = True
# os.environ['PLOTLY_USERNAME'] = 'your-plotly-username'
# os.environ['PLOTLY_API_KEY'] = 'your-plotly-api-key'
# os.environ['PLOTLY_DOMAIN'] = 'https://your-plotly-domain.com'
# os.environ['PLOTLY_API_DOMAIN'] = os.environ['PLOTLY_DOMAIN']
# PLOTLY_DASH_DOMAIN='https://my-dash-manager-plotly-domain.com'
# os.environ['PLOTLY_SSL_VERIFICATION'] = 'True'
TRAINED_MODEL_DIR = "trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
# TURBULENCE_DATA = "data/dow30_turbulence_index.csv"
# TESTING_DATA_FILE = "test.csv"

now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
TRAINED_MODEL_DIR = TRAINED_MODEL_DIR+'/'+now
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
# os.makedirs(TRAINED_MODEL_DIR)


## time_fmt = '%Y-%m-%d'
START_DATE = "1981-01-07"
END_DATE = datetime.datetime.utcnow().strftime("%Y-%m-%d")

START_TRADE_DATE = "2020-21-01"

## dataset default columns
DEFAULT_DATA_COLUMNS = ["time","symbol","open","high","low","close","tick_volume"]


## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = ["macd","boll_ub","boll_lb","rsi_30", "cci_30", "dx_30","close_30_sma","close_60_sma"]


## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00125,
    "batch_size": 15,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "batch_size": 15,
    "ent_coef": "auto_0.1",
}
CUSTOM_PARAMS = {}

################## ASSETS
DEFAULT_ASSETS = ["Boom 300 Index","Crash 300 Index","ETHUSD","LTCUSD"]