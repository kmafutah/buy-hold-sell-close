from typing import Counter
import gym
import json
from numpy import array
import pandas as pd
from ta import add_all_ta_features
from time import sleep
from libs.gymenv.TradingGymEnv import TradingEnv, TradingEnvAction
from libs.dwx_client import dwx_client
from datetime import datetime, timedelta
from gym import register
import matplotlib.pyplot as plt
import os, time
plt.style.use('seaborn')
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from stable_baselines3 import PPO

def get_data(symbol='#Ethereum',timeframe='M1'):
    MT4_files_dir = '/Volumes/Users/XXXXXXX/AppData/Roaming/MetaQuotes/Terminal/XXXXXXXXXXXXXXXXXXXXXXXXXX/MQL4/Files/'
    dwx = dwx_client(metatrader_dir_path=MT4_files_dir,verbose=False)
    end = datetime.now()
    start = end - timedelta(minutes=1440)  # last 24 hours
    print("Getting ...", start, end,datetime.now(),datetime.utcnow())
    dwx.get_historic_data(symbol, timeframe, start.timestamp(), end.timestamp())
    
    try:
        data =  pd.DataFrame.from_dict(dwx.historic_data[symbol+'_'+timeframe],orient='columns')
        data.T.to_csv('live_'+symbol+'-'+timeframe+'.csv',header=False)
        print("DF COUNT: ",len(data.T),flush=True)
        print(data.T.describe())
    except:
        data = pd.DataFrame()
        
    df = data.T
    return df


def PPO_Model():
    # start = datetime.utcnow()
    print(env.df.info(),env.df.describe(),flush=True)
    start = datetime.now()
    model_name = "ppo_TradingGymEnv-v6_Mac-Sep"
    if os.path.isfile(model_name+'.zip'):
        datecreated = datetime.fromtimestamp(os.path.getctime(model_name+'.zip'))
        print("File Create Date: ",datecreated + timedelta(minutes=15),":",start - datecreated)
    
    # print(datetime.fromtimestamp(time.ctime(os.path.getctime(model_name+'.zip'))) + timedelta(minutes=15))
    if not os.path.isfile(model_name+'.zip'):
        print('Train : ',start,flush=True)    
        model = PPO("MlpPolicy", env, verbose=False)
        model.learn(total_timesteps=len(env.df))
        print('Saving : ', datetime.now()-start,flush=False)
        model.save(model_name)
    else:
        #del model # remove to demonstrate saving and loading
        print('Loading : ',start,flush=False) 

        model = PPO.load(model_name,env=env,verbose=False)

        print('Done in : ',datetime.now()-start ,flush=False)
    return model

def my_process_data(df, window_size, frame_bound):
    start = frame_bound[0] - window_size
    end = frame_bound[1]
    df = add_all_ta_features(
    df, open="open", high="high", low="low", close="close", volume="tick_volume", fillna=True)
    prices = df.loc[:, 'close'].to_numpy()[start:end]
    feature_list =['open','high','low','close','tick_volume','volume_adi','volume_obv','volume_cmf','volume_fi','volume_mfi','volume_em','volume_sma_em','volume_vpt','volume_nvi',
    'volume_vwap','volatility_atr','volatility_bbm','volatility_bbh','volatility_bbl','volatility_bbw','volatility_bbp','volatility_bbhi','volatility_bbli',
    'volatility_kcc','volatility_kch','volatility_kcl','volatility_kcw','volatility_kcp','volatility_kchi','volatility_kcli','volatility_dcl','volatility_dch',
    'volatility_dcm','volatility_dcw','volatility_dcp','volatility_ui','trend_macd','trend_macd_signal','trend_macd_diff','trend_sma_fast','trend_sma_slow',
    'trend_ema_fast','trend_ema_slow','trend_adx','trend_adx_pos','trend_adx_neg','trend_vortex_ind_pos','trend_vortex_ind_neg','trend_vortex_ind_diff',
    'trend_trix','trend_mass_index','trend_cci','trend_dpo','trend_kst','trend_kst_sig','trend_kst_diff','trend_ichimoku_conv','trend_ichimoku_base',
    'trend_ichimoku_a','trend_ichimoku_b','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_aroon_up','trend_aroon_down','trend_aroon_ind',
    'trend_psar_up','trend_psar_down','trend_psar_up_indicator','trend_psar_down_indicator','trend_stc','momentum_rsi','momentum_stoch_rsi','momentum_stoch_rsi_k',
    'momentum_stoch_rsi_d','momentum_tsi','momentum_uo','momentum_stoch','momentum_stoch_signal','momentum_wr','momentum_ao','momentum_kama','momentum_roc',
    'momentum_ppo','momentum_ppo_signal','momentum_ppo_hist','others_dr','others_dlr','others_cr']
    signal_features = df.loc[:, feature_list].to_numpy()[start:end]
    # signal_features = df.loc[:, ['open', 'high', 'low', 'close', 'tick_volume']].to_numpy()[start:end]
    return prices, signal_features

class MyStocksEnv(TradingEnv):
    
    def __init__(self, prices, signal_features, **kwargs):
        self._prices = prices
        self._signal_features = signal_features
        super().__init__(**kwargs)

    def _process_data(self):
        return self._prices, self._signal_features

register(id='TradingGymEnv-v0',entry_point='libs.gymenv.TradingGymEnv:TradingEnv',max_episode_steps=1440)
model = None
while True:
    
    train_data =  get_data()

    if not train_data.empty:
        train_data.index = pd.to_datetime(train_data.index)

        prices, signal_features = my_process_data(df=train_data, window_size=15, frame_bound=(15, len(train_data)))
        env = MyStocksEnv(prices, signal_features, df=train_data, window_size=15, frame_bound=(15, len(train_data)))        
        if model is None:
            model = PPO_Model()
        else:
            model.env = env
        observation = env.reset()
        actions = []
        
        while True:
            action, _states = model.predict(observation)
            # print(TradingEnvAction(action).name,flush=True)
            actions.append(TradingEnvAction(action).name)
            observation, reward, done, info = env.step(action)
            # print(observation)
            if done:
                
                # print(TradingEnvAction(info['last_15_position_predictions'][-1:]))
                MT4_files_dir = '/Volumes/Users/XXXXXXX/AppData/Roaming/MetaQuotes/Terminal/XXXXXXXXXXXXXXXXXXXXXXXXXX/MQL4/Files/'
                dwx = dwx_client(metatrader_dir_path=MT4_files_dir,verbose=False)

                print(Counter(actions))
                print(info['position'])
                print(info['last_15_position_predictions']) 
                print(info['last_15_position_predictions'][-1])
                print('----------------------------------------')
                print('Trying to ' + TradingEnvAction(action).name)
                print('Orders Before Action: ',dwx.account_info)
                for val in dwx.open_orders:
                    print('Order = ',val,"|",dwx.open_orders[val]['lots'],"|",dwx.open_orders[val]['type'],"|",dwx.open_orders[val]['pnl'],"|",dwx.open_orders[val]['open_price'],"|",dwx.open_orders[val]['commission']+dwx.open_orders[val]['pnl'])

                if TradingEnvAction.BUY.value == action:
                    for val in dwx.open_orders:
                        if (dwx.open_orders[val]['pnl'] > 0) and (dwx.open_orders[val]['type'] == 'sell'):
                            print('closing sell: ',val,flush=True)
                            dwx.close_order(val)
                            sleep(1)                    
                    dwx.open_order(symbol='#Ethereum', order_type='buy', lots=0.02,magic=137)
                    sleep(1)
                    print('Orders After Action: ',dwx.account_info)
                    for val in dwx.open_orders:
                        print('Order = ',val,dwx.open_orders[val]['type'],"|",dwx.open_orders[val]['pnl'],"|",dwx.open_orders[val]['open_price'],"|",dwx.open_orders[val]['commission']+dwx.open_orders[val]['pnl'])

                elif TradingEnvAction.SELL.value == action:
                    for val in dwx.open_orders:
                        if (dwx.open_orders[val]['pnl'] > 0) and (dwx.open_orders[val]['type'] == 'buy'):
                            print('closing buy: ',val,flush=True)
                            dwx.close_order(val)
                            sleep(2)
                    dwx.open_order(symbol='#Ethereum', order_type='sell', lots=0.02,magic=137)
                    sleep(1)
                    print('Orders After Action: ',dwx.account_info)
                    for val in dwx.open_orders:
                        print('Order = ',val,dwx.open_orders[val]['type'],"|",dwx.open_orders[val]['pnl'],"|",dwx.open_orders[val]['open_price'],"|",dwx.open_orders[val]['commission']+dwx.open_orders[val]['pnl'])
                elif TradingEnvAction.STAY.value == action:
                    for val in dwx.open_orders:
                        print('Order = ',val,dwx.open_orders[val]['type'],"|",dwx.open_orders[val]['pnl'],"|",dwx.open_orders[val]['open_price'],"|",dwx.open_orders[val]['commission']+dwx.open_orders[val]['pnl'])
                    
                    print(dwx.account_info['equity'] - dwx.account_info['balance'])
                    print('--------------------------------------------------')
                    break
                elif TradingEnvAction.CLOSE.value == action:
                    print('Trying to close by 137')
                    for val in dwx.open_orders:
                        if dwx.open_orders[val]['pnl'] > 0:
                            print('closing : ',val,flush=True)
                            dwx.close_order(val)
                            sleep(2)
                    
                    print("ACC INFO: \n",dwx.account_info)
                    # sleep(5)
                    print("ORDERS AFTER CLOSE SIGNAL : ")
                    for val in dwx.open_orders:
                        print('Order = ',val,dwx.open_orders[val]['type'],"|",dwx.open_orders[val]['pnl'],"|",dwx.open_orders[val]['open_price'],"|",dwx.open_orders[val]['commission']+dwx.open_orders[val]['pnl'])
                    

                else:
                    pass
                sleep(5)
                break
    else:
        sleep(25)



    
    
#register(id='TradingGymEnv-v0',entry_point='libs.gymenv.TradingGymEnv:TradingEnv',max_episode_steps=train_data.count())

#Episodes = 1
#observation = []
#ENV_NAME = 'TradingGymEnv-v0'
#env = gym.make(ENV_NAME,df=train_data)

#observation = env.reset()
#while True:
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#    # env.render()
#    if done:
#        print("info:", info)
#        break

#plt.cla()
#env.render_all()
#plt.show()
