import gym
import json
import logging
import os
import time
import datetime
import pathlib
from typing import Optional
import config 
import glob
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from gym_mtsim import MtEnv,MtSimulator,SymbolInfo
from stable_baselines3 import A2C,PPO,TD3,SAC,DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class SymInfo():
    name = None
    currency_margin = 'EUR'
    currency_profit = 'USD'
    currencies= tuple(set([currency_margin, currency_profit]))
    trade_contract_size=1
    margin_rate  = 0.5  # MetaTrader info does not contain this value!
    volume_min = 0.01
    volume_max = 1
    volume_step = 0.01   

def get_model(model_name,
                    env,
                    policy="MlpPolicy",
                    policy_kwargs=None,
                    model_kwargs=None,
                    verbose=1):

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[temp_model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(temp_model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **temp_model_kwargs,
        )
        return model
    
def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        # Create Environment
        path_to_info_csv = f'{config.MT_DATA_PATH}AccountInfo/4299206-Kudakwashe-Mafutah.csv'
        acc_info = pd.read_csv(path_to_info_csv)
        sim = MtSimulator(
            unit=acc_info['currency'],
            balance=acc_info['balance'],
            leverage=acc_info['leverage'],
            hedge=True
        )

        
        for asset in config.DEFAULT_ASSETS:
            # print(asset)  
            df = pd.read_csv(max(glob.glob("./" + config.DATA_SAVE_DIR + "/" + '*' + "-" + asset +".csv") , key=os.path.getctime))
            df.index = pd.to_datetime(df.index)
            sy_info = SymInfo()
            sy_info.name = asset
            sy_info.currencies = tuple(set(['EUR','USD']))         
            del df['symbol']
            df['time'] = pd.to_datetime(df['time'])
            df.rename(columns={'open': 'Open', 'close': 'Close'}, inplace=True)
            df.set_index('time',inplace=True)
            sim.symbols_data[asset] = df
            sim.symbols_info[asset] = sy_info


        path_to_pkl = 'all_assets.pkl'
        sim.save_symbols(path_to_pkl)
        del gym.envs.registry.env_specs['mixed-hedge-v0']
        gym.envs.register(
            id='mixed-hedge-v0',
            entry_point='gym_mtsim.envs:MtEnv',
            kwargs={
                'original_simulator': MtSimulator(symbols_filename=path_to_pkl, hedge=True),
                'trading_symbols': config.DEFAULT_ASSETS,
                'window_size': 15,
                'symbol_max_orders': 3
            }
        )
        env = gym.make('mixed-hedge-v0')  
        # env = make_vec_env("mixed-hedge-v0", n_envs=4)

        # model =  get_model(model_name='a2c',env=env,policy="MultiInputPolicy")
        # model.learn(total_timesteps=25000)
        # model.save(config.TRAINED_MODEL_DIR+"/a2c_mixed-hedge-v0")
        # del model
        model = get_model(model_name='ppo',env=env,policy="MultiInputPolicy")
        model.learn(total_timesteps=50000)
        model.save(config.TRAINED_MODEL_DIR+"/ppo_mixed-hedge-v0")  
        del model
        # model = get_model(model_name='sac',env=env,policy="MultiInputPolicy")
        # model.learn(total_timesteps=25000)
        # model.save(config.TRAINED_MODEL_DIR+"/sac_mixed-hedge-v0")  
        # del model  
        # model = get_model(model_name='td3',env=env,policy="MultiInputPolicy")
        # model.learn(total_timesteps=25000)
        # model.save(config.TRAINED_MODEL_DIR+"/td3_mixed-hedge-v0")  
        # del model              
        # model = get_model(model_name='ddpg',env=env,policy="MultiInputPolicy")
        # model.learn(total_timesteps=25000)
        # model.save(config.TRAINED_MODEL_DIR+"/ddpg_mixed-hedge-v0")  
        # del model
                       
        

    elif options.mode == "download_data":
        for asset in config.DEFAULT_ASSETS:
            print(asset, end = '')
            path_to_csv = f'{config.MT_DATA_PATH}Export/*{asset.replace(" ","-")}.csv'
            csvFiles = glob.glob(path_to_csv)
            list_csv=[]
            for csvFile in csvFiles:
                print(".", end = '')
                tmp_df = pd.read_csv(csvFile)
                list_csv.append(tmp_df)


            df = pd.concat(list_csv,axis=0)
            df.drop_duplicates(inplace=True)
            df.set_index('time',inplace=True)
            now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
            df.to_csv("./" + config.DATA_SAVE_DIR + "/" + now + "-" + asset +".csv")
            print()
    elif options.mode == 'test':
        path_to_info_csv = f'{config.MT_DATA_PATH}AccountInfo/4299206-Kudakwashe-Mafutah.csv'
        acc_info = pd.read_csv(path_to_info_csv)
        sim = MtSimulator(
            unit=acc_info['currency'],
            balance=acc_info['balance'],
            leverage=acc_info['leverage'],
            hedge=True
        )        
        for asset in config.DEFAULT_ASSETS:
            files = sorted(glob.iglob(f'{config.MT_DATA_PATH}Export/*{asset.replace(" ","-")}.csv'), key=os.path.getctime, reverse=True)
            # print(files)
            list_csv=[]
            kount = 0
            for csvFile in files:
                kount +=1
                print(".", end = '')
                tmp_df = pd.read_csv(csvFile)
                list_csv.append(tmp_df)
                if kount == 3:
                    break


            df = pd.concat(list_csv,axis=0)
            df.drop_duplicates(inplace=True)
            df.sort_values(by=['time'],inplace=True)
            df.set_index('time',inplace=True)
            # print(df)            
        #     df = pd.read_csv(max(glob.glob(f'{config.MT_DATA_PATH}Export/*{asset.replace(" ","-")}.csv') , key=os.path.getctime))
        #     df.index = pd.to_datetime(df.index)
        #     df.to_csv("./" + config.DATA_SAVE_DIR + "/" + asset +"_live.csv",index=False)
            syminfo = SymInfo()
            syminfo.name = asset
            syminfo.currencies = tuple(set(['EUR','USD']))         
            del df['symbol']
            df.index= pd.to_datetime(df.index)
            df.rename(columns={'open': 'Open', 'close': 'Close'}, inplace=True)
            # df.set_index('time',inplace=True)
            

            # print(df.tail(15))            
            sim.symbols_data[asset] = df
            sim.symbols_info[asset] = syminfo  
            print(asset,len(df))          
        path_to_live_pkl = 'live_assets.pkl'
        sim.save_symbols(path_to_live_pkl)
        del sim
 
        path_to_info_csv = f'{config.MT_DATA_PATH}AccountInfo/4299206-Kudakwashe-Mafutah.csv'
        acc_info = pd.read_csv(path_to_info_csv)
        path_to_live_pkl = 'live_assets.pkl'
        sim = MtSimulator(
            unit=acc_info['currency'],
            balance=acc_info['balance'],
            leverage=acc_info['leverage'],
            hedge=True,
            symbols_filename=path_to_live_pkl
        ) 
        del gym.envs.registry.env_specs['mixed-hedge-v0']
        gym.envs.register(
            id='mixed-hedge-v0',
            entry_point='gym_mtsim.envs:MtEnv',
            kwargs={
                'original_simulator': MtSimulator(symbols_filename=path_to_live_pkl, hedge=True),
                'trading_symbols': config.DEFAULT_ASSETS,
                'window_size': 15,
                'symbol_max_orders': 3
            }
        )
        env = gym.make('mixed-hedge-v0')  
 
        print("env information:")
        for symbol in env.prices:
            print(f"> prices[{symbol}].shape:", env.prices[symbol].shape)
        print("> signal_features.shape:", env.signal_features.shape)
        print("> features_shape:", env.features_shape)
        model = PPO.load(path=config.TRAINED_MODEL_DIR+"/ppo_mixed-hedge-v0",env=env)   
        observation = env.reset()
        done = False
        while not done:
            # action = env.action_space.sample()
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)  

        env.render('advanced_figure', time_format="%m/%d %H:%M")  
                
if __name__ == "__main__":
    main()
