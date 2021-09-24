import gym
import pandas as pd
import numpy as np
from gym.utils import seeding
from gym import spaces
from enum import Enum
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('seaborn')

class TradingEnvAction(Enum):
    STAY = 0
    BUY = 1
    SELL = 2
    CLOSE = 3

class TradingEnvTicket(object):
    def __init__(self, order_type, open_price, take_profit, stop_loss, lots):
        self.order_type = order_type
        self.open_price = open_price
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.lots = lots
        self.trade_fee = 0.0003  # unit

class TradingEnvAccountInformation(object):
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.fixed_balance = initial_balance
        self.total_pips_buy = 10
        self.total_pips_sell = 10

    def items(self):
        return [('balance', self.balance), ('fixed_balance', self.fixed_balance), ('total_pips_buy', self.total_pips_buy), ('total_pips_sell', self.total_pips_sell)]

# class Actions(Enum):
#     Sell = 0
#     Buy = 1


# class Positions(Enum):
#     Short = 0
#     Long = 1

#     def opposite(self):
#         return Positions.Short if self == Positions.Long else Positions.Long


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, frame_bound):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # spaces
        self.action_space = spaces.Discrete(len(TradingEnvAction))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = TradingEnvAction.STAY.value
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.history = {}
        return self._get_observation()


    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        trade = False
        if (action != None):
            trade = True

        if trade:
            self._position = action
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,           
            position = Counter(self._position_history),
            last_15_position_predictions = self._position_history[-15:],
            position_predictions = self._position_history
        )
        self._update_history(info)

        return observation, step_reward, self._done, info


    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size):self._current_tick]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            marker = 'o'

            if position == TradingEnvAction.SELL.value:
                color = 'red'
                marker = 'v'
            elif position == TradingEnvAction.BUY.value:
                color = 'green'
                marker = '^'
            if position == TradingEnvAction.STAY.value:
                color = 'yellow'
                marker = 'o'
            elif position == TradingEnvAction.CLOSE.value:
                color = 'blue'
                marker = 'o'
            elif position == None:
                color = 'purple'                
            if color:
                plt.scatter(tick, self.prices[tick], marker=marker,color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        close_ticks = []
        stay_ticks = []        
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == TradingEnvAction.SELL.value:
                short_ticks.append(tick)
            elif self._position_history[i] == TradingEnvAction.BUY.value:
                long_ticks.append(tick)
            elif self._position_history[i] == TradingEnvAction.CLOSE.value:
                close_ticks.append(tick)
            elif self._position_history[i] == TradingEnvAction.STAY.value:
                stay_ticks.append(tick)

        
        plt.plot(short_ticks, self.prices[short_ticks], 'rv')
        plt.plot(long_ticks, self.prices[long_ticks], 'g^')
        plt.plot(close_ticks, self.prices[close_ticks], 'bo')
        plt.plot(stay_ticks, self.prices[stay_ticks], 'yo')        

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
        
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        step_reward = 0
        # print("p:",TradingEnvAction(self._position),"a:",TradingEnvAction(action),"-1:",TradingEnvAction(self._position_history[-1]),"c:",list(filter(None,self._position_history)))
        trade = False
        if (self._position != None and action != None):
            trade = True

        if trade:

            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price
            # print(action,self._position_history[-1:],price_diff)
            if list(filter(None,self._position_history)) == []:
                step_reward = -abs(price_diff)

            if (action == TradingEnvAction.BUY.value) or (action == TradingEnvAction.SELL.value):
                step_reward = abs(price_diff)

            if (action == TradingEnvAction.STAY.value) and (self._position_history[-1] == TradingEnvAction.BUY.value) and (current_price > last_trade_price ):
                step_reward += abs(price_diff)/15

            if (action == TradingEnvAction.STAY.value) and (self._position_history[-1] == TradingEnvAction.SELL.value) and (current_price < last_trade_price ):
                step_reward += abs(price_diff)/15

            if (action == TradingEnvAction.CLOSE.value) and (self._position_history[-1] == TradingEnvAction.SELL.value) and (current_price > last_trade_price ):
                step_reward += abs(price_diff)/15

            if (action == TradingEnvAction.CLOSE.value) and (self._position_history[-1] == TradingEnvAction.BUY.value) and (current_price < last_trade_price ):
                step_reward += abs(price_diff)/15                                            

            if (action == TradingEnvAction.STAY.value) and (self._position_history[-1] == TradingEnvAction.CLOSE.value):
                step_reward += -abs(price_diff) 
            
            if (action == TradingEnvAction.STAY.value) and (self._position_history[-1] == TradingEnvAction.STAY.value) and ((current_price < self.prices[-2]) or (current_price > self.prices[-2])):
                step_reward += abs(price_diff) 

            if (action == TradingEnvAction.STAY.value) and (self._position_history[-1] == TradingEnvAction.STAY.value):
                step_reward += abs(price_diff) 

    
        return step_reward


    def _update_profit(self, action):
        trade = False
        if (self._position != None and action != None):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == TradingEnvAction.BUY.value:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price



    def max_possible_profit(self):
        self.trade_fee = 0.0003  # unit
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = TradingEnvAction.SELL.value
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = TradingEnvAction.BUY.value

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self._position_history[-1] == TradingEnvAction.CLOSE.value:
                if position == TradingEnvAction.SELL.value:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self._position_history[-1] == TradingEnvAction.STAY.value:
                if position == TradingEnvAction.BUY.value:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit