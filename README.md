# buy-hold-sell-close
hands free trading - Don't Run on live account

The .MT4, .MT5 and dwx_client.py are files form https://github.com/darwinex/dwxconnect

TODO: 
TradingGymEnv - need to be fixed to mimic exchange env

chikwambo.py -  change to use on bar event, hist-data returns delayed data but good for quick dev

chikwambo.py - Use indicators to normalise data... **done - (see app.py)**

chikwambo.py - add ability to trade on 15M,1H and 4H to allow minimal balance.**50% done,pending 'mini Balance' - (see app.py)**

chikwambo.py -  on powerful PC allow model to be Trained every 15min/30min this has shown to have better results

<code>pip install Metatrader5 dash hashlib gym_mtsim</code> -  only for app.y - will try create non Metatrader5 dependent verion on app.py

<code>pip install stable-baselines3[extra]</code>

Change the two lines with <b>MT4_files_dir</b> to point to your MT4 or MT5 where the .MT4, .MT5 is running.

<code>MT4_files_dir = '/Volumes/Users/XXXXXXX/AppData/Roaming/MetaQuotes/Terminal/XXXXXXXXXXXXXXXXXXXXXXXXXX/MQL4/Files/' </code>

You can also run on two separate machines. just make sure the machine running chikwambo.py is pointing to machine running MetaTrader
