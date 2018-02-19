import os
import backtrader as bt
import numpy as np

from logbook import WARNING, INFO, DEBUG

from btfxwss.backtesting.datafeed.multi_asset_dataset import MultiAssetDataset
from btgym import BTgymEnv, BTgymDataset, BTgymRandomDataDomain
from btgym.strategy.observers import Reward, Position, NormPnL
from btgym.algorithms import Launcher, Unreal, AacStackedRL2Policy
from btgym.research import DevStrat_4_11

MyCerebro = bt.Cerebro()

MyCerebro.addstrategy(
    DevStrat_4_11,
    drawdown_call=10,  # max % to loose, in percent of initial cash
    target_call=10,  # max % to win, same
    skip_frame=10,
    gamma=0.99,
    reward_scale=7,  # gardient`s nitrox, touch with care!
    state_ext_scale=np.linspace(3e3, 1e3, num=5)
)

MyCerebro.broker.setcash(2000)
MyCerebro.broker.setcommission(commission=0.0001, leverage=10.0)  # commisssion to imitate spread
MyCerebro.addsizer(bt.sizers.SizerFix, stake=50, )

MyCerebro.addobserver(Reward)
MyCerebro.addobserver(Position)
MyCerebro.addobserver(NormPnL)

data_bitfinex = [
    './data/bitfinex_public_ohlcv.csv',
]

data_forex = [
    './data/DAT_ASCII_EURUSD_M1_201701.csv',
    './data/DAT_ASCII_EURUSD_M1_201702.csv',
    './data/DAT_ASCII_EURUSD_M1_201703.csv',
    './data/DAT_ASCII_EURUSD_M1_201704.csv',
    './data/DAT_ASCII_EURUSD_M1_201705.csv',
    './data/DAT_ASCII_EURUSD_M1_201706.csv',
]

parsing_params = dict(
    sep=',',
    header=0,
    index_col='timestamp',
    parse_dates=['timestamp'],
    names=['id', 'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'],

    # Pandas to BT.feeds params:
    timeframe=1,  # 1 minute.
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=-1,
    openinterest=-1,
)

BitfinexDataset = MultiAssetDataset(
    filename=data_bitfinex,
    symbol='OMGUSD',
    target_period={'days': 5, 'hours': 0, 'minutes': 0},
    trial_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 3, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 1, 'hours': 00},
        test_period={'days': 1, 'hours': 0, 'minutes': 0}),
    episode_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 0, 'hours': 12, 'minutes': 0},
        time_gap={'days': 0, 'hours': 5},
        start_00=False),
    parsing_params=parsing_params,
    log_level=INFO,
)

ForexDataset = BTgymRandomDataDomain(
    filename=data_forex,
    target_period={'days': 10, 'hours': 0, 'minutes': 0},
    trial_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 5, 'hours': 0, 'minutes': 0},
        start_00=False,
        time_gap={'days': 2, 'hours': 10},
        test_period={'days': 2, 'hours': 0, 'minutes': 0}),
    episode_params=dict(
        start_weekdays={0, 1, 2, 3, 4, 5, 6},
        sample_duration={'days': 0, 'hours': 23, 'minutes': 40},
        time_gap={'days': 0, 'hours': 10},
        start_00=False),
    log_level=INFO,
)

env_config = dict(
    class_ref=BTgymEnv,
    kwargs=dict(
        dataset=BitfinexDataset,
        engine=MyCerebro,
        render_modes=['episode', 'human', 'external', 'internal'],
        render_state_as_image=True,
        render_ylabel='OHL_diff. / Internals',
        render_size_episode=(12, 8),
        render_size_human=(9, 4),
        render_size_state=(11, 3),
        render_dpi=75,
        port=5000,
        data_port=4999,
        connect_timeout=90,
        verbose=2,
    )
)

cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=6,  # set according CPU's available or so
    num_ps=1,
    num_envs=1,
    log_dir=os.path.expanduser('~/tmp/test_4_11'),
)

policy_config = dict(
    class_ref=AacStackedRL2Policy,
    kwargs={
        'lstm_layers': (256, 256),
        'lstm_2_init_period': 60,
    }
)

trainer_config = dict(
    class_ref=Unreal,
    kwargs=dict(
        opt_learn_rate=[1e-4, 1e-4],  # random log-uniform
        opt_end_learn_rate=1e-5,
        opt_decay_steps=50 * 10 ** 6,
        model_gamma=0.99,
        model_gae_lambda=1.0,
        model_beta=0.01,  # entropy reg
        rollout_length=20,
        time_flat=True,
        episode_train_test_cycle=(10, 5),
        use_value_replay=False,
        model_summary_freq=100,
        episode_summary_freq=5,
        env_render_freq=20,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=False,
    max_env_steps=100 * 10 * 1000,
    root_random_seed=0,
    purge_previous=1,  # ask to override previously saved model and logs
    verbose=2
)

# Train it:
launcher.run()
