from btfxwss.robot import Robot
from configparser import ConfigParser

if __name__ == "__main__":
    config = ConfigParser()
    with open("settings.ini") as config_file:
        config.readfp(config_file)
        kwargs = {
            'API_KEY': config.get('settings', 'API_KEY'),
            'API_SECRET': config.get('settings', 'API_SECRET'),
            'REFRESH_TIME': config.getint('settings', 'REFRESH_TIME'),
            'BUY_WINDOW': config.getint('settings', 'BUY_WINDOW'),
            'BUY_ROC': config.getfloat('settings', 'BUY_ROC'),
            'SELL_WINDOW': config.getint('settings', 'SELL_WINDOW'),
            'SELL_POSITION_ROC': config.getfloat('settings', 'SELL_POSITION_ROC')
        }

    alg = Robot(**kwargs)
    alg.run()




