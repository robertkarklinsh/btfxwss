import time
import numpy as np
import telegram
from collections import defaultdict
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from btfxwss.algo.rapid_move import RapidMove
from btfxwss.client import BtfxWss
from btfxwss.queue_processor import QueueProcessor

import logging

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#                     level=logging.INFO)

TELEGRAM_CHAT_ID = 122363776


class KieraBot(telegram.Bot, QueueProcessor):
    def __init__(self, token, alg, data_q=None):
        telegram.Bot.__init__(self, token=token)
        QueueProcessor.__init__(self, data_q)
        self.updater = Updater(token=token)
        self.dispatcher = self.updater.dispatcher
        self.dispatcher.add_handler(CommandHandler('start', self.on_start))
        self.updater.start_polling()
        self.chat_id = None
        self.alg = alg
        self.timestamp_index = defaultdict(int)

    def on_start(self, bot, update):
        self.chat_id = update.message.chat_id
        bot.send_message(chat_id=update.message.chat_id, text="Hi Robert! Today smells like honey.")

    def _handle_candles(self, dtype, data, ts):
        self.log.debug("_handle_candles: %s - %s - %s", dtype, data, ts)
        channel_id, payload = data
        identifier = self.channel_directory[channel_id]
        symbol = identifier[1]
        payload = np.array(payload)
        if payload.size == 0:
            return
        # check for data containing single entity:
        if len(payload.shape) != 2:
            if self.timestamp_index[symbol] < payload[0]:
                self.timestamp_index[symbol] = payload[0]
                payload = payload.reshape((1, -1))
            else:
                return
        # formatting payload to [ts, o, h, l, c, v]
        payload = payload[:, [0, 1, 3, 4, 2, 5]]
        self.handle_candles(symbol, payload)

    def handle_candles(self, symbol, payload):
        payload = np.unique(payload, axis=0)
        for ohlcv in payload:
            if self.alg.trigger_condition(symbol, ohlcv):
                self.send_message(self.chat_id, symbol + ' changed by ' + str(self.alg.change_threshold))


if __name__ == '__main__':
    alg = RapidMove(1.0001, 5)
    bot = KieraBot('539127150:AAGFLwu3dRBiSbtjRWyJVwnmRl1n9KgD6ws', alg)
    bot.chat_id = TELEGRAM_CHAT_ID

    client = BtfxWss(queue_processor=bot, log_level=logging.DEBUG)
    client.start()
    time.sleep(2)
    client.subscribe_to_candles('BTCUSD')
    client.subscribe_to_candles('ETHUSD')
    client.subscribe_to_candles('OMGUSD')
    client.subscribe_to_candles('IOTUSD')
    client.subscribe_to_candles('LTCUSD')
    client.subscribe_to_candles('XMRUSD')
    client.subscribe_to_candles('EDOUSD')
    client.subscribe_to_candles('AVTUSD')
    client.subscribe_to_candles('ZECUSD')

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        bot.updater.stop()
