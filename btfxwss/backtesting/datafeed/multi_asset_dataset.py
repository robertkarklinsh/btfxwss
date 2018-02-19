import os
import pandas as pd
import numpy as np
from btgym import BTgymRandomDataDomain


class MultiAssetDataset(BTgymRandomDataDomain):
    """
    Support for csv files that contain ohlcv data for several symbols.

    """

    def read_csv(self, data_filename=None, force_reload=False):
        if self.data is not None and not force_reload:
            self.log.debug('data has been already loaded. Use `force_reload=True` to reload')
            return
        if data_filename:
            self.filename = data_filename  # override data source if one is given

        if type(self.filename) == str:
            self.filename = [self.filename]

        dataframes = []
        for filename in self.filename:
            try:
                assert filename and os.path.isfile(filename)
                current_dataframe = pd.read_csv(
                    filename,
                    sep=self.sep,
                    header=self.header,
                    index_col=self.index_col,
                    parse_dates=self.parse_dates,
                    names=self.names
                )

                current_dataframe.drop('id', axis=1, inplace=True)
                for _symbol, data in current_dataframe.groupby(['symbol']):
                    if self.symbol is None:
                        # If symbol not specified use first one
                        current_dataframe = data.iloc[:, 1:]
                        break
                    elif self.symbol == _symbol:
                        current_dataframe = data.iloc[:, 1:]
                current_dataframe.sort_index(inplace=True)

                # # Only for debugging no trades bug!
                # current_dataframe['volume'] = 0


                # Check and remove duplicate datetime indexes:
                duplicates = current_dataframe.index.duplicated(keep='first')
                how_bad = duplicates.sum()
                if how_bad > 0:
                    current_dataframe = current_dataframe[~duplicates]
                    self.log.warning('Found {} duplicated date_time records in <{}>.\
                     Removed all but first occurrences.'.format(how_bad, filename))

                dataframes += [current_dataframe]
                self.log.info('Loaded {} records from <{}>.'.format(dataframes[-1].shape[0], filename))

            except:
                msg = 'Data file <{}> not specified / not found.'.format(str(filename))
                self.log.error(msg)
                raise FileNotFoundError(msg)

        self.data = pd.concat(dataframes)
        range = pd.to_datetime(self.data.index)
        self.data_range_delta = (range[-1] - range[0]).to_pytimedelta()

    def __init__(
            self,
            symbol=None,
            **kwargs
    ):
        self.symbol = symbol

        super(MultiAssetDataset, self).__init__(
            **kwargs
        )