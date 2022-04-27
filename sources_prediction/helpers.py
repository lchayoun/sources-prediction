import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly, \
    plot_cross_validation_metric
from prophet.serialize import model_to_json, model_from_json


def load_dataset():
    return pd.read_csv('../raw_dataset_new.csv')


def save_model(m, group_name):
    with open('../models/' + group_name + '_serialized_model.json',
              'w') as fout:
        json.dump(model_to_json(m), fout)


def load_model(group_name):
    with open('../models/' + group_name + '_serialized_model.json', 'r') as fin:
        return model_from_json(json.load(fin))


def update_fitted_model(m, df, group_name):
    m = m.fit(df, init=load_model(group_name))
    save_model(m, group_name)
    return m


def prepare(orig_df):
    df = orig_df[
        ['FILE_NAME', 'LogicFile', 'START_TIME', 'START_TIME_epoc', 'STAT_DESC',
         'STATUS']]
    df = df[df['STAT_DESC'] != 'Processing']
    return df


def fit(group):
    m = Prophet()
    group = group.sort_values('START_TIME')
    group['DIFF'] = (
            group['START_TIME_epoc'].shift(-1) - group['START_TIME_epoc'])
    group['ds'] = group['START_TIME']
    group['y'] = group['DIFF']
    with suppress_stdout_stderr(), warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        m.fit(group)
    return m


def forecast(m, group_name):
    with suppress_stdout_stderr(), warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        future = m.make_future_dataframe(periods=0)
        forecast_group = m.predict(future)
    gp = forecast_group[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    lower = pd.Timestamp(gp.values[0][0].timestamp() + gp.values[0][2],
                         unit='s').strftime('%Y-%m-%d %X')
    upper = pd.Timestamp(gp.values[0][0].timestamp() + gp.values[0][3],
                         unit='s').strftime('%Y-%m-%d %X')
    result = gp.values[0][0].timestamp() + gp.values[0][1]
    predicted = pd.Timestamp(result,
                             unit='s').strftime('%Y-%m-%d %X')
    # print(
    #     group_name + " will arrive between " + lower + " to "
    #     + upper + ", predicted at: " + predicted)
    return result


# Validation code
def plot(m, forecast_group):
    fig1 = m.plot(forecast_group)
    ax = fig1.gca()
    ax.set_title("Time Diff", size=26)
    ax.set_xlabel("Date", size=22)
    ax.set_ylabel("dt [seconds]", size=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    m.plot_components(forecast_group)
    plot_plotly(m, forecast_group)
    plot_components_plotly(m, forecast_group)


def cv(m, group):
    with suppress_stdout_stderr():
        df_cv = cross_validation(m,
                                 horizon=str(int(group.size * 0.01)) + ' days')
    df_p = performance_metrics(df_cv)
    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_ylim(ymax=df_p['mape'].max() * 1.01)
    fig = plot_cross_validation_metric(df_cv, metric='mape', ax=ax)
    fig.show()


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
