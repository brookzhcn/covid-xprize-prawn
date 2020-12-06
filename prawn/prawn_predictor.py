import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from fbprophet import Prophet

DATA_URL = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
NPI_COLUMNS = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population']
NB_LOOKBACK_DAYS = 21
NB_TEST_DAYS = 14
WINDOW_SIZE = 7
US_PREFIX = "United States / "
NUM_TRIALS = 1
MAX_NB_COUNTRIES = 20


class DataProcessor:

    def __init__(self, data_url='data/OxCGRT_latest.csv'):
        self.df = self._prepare_dataframe(data_url)
        geos = self.df.GeoID.unique()
        # print(geos)
        self._geo_id_encoder = None
        self.geo_id_encoder = geos
        self.country_samples = self._create_country_samples(self.df, geos)

    def _prepare_dataframe(self, data_url: str) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :param data_url: the url containing the original data
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df = self._load_original_data(data_url)

        df = self._attach_population_context_df(df)

        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        # 从开始有病例算起
        # df = df[df['ConfirmedCases'] > 0]

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

        # Compute smoothed versions of new cases and deaths each day
        df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
        df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

        # Compute percent change in new cases and deaths each day
        df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1

        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        # 最终需要预测的值
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    @staticmethod
    def _load_original_data(data_url):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
        has_region_countries = latest_df[latest_df['RegionName'].notnull()]['CountryName'].unique()
        # ['Brazil', 'United Kingdom', 'United States']
        print(has_region_countries)
        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        return latest_df

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    @staticmethod
    def _attach_population_context_df(df):
        # File containing the population for each country
        # Note: this file contains only countries population, not regions
        additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                            usecols=['CountryName', 'Population'])
        additional_context_df['GeoID'] = additional_context_df['CountryName']

        # US states population
        additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                              usecols=['NAME', 'POPESTIMATE2019'])
        # Rename the columns to match measures_df ones
        additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
        # Prefix with country name to match measures_df
        additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_us_states_df)

        # UK population
        additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_uk_df)

        # Merge the 2 DataFrames
        df = df.merge(additional_context_df, on=['GeoID'], how='left', suffixes=('', '_y'))
        return df

    @staticmethod
    def _create_country_samples(df: pd.DataFrame, geos: list) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :return: a dictionary of train and test sets, for each specified country
        """
        context_column = 'PredictionRatio'
        action_columns = NPI_COLUMNS
        outcome_column = 'PredictionRatio'
        country_samples = {}
        for g in geos:
            cdf = df[df.GeoID == g]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            context_data = np.array(cdf[context_column])
            action_data = np.array(cdf[action_columns])
            outcome_data = np.array(cdf[outcome_column])
            context_samples = []
            action_samples = []
            outcome_samples = []
            nb_total_days = outcome_data.shape[0]
            for d in range(NB_LOOKBACK_DAYS, nb_total_days):
                context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
                action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
                outcome_samples.append(outcome_data[d])
            if len(outcome_samples) > 0:
                X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
                X_action = np.stack(action_samples, axis=0)
                y = np.stack(outcome_samples, axis=0)
                country_samples[g] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'y': y,
                    'X_train_context': X_context[:-NB_TEST_DAYS],
                    'X_train_action': X_action[:-NB_TEST_DAYS],
                    'y_train': y[:-NB_TEST_DAYS],
                    'X_test_context': X_context[-NB_TEST_DAYS:],
                    'X_test_action': X_action[-NB_TEST_DAYS:],
                    'y_test': y[-NB_TEST_DAYS:],
                }
        return country_samples

    @property
    def geo_id_encoder(self, ):
        return self._geo_id_encoder

    @geo_id_encoder.setter
    def geo_id_encoder(self, geos):
        encoder = preprocessing.LabelEncoder()
        encoder.fit(geos)
        self._geo_id_encoder = encoder




class RandomForestPredictor:
    def __init__(self, processor):
        self.model = RandomForestRegressor(max_depth=10, max_features=4, n_estimators=500)
        self.data_processor = processor


class ProphetPredictor:
    def __init__(self):
        m = Prophet(growth='logistic', weekly_seasonality=False)
        m.add_regressor()
        m.add_seasonality(name='halfly', period=180, fourier_order=5)
        m.add_country_holidays('US')
        self.model = m

    def train(self, df):
        df['ds'] = df['Date']
        df['y'] = df['NewCases']

        self.model.fit(df)

    def add_regressor(self, column_name):
        pass


if __name__ == '__main__':
    data_processor = DataProcessor()
    df = data_processor.df
    df = df[df['GeoID'] == 'United States / New York']
    predictor = ProphetPredictor()
    cap = max(df.iloc[0]['Population'] * 0.001, df['NewCases'].max()*1.2)
    df['cap'] = cap
    df['floor'] = 0
    predictor.train(df)
    future = predictor.model.make_future_dataframe(periods=30)
    print('cap:', cap)
    future['cap'] = cap
    future['floor'] = 0
    future.tail()
    forecast = predictor.model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = predictor.model.plot(forecast)
    fig1.show()

    fig2 = predictor.model.plot_components(forecast)


