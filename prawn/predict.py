# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.multioutput import MultiOutputRegressor


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def error_percent(pred, true):
    return np.mean(np.abs(pred - true) / true)


NB_LOOKBACK_DAYS = 30
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")
REGIONS_FILE = os.path.join(ROOT_DIR, 'data', "countries_regions.csv")
TOTAL_MODEL_FILE = os.path.join(ROOT_DIR, "models", "model.pkl")
ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
NPI_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            # 'E1_Income support',
            # 'E2_Debt/contract relief',
            # 'E3_Fiscal measures',
            # 'E4_International support',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            # 'H4_Emergency investment in healthcare',
            # 'H5_Investment in vaccines',
            'H6_Facial Coverings',
            # 'H7_Vaccination policy'
            ]


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    predictor = FinalPredictor(start_date_str=start_date, end_date_str=end_date,
                               path_to_ips_file=path_to_ips_file, verbose=True)
    preds_df = predictor.predict()
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")


class BaseModel:
    def __init__(self, model_file=None, predict_days_once=7, nb_lookback_days=21):
        self.model = None
        self.model_file = model_file
        # predict days_ahead new cases once a time
        self.predict_days_once = predict_days_once
        self.nb_lookback_days = nb_lookback_days

    def load_model(self):
        with open(self.model_file, 'rb') as model_file:
            model = pickle.load(model_file)
            self.model = model

    def extract_npis_features(self, *args, **kwargs):
        raise NotImplemented

    def extract_cases_features(self, gdf, **kwargs):
        raise NotImplemented

    def extract_extra_features(self, gdf, **kwargs):
        raise NotImplemented

    def extract_labels(self, gdf, cut_date):
        # used in train step
        return np.array(gdf[gdf['Date'] >= cut_date][CASES_COL])[:self.predict_days_once]

    def predict(self, X):
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    def fit(self, *args, **kwargs):
        pass


class TotalModel(BaseModel):
    def extract_npis_features(self, gdf: pd.DataFrame, **kwargs):
        # [start_date, end_date)
        num_split = 3
        npis_features = []
        start_date = kwargs.pop('start_date')
        end_date = kwargs.pop('end_date')
        start_index = gdf[gdf['Date'] == start_date].index[0] - gdf.index[0]
        end_index = gdf[gdf['Date'] == end_date].index[0] - gdf.index[0]
        # nb_total_days = len(gdf)
        all_npi_data = np.array(gdf[NPI_COLS])

        for d in range(start_index, end_index):
            d_features = []
            # the passed npis
            X_npis = all_npi_data[d - self.nb_lookback_days:d]

            for split in np.split(X_npis, num_split):
                # print(split)
                d_features += [np.median(split, axis=0), split.max(axis=0), split.min(axis=0)]

            # the future npis
            future_mean = np.median(all_npi_data[d:d + self.predict_days_once], axis=0)
            future_max = all_npi_data[d:d + self.predict_days_once].max(axis=0)
            future_min = all_npi_data[d:d + self.predict_days_once].min(axis=0)
            d_features.append(future_mean)
            d_features.append(future_max)
            d_features.append(future_min)
            npis_features.append(np.concatenate(d_features))

        return np.array(npis_features)

    def extract_cases_features(self, gdf, **kwargs):
        # all_case_data = np.array(gdf[CASES_COL])
        days_forward = kwargs.pop('days_forward')
        start_date = kwargs.pop('start_date', None)
        if start_date is None:
            start_index = gdf.index[-1]
        else:
            start_index = gdf[gdf['Date'] == start_date].index[0]
        cases_features = []
        case_lookback_days = 14
        for d in range(days_forward):
            X_cases = gdf.loc[start_index - case_lookback_days + 1:start_index, 'NewCases'].to_numpy()
            cases_features.append(X_cases)
            # move to next
            start_index += 1
        return np.array(cases_features)

    def extract_extra_features(self, gdf, **kwargs):
        # [start_date, end_date)
        start_date = kwargs.pop('start_date')
        end_date = kwargs.pop('end_date')
        geo_encoder = kwargs.pop('geo_encoder')
        initial_date = gdf[gdf['NewCases'] > 0]['Date'].iloc[0]
        weeks_since_initial = []
        for d in pd.date_range(start_date, end_date, freq='1D', closed='left'):
            days = (d - initial_date) / np.timedelta64(1, 'D')
            weeks = days // 7 + 1
            cap = 50
            weeks = min(max(weeks, 0), cap)
            weeks_since_initial.append(weeks)
        g = gdf['GeoID'].to_list()[0]
        geo_encoded = geo_encoder.transform([g] * len(weeks_since_initial))
        extra_features = np.array([geo_encoded,
                                   weeks_since_initial]).T
        return extra_features

    def extract_labels(self, gdf: pd.DataFrame, **kwargs):
        start_date = kwargs.pop('start_date')
        end_date = kwargs.pop('end_date')
        start_index = gdf[gdf['Date'] == start_date].index[0] - gdf.index[0]
        end_index = gdf[gdf['Date'] == end_date].index[0] - gdf.index[0]
        all_case_data = np.array(gdf[CASES_COL])
        y_samples = []
        for d in range(start_index, end_index):
            y_samples.append(all_case_data[d:d + self.predict_days_once].flatten())
        return y_samples

    def fit(self, hist_df: pd.DataFrame, unique_geo_ids: list, geo_encoder, holdout_num=14):
        X_train_samples = dict()
        X_test_samples = dict()
        y_train_samples = dict()
        y_test_samples = dict()
        # the start date can strongly affect the final result
        start_date = np.datetime64('2020-03-21')
        # start_date = start_date + np.timedelta64(self.nb_lookback_days, 'D')
        for g in unique_geo_ids:
            gdf = hist_df[hist_df.GeoID == g]
            end_date = gdf.Date.max() - np.timedelta64(self.predict_days_once - 1, 'D')
            days_forward = (end_date - start_date) // np.timedelta64(1, 'D')
            print(f'days forward: {days_forward}')
            npi_features = self.extract_npis_features(
                gdf,
                start_date=start_date,
                end_date=end_date
            )

            cases_features = self.extract_cases_features(
                gdf,
                start_date=start_date,
                days_forward=days_forward
            )

            extra_features = self.extract_extra_features(
                gdf,
                start_date=start_date,
                end_date=end_date,
                geo_encoder=geo_encoder

            )

            print('Train %s' % g)
            print('NPI: ', npi_features.shape)
            print('Cases:', cases_features.shape)
            print('Extra:', extra_features.shape)
            X_samples = np.concatenate([npi_features, cases_features, extra_features], axis=1)
            print('X_sample:', X_samples.shape)

            y_samples = self.extract_labels(
                gdf,
                start_date=start_date,
                end_date=end_date,
            )
            print(len(y_samples), len(y_samples[0]))
            X_train_samples[g] = X_samples[:-holdout_num]
            X_test_samples[g] = X_samples[-holdout_num:]

            y_train_samples[g] = y_samples[:-holdout_num]
            y_test_samples[g] = y_samples[-holdout_num:]

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for geo, val in X_train_samples.items():
            # if geo.startswith('United States'):
            X_train.append(val)
            y_train.append(y_train_samples[geo])

        for geo, val in X_test_samples.items():
            X_test.append(val)
            y_test.append(y_test_samples[geo])

        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        print('X_train: ', X_train.shape)
        print('y_train:', y_train.shape)

        model = RandomForestRegressor(max_depth=20, max_features='sqrt', n_estimators=200, min_samples_leaf=2,
                                      criterion='mse', random_state=301)
        # model = MultiOutputRegressor(model)
        model.fit(X_train, y_train)
        # Evaluate model
        train_preds = model.predict(X_train)
        train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases
        print('Train MAE:', mae(train_preds, y_train))

        test_preds = model.predict(X_test)
        test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
        print('Test MAE:', mae(test_preds, y_test))

        with open('models/model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)


GEO_MODEL_CONFIG = {

}


class FinalPredictor:
    def __init__(self, start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
        self.start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')

        # Load historical intervention plans, since inception
        hist_ips_df = pd.read_csv(path_to_ips_file,
                                  parse_dates=['Date'],
                                  encoding="ISO-8859-1",
                                  dtype={"RegionName": str},
                                  error_bad_lines=True)

        hist_ips_df['GeoID'] = hist_ips_df['CountryName'] + '__' + hist_ips_df['RegionName'].astype(str)

        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLS:
            hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        # Load historical data to use in making predictions in the same way
        # This is the data we trained on
        # We stored it locally as for predictions there will be no access to the internet
        hist_df = pd.read_csv(DATA_FILE,
                              parse_dates=['Date'],
                              encoding="ISO-8859-1",
                              dtype={"RegionName": str,
                                     "CountryName": str},
                              error_bad_lines=False)
        # Add RegionID column that combines CountryName and RegionName for easier manipulation of data
        hist_df['GeoID'] = hist_df['CountryName'] + '__' + hist_df['RegionName'].astype(str)
        # Add new cases column
        hist_df['NewCases'] = hist_df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        # Fill any missing case values by interpolation and setting NaNs to 0
        hist_df.update(hist_df.groupby('GeoID').NewCases.apply(
            lambda group: group.interpolate()).fillna(0))

        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLS:
            hist_df.update(hist_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        encoder = preprocessing.LabelEncoder()
        self.geo_id_encoder = encoder.fit(hist_df.GeoID.unique())
        # hist_df['GeoIDEncoded'] = self.geo_id_encoder.transform(np.array(hist_df['GeoID']))
        hist_df = hist_df[ID_COLS + CASES_COL + NPI_COLS]
        # Keep only the id and cases columns
        hist_cases_df = hist_df[ID_COLS + CASES_COL]

        # hist_ips_df['GeoIDEncoded'] = self.geo_id_encoder.transform(np.array(hist_ips_df['GeoID']))
        self.hist_ips_df = hist_ips_df

        # Intervention plans to forecast for: those between start_date and end_date
        self.ips_df = hist_ips_df[(hist_ips_df.Date >= self.start_date) & (hist_ips_df.Date <= self.end_date)]

        self.hist_cases_df = hist_cases_df
        self.hist_df = hist_df

        # include all the
        region_df = pd.read_csv(REGIONS_FILE, dtype={"CountryName": str, "RegionName": str},
                                error_bad_lines=False)

        region_df['GeoID'] = region_df['CountryName'] + '__' + region_df['RegionName'].astype(str)
        # only include the regions we care about
        self.unique_geo_ids = region_df['GeoID'].unique()

        self.verbose = verbose

    def fit_total(self):
        model = self.load_geo_model()
        model.fit(self.hist_df, self.unique_geo_ids, self.geo_id_encoder)

    def predict(self):
        geo_pred_dfs = []
        # the main api
        for g in self.hist_ips_df.GeoID.unique():
            if self.verbose:
                print('\nPredicting for', g)
            geo_pred_df = self.predict_geo(g)
            geo_pred_dfs.append(geo_pred_df)

        pred_df = pd.concat(geo_pred_dfs, ignore_index=True)
        if self.verbose:
            print('pred_df:\n', pred_df)
        return pred_df

    def predict_geo(self, g):
        # Make predictions for each country,region pair

        hist_cases_df = self.hist_cases_df
        # Pull out all relevant data for country c
        hist_cases_gdf = hist_cases_df[hist_cases_df.GeoID == g]
        last_known_date = hist_cases_gdf.Date.max()
        hist_ips_gdf = self.hist_ips_df[self.hist_ips_df.GeoID == g].copy()
        print('last know date', last_known_date)
        print('end date', self.end_date)

        # merge future and passed npis for feature extraction
        # future_index = pd.date_range(start=last_known_date + np.timedelta64(1, 'D'), end=self.end_date, freq='D')
        # future_npis = pd.DataFrame(data=np.zeros([future_index.size, len(NPI_COLS)]), index=future_index)
        # past_npis = past_npis.append(future_npis, verify_integrity=True)
        # past_npis.update(test_npis)

        # Make prediction for each several days

        # Start predicting from start_date, unless there's a gap since last known date
        current_date = min(last_known_date + np.timedelta64(1, 'D'), self.start_date)
        hist_cases_gdf = hist_cases_gdf[hist_cases_gdf.Date < current_date][['GeoID', 'Date', 'NewCases']].copy(
            deep=True)
        print(hist_cases_gdf)
        model = self.load_geo_model(g)

        interval = np.timedelta64(model.predict_days_once - 1, 'D')
        d1 = np.timedelta64(1, 'D')
        print('Train %s' % g)

        while current_date <= self.end_date:
            current_end_date = current_date + interval

            print(f"date range:{current_date.strftime('%Y-%m-%d')}-{current_end_date.strftime('%Y-%m-%d')}")

            npi_features = model.extract_npis_features(hist_ips_gdf, start_date=current_date,
                                                       end_date=current_date + d1)

            extra_features = model.extract_extra_features(
                self.hist_df,
                start_date=current_date,
                end_date=current_date + d1,
                geo_encoder=self.geo_id_encoder
            )
            # choose last one
            cases_features = model.extract_cases_features(hist_cases_gdf, days_forward=1)
            print('Train %s' % g)
            print('NPI: ', npi_features.shape)
            print('Cases:', cases_features.shape)
            print('Extra:', extra_features.shape)
            X = np.concatenate([npi_features, cases_features, extra_features], axis=1)
            print('X_sample:', X.shape)
            pred = model.predict(X)
            print('pred:', pred.shape, pred)
            tmp_case_df = pd.DataFrame({
                'Date': pd.date_range(current_date, current_end_date, freq='1D'),
                'NewCases': pred[0]
            })
            tmp_case_df['GeoID'] = g
            # in order to rollout predictions for further days.
            hist_cases_gdf = hist_cases_gdf.append(tmp_case_df, ignore_index=True)
            # Append the prediction and npi's for next day
            # past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + model.predict_days_once], axis=0)
            # days_ahead += model.predict_days_once
            # move on to next cycle
            current_date = current_end_date + d1
        hist_cases_gdf = hist_cases_gdf[
            (hist_cases_gdf.Date >= self.start_date) & (hist_cases_gdf.Date <= self.end_date)]
        country_name = hist_ips_gdf['CountryName'].iloc[0]
        region_name = hist_ips_gdf['RegionName'].iloc[0]
        geo_pred_df = pd.DataFrame(np.array([
            [country_name] * hist_cases_gdf.shape[0],
            [region_name] * hist_cases_gdf.shape[0],
            hist_cases_gdf['Date'],
            hist_cases_gdf['NewCases'],
        ]).T, columns=['CountryName', 'RegionName', 'Date', 'PredictedDailyNewCases'])
        print('Final:\n', geo_pred_df)
        return geo_pred_df

    @staticmethod
    def load_geo_model(geo=None) -> BaseModel:
        if geo is None or geo not in GEO_MODEL_CONFIG:
            model = TotalModel(model_file=TOTAL_MODEL_FILE)
            model.load_model()
            return model
        else:
            return GEO_MODEL_CONFIG[geo]


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
