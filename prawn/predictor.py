import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
import pickle


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
           'GeoIDEncoded',
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


class PrawnPredictor:
    def __init__(self, source_data='data/OxCGRT_latest.csv'):
        assert os.path.exists(source_data)
        self.source_data = source_data
        self.id_cols = ['CountryName',
                        'RegionName',
                        'GeoID',
                        'Date']
        self.cases_col = ['NewCases']
        self.npi_cols = ['C1_School closing',
                         'C2_Workplace closing',
                         'C3_Cancel public events',
                         'C4_Restrictions on gatherings',
                         'C5_Close public transport',
                         'C6_Stay at home requirements',
                         'C7_Restrictions on internal movement',
                         'C8_International travel controls',
                         'E1_Income support',
                         'E2_Debt/contract relief',
                         'E3_Fiscal measures',
                         'E4_International support',
                         'H1_Public information campaigns',
                         'H2_Testing policy',
                         'H3_Contact tracing',
                         'H4_Emergency investment in healthcare',
                         'H5_Investment in vaccines',
                         'H6_Facial Coverings',
                         'H7_Vaccination policy']
        self.X_test_samples = dict()
        self.X_train_samples = dict()
        self.y_test_samples = dict()
        self.y_train_samples = dict()
        self.submission_date = "2020-12-06"
        self.model = None
        self.df = self.parse_df()

    def get_geo_test_samples(self, geo):
        return np.array(self.X_test_samples[geo]), np.array(self.y_test_samples[geo])

    def get_geo_train_samples(self, geo):
        return np.array(self.X_train_samples[geo]), np.array(self.y_train_samples[geo])

    def parse_df(self):
        df = pd.read_csv(self.source_data,
                         parse_dates=['Date'],
                         encoding="ISO-8859-1",
                         dtype={"RegionName": str,
                                "RegionCode": str},
                         error_bad_lines=False)

        df = df[df.Date <= np.datetime64(self.submission_date)]

        # GeoID 作为地区的唯一识别号，用于后续的分组中
        df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

        # ConfirmedCases 确诊病例  NewCases 新增病例
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)

        # Keep only columns of interest
        id_cols = self.id_cols
        cases_col = self.cases_col
        npi_cols = self.npi_cols
        df = df[id_cols + cases_col + npi_cols]

        # Fill any missing case values by interpolation and setting NaNs to 0
        df.update(df.groupby('GeoID').NewCases.apply(
            lambda group: group.interpolate()).fillna(0))

        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in npi_cols:
            df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))
        return df

    def get_train_test_data(self):
        df = self.df
        geo_ids = df.GeoID.unique()
        encoder = preprocessing.LabelEncoder()
        encoder.fit(geo_ids)

        # Set number of past days to use to make predictions
        # 回看的天数，可以自行设置
        nb_lookback_days = 14

        # Create training data across all countries for predicting one day ahead
        # 用过去一个月的数据预测当天的值
        X_cols = self.cases_col + self.npi_cols
        y_col = self.cases_col

        X_train_samples = self.X_train_samples
        X_test_samples = self.X_test_samples

        y_train_samples = self.y_train_samples
        y_test_samples = self.y_test_samples

        # 使用未来多少天作为预测数据，用于评估模型
        holdout_num = 14

        # 向后预测的天数
        # 0表示向后预测第一天（单步） 1表示向后预测第2天
        PREDICT_DAYS = 0

        for g in geo_ids:
            X_samples = []
            y_samples = []

            # 筛选出特定区域的数据
            gdf = df[df.GeoID == g].copy()

            if gdf[gdf['NewCases'] > 0].empty:
                # 如果该地区历史数据新增病例全为零，则应该剔除该地区，直接跳到下一个地区的循环
                continue
            # 剔除新增病例为零的点，因为实测中发现新增病例为零的有可能是没有更新数据
            # gdf = gdf[gdf['NewCases'] > 100]

            initial_date = gdf[gdf['NewCases'] > 0]['Date'].iloc[0]
            # 距离首次爆发的时间（天数）
            gdf['days_since_initial'] = (gdf['Date'] - initial_date).apply(lambda x: x / np.timedelta64(1, 'D'))
            # 使用新增病例增长率来预测
            gdf['NewCasesRatio'] = gdf.NewCases.pct_change(
            ).fillna(0).replace(np.inf, 0) + 1

            all_case_data = np.array(gdf[self.cases_col])
            all_npi_data = np.array(gdf[self.npi_cols])
            geo_data = encoder.transform(np.array(gdf['GeoID']))
            # print(geo_data)
            # Create one sample for each day where we have enough data
            # Each sample consists of cases and npis for previous nb_lookback_days
            # 每个样本包括过去一个月病例和干预措施，相当于当天的预测只参考于过去一个月的数据， K阶马尔可夫
            nb_total_days = len(gdf)
            for d in range(nb_lookback_days, nb_total_days - PREDICT_DAYS):
                X_cases = all_case_data[d - nb_lookback_days:d]
                # X_cases = all_case_data[d-1:d]
                # Take negative of npis to support positive
                # weight constraint in Lasso.
                # 取负值，是为了让权重变成正的
                X_npis = -all_npi_data[d - nb_lookback_days:d]
                # 需要观察的天到过去第7天前
                X_npis_mean = np.mean(-all_npi_data[d - nb_lookback_days:d - 7], axis=0)
                X_npis_min = np.min(-all_npi_data[d - nb_lookback_days:d - 7], axis=0)
                # 过去7天的平均值
                X_npis_mean_7 = np.mean(-all_npi_data[d - 7:d], axis=0)
                X_npis_min_7 = np.min(-all_npi_data[d - 7:d], axis=0)

                # 当天及中间天的干预措施
                X_nips_current_day = -all_npi_data[d + PREDICT_DAYS]
                # PREDICT_DAYS=0 时为空
                X_nips_future_day = -all_npi_data[d:d + PREDICT_DAYS]

                # X_cases 是一个矩阵，需要拉平为一个向量
                # Flatten all input data so it fits Lasso input format.
                extra_data = [geo_data[0], gdf['days_since_initial'].iloc[d]]
                all_features = [X_cases.flatten(),
                                X_npis_mean,
                                X_npis_min,
                                X_npis_mean_7,
                                X_npis_min_7,
                                X_nips_current_day,
                                # X_npis.flatten(),
                                extra_data]
                if X_nips_future_day.size > 0:
                    all_features.append(np.mean(X_nips_future_day, axis=0))
                    all_features.append(np.min(X_nips_future_day, axis=0))
                X_sample = np.concatenate(all_features)
                y_sample = all_case_data[d + PREDICT_DAYS]
                X_samples.append(X_sample)
                y_samples.append(y_sample)

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
            X_train += val
            y_train += y_train_samples[geo]

        for geo, val in X_test_samples.items():
            X_test += val
            y_test += y_test_samples[geo]

        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()

        X_test = np.array(X_test)
        y_test = np.array(y_test).flatten()

        return X_train, y_train, X_test, y_test

    def _train(self, X_train, y_train, X_test, y_test):
        model = RandomForestRegressor(max_depth=15, max_features='sqrt', n_estimators=200, min_samples_leaf=3,
                                      criterion='mse')
        model.fit(X_train, y_train)
        self.model = model
        # Evaluate model
        train_preds = model.predict(X_train)
        train_preds = np.maximum(train_preds, 0)  # Don't predict negative cases
        print('Train MAE:', mae(train_preds, y_train))

        test_preds = model.predict(X_test)
        test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
        print('Test MAE:', mae(test_preds, y_test))
        return model

    def train(self):
        X_train, y_train, X_test, y_test = self.get_train_test_data()
        self._train(X_train, y_train, X_test, y_test)

    def train_geo(self, geo):
        X_train, y_train = self.get_geo_train_samples(geo)
        X_test, y_test = self.get_geo_test_samples(geo)
        self._train(X_train, y_train.ravel(), X_test, y_test.ravel())

    def test_geo(self, geo):
        X_geo_test, y_geo_test = self.get_geo_test_samples(geo)
        test_preds = self.model.predict(X_geo_test)
        test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
        print('Test MAE:', mae(test_preds, y_geo_test))
        test_df = pd.DataFrame({'test_preds': test_preds, 'y_test': y_geo_test.flatten()})
        print(test_df)


class BaseModel:
    def __init__(self, model_file=None, predict_days_once=7, nb_lookback_days=21):
        if model_file is not None:
            with open(model_file, 'rb') as model_file:
                model = pickle.load(model_file)
                self.model = model
        else:
            self.model = None
        # predict days_ahead new cases once a time
        self.predict_days_once = predict_days_once
        self.nb_lookback_days = nb_lookback_days

    def extract_npis_features(self, past_npis: pd.DataFrame, future_npis: pd.DataFrame, **kwargs):
        raise NotImplemented

    def extract_cases_features(self, past_cases_df, **kwargs):
        raise NotImplemented

    def extract_extra_features(self, gdf):
        pass

    def extract_labels(self, gdf, cut_date):
        # used in train step
        return np.array(gdf[gdf['Date'] >= cut_date][CASES_COL])[:self.predict_days_once]

    def predict(self, X):
        preds = self.model.predict(X)
        return np.maximum(preds, 0)

    # def fit(self, hist_cases_df: pd.DataFrame, unique_geo_ids: list):
    #     pass


class TotalModel(BaseModel):
    def extract_npis_features(self, gdf: pd.DataFrame, **kwargs):
        num_split = 3
        npis_features = []
        start_date = kwargs.pop('start_date')
        end_date = kwargs.pop('end_date')
        start_index = gdf[gdf['Date'] == start_date].index[0] - gdf.index[0]
        end_index = gdf[gdf['Date'] == end_date].index[0] - gdf.index[0]
        # nb_total_days = len(gdf)
        all_npi_data = np.array(gdf[NPI_COLS])

        for d in range(start_index, end_index + 1):
            d_features = []
            # the passed npis
            X_npis = all_npi_data[d - self.nb_lookback_days:d]

            for split in np.split(X_npis, num_split):
                # print(split)
                d_features += [split.mean(axis=0), split.max(axis=0), split.min(axis=0)]

            # the future npis
            future_mean = all_npi_data[d:d + self.predict_days_once].mean(axis=0)
            future_max = all_npi_data[d:d + self.predict_days_once].max(axis=0)
            future_min = all_npi_data[d:d + self.predict_days_once].min(axis=0)
            d_features.append(future_mean)
            d_features.append(future_max)
            d_features.append(future_min)
            npis_features.append(np.concatenate(d_features))

        return np.array(npis_features)

    def extract_cases_features(self, past_cases_df, **kwargs):
        return past_cases_df

    def extract_labels(self, gdf: pd.DataFrame, **kwargs):
        start_date = kwargs.pop('start_date')
        end_date = kwargs.pop('end_date')
        start_index = gdf[gdf['Date'] == start_date].index[0] - gdf.index[0]
        end_index = gdf[gdf['Date'] == end_date].index[0] - gdf.index[0]
        all_case_data = np.array(gdf[CASES_COL])
        y_samples = []
        for d in range(start_index, end_index + 1):
            y_samples.append(all_case_data[d:d + self.predict_days_once].flatten())
        return y_samples

    def fit(self, hist_df: pd.DataFrame, unique_geo_ids: list):
        start_date = np.datetime64('2020-01-01')
        start_date = start_date + np.timedelta64(self.nb_lookback_days, 'D')
        for g in unique_geo_ids:
            X_samples = []
            y_samples = []
            gdf = hist_df[hist_df.GeoID == g]
            end_date = gdf.Date.max() - np.timedelta64(self.predict_days_once, 'D')

            npi_features = self.extract_npis_features(
                gdf,
                start_date=start_date,
                end_date=end_date
            )
            X_sample = np.concatenate([npi_features])
            X_samples.append(X_sample)
            print('Train %s' % g)
            print(npi_features.shape)

            y_samples = self.extract_labels(
                gdf,
                start_date=start_date,
                end_date=end_date
            )
            print(len(y_samples), len(y_samples[0]))


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
        hist_df['GeoIDEncoded'] = self.geo_id_encoder.transform(np.array(hist_df['GeoID']))
        hist_df = hist_df[ID_COLS + CASES_COL + NPI_COLS]
        # Keep only the id and cases columns
        hist_cases_df = hist_df[ID_COLS + CASES_COL]

        hist_ips_df['GeoIDEncoded'] = self.geo_id_encoder.transform(np.array(hist_ips_df['GeoID']))
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
        model.fit(self.hist_df, self.unique_geo_ids)

    def predict(self):
        # the main api
        for g in self.hist_ips_df.GeoID.unique():
            if self.verbose:
                print('\nPredicting for', g)
            self.predict_geo(g)

    def predict_geo(self, g):
        # Make predictions for each country,region pair
        geo_preds = []
        geo_pred_dfs = []
        hist_cases_df = self.hist_cases_df
        ips_df = self.ips_df
        # Pull out all relevant data for country c
        hist_cases_gdf = hist_cases_df[hist_cases_df.GeoID == g].copy()

        initial_date = hist_cases_df[hist_cases_df['NewCases'] > 0]['Date'].iloc[0]
        # days since initial break
        hist_cases_gdf['days_since_initial'] = (hist_cases_gdf['Date'] - initial_date).apply(
            lambda x: x / np.timedelta64(1, 'D'))

        last_known_date = hist_cases_gdf.Date.max()
        ips_gdf = ips_df[ips_df.GeoID == g]
        # past_cases = np.array(hist_cases_gdf[CASES_COL])
        past_cases = pd.DataFrame(data=hist_cases_gdf[CASES_COL], index=hist_cases_gdf['Date'].to_list(),
                                  columns=CASES_COL)
        hist_ips_gdf = self.hist_ips_df[self.hist_ips_df.GeoID == g].copy()
        past_npis = pd.DataFrame(data=hist_ips_gdf[NPI_COLS].to_numpy(), index=hist_ips_gdf['Date'].to_list(),
                                 columns=NPI_COLS)

        # past_npis = np.array(self.hist_ips_df[NPI_COLS])
        # test_npis = pd.DataFrame(data=ips_gdf[NPI_COLS], index=ips_gdf['Date'].to_list())
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

        model = self.load_geo_model(g)

        npis_features = model.extract_npis_features(hist_ips_gdf,
                                                    start_date=self.start_date,
                                                    end_date=self.end_date)

        print(npis_features.shape)
        print(npis_features)
        return
        days_ahead = 0
        while current_date <= self.end_date:
            next_date = current_date + np.timedelta64(model.predict_days_once, 'D')
            case_features = model.extract_cases_features(past_cases, start_date=current_date, end_date=next_date)

            X = np.concatenate([case_features.flatten(),
                                npis_features.flatten()])
            pred = model.predict(X)

            # Add if it's a requested date
            if current_date >= self.start_date:
                geo_preds.append(pred)
                if self.verbose:
                    print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
            else:
                if self.verbose:
                    print(f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

            # Append the prediction and npi's for next day
            # in order to rollout predictions for further days.
            past_cases = np.append(past_cases, pred)
            past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + model.predict_days_once], axis=0)
            days_ahead += model.predict_days_once
            # move on to next cycle
            current_date = next_date

    @staticmethod
    def load_geo_model(geo=None) -> BaseModel:
        if geo is None or geo not in GEO_MODEL_CONFIG:
            model = TotalModel(model_file=TOTAL_MODEL_FILE)
            return model
        else:
            return GEO_MODEL_CONFIG[geo]


if __name__ == '__main__':
    predictor = PrawnPredictor()
    predictor.get_train_test_data()
    # predictor.train()
    # X_geo_test, y_geo_test = predictor.get_geo_test_samples('United States__New York')
    predictor.train_geo('United States__New York')
    # test_preds = model.predict(X_geo_test)
    # test_preds = np.maximum(test_preds, 0)  # Don't predict negative cases
    # print('Test MAE:', mae(test_preds, y_geo_test))
    # test_df = pd.DataFrame({'test_preds': test_preds, 'y_test': y_geo_test.flatten()})
    # geo_ids = predictor.df.GeoID.unique()
    # for geo in geo_ids:
    #     try:
    #         print('%s：' % geo)
    #         X_geo_test, y_geo_test = predictor.get_geo_test_samples(geo)
    #         if X_geo_test.size == 0:
    #             print('%s Test case is 0' % geo)
    #         else:
    #             test_preds = model.predict(X_geo_test)
    #             test_preds = np.maximum(test_preds, 0)
    #             print(mae(test_preds, y_geo_test))
    #     except KeyError:
    #         print('%s has no new case' % geo)
