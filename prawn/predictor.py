import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def error_percent(pred, true):
    return np.mean(np.abs(pred - true)/true)


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
