from prawn.prawn_prescribe import run_geo
from joblib import Parallel, delayed
from prawn.standard_predictor.xprize_predictor import NPI_COLUMNS, XPrizePredictor
import pandas as pd


if __name__ == '__main__':
    start_date_str = '2020-08-01'
    end_date_str = '2020-08-31'
    geo_list = [
                'Albania',
                'Algeria',
                'Andorra',
                'Afghanistan',
                'Angola',
                'Argentina',
                'Aruba',
                'Australia',
                'Austria',
                'Azerbaijan']
    outputs = Parallel(backend='loky', n_jobs=12)(delayed(run_geo)(geo, start_date_str, end_date_str) for geo in geo_list)
    df = pd.concat(outputs)
    df.to_csv('result.csv', index=False)
