from prawn.prawn_prescribe import run_geo, add_geo_id
from joblib import Parallel, delayed
from prawn.standard_predictor.xprize_predictor import NPI_COLUMNS, XPrizePredictor
import pandas as pd


if __name__ == '__main__':
    path_to_prior_ips_file = 'data/2020-09-30_historical_ip.csv'
    path_to_cost_file = 'data/uniform_random_costs.csv'
    cost_df = pd.read_csv(path_to_cost_file, dtype={"RegionName": str, "RegionCode": str})
    add_geo_id(cost_df)
    geo_list = cost_df.GeoID.unique().tolist()
    start_date_str = '2020-08-01'
    end_date_str = '2020-08-10'
    outputs = Parallel(backend='loky', n_jobs=12)(delayed(run_geo)(geo, start_date_str, end_date_str,
                                                                   path_to_cost_file, path_to_prior_ips_file)
                                                  for geo in geo_list)
    df = pd.concat(outputs)
    df.to_csv('result.csv', index=False)
