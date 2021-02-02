from prawn_prescribe import run_geo, add_geo_id, get_country_region, PrawnPrescribe
from joblib import Parallel, delayed
from prawn.standard_predictor.xprize_predictor import NPI_COLUMNS, XPrizePredictor
import pandas as pd
import time

if __name__ == '__main__':
    path_to_prior_ips_file = 'data/all_2020_ips.csv'
    path_to_cost_file = 'data/uniform_random_costs.csv'
    x_predictor = XPrizePredictor = XPrizePredictor()
    cost_df = pd.read_csv(path_to_cost_file, dtype={"RegionName": str, "RegionCode": str})
    add_geo_id(cost_df)
    geo_list = cost_df.GeoID.unique().tolist()
    start_date_str = '2021-01-01'
    end_date_str = '2021-01-10'
    prescribe1 = PrawnPrescribe(start_date_str=start_date_str, end_date_str=end_date_str,
                                path_to_prior_ips_file=path_to_prior_ips_file,
                                path_to_cost_file=path_to_cost_file, predictor=x_predictor,
                                interval=14
                                )
    zero_geos, others = prescribe1.filter_geos()
    zero_outputs = []
    date_range = prescribe1.date_range
    num_of_days = prescribe1.num_of_days
    for zero_geo in zero_geos:
        c, r = get_country_region(zero_geo)
        zero_df = pd.DataFrame({
            'PrescriptionIndex': [0] * num_of_days,
            'CountryName': [c] * num_of_days,
            'RegionName': [r] * num_of_days,
            'Date': date_range
        })
        zero_df.loc[:, NPI_COLUMNS] = 0
        zero_outputs.append(zero_df)

    print(zero_geos)
    print(others)

    ratio = 50
    s = time.time()
    outputs = Parallel(backend='loky', n_jobs=6)(delayed(run_geo)(geo, start_date_str, end_date_str,
                                                                  path_to_cost_file, path_to_prior_ips_file, ratio)
                                                 for geo in others)
    outputs += zero_outputs
    df = pd.concat(outputs)
    df.to_csv('result.csv', index=False)
    e = time.time()
    print(f'Total seconds {e - s}')
