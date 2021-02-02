# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import pandas as pd
from joblib import Parallel, delayed
import time
from prawn.prawn_prescribe import add_geo_id, run_geo


def prescribe(start_date: str,
              end_date: str,
              path_to_prior_ips_file: str,
              path_to_cost_file: str,
              output_file_path) -> None:
    """
    Generates and saves a file with daily intervention plan prescriptions for the given countries, regions and prior
    intervention plans, between start_date and end_date, included.
    :param start_date: day from which to start making prescriptions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making prescriptions, as a string, format YYYY-MM-DDD
    :param path_to_prior_ips_file: path to a csv file containing the intervention plans between inception date
    (Jan 1 2020) and end_date, for the countries and regions for which a prescription is needed
    :param path_to_cost_file: path to a csv file containing the cost of each individual intervention, per country
    See covid_xprize/validation/data/uniform_random_costs.csv for an example
    :param output_file_path: path to file to save the prescriptions to
    :return: Nothing. Saves the generated prescriptions to an output_file_path csv file
    See 2020-08-01_2020-08-04_prescriptions_example.csv for an example
    """
    # !!! YOUR CODE HERE !!!

    cost_df = pd.read_csv(path_to_cost_file, dtype={"RegionName": str, "RegionCode": str})
    add_geo_id(cost_df)
    geo_list = cost_df.GeoID.unique().tolist()
    s = time.time()
    outputs = Parallel(backend='loky', n_jobs=2)(delayed(run_geo)(geo, start_date, end_date,
                                                                  path_to_cost_file, path_to_prior_ips_file)
                                                 for geo in geo_list)
    df = pd.concat(outputs)
    df.to_csv(output_file_path, index=False)
    e = time.time()
    print(f'Total seconds {e-s}')


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to prescribe, included, as YYYY-MM-DD."
                             "For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prescription, included, as YYYY-MM-DD."
                             "For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_past",
                        dest="prior_ips_file",
                        type=str,
                        required=True,
                        help="The path to a .csv file of previous intervention plans")
    parser.add_argument("-c", "--intervention_costs",
                        dest="cost_file",
                        type=str,
                        required=True,
                        help="Path to a .csv file containing the cost of each IP for each geo")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    args = parser.parse_args()
    print(f"Generating prescriptions from {args.start_date} to {args.end_date}...")
    prescribe(args.start_date, args.end_date, args.prior_ips_file, args.cost_file, args.output_file)
    print("Done!")

