import pandas as pd
import numpy as np
from prawn.standard_predictor.xprize_predictor import NPI_COLUMNS, XPrizePredictor
import math
import pygad
import time
import multiprocessing
from joblib import Parallel, delayed

weekend_related_index = [0, 1]


def add_geo_id(df):
    # use standard predictor geo id
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])


class PrawnPrescribe:
    def __init__(self, start_date_str: str, end_date_str: str, path_to_prior_ips_file,
                 path_to_cost_file, predictor: XPrizePredictor, interval=7):
        self.start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.date_range = pd.date_range(self.start_date, self.end_date)
        self.cost_df = self.load_cost_df(path_to_cost_file)
        ips_df = self.load_ips_df(path_to_prior_ips_file)
        self.ips_df = ips_df
        cost_dict = {}
        self.geo_list = self.cost_df.GeoID.unique().tolist()
        for geo_id in self.geo_list:
            cost_dict[geo_id] = self.cost_df[self.cost_df['GeoID'] == geo_id][NPI_COLUMNS]
        self.cost_dict = cost_dict
        self.num_of_days = (self.end_date - self.start_date).days + 1
        # the policy change according to this interval
        self.interval = interval
        self.num_of_intervals = math.ceil(self.num_of_days / self.interval)
        self.predictor = predictor

    @staticmethod
    def load_cost_df(path_to_cost_file):
        df = pd.read_csv(path_to_cost_file, dtype={"RegionName": str, "RegionCode": str})
        # df['RegionName'] = df['RegionName'].fillna("")
        add_geo_id(df)
        return df

    @staticmethod
    def load_ips_df(data_url):
        ips_df = pd.read_csv(data_url,
                             parse_dates=['Date'],
                             encoding="ISO-8859-1",
                             dtype={"RegionName": str,
                                    "RegionCode": str},
                             error_bad_lines=False)
        add_geo_id(ips_df)
        return ips_df

    def compute_stringency(self, geo_id, npi_array):
        return self.cost_dict[geo_id].dot(npi_array)

    @staticmethod
    def _random_policy(max_value=6):
        return np.random.randint(0, max_value, size=len(NPI_COLUMNS))

    @staticmethod
    def _trans_to_str(rp):
        return ''.join(str(i) for i in rp)

    @staticmethod
    def _trans_to_arr(str_rp):
        """
        represent Gene of GA
        """
        return np.array([int(i) for i in str_rp])

    def get_interval_policy(self):
        """
        we keep policy same in {interval} days
        """
        policy = []
        for interval_index in range(self.num_of_intervals):
            rp = self._random_policy()
            policy.append(rp)
        return np.array(policy)

    def get_interval_policy_str(self):
        """
        for GA input
        """
        policy = []
        for interval_index in range(self.num_of_intervals):
            rp = self._random_policy()
            policy.append(self._trans_to_str(rp))
        return policy

    def get_interval_policy_flat(self):
        policy = []
        for interval_index in range(self.num_of_intervals):
            rp = self._random_policy()
            policy.append(rp)
        return np.concatenate(policy)

    def _transform_to_total_policy(self, interval_policy):
        total = interval_policy.repeat(self.interval, axis=0)
        return total[:self.num_of_days]

    @staticmethod
    def set_policy_df(gdf, total_policy):
        """
        gdf should contain only data in [start_date, end_date]
        """
        gdf.loc[:, NPI_COLUMNS] = total_policy

    def predict(self, gdf, policy):
        """
        policy: interval policy
        """
        policy = self._transform_to_total_policy(policy)
        self.set_policy_df(gdf, policy)
        pred_df = self.predictor.predict_from_df(self.start_date_str, self.end_date_str, gdf)
        return pred_df

    ################################################################################################
    # The follow methods implement GA
    ################################################################################################
    def transfer_to_real_policy(self, solution):
        return np.array(np.split(solution, self.num_of_intervals))

    def get_fitness_func(self, gdf, ratio=1):
        def fitness_func(solution, solution_index):
            """
            solution: interval policy
            """
            geo_id = gdf['GeoID'].iloc[0]
            real_policy = self.transfer_to_real_policy(solution=solution)
            w = self.cost_dict[geo_id]
            avg_stringency = w.dot(real_policy.T).mean(axis=1).iloc[0]
            pred_df = self.predict(gdf, self._transform_to_total_policy(real_policy))
            avg_new_case = pred_df['PredictedDailyNewCases'].mean()
            val = -avg_new_case - ratio * avg_stringency
            print(
                "{} Solution {}: {:.2f},  avg_stringency: {:.2f}  avg_new_case: {:.2f} \n".format(
                    geo_id,
                    solution_index,
                    val,
                    avg_stringency,
                    avg_new_case
                )
            )
            return val

        return fitness_func

    def get_on_start(self):
        def on_start(ga_instance):
            print('On start')
            print(ga_instance)

        return on_start

    def get_on_fitness(self):

        def on_fitness(ga_instance, population_fitness):
            print()
            print("on_fitness()")
            # print(f'min: {population_fitness.min()} max: {population_fitness.max()}')

        return on_fitness

    def get_on_parents(self):
        def on_parents(ga_instance, selected_parents):
            print("on_parents()")
            print(f'parent number {selected_parents.shape[0]}')

        return on_parents

    def get_on_crossover(self):

        def on_crossover(ga_instance, offspring_crossover):
            print("on_crossover()")

        return on_crossover

    def get_on_mutation(self):
        def on_mutation(ga_instance, offspring_mutation):
            print("on_mutation()")

        return on_mutation

    def get_on_generation(self):
        def on_generation(ga_instance):
            print("on_generation()")

        return on_generation

    def get_on_stop(self):
        def on_stop(ga_instance, last_population_fitness):
            print("on_stop()")

        return on_stop

    def run_geo(self, geo, prescription_index=0):
        ip = self.get_interval_policy()
        ips_df = self.ips_df[(self.ips_df.Date >= self.start_date) &
                             (self.ips_df.Date <= self.end_date)]
        gdf = ips_df[ips_df.GeoID == geo].copy()
        # print(gdf)
        pred_df = self.predict(gdf, ip)
        average = pred_df['PredictedDailyNewCases'].mean()
        # flat_policy = self.get_interval_policy_flat()
        # print(flat_policy)

        print(average)
        num_of_initial_policies = 50
        initial_population = []
        for _ in range(num_of_initial_policies):
            p = self.get_interval_policy_flat()
            initial_population.append(p)

        num_generations = 10  # Number of generations.
        num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.
        fitness_func = self.get_fitness_func(gdf, 10)
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               initial_population=initial_population,
                               gene_type=int,
                               parent_selection_type='rank',
                               mutation_type='random',
                               crossover_type='two_points',
                               mutation_percent_genes=10,
                               gene_space=[0, 1, 2, 3, 4, 5],
                               # on_start=prescribe.get_on_start(),
                               # on_fitness=prescribe.get_on_fitness(),
                               # on_parents=prescribe.get_on_parents(),
                               # on_crossover=prescribe.get_on_crossover(),
                               # on_mutation=prescribe.get_on_mutation(),
                               # on_generation=prescribe.get_on_generation(),
                               # on_stop=prescribe.get_on_stop(),
                               save_best_solutions=True
                               )
        s = time.time()
        ga_instance.run()
        country_name = gdf['CountryName'].iloc[0]
        region_name = gdf['RegionName'].iloc[0]
        num = len(self.date_range)
        best_solution = ga_instance.best_solutions[-1]
        real_policy = self.transfer_to_real_policy(best_solution)
        prescription_df = pd.DataFrame({
            'PrescriptionIndex': [prescription_index] * num,
            'CountryName': [country_name] * num,
            'RegionName': [region_name] * num,
            'Date': [d.strftime("%Y-%m-%d") for d in self.date_range]
        })
        prescription_df[NPI_COLUMNS] = self._transform_to_total_policy(real_policy)
        e = time.time()
        # median_policy = np.median(ga_instance.best_solutions, axis=0)
        # median_fitness_val = fitness_func(median_policy, -1)
        # print(f'median_fitness_val: {median_fitness_val}')
        print(f'finish in {e - s} seconds')
        print(f'Best fitness {ga_instance.best_solutions_fitness}')
        prescription_df.to_csv(f'{geo}-{self.interval}.csv', index=False)
        return prescription_df


def start_process():
    print('Starting, ', multiprocessing.current_process().name)


def run_geo(geo, start_date, end_date):
    x_predictor = XPrizePredictor()
    prescribe = PrawnPrescribe(start_date_str=start_date, end_date_str= end_date,
                               path_to_prior_ips_file='data/2020-09-30_historical_ip.csv',
                               path_to_cost_file='data/uniform_random_costs.csv', predictor=x_predictor,
                               interval=14
                               )

    return prescribe.run_geo(geo)
    # print(f'pool size {pool_size}')
    # pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)


if __name__ == '__main__':
    x_predictor = XPrizePredictor()

    prescribe1 = PrawnPrescribe(start_date_str='2020-08-01', end_date_str='2020-08-31',
                                path_to_prior_ips_file='data/2020-09-30_historical_ip.csv',
                                path_to_cost_file='data/uniform_random_costs.csv', predictor=x_predictor,
                                interval=14
                                )

    prescribe1.run_geo('Afghanistan')
    # pool_size = multiprocessing.cpu_count() * 2
    # print(f'pool size {pool_size}')
    # pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    # geo_list = prescribe1.geo_list[:10]
    # Parallel(backend='loky', n_jobs=6)(delayed(prescribe1.run_geo)(geo) for geo in geo_list)
    # print(geo_list)
    # pool_outputs = pool.map(delayed(prescribe.run_geo)(geo) for geo in geo_list)
    # pool.close()
    # pool.join()
