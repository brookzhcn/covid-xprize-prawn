import pandas as pd
import numpy as np
from standard_predictor.xprize_predictor import NPI_COLUMNS, XPrizePredictor
import math
import pygad
import time
import multiprocessing
import numpy
import random

# weekend_related_index = [5, 6]

IP_MAX_VALUES = {
    'C1_School closing': 3,
    'C2_Workplace closing': 3,
    'C3_Cancel public events': 2,
    'C4_Restrictions on gatherings': 4,
    'C5_Close public transport': 2,
    'C6_Stay at home requirements': 3,
    'C7_Restrictions on internal movement': 2,
    'C8_International travel controls': 4,
    'H1_Public information campaigns': 2,
    'H2_Testing policy': 3,
    'H3_Contact tracing': 2,
    'H6_Facial Coverings': 4
}
MAX_VALUES = [3, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]

GENE_SPACE = [list(range(m+1)) for m in MAX_VALUES]


class MyGa(pygad.GA):

    def run(self):
        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        if self.valid_parameters == False:
            raise ValueError(
                "ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        if not (self.on_start is None):
            self.on_start(self)

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness()
            fitness_num = len(fitness)
            # for some cases for zero
            if fitness_num < self.sol_per_pop:
                return self.population[fitness_num - 1]

            if not (self.on_fitness is None):
                self.on_fitness(self, fitness)

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=fitness)
            if best_solution_fitness >= -5.0:
                return best_solution

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Appending the best solution to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)
            # early stop
            print(f"############################# generation {generation}")
            if len(self.best_solutions_fitness) >= 2:
                f1 = self.best_solutions_fitness[-1]
                f2 = self.best_solutions_fitness[-2]
                d = abs(f1 - f2)
                p = abs(d / f2)
                print(f"improve percent {p:.3f} {f1 - f2:.2f}")
                if p < 0.01 or d < 5:
                    print('Early stop')
                    return self.best_solutions[-1]

            # Selecting the best parents in the population for mating.
            parents = self.select_parents(fitness, num_parents=self.num_parents_mating)
            if not (self.on_parents is None):
                self.on_parents(self, parents)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.num_offspring <= self.keep_parents:
                    offspring_crossover = parents[0:self.num_offspring]
                else:
                    offspring_crossover = numpy.concatenate(
                        (parents, self.population[0:(self.num_offspring - parents.shape[0])]))
            else:
                # Generating offspring using crossover.
                offspring_crossover = self.crossover(parents,
                                                     offspring_size=(self.num_offspring, self.num_genes))
                if not (self.on_crossover is None):
                    self.on_crossover(self, offspring_crossover)
            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                offspring_mutation = offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                offspring_mutation = self.mutation(offspring_crossover)
                if not (self.on_mutation is None):
                    self.on_mutation(self, offspring_mutation)

            if (self.keep_parents == 0):
                self.population = offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:parents.shape[0], :] = parents
                self.population[parents.shape[0]:, :] = offspring_mutation
            elif (self.keep_parents > 0):
                parents_to_keep = self.steady_state_selection(fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = offspring_mutation

            self.generations_completed = generation + 1  # The generations_completed attribute holds the number of the last completed generation.

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            if not (self.on_generation is None):
                r = self.on_generation(self)
                if type(r) is str and r.lower() == "stop":
                    return self.best_solutions[-1]
                    # # Before aborting the loop, save the fitness value of the best solution.
                    # _, best_solution_fitness, _ = self.best_solution()
                    # self.best_solutions_fitness.append(best_solution_fitness)
                    # break

            time.sleep(self.delay_after_gen)

        last_gen_fitness = self.cal_pop_fitness()
        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=last_gen_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        self.best_solution_generation = \
            numpy.where(
                numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][
                0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True  # Set to True only after the run() method completes gracefully.

        if not (self.on_stop is None):
            self.on_stop(self, last_gen_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = numpy.array(self.best_solutions)
        return self.best_solutions[-1]

    def cal_pop_fitness(self):

        """
        Calculating the fitness values of all solutions in the current population.
        It returns:
            -fitness: An array of the calculated fitness values.
        """

        if self.valid_parameters == False:
            raise ValueError(
                "ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")
        # keep the best parent and avoid duplicate compute
        has_best_parent = len(self.best_solutions_fitness) > 0
        pop_fitness = [self.best_solutions_fitness[-1]] if has_best_parent else []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(self.population):
            if has_best_parent and sol_idx == 0:
                print('ignore first')
                continue
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

            if abs(fitness) < 50:
                print(f'fitness {fitness} found')
                break
        pop_fitness = numpy.array(pop_fitness)

        return pop_fitness


def add_geo_id(df):
    # use standard predictor geo id
    df["GeoID"] = np.where(df["RegionName"].isnull(),
                           df["CountryName"],
                           df["CountryName"] + ' / ' + df["RegionName"])


def get_country_region(geo_id):
    parts = geo_id.split('/')
    if len(parts) == 1:
        return parts[0], None
    return parts[0].rstrip(), parts[1].lstrip()


class PrawnPrescribe:
    def __init__(self, start_date_str: str, end_date_str: str, path_to_prior_ips_file,
                 path_to_cost_file, predictor: XPrizePredictor, interval=7, verbose=True):
        self.start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        # self.start_dayofweek = self.start_date.dayofweek
        # sat_index = 5 - self.start_dayofweek
        # if sat_index < 0:
        #     sat_index += 7
        # sun_index = 6 - self.start_dayofweek

        self.end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.date_range = pd.date_range(self.start_date, self.end_date)
        self.cost_df = self.load_cost_df(path_to_cost_file)
        ips_df = self.load_ips_df(path_to_prior_ips_file)
        self.ips_df = ips_df
        # self.ips_df = ips_df[(ips_df['Date'] >= self.start_date_str) & (ips_df['Date'] <= self.end_date_str)]
        cost_dict = {}
        self.geo_list = self.cost_df.GeoID.unique().tolist()
        for geo_id in self.geo_list:
            cost_dict[geo_id] = self.cost_df[self.cost_df['GeoID'] == geo_id][NPI_COLUMNS]
        self.cost_dict = cost_dict
        self.num_of_days = (self.end_date - self.start_date).days + 1
        # the policy change according to this interval
        self.interval = interval
        self.num_of_intervals = math.ceil(self.num_of_days / self.interval)
        # weekend_index = []
        # while sun_index < self.num_of_days or sat_index < self.num_of_days:
        #     if sat_index < self.num_of_days:
        #         weekend_index.append(sat_index)
        #     if sun_index < self.num_of_days:
        #         weekend_index.append(sun_index)
        #
        #     sat_index += 7
        #     sun_index += 7
        #
        # self.weekend_index = weekend_index

        self.predictor = predictor
        self.verbose = verbose
        self.generation = 0

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
    def _random_policy():
        return [random.randint(0, m) for m in MAX_VALUES]

    @staticmethod
    def _max_policy():
        return np.array(MAX_VALUES)

    @staticmethod
    def _fix_value_policy(val):
        return np.array([val] * len(NPI_COLUMNS))

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

    def get_max_policy_flat(self):
        policy = []
        for interval_index in range(self.num_of_intervals):
            rp = np.array(MAX_VALUES)
            policy.append(rp)
        return np.concatenate(policy)

    def get_fix_value_policy_flat(self, val):
        policy = []
        for interval_index in range(self.num_of_intervals):
            rp = self._fix_value_policy(val)
            policy.append(rp)
        return np.concatenate(policy)

    def get_two_value_policy_flat(self, val, last_interval_val):
        policy = []
        for interval_index in range(self.num_of_intervals - 1):
            rp = self._fix_value_policy(val)
            policy.append(rp)
        policy.append(self._fix_value_policy(last_interval_val))
        return np.concatenate(policy)

    def get_column_policy_flat(self, column_index):
        policy = []
        for interval_index in range(self.num_of_intervals - 1):
            rp = self._fix_value_policy(5)
            rp[column_index] = 0
            policy.append(rp)
        return np.concatenate(policy)

    def transform_to_total_policy(self, interval_policy):
        total = interval_policy.repeat(self.interval, axis=0)
        return total[:self.num_of_days]

    def set_policy_df(self, gdf, total_policy):
        """
        gdf should contain only data in [start_date, end_date]
        """
        gdf.loc[:, NPI_COLUMNS] = total_policy
        # gdf.iloc[self.weekend_index, weekend_related_index] = 0

    def predict(self, gdf, policy):
        """
        policy: total policy
        """
        # policy = self.transform_to_total_policy(policy)
        self.set_policy_df(gdf, policy)
        pred_df = self.predictor.predict_from_df(self.start_date_str, self.end_date_str, gdf)
        return pred_df

    ################################################################################################
    # The follow methods implement GA
    ################################################################################################
    def transfer_to_real_policy(self, solution):
        return np.array(np.split(solution, self.num_of_intervals))

    def get_fitness_func(self, gdf, geo_id, ratio=1):
        def fitness_func(solution, solution_index):
            """
            solution: interval policy
            """
            real_policy = self.transfer_to_real_policy(solution=solution)
            w = self.cost_dict[geo_id]
            total_policy = self.transform_to_total_policy(real_policy)
            avg_stringency = w.dot(total_policy.T).mean(axis=1).iloc[0]
            pred_df = self.predict(gdf, total_policy)
            avg_new_case = pred_df['PredictedDailyNewCases'].mean()
            # print(pred_df['PredictedDailyNewCases'])
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
            print(f"############################# start generation {self.generation}")

        return on_start

    def get_on_fitness(self):
        def on_fitness(ga_instance, population_fitness):
            pass

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
            pass

        return on_generation

    def get_on_stop(self):
        def on_stop(ga_instance, last_population_fitness):
            print("on_stop()")

        return on_stop

    def get_gene_space(self):
        space = []
        for _ in range(self.num_of_intervals):
            space += GENE_SPACE
        return space

    def run_geo(self, geo, prescription_index=0, ratio=20):
        # ip = self.get_interval_policy()
        ips_df = self.ips_df
        # gdf = ips_df[ips_df.GeoID == geo]
        country_name, region_name = get_country_region(geo)
        # print(gdf)
        # pred_df = self.predict(gdf, ip)
        # average = pred_df['PredictedDailyNewCases'].mean()
        # flat_policy = self.get_interval_policy_flat()
        # print(flat_policy)
        # print(average)
        num_of_initial_policies = 15
        initial_population = []
        # 0, 1, 2
        for v in range(3):
            initial_population.append(self.get_fix_value_policy_flat(v))

        for v in range(1, 3):
            p = self.get_two_value_policy_flat(v, last_interval_val=v - 1)
            initial_population.append(p)
        initial_population.append(self.get_max_policy_flat())

        for _ in range(num_of_initial_policies):
            p = self.get_interval_policy_flat()
            initial_population.append(p)

        gdf = pd.DataFrame({
            'CountryName': [country_name] * self.num_of_days,
            'RegionName': [region_name] * self.num_of_days,
            'GeoID': [geo] * self.num_of_days,
            'Date': pd.date_range(self.start_date_str, self.end_date_str),
        })
        num_generations = 10  # Number of generations.
        num_parents_mating = 10  # Number of solutions to be selected as parents in the mating pool.
        fitness_func = self.get_fitness_func(gdf, geo, ratio)
        ga_instance = MyGa(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           initial_population=initial_population,
                           gene_type=int,
                           parent_selection_type='sss',
                           mutation_type='random',
                           crossover_type='single_point',
                           gene_space=self.get_gene_space(),
                           num_genes=self.num_of_intervals * len(MAX_VALUES),
                           # on_start=prescribe.get_on_start(),
                           # on_fitness=self.get_on_fitness(),
                           # on_parents=prescribe.get_on_parents(),
                           # on_crossover=prescribe.get_on_crossover(),
                           # on_mutation=prescribe.get_on_mutation(),
                           # on_generation=self.get_on_generation(),
                           # on_stop=prescribe.get_on_stop(),
                           suppress_warnings=True,
                           save_best_solutions=True,
                           keep_parents=1
                           )
        ga_instance.improve_percents = 1
        s = time.time()
        solution = ga_instance.run()
        country_name = gdf['CountryName'].iloc[0]
        region_name = gdf['RegionName'].iloc[0]
        num = len(self.date_range)
        # best_solution = ga_instance.best_solutions[-1]
        real_policy = self.transfer_to_real_policy(solution)
        prescription_df = pd.DataFrame({
            'PrescriptionIndex': [prescription_index] * num,
            'CountryName': [country_name] * num,
            'RegionName': [region_name] * num,
            'Date': [d.strftime("%Y-%m-%d") for d in self.date_range]
        })
        prescription_df[NPI_COLUMNS] = self.transform_to_total_policy(real_policy)
        e = time.time()
        # median_policy = np.median(ga_instance.best_solutions, axis=0)
        # median_fitness_val = fitness_func(median_policy, -1)
        # print(f'median_fitness_val: {median_fitness_val}')
        print(f'finish in {e - s} seconds')
        print(f'Best fitness {ga_instance.best_solutions_fitness}')
        # prescription_df.to_csv(f'{country_name}-{region_name}-{self.interval}.csv', index=False)
        return prescription_df

    def filter_geos(self, threshold=50, test_days=14):
        empty_geos = []
        others = []
        dr = pd.date_range(self.start_date_str, periods=test_days)
        end_date_str = dr[-1].strftime('%Y-%m-%d')
        for geo in self.geo_list:
            c, r = get_country_region(geo)
            zero_df = pd.DataFrame({
                'PrescriptionIndex': [0] * test_days,
                'CountryName': [c] * test_days,
                'RegionName': [r] * test_days,
                'GeoID': [geo] * test_days,
                'Date': dr
            })
            zero_df.loc[:, NPI_COLUMNS] = 0
            pred_df = self.predictor.predict_from_df(self.start_date_str, end_date_str, zero_df)
            avg_new_case = pred_df['PredictedDailyNewCases'].mean()
            if avg_new_case < threshold:
                print(f'zero policy for {geo}: {avg_new_case}')
                empty_geos.append(geo)
            else:
                others.append(geo)
        return empty_geos, others


def start_process():
    print('Starting, ', multiprocessing.current_process().name)


def run_geo(geo, start_date, end_date, path_to_cost_file, path_to_prior_ips_file, ratio=10):
    print(f'run {geo}')
    x_predictor = XPrizePredictor()
    prescribe = PrawnPrescribe(start_date_str=start_date, end_date_str=end_date,
                               path_to_prior_ips_file=path_to_prior_ips_file,
                               path_to_cost_file=path_to_cost_file,
                               predictor=x_predictor,
                               interval=14
                               )

    return prescribe.run_geo(geo, ratio=ratio)
    # print(f'pool size {pool_size}')
    # pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)


class GAWrapper(pygad.GA):
    def __init__(self, num_generations, num_parents_mating, fitness_func, initial_population=None, sol_per_pop=None,
                 num_genes=None, init_range_low=-4, init_range_high=4, gene_type=float, parent_selection_type="sss",
                 keep_parents=-1, K_tournament=3, crossover_type="single_point", crossover_probability=None,
                 mutation_type="random", mutation_probability=None, mutation_by_replacement=False,
                 mutation_percent_genes='default', mutation_num_genes=None, random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0, gene_space=None, on_start=None, on_fitness=None, on_parents=None,
                 on_crossover=None, on_mutation=None, callback_generation=None, on_generation=None, on_stop=None,
                 delay_after_gen=0.0, save_best_solutions=False, suppress_warnings=False, geo_id=None):
        super().__init__(num_generations, num_parents_mating, fitness_func, initial_population, sol_per_pop, num_genes,
                         init_range_low, init_range_high, gene_type, parent_selection_type, keep_parents, K_tournament,
                         crossover_type, crossover_probability, mutation_type, mutation_probability,
                         mutation_by_replacement, mutation_percent_genes, mutation_num_genes, random_mutation_min_val,
                         random_mutation_max_val, gene_space, on_start, on_fitness, on_parents, on_crossover,
                         on_mutation, callback_generation, on_generation, on_stop, delay_after_gen, save_best_solutions,
                         suppress_warnings)
        self.geo_id = geo_id


class GACluster:
    def __init__(self, *ga_instances: GAWrapper, num_generations=10, prescribe_instance: PrawnPrescribe = None,
                 ratio=10):
        self.ga_instances = ga_instances
        self.num_generations = num_generations
        self.prescribe_instance = prescribe_instance
        self.ratio = ratio

    def cal_pop_fitness(self):
        prescribe_instance = self.prescribe_instance
        ips_df = prescribe_instance.ips_df
        num_of_population = len(self.ga_instances[0].population)
        fitness_values = {geo_id: [] for geo_id in prescribe_instance.geo_list}
        for pop_index in range(num_of_population):
            policy_list = []
            policy_dict = {}
            for ga in self.ga_instances:
                solution = ga.population[pop_index]
                real_policy = prescribe_instance.transfer_to_real_policy(solution=solution)
                # w = self.cost_dict[geo_id]
                total_policy = prescribe_instance.transform_to_total_policy(real_policy)
                policy_list.append(total_policy)
                policy_dict[ga.geo_id] = total_policy

            ips_df.loc[:, NPI_COLUMNS] = np.concatenate(policy_list)
            print(f'Predict population index {pop_index}')
            pred_df = prescribe_instance.predictor.predict_from_df(prescribe_instance.start_date_str,
                                                                   prescribe_instance.end_date_str,
                                                                   ips_df)
            add_geo_id(pred_df)
            for geo_id, w in prescribe_instance.cost_dict.items():
                pred_gdf = pred_df[pred_df['GeoID'] == geo_id]
                avg_new_case = pred_gdf['PredictedDailyNewCases'].mean()

                avg_stringency = w.dot(policy_dict[geo_id].T).mean(axis=1).iloc[0]

                val = -avg_new_case - self.ratio * avg_stringency
                fitness_values[geo_id].append(val)
        return fitness_values

    def run(self):
        for generation in range(self.num_generations):
            print(f'Generation {generation}')
            # Measuring the fitness of each chromosome in the population.
            fitness_dict = self.cal_pop_fitness()
            for ga_instance in self.ga_instances:
                fitness = fitness_dict[ga_instance.geo_id]
                best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution(pop_fitness=fitness)
                # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
                ga_instance.best_solutions_fitness.append(best_solution_fitness)

                # Appending the best solution to the best_solutions list.
                if ga_instance.save_best_solutions:
                    ga_instance.best_solutions.append(best_solution)

                # Selecting the best parents in the population for mating.
                parents = ga_instance.select_parents(fitness, num_parents=ga_instance.num_parents_mating)

                # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
                if ga_instance.crossover_type is None:
                    if ga_instance.num_offspring <= ga_instance.keep_parents:
                        offspring_crossover = parents[0:ga_instance.num_offspring]
                    else:
                        offspring_crossover = numpy.concatenate(
                            (parents, ga_instance.population[0:(ga_instance.num_offspring - parents.shape[0])]))
                else:
                    # Generating offspring using crossover.
                    offspring_crossover = ga_instance.crossover(parents, offspring_size=(
                        ga_instance.num_offspring, ga_instance.num_genes))

                # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
                if ga_instance.mutation_type is None:
                    offspring_mutation = offspring_crossover
                else:
                    # Adding some variations to the offspring using mutation.
                    offspring_mutation = ga_instance.mutation(offspring_crossover)

                if (ga_instance.keep_parents == 0):
                    ga_instance.population = offspring_mutation
                elif (ga_instance.keep_parents == -1):
                    # Creating the new population based on the parents and offspring.
                    ga_instance.population[0:parents.shape[0], :] = parents
                    ga_instance.population[parents.shape[0]:, :] = offspring_mutation
                elif (ga_instance.keep_parents > 0):
                    parents_to_keep = ga_instance.steady_state_selection(fitness, num_parents=ga_instance.keep_parents)
                    ga_instance.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                    ga_instance.population[parents_to_keep.shape[0]:, :] = offspring_mutation

                ga_instance.generations_completed = generation + 1  # The generations_completed attribute holds the number of the last completed generation.

                time.sleep(ga_instance.delay_after_gen)

            last_gen_fitness = self.cal_pop_fitness()
            for ga_instance in self.ga_instances:
                # Save the fitness value of the best solution.
                _, best_solution_fitness, _ = ga_instance.best_solution(pop_fitness=last_gen_fitness)
                ga_instance.best_solutions_fitness.append(best_solution_fitness)

                # self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
                # After the run() method completes, the run_completed flag is changed from False to True.
                ga_instance.run_completed = True  # Set to True only after the run() method completes gracefully.

                # Converting the 'best_solutions' list into a NumPy array.
                ga_instance.best_solutions = numpy.array(self.best_solutions)


if __name__ == '__main__':
    x_predictor = XPrizePredictor()
    base_dir = '/Users/brook/PycharmProjects/covid-xprize-prawn/prawn/'
    prescribe1 = PrawnPrescribe(start_date_str='2021-01-01', end_date_str='2021-01-31',
                                path_to_prior_ips_file=f'{base_dir}/data/all_2020_ips.csv',
                                path_to_cost_file=f'{base_dir}/data/uniform_random_costs.csv', predictor=x_predictor,
                                interval=14
                                )

    r = prescribe1.run_geo('Argentina', ratio=50)
    print(r)
    # s = time.time()
    # for geo_id in prescribe1.geo_list:
    #     print(f"\n######################### run {geo_id}")
    #     prescribe1.run_geo(geo_id)
    # e = time.time()
    # print(f'Total seconds {e-s}')
    # ga_instance_list = []
    # num_generations = 15
    # num_parents_mating = 10
    #
    # num_of_initial_policies = 50
    # initial_population = []
    # for v in range(6):
    #     initial_population.append(prescribe1.get_fix_value_policy_flat(v))
    #
    # for _ in range(num_of_initial_policies):
    #     p = prescribe1.get_interval_policy_flat()
    #     initial_population.append(p)
    #
    # for geo_id in prescribe1.geo_list:
    #     ga = GAWrapper(
    #         num_generations=num_generations,
    #         num_parents_mating=num_parents_mating,
    #         fitness_func=lambda x, y: 1,
    #         initial_population=initial_population,
    #         gene_type=int,
    #         parent_selection_type='rank',
    #         mutation_type='random',
    #         crossover_type='two_points',
    #         gene_space=[0, 1, 2, 3, 4, 5],
    #         save_best_solutions=True,
    #         geo_id=geo_id,
    #     )
    #     ga_instance_list.append(ga)
    #
    # ga_cluster = GACluster(*ga_instance_list, num_generations=num_generations, prescribe_instance=prescribe1, ratio=10)
    # ga_cluster.run()
    # pool_size = multiprocessing.cpu_count() * 2
    # print(f'pool size {pool_size}')
    # pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    # geo_list = prescribe1.geo_list[:10]
    # Parallel(backend='loky', n_jobs=6)(delayed(prescribe1.run_geo)(geo) for geo in geo_list)
    # print(geo_list)
    # pool_outputs = pool.map(delayed(prescribe.run_geo)(geo) for geo in geo_list)
    # pool.close()
    # pool.join()
