from prawn.predict import FinalPredictor
import unittest
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output')


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("this setupclass() method only called once.\n")

    @classmethod
    def tearDownClass(cls):
        print("this tearDownClass() method only called once too.\n")

    def setUp(self) -> None:
        self.predictor = FinalPredictor(start_date_str='2020-12-01', end_date_str='2020-12-31',
                                        path_to_ips_file=os.path.join(DATA_PATH, 'future_ip.csv'),
                                        verbose=True)

        print('set up test for predict API\n')

    def tearDown(self) -> None:
        print('exit test\n')

    def test_predict(self):
        preds_df = self.predictor.predict()
        # Save to a csv file
        preds_df.to_csv(os.path.join(DATA_PATH, 'results.csv'), index=False)

    def test_fit_total(self):
        self.predictor.fit_total()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPredict("test_predict_month_oct"))
    suite.addTest(TestPredict("test_predict_month_sep"))
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
