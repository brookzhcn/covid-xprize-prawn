from prawn.predict import predict
import unittest
import os
from prawn.predictor import FinalPredictor

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output')


def predict_wrapper(start_date, end_date, ip_file):
    ips_file_path = os.path.join(DATA_PATH, ip_file)
    assert os.path.exists(ips_file_path), 'ips file not exists'
    output_filename = '%s_%s.csv' % (start_date, end_date)
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    # first clean output file
    if os.path.exists(output_filepath):
        print('remove exist output file: %s' % output_filepath)
        os.remove(output_filepath, )
    predict(start_date, end_date, ips_file_path, output_filepath)
    # to check generate the desired file
    exists = os.path.exists(output_filepath)
    return exists


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

    def test_extract_npis_feature(self):
        self.predictor.predict()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPredict("test_predict_month_oct"))
    suite.addTest(TestPredict("test_predict_month_sep"))
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
