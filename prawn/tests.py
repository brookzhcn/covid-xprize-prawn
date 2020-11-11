from prawn.predict import predict
import unittest
import os

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
        print('set up test for predict API\n')

    def tearDown(self) -> None:
        print('exit test\n')

    def test_predict_month_oct(self):
        start_date = '2020-10-1'
        end_date = '2020-10-31'
        ip_file = 'ip_file1.csv'
        status = predict_wrapper(start_date, end_date, ip_file)
        self.assertTrue(status)

    def test_predict_month_sep(self):
        start_date = '2020-9-1'
        end_date = '2020-9-30'
        ip_file = 'ip_file2.csv'
        status = predict_wrapper(start_date, end_date, ip_file)
        self.assertTrue(status)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPredict("test_predict_month_oct"))
    suite.addTest(TestPredict("test_predict_month_sep"))
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(suite)
