from prawn.predict import predict
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
        print("this teardownclass() method only called once too.\n")

    def setUp(self) -> None:
        print('set up test for predict API')

    def tearDown(self) -> None:
        print('exit test')

    def test_predict_month_oct(self):
        ips_file_path = os.path.join(DATA_PATH, 'ip_file.csv')
        self.assertTrue(os.path.exists(ips_file_path))
        start_date = '2020-10-1'
        end_date = '2020-10-31'
        output_filename = '%s_%s.csv' % (start_date, end_date)
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)
        # first clean output file
        if os.path.exists(output_filepath):
            print('remove exist output file: %s' % output_filepath)
            os.remove(output_filepath, )
        predict(start_date, end_date, ips_file_path, output_filepath)
        # to check generate the desired file
        exists = os.path.exists(output_filepath)
        self.assertTrue(exists)
