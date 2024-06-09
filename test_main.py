import unittest
import os
import yaml
from unittest.mock import patch
from main import parser

class TestMain(unittest.TestCase):
    @patch('main.argparse.ArgumentParser.parse_args',
           return_value=argparse.Namespace(param='smcc_MacOpt_gorilla', env=1, i=200))
    def test_argument_parsing(self, mock_args):
        args = parser.parse_args()
        self.assertEqual(args.param, 'smcc_MacOpt_gorilla')
        self.assertEqual(args.env, 1)
        self.assertEqual(args.i, 200)

    def test_config_loading(self):
        config_path = 'params/smcc_MacOpt_gorilla.yaml'
        with open(config_path, 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        self.assertIsInstance(params, dict)
        self.assertIn('experiment', params)
        self.assertIn('env', params)

if __name__ == '__main__':
    unittest.main()

