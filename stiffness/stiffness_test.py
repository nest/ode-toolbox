import json
import unittest
from stiffness import *


class TestStiffnessChecker(unittest.TestCase):

    def test_with_iaf_cond_alpha(self):
        with open("test/iaf_cond_alpha_imicit.json") as infile:
            input = json.load(infile)
        #  print(check_ode_system_for_stiffness(odes, default_values, threshold_body))



if __name__ == '__main__':
    unittest.main()
