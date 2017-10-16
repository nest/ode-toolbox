import json
import unittest

from stiffness import check_ode_system_for_stiffness


class TestStiffnessChecker(unittest.TestCase):

    def test_with_iaf_cond_alpha(self):
        with open("iaf_cond_alpha_implicit.json") as infile:
            input = json.load(infile)
            print(check_ode_system_for_stiffness(input))


if __name__ == '__main__':
    unittest.main()
