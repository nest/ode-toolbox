import json
import unittest

from stiffness import check_ode_system_for_stiffness


class TestStiffnessChecker(unittest.TestCase):

    def test_with_iaf_cond_alpha(self):
        with open("iaf_cond_alpha_implicit.json") as infile:
            json_input = json.load(infile)
            self.assertEquals("explicit", check_ode_system_for_stiffness(json_input))

    def test_with_aeif_cond_alpha(self):
        with open("aeif_cond_alpha_implicit.json") as infile:
            json_input = json.load(infile)
            self.assertEquals("explicit", check_ode_system_for_stiffness(json_input))


if __name__ == '__main__':
    unittest.main()
