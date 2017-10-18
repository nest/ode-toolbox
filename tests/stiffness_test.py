import json
import unittest

from stiffness import check_ode_system_for_stiffness


class TestStiffnessChecker(unittest.TestCase):

    def test_with_iaf_cond_alpha(self):
        with open("iaf_cond_alpha_odes_stiff.json") as infile:
            json_input = json.load(infile)
            self.assertEquals("implicit", check_ode_system_for_stiffness(json_input))

        with open("iaf_cond_alpha_odes.json") as infile:
            json_input = json.load(infile)
            self.assertEquals("explicit", check_ode_system_for_stiffness(json_input))

        with open("iaf_cond_alpha_odes_threshold.json") as infile:
            json_input = json.load(infile)
            self.assertEquals("explicit", check_ode_system_for_stiffness(json_input))


if __name__ == '__main__':
    unittest.main()
