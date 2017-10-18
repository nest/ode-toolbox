import unittest

import shapes


class TestShapeFunction(unittest.TestCase):

    def test_shape_to_odes(self):
        shape_inh = shapes.shape_from_function("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = shapes.shape_from_function("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")
        self.assertIsNotNone(shape_inh.ode_definition)
        self.assertIsNotNone(shape_exc.ode_definition)

        print(shape_inh.ode_definition)
        print(shape_exc.ode_definition)


class TestShapeODE(unittest.TestCase):

    def test_ode_shape(self):

        shape_inh = shapes.shape_from_ode("alpha", "-1/tau**2 * alpha -2/tau * alpha'", ["0", "e/tau"])
        self.assertIsNotNone(shape_inh.derivative_factors)

        print shape_inh.derivative_factors
        
        
if __name__ == '__main__':
    unittest.main()
