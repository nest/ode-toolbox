import unittest

import shapes


class TestShapeFunction(unittest.TestCase):

    def test_shape_to_odes(self):
        shape_inh = ShapeFunction("I_in", "(e/tau_syn_in) * t * exp(-t/tau_syn_in)")
        shape_exc = ShapeFunction("I_ex", "(e/tau_syn_ex) * t * exp(-t/tau_syn_ex)")

        print(shape_inh.nestml_ode_form)
        print(shape_exc.nestml_ode_form)


class TestShapeODE(unittest.TestCase):

    def test_ode_shape(self):

        shape_inh = ShapeODE("alpha", "-1/tau**2 * alpha -2/tau * alpha'", ["0", "e/tau"])

        print shape_inh
        
        
if __name__ == '__main__':
    unittest.main()
