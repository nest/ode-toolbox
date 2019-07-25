from sympy.printing import StrPrinter


class SympyPrinter(StrPrinter):

    def _print_Exp1(self, expr):
        return 'e'
