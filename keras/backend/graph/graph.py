class Placeholder(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

    def eval(self):
        try:
            return self.value
        except AttributeError:
            try:
                self.value = self.op(self.inputs)
                return self.value
            except AttributeError:
                raise Exception('Uable to evaluate. No value/inputs provided.')


    def set_value(self, value):
        self.value = value


class Variable(Placeholder):

    def __init__(self, value, **kwargs):
        self.value = value
        super(Variable, self).__init__(**kwargs)


class Op(object):

    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        xt = type(x)
        if xt in (Placeholder, Variable):
            y = Placeholder()
            y.inputs = x
            y.op = self
            return y
        if xt is list:
            x = [x.eval() if type(x) is Placeholder else x]
        elif xt is tuple:
            x = tuple([x.eval() if type(x) is Placeholder else x])
        elif xt is Placeholder:
            x = x.eval()
        return self.f(x)
