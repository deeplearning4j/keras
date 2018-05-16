class Placeholder(object):

    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def eval(self):
        try:
            return self.value
        except AttributeError:
            try:
                ags = list(self.ags)
                for i, a in enumerate(args):
                    if isinstance(a, Placeholder):
                        args[i] = a.eval()
                for k in kwargs:
                    v = kwargs[k]
                    if isinstance(v, Placeholder):
                        kwargs[k] = v.eval()
                self.value = self.op.f(*args, **kwargs)
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

    def __call__(self, *args, **kwargs):
        y = Placeholder()
        y.op = self
        y.args = args
        y.kwargs = kwargs
        return y
