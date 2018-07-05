class Placeholder(object):

    def __init__(self, **kwargs):
        self.index = None
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def eval(self):
        if hasattr(self, 'op'):
            args = list(self.args)
            kwargs = self.kwargs
            for i, a in enumerate(args):
                if isinstance(a, Placeholder):
                    args[i] = a.eval()
            for k in kwargs:
                v = kwargs[k]
                if isinstance(v, Placeholder):
                    kwargs[k] = v.eval()
            output = self.op.f(*args, **kwargs)
            idx = self.index
            if idx is None:
                self.value = output
            else:
                self.value = output[idx]
            return self.value
        else:
            try:
                return self.value
            except AttributeError as ex:
                raise Exception("Uable to evaluate. No value/inputs provided.")


    def set_value(self, value):
        self.value = value


class Variable(Placeholder):

    def __init__(self, value, **kwargs):
        self.value = value
        super(Variable, self).__init__(**kwargs)


class Op(object):

    def __init__(self, f, num_outputs=1):
        self.num_outputs = num_outputs
        self.f = f

    def __call__(self, *args, **kwargs):
        if self.num_outputs == 1:
            y = Placeholder()
            y.op = self
            y.args = args
            y.kwargs = kwargs
            y.index = None
            return y
        outputs = []
        for i in range(self.num_outputs):
            y = Placeholder()
            y.op = self
            y.args = args
            y.kwargs = kwargs
            y.index = i
            outputs.append(y)
        return outputs


class Graph(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, inputs):
        for val, ph in zip(inputs, self.inputs):
            ph.set_value(val)
        
        return [o.eval() for o in self.outputs]
