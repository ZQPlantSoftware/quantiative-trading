from Perception import Perception

# define activation function f
f = lambda x: x

class LinearUnit(Perception):
    def __init__(self, input_num):
        '''
        Init linear unit, set input param's number
        :param input_num:
        '''
        Perception.__init__(self, input_num, f)