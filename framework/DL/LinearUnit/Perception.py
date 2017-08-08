class Perception(object):
    def __init__(self, input_num, activator):
        '''
        init perceptron
        :param input_num:
        :param activator:
        '''
        self.activator = activator
        # weight vector initialize to 0
        self.weights = [0.0 for _ in range(input_num)]
        # bias unit initialize to 0
        self.bias = 0.0

    def __str__(self):
        '''
        print weight and bias from learnng
        :return:
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        '''
        Input vector and compute perceptron result

        Put input_vec[x1, x2, x3...] and weights [w1, w2, w3, ... ] together
        and turn to [(x1, w1), (x2, w2), (x3, w3), ... ]
        Then we use function map to compute [x1 * w1, x2 * w2, x3 * w3, ...]
        Finally we use reduce method to get sum.
        :param input_vec:
        :return:
        '''
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,
                       zip(input_vec, self.weights))
                   , 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        '''
        :param input_vecs: Example vector
        :param labels: Label vector
        :param iteration: Iteration number
        :param rate: Learn rate
        :return:
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        '''
        One iteration train all the data
        Put input and output together to label list[(input_vec, label), ...]
        Every train example is (input_vec, label)
        :param input_vecs:
        :param labels:
        :param rate:
        :return:
        '''
        samples = zip(input_vecs, labels)
        # for every training example, update the weight by perceptron
        for (input_vec, label) in samples:
            # Compute perceptron output
            output = self.predict(input_vec)
            # Update weight
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        '''
        Update weight
        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        '''
        # Put input_vec[x1, x2, x3, ...] and weights[w1, w2, w3, ...] together
        # Turn to [(x1, w1), (x2, w2), (x3, w3), ...]
        # And then use perceptron rule to update weight
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights))

        self.bias += rate * delta

