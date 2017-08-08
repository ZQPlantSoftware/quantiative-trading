from Perception import Perception

def f(x):
    return 1 if x > 0 else 0

def train_and_perception():
    # Create perception, Input params is 2, activity method is f
    p = Perception(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

def get_training_dataset():
    '''
    Base on and, training data
    :return:
    '''

    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]

    # Compare with output list
    # [1, 1] -> 1, [0, 0] -> 0, [1, 0] -> 0, [0, 1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels

if __name__ == '__main__':
    and_perception = train_and_perception()
    print and_perception

    print '1 and 1 = %d' % and_perception.predict([1, 1])
    print '0 and 0 = %d' % and_perception.predict([0, 0])
    print '1 and 0 = %d' % and_perception.predict([1, 0])
    print '0 and 1 = %d' % and_perception.predict([0, 1])
