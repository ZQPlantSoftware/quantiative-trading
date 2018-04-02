from build_model import build_model
from tools.model_persistence import save_model
import matplotlib.pyplot as plt
import numpy as np


def train_model(lstm_training_inputs, lstm_training_outputs, need_save_model=True):
    np.random.seed(202)
    model = build_model(lstm_training_inputs, output_size=1, neurons=20)
    eth_history = model.fit(lstm_training_inputs, lstm_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)

    scores = model.evaluate(lstm_training_inputs, lstm_training_outputs, verbose=0)

    print('scores:', scores)

    if need_save_model:
        save_model(model)

    def plot_error():
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(eth_history.epoch, eth_history.history['loss'])
        ax1.set_title('Training Error')

        if model.loss == 'mae':
            ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
        # just in case you decided to change the model loss calculation
        else:
            ax1.set_ylabel('Model Loss', fontsize=12)
        ax1.set_xlabel('# Epochs', fontsize=12)
        plt.show()

    return model
