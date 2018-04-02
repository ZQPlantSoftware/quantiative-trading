from keras.models import model_from_json

def save_model(model):
    model_json = model.to_json()
    path = "model.json"
    with open(path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print('- Save model success')
    return path


def load_model(path):
    with open(path, encoding='utf-16') as f:
        loaded_model_json = f.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")

    print("- Loading model success")
    return loaded_model
