import operator
import numpy as np
import json
import matplotlib.pyplot as plt
from models.NeuralNetModel import NeuralNetModel
from models.PolynomialRegressionModel import PolynomialRegressionModel

def get_training_set_from_file(file):
    return np.genfromtxt(file, delimiter=';').astype(np.int32)

def get_model(x, y, model_config):
    if model_config["model"]["type"] == "regression":
        regression_model = PolynomialRegressionModel(model_config["model_name"], model_config["model"]["polynomial_degree"])
        regression_model.train(x, y)

        return regression_model
    elif model_config["model"]["type"] == "neural_net":
        neural_net_model = NeuralNetModel(model_config["model_name"])
        neural_net_model.train(x, y, model_config["model"])
        
        return neural_net_model
    
    return None

def plot_graph(model_name, x, y, y_pred):
    plt.scatter(x, y, s=10)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_pred), key=sort_axis)
    x, y_pred = zip(*sorted_zip)
    
    plt.plot(x, y_pred, color='m')
    plt.title(f"Amount of {model_name} in each day")
    plt.xlabel("Day")
    plt.ylabel(model_name)
    plt.show()

def print_forecast(model_name, model, beginning_day=0, limit=10):
    next_days_x = np.array(range(beginning_day, beginning_day + limit)).reshape(-1, 1)
    next_days_pred = model.get_predictions(next_days_x)

    print(f"The forecast for {model_name} in the following {str(limit)} days is:")

    for i in range(0, limit):
        print(f"Day {str(i + 1)}: {str(next_days_pred[i])}")

def print_stats(model_config, x, y, model):
    y_pred = model.get_predictions(x)

    print_forecast(model_config["model_name"], model, beginning_day=len(x), limit=model_config["days_to_predict"])

    if isinstance(model, PolynomialRegressionModel):
        print(f"The {model_config['model_name']} model function is: f(x) = {model.get_model_polynomial_str()}")

    plot_graph(model_config["model_name"], x, y, y_pred)
    print("")


def model_handler(model_config):
    training_set = get_training_set_from_file(model_config["dataset"])
    x = training_set[:, 0].reshape(-1, 1)
    y = training_set[:, 1]
    model = get_model(x, y, model_config)

    print_stats(model_config, x, y, model)

if __name__ == "__main__":
    config = {}

    with open("config.json", "r") as f:
        config = json.load(f)
    
    for model_config in config["models"]:
        if "enabled" in model_config and model_config["enabled"] == False:
            continue
        
        model_handler(model_config)