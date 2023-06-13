from data_load import get_dataset, load_data
from constants import *
from training import train_model


print("Reading images")
sandals = load_data(PATH_SANDALS)
shoes = load_data(PATH_SHOES)
boots = load_data(PATH_BOOTS)

print("Creating dataset")
x_train_set, x_test_set, x_val_set, y_train_set, y_test_set, y_val_set = get_dataset(boots, shoes, sandals)

print("Training")
train_model(x_train_set, x_val_set, y_train_set, y_val_set)
