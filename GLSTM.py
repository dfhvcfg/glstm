import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense
from keras.models import Model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

np.random.seed(1120)


data = pd.read_csv('train.csv')
data = np.reshape(np.array(data['wp1']),(len(data['wp1']),1))
print(data[:10])
train_data = data[0:17257]
test_data = data[17257:]

def prepare_dataset(data, window_size):
    X, Y = np.empty((0,window_size)), np.empty((0))
    for i in range(len(data)-window_size-1):
        X = np.vstack([X,data[i:(i + window_size),0]])
        Y = np.append(Y,data[i + window_size,0])
    X = np.reshape(X,(len(X),window_size,1))
    Y = np.reshape(Y,(len(Y),1))
    return X, Y

X_train,y_train = prepare_dataset(train_data,3)
print(X_train)
print(y_train)


def train_evaluate(ga_individual_solution):
    # Decode the Genetic Algorithm solution to get the window size and number of bits
    window_size_bits = BitArray(ga_individual_solution[0:6])
    num_units_bits = BitArray(ga_individual_solution[6:])
    window_size = window_size_bits.uint
    num_of_units = num_units_bits.uint
    print('\nWindow Size: ', window_size, ', Num of Units: ', num_of_units)

    # Return fitness score of 100 if window_size or num_unit is zero
    if window_size == 0 or num_of_units == 0:
        return 100,

        # Segment the train_data based on new window_size;
    # Split the dataset into train set(80) and validation set(20)
    X_data, Y_data = prepare_dataset(train_data, window_size)
    X_train, X_val, y_train, y_val = split(X_data, Y_data, test_size=0.20, random_state=1120)

    # Design an LSTM model to train on training data and predict on validation data
    input_ph = Input(shape=(window_size, 1))
    x = LSTM(num_of_units, input_shape=(window_size, 1))(input_ph)
    predicted_values = Dense(1, activation='tanh')(x)
    model = Model(inputs=input_ph, outputs=predicted_values)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=20, shuffle=True)
    y_pred = model.predict(X_val)

    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print('Validation RMSE: ', rmse, '\n')

    return rmse,


population_size = 4
num_generations = 4
gene_length = 10

#Implementation of Genetic Algorithm using DEAP python library.

#Since we try to minimise the loss values, we use the negation of the root mean squared loss as fitness function.
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

#initialize the variables as bernoilli random variables
toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

#Ordered cross-over used for mating
toolbox.register('mate', tools.cxOrdered)
#Shuffle mutation to reorder the chromosomes
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
#use roulette wheel selection algorithm
toolbox.register('select', tools.selRoulette)
#training function used for evaluating fitness of individual solution.
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n = population_size)
r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)


optimal_individuals_data = tools.selBest(population,k = 1) #select top 1 solution
optimal_window_size = None
optimal_num_units = None

for bi in optimal_individuals_data:
    window_size_bits = BitArray(bi[0:6])
    num_units_bits = BitArray(bi[6:])
    optimal_window_size = window_size_bits.uint
    optimal_num_units = num_units_bits.uint
    print('\n Best Window Size: ', optimal_window_size, ', Best Num of Units: ', optimal_num_units)

#print(optimal_window_size, optimal_num_units)


#hence train the model with the optimal number of lstm units and optimal window size for prediction
X_train,y_train = prepare_dataset(train_data,optimal_window_size)
X_test, y_test = prepare_dataset(test_data,optimal_window_size)

inputs = Input(shape=(optimal_window_size,1))
x = LSTM(optimal_num_units, input_shape=(optimal_window_size,1))(inputs)
predictions = Dense(1, activation='tanh')(x)
model = Model(inputs = inputs, outputs = predictions)
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=20,shuffle=True)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: ', rmse)

