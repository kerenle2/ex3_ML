import numpy as np

# define all hyper-parameters here:

output_layer_size = 10
hidden_layer_size = 100 # ?
input_layer_size = 784
data_min_val = 0
data_max_val = 255
validation = 0.2 # ? can change...
# alpha =
# learning_rate =
num_epochs = 50

def normalize(x):
    return np.divide(x, data_max_val)

def load_data():
    train_y = np.loadtxt("train_y")
    train_y = np.reshape(train_y, (len(train_y), 1))
    train_x = np.loadtxt("train_x")
    test_x = np.loadtxt("test_x")
    return train_x, train_y, test_x

sigmoid = lambda x: 1 / (1 + np.exp(-x))

relu = lambda x: np.maximum(x, np.zeros(np.shape(x)))

def drelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def init_params():
    params = {'W1' : np.random.rand(hidden_layer_size, input_layer_size) *0.2 - 0.1,
              'b1' : np.random.rand(hidden_layer_size, 1)*0.2 - 0.1,
              'W2' : np.random.rand(output_layer_size, hidden_layer_size)*0.2 - 0.1,
              'b2' : np.random.rand(output_layer_size, 1)*0.2 - 0.1}
    return params

def train_one_epoch(x, y, params):
    epoch_loss = []
    x, y = shuffle2arr(x, y)
    for example, label in zip(train_x, train_y):
        example = normalize(example)
        ff_cache = feed_forward(example, label, params)
        bp_cache = back_prop()  # IMPLEMENT THIS
        # need to continue..
        #update params
    return params


def feed_forward(x, y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    x.shape = (input_layer_size, 1) # this makes x transpose (x as a columne, not as a row)
    z1 = np.dot(W1, x) + b1
    h1 = relu(z1)
    # do we want more layres? with what activation functions? if so, add here.
    # .....
    z_output = np.dot(W2, h1) + b2
    h_output = softmax(z_output)

    y_vec = np.zeros(output_layer_size)
    y_vec[np.int(y)] = 1

    loss = -(np.dot(y_vec, np.log(h_output)) + np.dot((1 - y_vec), np.log(1 - h_output)))

    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z_output': z_output, 'h_output': h_output, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def back_prop():
    pass

def shuffle2arr(x, y):
    alldata = np.append(x, y, axis=1)
    np.random.seed(0)
    alldata = np.random.permutation(alldata)

    datax = alldata[:, :-1]
    datay = alldata[:, -1]

    return datax, datay


if __name__ == "__main__":
    params = init_params()
    data_x, data_y, test_x = load_data()
    data_x, data_y = shuffle2arr(data_x, data_y)

    # data_x = normalize(data_x)

    data_size = len(data_y) #DEBUG! OR np.size(data_y)
    train_size = int(data_size * (1 - validation))

    train_x = data_x[:train_size, :]
    train_y = data_y[:train_size]
    train_y = np.reshape(train_y, (len(train_y), 1))

    validation_x = data_x[train_size:, :]
    validation_y = data_y[train_size:]
    validation_y = np.reshape(validation_y, (len(validation_y), 1))

    # train the model:
    for epoch in range(num_epochs):
        params = train_one_epoch(train_x, train_y, params)
