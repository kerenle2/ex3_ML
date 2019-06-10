import numpy as np

# define all hyper-parameters here:

output_layer_size = 10
hidden_layer_size = 100 # ?
input_layer_size = 784
data_min_val = 0
data_max_val = 255
validation = 0.2 # ? can change...
# alpha =
learning_rate = 0.005
num_epochs = 50


def load_data():
    # train_x = np.loadtxt("train_x")
    # train_y = np.loadtxt("train_y")
    train_x = load_10000_data("train_x")
    train_y = load_10000_data("train_y")
    train_y = train_y[:,0]
    # train_y = np.reshape(train_y, (len(train_y), 1))
    test_x = np.loadtxt("test_x")
    return train_x, train_y, test_x

#############load train x - just 1000 lines - tempuarly!!!!!1##########
def load_10000_data(file):
    rows = 0
    for line in open(file):
        rows += 1
    data = np.zeros((10000,784))
    row = 0
    for line in open(file):
        if row < 10000:
            data[row] = np.fromstring(line, sep=" ")
            row += 1
    return data

def normalize(x):
    return np.divide(x, data_max_val)

# def load_data():
#     train_y = np.loadtxt("train_y")
#     train_y = np.reshape(train_y, (len(train_y), 1))
#     train_x = np.loadtxt("train_x")
#     test_x = np.loadtxt("test_x")
#     return train_x, train_y, test_x

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
    x, y = shuffle2arr(x, y) ## why shuffle again????
    for example, label in zip(x, y):
        example = normalize(example)
        ff_cache = feed_forward(example, label, params)
        bp_cache = back_prop(ff_cache)  # IMPLEMENT THIS
        # need to continue..
        params = update_params(bp_cache, learning_rate)
        epoch_loss.append(ff_cache['loss']) #DEBUG = check if the
    return params, epoch_loss

def update_params(bp_cache, learning_rate):
    b1, W1, b2, W2 = [params[key] for key in ('b1', 'W1', 'b2', 'W2')]
    b1 = b1 - learning_rate * b1
    W1 = W1 - learning_rate * W1
    b2 = b2 - learning_rate * b2
    W2 = W2 - learning_rate * W2
    ret = {'b1': b1, 'W1': W1, 'b2': b2, 'W2': W2 }
    for key in params:
        ret[key] = params[key]
    return ret

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
    # print("h_oupput is ", str(h_output[:,0]))
    # print("y_vec is ", str(y_vec))
    loss = -(np.dot(y_vec, np.log(h_output)) + np.dot((1 - y_vec), np.log(1 - h_output)))

    ret = {'x': x, 'y_real': y_vec, 'z1': z1, 'h1': h1, 'z_output': z_output, 'h_output': h_output, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def back_prop(ff_cache):
    x, y_real, z1, h1, z_output, h_output, loss = [ff_cache[key] for key in ('x', 'y_real', 'z1', 'h1', 'z_output', 'h_output', 'loss')]
    #transpose dimensions of y_real to fit h_output
    y_real = np.reshape(y_real,(len(y_real),1))
    dz2 = (h_output - y_real)  # dL/dz2
    dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
    db2 = dz2  # dL/dz2 * dz2/db2
    # dh1 =
    dz1 = np.dot(ff_cache['W2'].T, dz2) * drelu(z1)  # dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    ret = {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2 }
    return ret


def predict_y(y, params):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    y.shape = (input_layer_size, 1)  # this makes x transpose (x as a columne, not as a row)
    z1 = np.dot(W1, y) + b1
    h1 = relu(z1)
    z_output = np.dot(W2, h1) + b2
    h_output = softmax(z_output)
    return np.argmax(h_output)


def evaluate(x,y,params):
    counter = 0 #count the correct evaluates
    for example, label in zip(x, y):
        example = normalize(example)
        y_hat = predict_y(example,params)
        if y_hat == label:
            counter += 1
    correctness = counter / len(x)
    return correctness


################
#shuffle rows in x and y in the same order
#############
def shuffle2arr_old(x,y):
    # add x and y together:
    numFeatures = len(x[0])
    alldata = np.zeros((len(x[:, 1]), numFeatures + 1))
    # alldata = np.append(x, y, axis=1)
    alldata[:,0:-1] = x
    alldata[:, -1] = y
    np.random.seed(0)
    alldata = np.random.permutation(alldata)
    datax = alldata[:, :-1]
    datay = alldata[:, -1]
    return datax, datay

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
    data_x, data_y = shuffle2arr_old(data_x, data_y)

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
        params, epoch_loss = train_one_epoch(train_x, train_y, params)
        mean_loss = np.mean(epoch_loss)
        correctness = evaluate(validation_x,validation_y,params)
        print("loss is: " + str(mean_loss))
        print("correctness is: " + str(correctness))
        if correctness > 0.9:
            break