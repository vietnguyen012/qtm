import typing
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from ..dag import circuit_to_adjacency_matrix
from .. import constant

import numpy as np

def max_pooling(input_data, pool_size):
    """
    Perform max pooling on the input data.

    Args:
    - input_data: A 2D numpy array representing the input feature map.
    - pool_size: The size of the pooling window (e.g., (2, 2) for a 2x2 window).

    Returns:
    - pooled_data: A 2D numpy array representing the result of max pooling.
    """
    input_height, input_width = input_data.shape
    pool_height, pool_width = pool_size
    output_height = input_height // pool_height
    output_width = input_width // pool_width

    pooled_data = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            # Extract the local region for pooling
            start_row = i * pool_height
            end_row = start_row + pool_height
            start_col = j * pool_width
            end_col = start_col + pool_width

            # Perform max pooling within the local region
            local_region = input_data[start_row:end_row, start_col:end_col]
            pooled_data[i, j] = np.max(local_region)

    return pooled_data

class Predictor():
    def __init__(self, params: typing.Union[typing.Dict, str], circuits, fitnesss):
        self.params = params
        self.rate_train = params['rate_train']
        self.rate_val = params['rate_val']
        self.rate_test = params['rate_test']
        self.circuits = circuits
        self.xs = []
        self.ys = fitnesss
        self.num_item = len(self.ys)
        print(self.num_item)
        for i in range(0, self.num_item):
            adj_matrix = circuit_to_adjacency_matrix(self.circuits[i])
            self.xs.append(adj_matrix)

        self.num_node = 0
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = [
        ], [], [], [], [], []
        self.history = None
        self.model = None
    def prepare_dataset(self):
        self.num_node = self.circuits[0].num_qubits * self.circuits[0].depth()
        self.x_train = np.array(self.xs[:int(self.num_item * self.rate_train)])
        self.y_train = np.array(self.ys[:int(self.num_item * self.rate_train)])
        self.x_val = np.array(self.xs[int(self.num_item * self.rate_train):int(self.num_item * (self.rate_train + self.rate_val))])
        self.y_val = np.array(self.ys[int(self.num_item * self.rate_train):int(self.num_item * (self.rate_train + self.rate_val))])
        self.x_test = np.array(
            self.xs[int(self.num_item * (self.rate_train + self.rate_val)):])
        self.y_test = np.array(
            self.ys[int(self.num_item * (self.rate_train + self.rate_val)):])

    def fit(self):
        # self.model = tf.keras.models.Sequential([
        #     # tf.keras.layers.Conv2D(
        #     #     32, (3, 3), 
        #     #     activation='relu', 
        #     #     input_shape=(self.num_node, self.num_node, 1), 
        #     #     kernel_regularizer=tf.keras.regularizers.l2(constant.L2_REGULARIZER_RATE)
        #     # ),
        #     # tf.keras.layers.AveragePooling2D((2, 2)),
        #     tf.keras.layers.Flatten(input_shape=(self.num_node, self.num_node)),
        #     # tf.keras.layers.Dropout(constant.DROP_OUT_RATE),
        #     tf.keras.layers.Dense(
        #         128, 
        #         activation='relu', 
        #         kernel_regularizer=tf.keras.regularizers.l2(constant.L2_REGULARIZER_RATE)
        #     ),
        #     tf.keras.layers.Dense(
        #         64, 
        #         activation='relu', 
        #         kernel_regularizer=tf.keras.regularizers.l2(constant.L2_REGULARIZER_RATE)
        #     ),
        #     tf.keras.layers.Dropout(constant.DROP_OUT_RATE),
        #     tf.keras.layers.Dense(4, activation='softmax')
        # ])

        # self.model.compile(loss='categorical_crossentropy',
        #                     optimizer='adam',
        #                     metrics=['accuracy'])
        # Train the model
        self.model = models.Sequential([
            #tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(input_shape=(self.num_node, self.num_node)),  # Biến đổi ma trận 10x10 thành vector 100 phần tử
            tf.keras.layers.Dense(64, activation='relu'),  # Tầng fully connected với 128 đơn vị ẩn và hàm kích hoạt ReLU
            tf.keras.layers.Dropout(constant.DROP_OUT_RATE),
            tf.keras.layers.Dense(32, activation='relu'),  # Tầng fully connected với 128 đơn vị ẩn và hàm kích hoạt ReLU
            tf.keras.layers.Dropout(constant.DROP_OUT_RATE),
            tf.keras.layers.Dense(4, activation='softmax')  # Tầng fully connected với 4 đơn vị đầu ra và hàm kích hoạt softmax
        ])

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(self.x_train, self.y_train, epochs=constant.NUM_EPOCH,
                            batch_size=1, validation_data=(self.x_val, self.y_val))
        return
    def plot(self):
        plt.plot(self.history.history['loss'], label = 'Losss')
        plt.plot(self.history.history['val_loss'],  label = 'Val loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        return
    
    def augumented_data(self, num_augmented, noise_stddev):
        def argument(data, noise_stddev):
            def add_noise(matrix, stddev):
                noise = np.random.normal(0, stddev, matrix.shape)
                noisy_data = matrix + noise
                return noisy_data
            augmented_data = add_noise(data, noise_stddev)
            return augmented_data
        for i in range(0, num_augmented):
            j = np.random.randint(0, self.num_item, 1)[0]
            new_data = argument(self.xs[j], noise_stddev)
            self.xs.append(new_data)
            self.ys.append(self.ys[j])
        return
    def save_model(self, filename: str):
        self.model.save(filename)
        return
        
    def load_model(self, filename: str):
        self.model = tf.keras.models.load_model(filename)
        return
    
    def predict(self):
        return self.model.predict(self.x_test)
    def predict_x(self, circuits):
        xs = []
        for i in range(0, len(circuits)):
            adj_matrix = circuit_to_adjacency_matrix(self.circuits[i])
            xs.append(adj_matrix)
        return self.model.predict(np.array(xs))