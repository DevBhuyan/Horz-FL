#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_federated as tff

# Define the Keras model for the classifier
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define the TFF model for the classifier
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=data_preprocessing_fn().get_model_input_spec(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# Define the data preprocessing function
def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            x=element['x'], y=element['y'])
    return dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE).map(
        batch_format_fn)


# Define the Federated Averaging process
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

# Define the initial state of the Federated Averaging process
state = iterative_process.initialize()


# Define the training loop
for round_num in range(NUM_ROUNDS):
    # Sample a set of clients to participate in this round
    sampled_clients = np.random.choice(train_client_ids, size=NUM_CLIENTS_PER_ROUND, replace=False)
    
    # Construct the client datasets for this round
    client_datasets = [
        preprocess(train_data.create_tf_dataset_for_client(client_id))
        for client_id in sampled_clients]
    
    # Run one round of training on the client datasets
    state, metrics = iterative_process.next(state, client_datasets)
    
    # Print the results for this round
    print('Round {:2d}, metrics={}'.format(round_num, metrics))

# Define the evaluation loop
evaluation = tff.learning.build_federated_evaluation(model_fn)
test_metrics = evaluation(state.model, [preprocess(test_data.create_tf_dataset_for_client(client_id)) for client_id in test_client_ids])
print('Test metrics={}'.format(test_metrics))
