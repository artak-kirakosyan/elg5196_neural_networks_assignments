import numpy as np
from random import randint


# all those functions are from assignment 1 with small modifications.
def sigmoid_value(x):
    return 1 / (1 + np.exp(-1 * x))


def derivative_of_sigmoid(x):
    return sigmoid_value(x) * (1 - sigmoid_value(x))


def calculate_hidden_node_values(input_pattern, hidden_weights, hidden_node_thetas):
    hidden_sum = np.dot(hidden_weights.T, input_pattern) + hidden_node_thetas
    return np.array([sigmoid_value(i) for i in hidden_sum]), hidden_sum


def calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas):
    output_sum = np.dot(output_weights.T, hidden_node_values) + output_node_thetas
    return np.array([sigmoid_value(i) for i in output_sum]), output_sum


def calculate_current_error(desired_output, current_output):
    current_error = 1 / 2 * sum([i ** 2 for i in (desired_output - current_output)])
    return current_error


def calculate_output_error_terms(desired_output, output_node_values, output_sum):
    output_error_terms = np.array(
        [(derivative_of_sigmoid(output_sum[i]) * (desired_output[i] - output_node_values[i])) for i in
         range(len(desired_output))])
    return output_error_terms


def calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum):
    hidden_error_terms = np.array(
        [(derivative_of_sigmoid(hidden_sum[i]) * np.dot(output_error_terms, output_weights[i])) for i in
         range(len(hidden_sum))])
    return hidden_error_terms


def update_and_return_output_weights(output_weights, output_error_terms, hidden_node_values, learning_rate):
    for output_node_index in range(len(output_weights)):
        for hidden_node_index in range(len(output_weights[0])):
            output_weights[output_node_index][hidden_node_index] += learning_rate * output_error_terms[
                hidden_node_index] * hidden_node_values[output_node_index]
    return output_weights


def update_and_return_hidden_weights(hidden_weights, hidden_error_terms, current_pattern, learning_rate):
    for hidden_node_index in range(len(hidden_weights)):
        for input_index in range(len(hidden_weights[0])):
            hidden_weights[hidden_node_index][input_index] += learning_rate * hidden_error_terms[input_index] * \
                                                              current_pattern[hidden_node_index]
    return hidden_weights


def update_and_return_output_node_thetas(output_node_thetas, output_error_terms, learning_rate):
    for i in range(len(output_node_thetas)):
        output_node_thetas[i] += learning_rate * output_error_terms[i]
    return output_node_thetas


def update_and_return_hidden_node_thetas(hidden_node_thetas, hidden_error_terms, learning_rate):
    for i in range(len(hidden_node_thetas)):
        hidden_node_thetas[i] += learning_rate * hidden_error_terms[i]
    return hidden_node_thetas


def train_and_present_test_set(training_inputs, training_outputs, testing_inputs, testing_outputs, hidden_nodes,
                               learning_rate, number_of_iterations):
    """ this function is the main one, it takes in training and testing sets, number of hidden nodes, learning rate and
        number of iterations, learns, tests on testing set and returns the learning curve
    """
    number_of_inputs = len(training_inputs[0])
    number_of_hidden_nodes = hidden_nodes
    number_of_output_nodes = len(training_outputs[0])

    hidden_weights = np.random.random((number_of_inputs, number_of_hidden_nodes))
    output_weights = np.random.random((number_of_hidden_nodes, number_of_output_nodes))
    hidden_node_thetas = np.random.random(number_of_hidden_nodes)
    output_node_thetas = np.random.random(number_of_output_nodes)

    error_curve = []
    counter = 0
    error_counter = 0

    for i in range(number_of_iterations):

        counter += 1
        error_counter += 1

        if error_counter > 100:
            # every 100 iterations we calculate the average error of the epoch and append it to the list to return
            current_error = 0
            for current_pattern, desired_output in zip(training_inputs, training_outputs):
                hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights,
                                                                              hidden_node_thetas)
                output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights,
                                                                              output_node_thetas)
                current_error += calculate_current_error(desired_output, output_node_values)
            error_curve.append(current_error/len(training_outputs))
            error_counter = 0

        if counter > 10000:
            print("10000 iterations done\n")
            for current_pattern, desired_output in zip(training_inputs, training_outputs):
                print(desired_output)
                hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights,
                                                                              hidden_node_thetas)
                output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights,
                                                                              output_node_thetas)
                print(output_node_values)
            print("iteration over\n")
            counter = 0

        index = randint(0, len(training_inputs) - 1)
        current_pattern = training_inputs[index]
        desired_output = training_outputs[index]
        hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights,
                                                                      hidden_node_thetas)
        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights,
                                                                      output_node_thetas)

        output_error_terms = calculate_output_error_terms(desired_output, output_node_values, output_sum)
        hidden_error_terms = calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum)

        output_weights = update_and_return_output_weights(output_weights, output_error_terms, hidden_node_values,
                                                          learning_rate)
        hidden_weights = update_and_return_hidden_weights(hidden_weights, hidden_error_terms, current_pattern,
                                                          learning_rate)

        output_node_thetas = update_and_return_output_node_thetas(output_node_thetas, output_error_terms,
                                                                  learning_rate)
        hidden_node_thetas = update_and_return_hidden_node_thetas(hidden_node_thetas, hidden_error_terms,
                                                                  learning_rate)

    print("Final iteration over training set\n")
    for current_pattern, desired_output in zip(training_inputs, training_outputs):
        hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights,
                                                                      hidden_node_thetas)

        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights,
                                                                      output_node_thetas)
        print(desired_output)
        print(output_node_values)
    print("Final iteration done\n")
    print("Presenting testing set\n")
    for testing_example, desired_output in zip(testing_inputs, testing_outputs):
        hidden_node_values, hidden_sum = calculate_hidden_node_values(testing_example, hidden_weights,
                                                                      hidden_node_thetas)

        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights,
                                                                      output_node_thetas)
        print(desired_output)
        print(output_node_values)
        print()
    print("Test set presentation done\n")
    return error_curve
