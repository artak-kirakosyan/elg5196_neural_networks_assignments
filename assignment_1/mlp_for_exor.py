#!/usr/bin/env python3.6

import numpy as np


def sigmoid_value(x):
  return 1/(1 + np.exp(-1 * x))

def derivative_of_sigmoid(x):
  return sigmoid_value(x) * (1 - sigmoid_value(x))

def calculate_hidden_node_values(input_pattern, hidden_weights, hidden_node_thetas):
  hidden_sum = np.dot(hidden_weights.T,input_pattern) + hidden_node_thetas
  return np.array([sigmoid_value(i) for i in hidden_sum]), hidden_sum

def calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas):
  output_sum = np.dot(output_weights.T, hidden_node_values) + output_node_thetas
  return np.array([sigmoid_value(i) for i in output_sum]), output_sum

def calculate_output_error_terms(desired_output, output_node_values, output_sum):
  return np.array([(derivative_of_sigmoid(output_sum[i]) * (desired_output[i] - output_node_values[i])) for i in range(len(desired_output))])

def calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum):
  return np.array([(derivative_of_sigmoid(hidden_sum[i]) * np.dot(output_error_terms, output_weights[i])) for i in range(len(hidden_sum))])

def update_and_return_output_weights(output_weights, output_error_terms, hidden_node_values, small_number):
  for output_node_index in range(len(output_weights)):
    for hidden_node_index in range(len(output_weights[0])):
      output_weights[output_node_index][hidden_node_index] += small_number * output_error_terms[hidden_node_index] * hidden_node_values[output_node_index]
  return output_weights

def update_and_return_hidden_weights(hidden_weights, hidden_error_terms, current_pattern, small_number):
  for hidden_node_index in range(len(hidden_weights)):
    for input_index in range(len(hidden_weights[0])):
      hidden_weights[hidden_node_index][input_index] += small_number * hidden_error_terms[input_index] * current_pattern[hidden_node_index]
  return hidden_weights

def update_and_return_output_node_thetas(output_node_thetas, output_error_terms, small_number):
  for i in range(len(output_node_thetas)):
    output_node_thetas[i] += small_number * output_error_terms[i]
  return output_node_thetas
def update_and_return_hidden_node_thetas(hidden_node_thetas, hidden_error_terms, small_number):
  for i in range(len(hidden_node_thetas)):
    hidden_node_thetas[i] += small_number* hidden_error_terms[i]
  return hidden_node_thetas

def main():
  training_inputs = np.array([[0,0], [0,1],[1,0],[1,1]])
  training_outputs = np.array([[0],[1],[1],[0]])

  number_of_inputs = 2
  number_of_hidden_nodes = 2
  number_of_output_nodes = 1
  small_number = 0.5
  number_of_iterations = 8000

  hidden_weights = np.random.random((number_of_inputs, number_of_hidden_nodes))
  output_weights = np.random.random((number_of_hidden_nodes, number_of_output_nodes))
  hidden_node_thetas = np.random.random(number_of_hidden_nodes)
  output_node_thetas = np.random.random(number_of_output_nodes)

  for i in range(int(number_of_iterations)):
    for current_pattern, desired_output in zip(training_inputs, training_outputs):
        hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights, hidden_node_thetas)
        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
        output_error_terms = calculate_output_error_terms(desired_output, output_node_values, output_sum)
        hidden_error_terms = calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum)
        output_weights = update_and_return_output_weights(output_weights, output_error_terms, hidden_node_values, small_number)
        hidden_weights = update_and_return_hidden_weights(hidden_weights, hidden_error_terms, current_pattern, small_number)

        output_node_thetas = update_and_return_output_node_thetas(output_node_thetas, output_error_terms, small_number)
        hidden_node_thetas = update_and_return_hidden_node_thetas(hidden_node_thetas, hidden_error_terms, small_number)

  print("hidden_weights")
  print(hidden_weights)
  print("output_weights")
  print(output_weights)
  print("hidden_node_thetas")
  print(hidden_node_thetas)
  print("output_node_thetas")
  print(output_node_thetas)
  print("FINAL ITERATION")
  for current_pattern, desired_output in zip(training_inputs, training_outputs):
    hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights, hidden_node_thetas)
    output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
    print(current_pattern)
    print(output_node_values) 
  print("FINAL ITERATION")

if __name__ == "__main__":
  main()
