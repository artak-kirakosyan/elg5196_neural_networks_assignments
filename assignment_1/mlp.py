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
  output_error_terms = np.array([(derivative_of_sigmoid(output_sum[i]) * (desired_output[i] - output_node_values[i])) for i in range(len(desired_output))])
  return output_error_terms

def calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum):
  hidden_error_terms = np.array([(derivative_of_sigmoid(hidden_sum[i]) * np.dot(output_error_terms, output_weights[i])) for i in range(len(hidden_sum))])
  return hidden_error_terms

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
  one = [0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0,\
         0,0,0,0,1,0,0,0]

  two = [1,1,1,1,1,1,1,1,\
         0,0,0,0,0,0,0,1,\
         0,0,0,0,0,0,0,1,\
         1,1,1,1,1,1,1,1,\
         1,0,0,0,0,0,0,0,\
         1,0,0,0,0,0,0,0,\
         1,0,0,0,0,0,0,0,\
         1,1,1,1,1,1,1,1]

  three =  [1,1,1,1,1,1,1,1,\
            0,0,0,0,0,0,0,1,\
            0,0,0,0,0,0,0,1,\
            1,1,1,1,1,1,1,1,\
            0,0,0,0,0,0,0,1,\
            0,0,0,0,0,0,0,1,\
            0,0,0,0,0,0,0,1,\
            1,1,1,1,1,1,1,1]     

  four = [1,0,0,0,0,0,0,1,\
          1,0,0,0,0,0,0,1,\
          1,0,0,0,0,0,0,1,\
          1,0,0,0,0,0,0,1,\
          1,1,1,1,1,1,1,1,\
          0,0,0,0,0,0,0,1,\
          0,0,0,0,0,0,0,1,\
          0,0,0,0,0,0,0,1]

  five = [1,1,1,1,1,1,1,1,\
          1,0,0,0,0,0,0,0,\
          1,0,0,0,0,0,0,0,\
          1,1,1,1,1,1,1,1,\
          0,0,0,0,0,0,0,1,\
          0,0,0,0,0,0,0,1,\
          0,0,0,0,0,0,0,1,\
          1,1,1,1,1,1,1,1]

  tilted_one = [0,0,0,1,0,0,0,0,\
                0,0,0,1,0,0,0,0,\
                0,0,0,1,0,0,0,0,\
                0,0,0,1,0,0,0,0,\
                0,0,0,0,1,0,0,0,\
                0,0,0,0,1,0,0,0,\
                0,0,0,0,1,0,0,0,\
                0,0,0,0,1,0,0,0]

  tilted_two = [0,1,1,1,1,1,1,0,\
                1,0,0,0,0,0,0,1,\
                0,0,0,0,0,0,0,1,\
                0,0,0,0,1,1,1,0,\
                0,1,1,1,0,0,0,0,\
                1,0,0,0,0,0,0,0,\
                1,0,0,0,0,0,0,0,\
                1,1,1,1,1,1,1,1]

  tilted_three = [0,1,1,1,1,1,1,0,\
                  1,0,0,0,0,0,0,1,\
                  0,0,0,0,0,0,0,1,\
                  0,0,0,0,1,1,1,1,\
                  0,0,0,0,0,0,0,1,\
                  0,0,0,0,0,0,0,1,\
                  1,0,0,0,0,0,0,1,\
                  0,1,1,1,1,1,1,0]

  tilted_four = [0,0,0,0,0,1,1,0,\
                 0,0,0,0,1,0,1,0,\
                 0,0,1,1,0,0,1,0,\
                 0,1,0,0,0,0,1,0,\
                 1,1,1,1,1,1,1,1,\
                 0,0,0,0,0,0,1,0,\
                 0,0,0,0,0,0,1,0,\
                 0,0,0,0,0,0,1,0]

  tilted_five = [1,1,1,1,1,1,1,1,\
                 1,0,0,0,0,0,0,0,\
                 1,0,0,0,0,0,0,0,\
                 1,1,1,1,0,0,0,0,\
                 0,0,0,0,1,1,1,1,\
                 0,0,0,0,0,0,0,1,\
                 1,0,0,0,0,0,0,1,\
                 0,1,1,1,1,1,1,0] 

  tilted_samles = [tilted_one, tilted_two, tilted_three, tilted_four, tilted_five]

  training_inputs = np.array([one, two, three, four, five])
  training_outputs = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
  number_of_inputs = 64
  number_of_hidden_nodes = 20
  number_of_output_nodes = 5
  small_number = 0.5
  number_of_iterations = 40000

  hidden_weights = np.random.random((number_of_inputs, number_of_hidden_nodes))
  output_weights = np.random.random((number_of_hidden_nodes, number_of_output_nodes))
  hidden_node_thetas = np.random.random(number_of_hidden_nodes)
  output_node_thetas = np.random.random(number_of_output_nodes)
  
  counter = 0

  for i in range(int(number_of_iterations)):
    counter += 1
    if counter > 1000:
      print("iteration start")
      for current_pattern, desired_output in zip(training_inputs, training_outputs):
        print(desired_output)
        hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights, hidden_node_thetas)
        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
        print(output_node_values)
      print("iteration over")

      counter = 0

    for current_pattern, desired_output in zip(training_inputs, training_outputs):
        hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights, hidden_node_thetas)
        output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)

        output_error_terms = calculate_output_error_terms(desired_output, output_node_values, output_sum)
        hidden_error_terms = calculate_hidden_error_terms(output_error_terms, output_weights, hidden_sum)

        output_weights = update_and_return_output_weights(output_weights, output_error_terms, hidden_node_values, small_number)
        hidden_weights = update_and_return_hidden_weights(hidden_weights, hidden_error_terms, current_pattern, small_number)

        output_node_thetas = update_and_return_output_node_thetas(output_node_thetas, output_error_terms, small_number)
        hidden_node_thetas = update_and_return_hidden_node_thetas(hidden_node_thetas, hidden_error_terms, small_number)

  print("FINAL ITERATION")
  for current_pattern, desired_output in zip(training_inputs, training_outputs):
    hidden_node_values, hidden_sum = calculate_hidden_node_values(current_pattern, hidden_weights, hidden_node_thetas)
    output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
    print(output_node_values) 
  print("FINAL ITERATION")

  hidden_node_values, hidden_sum = calculate_hidden_node_values(tilted_one, hidden_weights, hidden_node_thetas)
  output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
  print("Presenting tilted examples")
  for tilted_sample in tilted_samles:
    hidden_node_values, hidden_sum = calculate_hidden_node_values(tilted_sample, hidden_weights, hidden_node_thetas)
    output_node_values, output_sum = calculate_output_node_values(hidden_node_values, output_weights, output_node_thetas)
    print(output_node_values)

if __name__ == "__main__":
  main()
