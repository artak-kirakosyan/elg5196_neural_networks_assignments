#!/usr/bin/env python3.6

from matplotlib import pyplot as plt

from convolution_helpers import *
from mlp_helpers import *
from samples import *

np.set_printoptions(precision=1)


def main():

    # those 2 variables control the execution of one and two stage convolutions respectively
    one_step_convolution_needed = 1
    two_step_convolution_needed = 1

    # defining filters and creating an array of all filter to pass them to the convolution functions
    horizontal_line = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
    vertical_line = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
    degree_45_line = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
    degree_135_line = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

    kernels = [horizontal_line, vertical_line, degree_45_line, degree_135_line]

    # those arrays are defined in samples.py file, here we create a full training and testing set
    training_inputs = np.array([one, two, three, four, five])
    training_outputs = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    testing_inputs = np.array(
        [tilted_one, tilted_one_2, tilted_two, tilted_two_2, tilted_three, tilted_three_2, tilted_four, tilted_four_2,
         tilted_five, tilted_five_2])
    testing_outputs = np.array(
        [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]])

    # defining stride and pool settings for one stage convolution
    stride = (2, 2)
    pool = (2, 2)
    if one_step_convolution_needed:
        # creating one step convolved training and testing samples

        one_step_convolved_inputs = []

        for i in training_inputs:
            one_step_convolved_inputs.append(one_step_convolution_manager(i.reshape((16, 16)), kernels, stride, pool))
        one_step_convolved_inputs = np.array(one_step_convolved_inputs)

        one_step_convolved_testing_set = []

        for i in testing_inputs:
            one_step_convolved_testing_set.append(
                one_step_convolution_manager(i.reshape((16, 16)), kernels, stride, pool))
        one_step_convolved_testing_set = np.array(one_step_convolved_testing_set)
        print(one_step_convolved_inputs[0].shape)
        print("Training on one step convolved sets\n")
        learning_curve_one_step = train_and_present_test_set(one_step_convolved_inputs, training_outputs,
                                                             one_step_convolved_testing_set, testing_outputs, 8, 0.5,
                                                             120000)

        print("Training on one step convolved sets finished \n")
        fig, ax = plt.subplots()
        ax.plot(learning_curve_one_step)
        plt.savefig("one_step_convolution.png")

    # defining stride and pool settings for the convolution
    stride = (3, 3)
    pool = (1, 1)

    # training MLP on two step convolved sets
    if two_step_convolution_needed:
        # creating two step convolved_training and testing samples
        two_step_convolved_inputs = []
        for i in training_inputs:
            two_step_convolved_inputs.append(two_step_convolution_manager(i.reshape((16, 16)), kernels, stride, pool))
        two_step_convolved_inputs = np.array(two_step_convolved_inputs)
        two_step_convolved_testing_set = []
        for i in testing_inputs:
            two_step_convolved_testing_set.append(
                two_step_convolution_manager(i.reshape((16, 16)), kernels, stride, pool))
        two_step_convolved_testing_set = np.array(two_step_convolved_testing_set)

        print("Training on two step convolved sets\n")
        print(two_step_convolved_inputs[0].shape)
        learning_curve_two_step = train_and_present_test_set(two_step_convolved_inputs, training_outputs,
                                                             two_step_convolved_testing_set, testing_outputs, 3, 0.3,
                                                             50000)

        print("Training on two step convolved sets done\n")

        fig, ax = plt.subplots()
        ax.plot(learning_curve_two_step)
        plt.savefig("two_step_convolution.png")


if "__main__" == __name__:
    main()
