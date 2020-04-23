import numpy as np


def return_max_element(input_matrix):
    """

    :param input_matrix: any matrix
    :return: biggest value in the input matrix
    """
    return max(input_matrix.flat)


def apply_relu(input_matrix):
    """
    This function takes in a matrix, checks if it is not a 2D matrix, then return it, if not, applies ReLU on it and
    returns
    """
    if len(input_matrix.shape) != 2:
        print("Not a 2d array")
        return input_matrix

    for row_index in range(input_matrix.shape[0]):
        for col_index in range(input_matrix.shape[1]):
            input_matrix[row_index][col_index] = max(0, input_matrix[row_index][col_index])
    return input_matrix


def detect_max_outcome_of_a_kernel(kernel):
    """

    :param kernel: kernel matrix
    :return: biggest value which can be produced by the kernel
    """
    return sum(apply_relu(kernel).flat)


def apply_filter_on_sub_matrix(kernel, submatrix):
    """
    This function takes in a kernel and a submatrix of the same size and returns the convolution sum of those 2 matrices
  """
    if kernel.shape != submatrix.shape:
        return 0
    else:
        return (kernel * submatrix).sum()


def pad_with_zeros_from_right(input_matrix, horizontal, vertical):
    """
    This function takes a matrix and 2 numbers and pads the matrix with required amount of zeros
  """
    i = 0
    while i < horizontal:
        input_matrix = np.vstack((input_matrix, np.zeros(input_matrix.shape[1])))
        i += 1
    i = 0
    while i < vertical:
        input_matrix = np.vstack((input_matrix.transpose(), np.zeros(input_matrix.shape[0]))).transpose()
        i += 1
    return input_matrix


def max_pooling(input_matrix, horizontal_pool_size=2, vertical_pool_size=2):
    """

    :param input_matrix: input matrix on which pooling to be done
    :param horizontal_pool_size: size of the pool(horizontal)
    :param vertical_pool_size: size of the pool(vertical)
    :return: max pooled matrix
    """
    # as in the convolution, here also, keep 2 set of indexes, one for input, one for output
    output_current_position = [0, 0]
    input_current_position = [0, 0]
    input_dimensions = input_matrix.shape

    # this part determines if padding with zeros is needed or not
    horizontal_padding = input_dimensions[0] % horizontal_pool_size
    vertical_padding = input_dimensions[1] % vertical_pool_size

    # if padding is needed, determining the size of the padding
    if 0 == horizontal_padding:
        output_horizontal_size = input_dimensions[0] // horizontal_pool_size
    else:
        horizontal_padding = horizontal_pool_size - horizontal_padding
        output_horizontal_size = input_dimensions[0] // horizontal_pool_size + 1

    if 0 == vertical_padding:
        output_vertical_size = input_dimensions[1] // vertical_pool_size
    else:
        vertical_padding = vertical_pool_size - vertical_padding
        output_vertical_size = input_dimensions[1] // vertical_pool_size + 1

    result = np.zeros((output_horizontal_size, output_vertical_size))
    # applying the padding
    input_matrix = pad_with_zeros_from_right(input_matrix, horizontal_padding, vertical_padding)
    input_dimensions = input_matrix.shape
    # main part of the pooling. iterate over input and create the output
    while input_current_position[0] <= input_dimensions[0] - horizontal_pool_size:

        while input_current_position[1] <= input_dimensions[1] - vertical_pool_size:
            row_start_index = input_current_position[0]
            row_end_index = input_current_position[0] + horizontal_pool_size
            column_start_index = input_current_position[1]
            column_end_index = input_current_position[1] + vertical_pool_size

            submatrix = np.array(
                [i[column_start_index:column_end_index] for i in input_matrix[row_start_index:row_end_index]])

            result[output_current_position[0]][output_current_position[1]] = return_max_element(submatrix)
            input_current_position[1] += vertical_pool_size
            output_current_position[1] += 1

        input_current_position[1] = 0
        output_current_position[1] = 0
        input_current_position[0] += horizontal_pool_size
        output_current_position[0] += 1
    return result


def convolution(input_matrix, kernel, horizontal_stride=1, vertical_stride=1):
    """
    Function takes in the input data, a filter horizontal and vertical strides 
    and returns the result of the 2D convolution.
  """
    # we keep 2 sets of positions: one to run over the input, another one for filling in the output
    current_position_in_input = [0, 0]
    current_position_in_output = [0, 0]

    kernel_dimensions = kernel.shape
    input_dimensions = input_matrix.shape

    # this lines check if padding with zeros is necessary from the right side
    horizontal_padding = (input_dimensions[0] - kernel_dimensions[0]) % horizontal_stride
    vertical_padding = (input_dimensions[1] - kernel_dimensions[1]) % vertical_stride

    # if padding is required, the number of padding arrays is calculated alongside with the dimensions of the output
    # matrix
    if 0 == horizontal_padding:
        output_horizontal_size = (input_dimensions[0] - kernel_dimensions[0]) // horizontal_stride + 1
    else:
        horizontal_padding = horizontal_stride - horizontal_padding
        output_horizontal_size = (input_dimensions[0] - kernel_dimensions[0]) // horizontal_stride + 2

    if 0 == vertical_padding:
        output_vertical_size = (input_dimensions[1] - kernel_dimensions[1]) // vertical_stride + 1
    else:
        output_vertical_size = (input_dimensions[1] - kernel_dimensions[1]) // vertical_stride + 2
        vertical_padding = vertical_stride - vertical_padding
    # here we pad the input with zeros and update input_dimensions
    input_matrix = pad_with_zeros_from_right(input_matrix, horizontal_padding, vertical_padding)
    input_dimensions = input_matrix.shape

    # we create an array filled with zeros with necessary size
    result = np.zeros((output_horizontal_size, output_vertical_size))

    # if the kernel size is bigger than that of input --> no point of doing the convolution
    if (kernel_dimensions[0] > input_dimensions[0]) and (kernel_dimensions[1] > input_dimensions[1]):
        return input_matrix

    # this loop iterates over the rows of the input matrix until we reach last possible row for convolution
    while current_position_in_input[0] <= input_dimensions[0] - kernel_dimensions[0]:

        # this loop iterates over the columns of the input matrix until we reach last possible column for convolution
        while current_position_in_input[1] <= input_dimensions[1] - kernel_dimensions[1]:
            # those 4 rows calculate the starting and ending point to take the current sub-matrix of interest to do
            # the convolution summation
            row_start_index = current_position_in_input[0]
            row_end_index = current_position_in_input[0] + kernel_dimensions[0]
            column_start_index = current_position_in_input[1]
            column_end_index = current_position_in_input[1] + kernel_dimensions[1]

            # this row takes the sub-matrix of interest from the input matrix
            submatrix = np.array(
                [i[column_start_index:column_end_index] for i in input_matrix[row_start_index:row_end_index]])
            # this line gives the submatrix and the kernel to the function, which calculates and returns the
            # convolution sum for current case
            result[current_position_in_output[0]][current_position_in_output[1]] = apply_filter_on_sub_matrix(kernel,
                                                                                                              submatrix)

            # here we increase the input vertical position by vertical_stride and output vertical position by 1 to be
            # consistent
            current_position_in_input[1] += vertical_stride
            current_position_in_output[1] += 1

        # this 4 lines execute, when one line of the convolution is done, and we need to proceed with the next line
        # we set vertical position of both input and output matrices to start from the start of the next line
        current_position_in_input[1] = 0
        current_position_in_output[1] = 0
        # and also do similar increase in horizontal input and output positions by horizontal_stride and 1
        # correspondingly
        current_position_in_input[0] += horizontal_stride
        current_position_in_output[0] += 1
    # after everything is done, we return the result
    return result


def perform_full_step(input_matrix, kernel, stride, pool):
    """
    Perform 1 step of convolution, relu, max pooling and normalizing.
    :param input_matrix: to apply steps
    :param kernel: kernel to use
    :param stride: stride dimensions
    :param pool: pool dimensions
    :return: result of all steps
    """
    convolved_matrix = convolution(input_matrix, kernel, stride[0], stride[1])
    after_relu = apply_relu(convolved_matrix)
    after_max_pool = max_pooling(after_relu, pool[0], pool[1])
    kernel_max = detect_max_outcome_of_a_kernel(kernel)
    if 0 == kernel_max:
        print("Kernel has a greatest value of 0")
    else:
        for (row, col), value in np.ndenumerate(after_max_pool):
            after_max_pool[row][col] /= kernel_max
    return after_max_pool


def one_step_convolution_manager(input_matrix, kernels, stride, pool):
    """
    apply 1 level of convolution on input matrix
    :param input_matrix: input
    :param kernels: array of kernels
    :param stride: stride dimensions
    :param pool: pool size
    :return: result
    """
    result = []
    for kernel in kernels:
        current_result = perform_full_step(input_matrix, kernel, stride, pool)
        result.extend(current_result.reshape(current_result.shape[0] * current_result.shape[1]))
    return np.array(result)


def two_step_convolution_manager(input_matrix, kernels, stride, pool):
    """
    apply 2 level convolution on input matrix
    :param input_matrix: input
    :param kernels: array of kernels
    :param stride: stride dimensions
    :param pool: pool dimensions
    :return: result
    """
    result = []
    for kernel in kernels:
        first_step_result = perform_full_step(input_matrix, kernel, stride, pool)
        second_step_result = perform_full_step(first_step_result, kernel, stride, pool)
        result.extend(second_step_result.reshape(second_step_result.shape[0] * second_step_result.shape[1]))
    return result
