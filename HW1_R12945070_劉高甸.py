import numpy as np
import math
import matplotlib.pyplot as plt

size = 17
frequency_sampling = 6000
passband = [i / frequency_sampling for i in range(1, 1200)]
transition_band = [i / frequency_sampling for i in range(1200, 1500)]
passband_2 = [i / frequency_sampling for i in range(1501, 3000)]
delta = 0.0001

half_size = int((size - 1) / 2)
F1 = np.linspace(0.0, 1200 / frequency_sampling, int((half_size + 2) / 2))
F2 = np.linspace(1500 / frequency_sampling, 3000 / frequency_sampling, int((half_size + 2) / 2))
F0 = np.array([0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

max_error_list = [10]

while True:
    matrix_A = np.zeros([half_size + 2, half_size + 2])
    desired_response = np.zeros(half_size + 2)
    for i in range(0, half_size + 2):
        for j in range(0, half_size + 1):
            matrix_A[i, j] = math.cos(2 * math.pi * j * F0[i])
        if F0[i] <= 1350 / frequency_sampling:
            matrix_A[i, half_size + 1] = 1 / 1 * float((-1) ** i)
            desired_response[i] = 1
        else:
            matrix_A[i, half_size + 1] = (1 / 0.6) * float((-1) ** i)
            desired_response[i] = 0

    inverse_A = np.linalg.inv(matrix_A)
    solution_s = inverse_A.dot(desired_response)

    frequency_range = np.linspace(0, 0.5, 1000)
    frequency_response = np.zeros(frequency_range.shape[0])
    desired_response = np.zeros(frequency_range.shape[0])
    weighting_function = np.zeros(frequency_range.shape[0])
    for i, frequency in enumerate(frequency_range):
        temp_sum = 0
        for j in range(0, half_size + 1):
            temp_sum += solution_s[j] * math.cos(2 * math.pi * j * frequency)
        frequency_response[i] = temp_sum
        if frequency <= 1350 / frequency_sampling:
            desired_response[i] = 1
        else:
            desired_response[i] = 0
        if frequency <= 1200 / frequency_sampling:
            weighting_function[i] = 1
        elif frequency >= 1500 / frequency_sampling:
            weighting_function[i] = 0.6
        else:
            weighting_function[i] = 0

    error = (frequency_response - desired_response) * weighting_function
    extremal_frequency = []
    extremal_response = []
    boundary_frequency = []
    boundary_response = []

    for i in range(0, frequency_range.shape[0]):
        if i == 0:
            if (error[i] > 0) and (error[i] > error[i + 1]):
                boundary_frequency.append(frequency_range[i])
                boundary_response.append(abs(error[i]))
            elif (error[i] < 0) and (error[i] <= error[i + 1]):
                boundary_frequency.append(frequency_range[i])
                boundary_response.append(abs(error[i]))
        elif i == (frequency_range.shape[0] - 1):
            if (error[i] > error[i - 1]) and (error[i] > 0):
                boundary_frequency.append(frequency_range[i])
                boundary_response.append(abs(error[i]))
            elif (error[i] < error[i - 1]) and (error[i] < 0):
                boundary_frequency.append(frequency_range[i])
                boundary_response.append(abs(error[i]))
        else:
            if (error[i] > error[i + 1]) and (error[i] > error[i - 1]):
                extremal_frequency.append(frequency_range[i])
                extremal_response.append(error[i])
            elif (error[i] < error[i + 1]) and (error[i] < error[i - 1]):
                extremal_frequency.append(frequency_range[i])
                extremal_response.append(error[i])

    while len(extremal_response) < (half_size + 2):
        max_response = max(boundary_response)
        index = boundary_response.index(max_response)
        boundary_response.remove(max_response)
        extremal_response.append(max_response)
        temp_frequency = boundary_frequency[index]
        boundary_frequency.remove(temp_frequency)
        extremal_frequency.append(temp_frequency)

    extremal_frequency, extremal_response = zip(*sorted(zip(extremal_frequency, extremal_response)))

    F0 = np.array(extremal_frequency)

    plt.plot(frequency_range, frequency_response, frequency_range, desired_response)
    plt.show()

    plt.plot(frequency_range, error, extremal_frequency, extremal_response, 'o')
    plt.show()

    max_error_list.append(max(abs(error)))
    if ((max_error_list[-2] - max_error_list[-1]) < delta) and ((max_error_list[-2] - max_error_list[-1]) > 0):
        break

error = (frequency_response - desired_response) * weighting_function
extremal_frequency = []
extremal_response = []
boundary_frequency = []
boundary_response = []

for i in range(0, frequency_range.shape[0]):
    if i == 0:
        if (error[i] > 0) and (error[i] > error[i + 1]):
            boundary_frequency.append(frequency_range[i])
            boundary_response.append(abs(frequency_response[i]))
        elif (error[i] < 0) and (error[i] <= error[i + 1]):
            boundary_frequency.append(frequency_range[i])
            boundary_response.append(abs(frequency_response[i]))
    elif i == (frequency_range.shape[0] - 1):
        if (error[i] > error[i - 1]) and (error[i] > 0):
            boundary_frequency.append(frequency_range[i])
            boundary_response.append(abs(frequency_response[i]))
        elif (error[i] < error[i - 1]) and (error[i] < 0):
            boundary_frequency.append(frequency_range[i])
            boundary_response.append(abs(frequency_response[i]))
    else:
        if (error[i] > error[i + 1]) and (error[i] > error[i - 1]):
            extremal_frequency.append(frequency_range[i])
            extremal_response.append(frequency_response[i])
        elif (error[i] < error[i + 1]) and (error[i] < error[i - 1]):
            extremal_frequency.append(frequency_range[i])
            extremal_response.append(frequency_response[i])

while (len(extremal_response) < (half_size + 2)):
    max_response = max(boundary_response)
    index = boundary_response.index(max_response)
    boundary_response.remove(max_response)
    extremal_response.append(max_response)
    temp_frequency = boundary_frequency[index]
    boundary_frequency.remove(temp_frequency)
    extremal_frequency.append(temp_frequency)

(extremal_frequency, extremal_response) = (list(t) for t in zip(*sorted(zip(extremal_frequency, extremal_response))))

F0 = np.array(extremal_frequency)
plt.plot(frequency_range, frequency_response, frequency_range, desired_response, extremal_frequency, extremal_response, 'o')
plt.xlabel('Frequency')
plt.legend(['H(F)', 'H_d(F)'])
plt.title('Frequency Response')
plt.show()

plt.plot(frequency_range, error)
plt.show()

print('    Iteration   | Max Error')
for i in range(1, len(max_error_list)):
    print(f'\t{i}\t| {max_error_list[i]:.4f}')

impulse_response = np.zeros([size])
y = np.zeros([size])
for i in range(0, solution_s.shape[0] - 1):
    impulse_response[half_size - i] = solution_s[i]
    impulse_response[half_size + i] = solution_s[i]

plt.plot(impulse_response, color='lightgrey')
n = np.linspace(0, 16, 17)
plt.plot(n, y, color='black')
plt.bar(n, impulse_response, width=0.2, color='lightgrey')
plt.scatter(n, impulse_response, color='lightgrey')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Impulse Response h[n]')
plt.show()