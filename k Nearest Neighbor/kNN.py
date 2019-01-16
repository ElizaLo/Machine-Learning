import numpy as np
import matplotlib.pyplot as plt


train_data = np.genfromtxt("mnist_train.csv", delimiter = ',')
test_data = np.genfromtxt("mnist_test.csv", delimiter = ',')

print(train_data.shape)
print(test_data.shape, "\n")

#Number, 784 = 28*28 pixel values

#train data - 60'000 vectors
#test data - 1'000 vectors
#vector has 785 values - first one is the tag (a digit from 0 to 9) and
#next 784 numbers represent the image of this digit (actually it is reshaped 28x28 matrix of pixels,
#modified for easier processing)

N_train = 1000
N_test = 100
IMAGE_LENGTH = test_data.shape[1] - 1

def get_rand_sub(data, new_size):
    old_size = data.shape[0]
    indexes = np.random.choice(old_size, new_size, replace = False)
    labels = data[indexes, 0].astype(int)
    images = data[indexes, 1:]
    return labels, images

train_labels, train_images = get_rand_sub(train_data, N_train)
test_labels, test_images = get_rand_sub(test_data, N_test)

#train 1000x1x784
#test 1x100x784
#train - test 1000x100x784
#square, sum(axis=2)
#a[:,j] 1000 расстояние от тестового j ко всем тренировочным
#argmin(axis=0) 100

def classify_images(test_images, train_images, train_labels):
    DM = np.square(test_images - train_images).sum(axis = 1)
    index = DM.argmin(axis = 0)
    return train_labels[index]

test_predicted = [classify_images(test, train_images, train_labels)
                     for test in test_images]
accuracy = (test_predicted == test_labels).sum()/N_test #Точность

print("N_train:  ", N_train)
print("N_test:   ", N_test)
print("Accuracy: ", accuracy * 100, '%', "\n")

#how accuracy depends on size of train subset


def get_average_accuracy(test_images, train_data, train_subset_size, iterations_count):
    total_accuracy = 0

    for i in range(iterations_count):
        train_labels, train_images = get_rand_sub(train_data, train_subset_size)
        test_predicted = [classify_images(test, train_images, train_labels)
                     for test in test_images]
        accuracy = (test_predicted == test_labels).sum()/N_test #Точность
        total_accuracy += accuracy / test_labels.shape[0]

    return total_accuracy / iterations_count

size_of_subset_v = [50,  200, 500, 1000, 2000, 5000, 10000]
count_of_iters_v = [100, 40,  20,  15,   10,   3,    1]

accuracy_info = [get_average_accuracy(test_images, train_data, size, iters)
                 for size, iters in zip(size_of_subset_v, count_of_iters_v)]


plt.semilogx(size_of_subset_v, accuracy_info, color = "g")
plt.scatter(size_of_subset_v, accuracy_info, color = "b")

#x-axis is logarithmically scaled

plt.xlabel("Size of train subset")
plt.ylabel("Accuracy")
plt.grid(axis="y", which="both", linestyle='--')

plt.show()
