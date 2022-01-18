import math
import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
from scipy import stats

training_impath = 'train-images.idx3-ubyte'
training_labpath = 'train-labels.idx1-ubyte'
test_impath = 't10k-images.idx3-ubyte'
test_labpath = 't10k-labels.idx1-ubyte'


def read_images_labels(images_filepath, labels_filepath):
    labels_file = open(labels_filepath, 'rb')
    struct.unpack(">II", labels_file.read(8))
    labels = array("B", labels_file.read())
    images_file = open(images_filepath, 'rb')
    magic_number, size, rows, cols = struct.unpack(">IIII", images_file.read(16))
    image_data = array("B", images_file.read())
    images = []
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        images.append(img.reshape(28, 28))
    return images, labels


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(25, 25))
    index = 1
    for x in zip(images, title_texts):
        image = x[0].reshape(28, 28)
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15);
        index += 1


def show_single_images(image, title):
    x = int(math.sqrt(image.shape[0]))
    plt.imshow(image.reshape(x, x), cmap=plt.cm.gray)
    if title != '':
        plt.title(title, fontsize=50)
    plt.show()


def show_image_by_index(set, index):
    if set == 'train':
        plt.imshow(x_train[index], cmap=plt.cm.gray)
        plt.title('train image [' + str(index) + '] = ' + str(y_train[index]), fontsize=30);
    else:
        plt.imshow(x_test[index], cmap=plt.cm.gray)
        plt.title('test image [' + str(index) + '] = ' + str(y_test[index]), fontsize=30);
    plt.show()


def eigen_calc(images_set):
    mean = np.mean(images_set, axis=0)
    centralized_images = images_set - mean
    # mean subtracted
    sample_size = images_set.__len__()
    centralized_images = centralized_images.reshape(sample_size, -1)
    cov_matrix_small = np.dot(centralized_images, centralized_images.transpose())
    # k * k covariance matrix'
    eigen_val, eigen_small = np.linalg.eig(cov_matrix_small)
    eigen_vec = np.dot(centralized_images.transpose(), eigen_small)

    return eigen_val, eigen_vec, centralized_images


def sort_eigen(eigvls, eigvcs):
    indices = np.argsort(eigvls)
    descending_eigval = np.zeros(eigvls.size)
    descending_eigvec = np.zeros((eigvcs.shape[0], eigvls.size))
    for i in range(eigvls.size):
        descending_eigval[eigvls.size - 1 - i] = eigvls[indices[i]]
        descending_eigvec[:, eigvls.size - 1 - i] = eigvcs[:, indices[i]]
    return descending_eigval, descending_eigvec
    # eigval 784, normalized eigvec k*784


def normalize(vectors):
    vectors = vectors / np.linalg.norm(vectors, axis=-1)[:, np.newaxis]
    # print(np.sum(structure ** 2, axis=-1))
    return vectors


def cut_top_eigenvc(eigvcs, rank):
    top_eigvec = np.zeros((rank, eigvcs.shape[1]))
    for i in range(rank):
        top_eigvec[i, :] = eigvcs[i, :]
    return top_eigvec


# Knn in with euclidean distance and output a matrix with the index of training samples with the descending order
# of nearness.
def k_nearest_neighbours(train_projected, test_projected, k):
    nearest_neighbours_matrix = np.zeros((test_projected.shape[1], k))
    for i in range(0, test_projected.shape[1] - 1):
        dists = np.zeros(train_projected.shape[1])
        test_img = test_projected[:, i]
        for j in range(train_projected.shape[1]):
            train_img = train_projected[:, j]
            diff_img = train_img - test_img
            dists[j] = np.linalg.norm(diff_img)
        index = np.argsort(dists)
        # pick only the nearest k indices
        nearest_neighbours_matrix[i, :] = index[0: k]

    return nearest_neighbours_matrix


def test_assign_labels(knn_matrix, train_set_labels):
    test_set_size = knn_matrix.shape[0]
    test_set_labels = np.zeros(test_set_size)
    for i in range(0, test_set_size - 1):
        k_labels = []
        for j in range(knn_matrix.shape[1]):
            label = train_set_labels[np.int_(knn_matrix[i, j])]
            k_labels.append(label)
        # list out the picked training samples which are the nearest and pick the mode from the list and label test
        # input using that result
        test_set_labels[i] = stats.mode(k_labels)[0][0]
    return test_set_labels


def model_stats(test_set_labels, test_set_labels_answer):
    test_set_size = test_set_labels.shape[0]
    count = 0
    for i in range(0, test_set_size - 1):
        # print("result " + str(test_set_labels[i]) + "answer " + str(actual_test_set_labels[i]))
        if test_set_labels[i] == test_set_labels_answer[i]:
            count += 1
            # return the accuracy in float.
    return float(count * 100) / float(test_set_size)


def pca_calculate(image_set, number_of_components):
    eigenvalue, eigenvector, centralized_image = eigen_calc(image_set)
    sorted_eigenvalue, sorted_eigenvector = sort_eigen(eigenvalue, eigenvector)
    normalized_sorted_eigenvector = normalize(sorted_eigenvector.transpose())
    number_of_components = min(number_of_components, normalized_sorted_eigenvector.shape[0])

    top_eigenvector = cut_top_eigenvc(normalized_sorted_eigenvector, number_of_components)
    projected_image = np.dot(top_eigenvector, centralized_image.transpose())
    # result: component * k, one column shows how much each component is included in this sample
    return top_eigenvector, projected_image, centralized_image


x_train, y_train = read_images_labels(training_impath, training_labpath)
x_test, y_test = read_images_labels(test_impath, test_labpath)

plotdata_x_0 = []
plotdata_y_all = np.zeros((4, 10))


def classifier(train_sample_number, test_sample_number, reduced_dimension_number, k):
    image_train = []
    label_train = []
    image_test = []
    label_test = []

    for i in range(train_sample_number):
        index = np.random.randint(0, 50000)
        image_train.append(x_train[index])
        label_train.append(y_train[index])

    for i in range(test_sample_number):
        index = 5000 + i
        image_test.append(x_test[index])
        label_test.append(y_test[index])
        # to ensure every time the test samples are the same

    """
    training_samples_2_show = []
    training_samples_titles_2_show = []
    listed_set_index = []
    
    for i in range(20):
        index = np.random.randint(0, 300)
        listed_set_index.append(index)
        training_samples_2_show.append(x_train[index])
        training_samples_titles_2_show.append('train image [' + str(index) + '] = ' + str(y_train[index]))
    show_images(training_samples_2_show, training_samples_titles_2_show)
    plt.savefig(str(dimension) + "samples_train_set.png", dpi=300, format='png')
    """

    principle_components, pca_result, train_centralized_image = pca_calculate(image_train, reduced_dimension_number)
    reconstructed = np.dot(pca_result.transpose(), principle_components)
    unused_test_eigenval, unused_test_eigenvec, centralized_test_image = eigen_calc(image_test)
    projected_test_images = np.dot(principle_components, centralized_test_image.transpose())

    """
    mean = np.mean(image_train, axis=0)
    reconstructed = np.dot(pca_result.transpose(), principle_components)
    plt.figure()
    training_samples_mean_2_show = []
    training_samples__mean_titles_2_show = []
    
    for i in range(20):
        index = listed_set_index[i]
        training_samples_mean_2_show.append(train_centralized_image[index])
        training_samples__mean_titles_2_show.append('image - mean [' + str(index) + '] = ' + str(y_train[index]))
    show_images(training_samples_mean_2_show, training_samples__mean_titles_2_show)
    plt.savefig(str(dimension) + "samples_train_mean_set.png", dpi=300, format='png')
    """

    """eigenvec_2_show = []
    eigenvec_title_2_show = []
    plt.figure()
    for i in range(20):
        eigenvec_2_show.append(principle_components[i].reshape(28, 28))
        eigenvec_title_2_show.append('P Eigenvector [' + str(i) + ']')
    show_images(eigenvec_2_show, eigenvec_title_2_show)
    plt.savefig('m' + str(dimension) + "_eigenvec_2_show.png", dpi=300, format='png')
    
    reconstructed_2_show = []
    reconstructed_title_2_show = []
    plt.figure()
    for i in range(20):
        index = listed_set_index[i]
        reconstructed_2_show.append(reconstructed[index].reshape(28, 28))
        reconstructed_title_2_show.append('Reconstructed [' + str(index) + ']')
    show_images(reconstructed_2_show, reconstructed_title_2_show)
    plt.savefig('m' + str(dimension) + "_reconstructed_2_show.png", dpi=300, format='png')
    """

    knn_matrix = k_nearest_neighbours(pca_result, projected_test_images, k)
    classified_test_labels = test_assign_labels(knn_matrix, label_train)
    accuracy = model_stats(classified_test_labels, label_test)
    # print(" {{ Train sample size = " + str(train_sample_number) + ", m = " + str(
    #    reduced_dimension_number) + ", k = " + str(k) + " }} Accuracy == " + str(accuracy))
    return accuracy


dimension = [20, 50, 100, 300]

for i in range(1, 11):
    x = 100 * i
    plotdata_x_0.append(x)
    for j in range(4):
        plotdata_y_all[j][i - 1] = classifier(x, 3000, dimension[j], 20)

l0 = plt.plot(plotdata_x_0, plotdata_y_all[0], 'r--', label='eigenvector size = ' + str(20))
l1 = plt.plot(plotdata_x_0, plotdata_y_all[1], 'g--', label='eigenvector size = ' + str(50))
l2 = plt.plot(plotdata_x_0, plotdata_y_all[2], 'b--', label='eigenvector size = ' + str(100))
l3 = plt.plot(plotdata_x_0, plotdata_y_all[3], 'c--', label='eigenvector size = ' + str(300))
plt.title('Training Size VS. Accuracy (k = 20)')
plt.xlabel('Training size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
