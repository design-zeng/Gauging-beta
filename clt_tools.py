import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import os
import arff
import h5py
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_test_cases_auto(folder_path):
    test_cases = {}
    try:
        # List all files in the folder
        file_names = os.listdir(folder_path)
    except Exception as e:
        print("Error:", e)

    for file_name in file_names:
        test_cases[os.path.splitext(file_name)[0]] = folder_path + file_name
    return test_cases



def read_data(filePath, has_labels=True):
    data = []
    labels = []
    with open(filePath, 'r+') as fr:
        for line in fr:
            value = line.split()
            value = np.float32(np.array(value))
            data.append(value[:-1])
            if has_labels:
                labels.append(value[-1])
    return np.array(data), np.array(labels)


def draw_clusters(data, labels):
    unique_labels = np.unique(labels)
    f = plt.figure()
    dimension = len(data[0])

    if dimension == 2:
        for label in unique_labels:
            plt.scatter(data[(labels == label), 0], data[(labels == label), 1], s=12)

        ax = plt.gca()  # Get current axis
        ax.set_xticks([])  # Remove x-axis tick labels
        ax.set_yticks([])  # Remove y-axis tick labels
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    if dimension == 3:
        ax = f.add_subplot(111, projection='3d')

        for label in unique_labels:
            ax.scatter(data[labels == label, 0], data[labels == label, 1], data[labels == label, 2], s=12)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    return f


def plot_data(data):
    f = plt.figure()
    dimension = len(data[0])

    if dimension == 2:
        plt.scatter(data[:, 0], data[:, 1], s=12)

        ax = plt.gca()  # Get current axis
        ax.set_xticks([])  # Remove x-axis tick labels
        ax.set_yticks([])  # Remove y-axis tick labels
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
    if dimension == 3:
        ax = f.add_subplot(111, projection='3d')

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=12)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    return f


def plot(self, withline=False):
    plt.figure(figsize=(12, 12))
    for k, cluster in self.clusters.items():
        if self.DIMENSION == 2:
            plt.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1])
            if withline:
                for track in cluster['traces']:
                    # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
                    #                 color=scatter.get_facecolor()[0], linestyle='-')
                    plt.plot(self.X[list(track)][:, 0], self.X[list(track)][:, 1], 'k-')
        else:
            plt.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1],
                                  self.X[cluster['data']][:, 2])
    plt.savefig(f'{self.log_path}/{len(self.clusters)}.png')
    plt.close('all')


def load_image(file_name):
    image = plt.imread(file_name)
    image = cv2.resize(image,(50, 50))
    RGB = image.reshape(-1, 3)
    plt.imshow(image)


    X = np.arange(image.shape[0])
    Y = np.arange(image.shape[1])
    x, y = np.meshgrid(X, Y)
    RGBXY = np.column_stack((RGB, x.reshape(-1, 1), y.reshape(-1, 1)))
    RGBXY *= [1, 1, 1, 0.01, 0.01]
    RGBXY = np.float32(RGBXY)
    return RGBXY


def load_clustering_data(file_name):
    # plt.clf('all')
    data = np.loadtxt(file_name)
    # img = np.empty((50, 50, 3))
    # img = np.empty((50, 75, 3))
    clusters = {}
    centroids = {}
    for x in data:
        if int(x[-1]) not in clusters:
            # clusters[int(x[-1])] = [denormalize_imagexy(x[:-1], file_name)]
            clusters[int(x[-1])] = [x[:-1]]
        else:
            # clusters[int(x[-1])].append(denormalize_imagexy(x[:-1], file_name))
            clusters[int(x[-1])].append(x[:-1])

    for k, v in clusters.items():
        v = np.array(v)
        ave_c = np.mean(v[:, :3], axis=0)
        centroids[k] = ave_c
    RGB_recolored = data[:, :3]
    for i in range(RGB_recolored.shape[0]):
        RGB_recolored[i, :] = centroids[data[i, 5]]

    plt.imshow(RGB_recolored.reshape((50,50,3)))
    # plt.show()
    plt.savefig(file_name.split('.')[0] + '.png')


def denormalize_imagexy(imagexy, path):
    # data = np.loadtxt(path.split('.')[0][:-1]+'normalization.out')
    # diffs, mins = data[:5], data[-5:]
    #
    # imagexy /= [50, 50, 50, max(diffs[:3]), max(diffs[:3])]
    # imagexy += mins

    imagexy /= [1, 1, 1, 4, 4]
    return imagexy


def process_mnist():
    mnist = fetch_openml('mnist_784')
    image = mnist.data.to_numpy()
    for i in range(100):

        plt.imshow((image[i].reshape(28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        # plt.show()
        plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_3\mnist\mnist{i}.png')


def load_mnist(file_name):
    data = np.loadtxt(file_name)
    clusters = {}
    for x in data:
        if int(x[-1]) not in clusters:
            clusters[int(x[-1])] = [x[:-1]]
        else:
            clusters[int(x[-1])].append(x[:-1])

    root = file_name.split('.')[0]
    for k, cs in clusters.items():
        for i, c in enumerate(cs):
            plt.imshow((c.reshape(28, 28) * 255), cmap=plt.cm.gray_r,
                       interpolation='nearest')
            # plt.show()
            plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_4\\res\{k}_{i}.png')


def load_cifar(file_name):
    data = np.loadtxt(file_name)
    clusters = {}
    for x in data:
        if int(x[-1]) not in clusters:
            clusters[int(x[-1])] = [x[:-1]]
        else:
            clusters[int(x[-1])].append(x[:-1])

    root = file_name.split('.')[0]
    for k, cs in clusters.items():
        for i, c in enumerate(cs):
            plt.imshow((c.reshape(3, 32, 32).transpose(1, 2, 0) / 255))
            # plt.show()
            plt.savefig(f'D:\python projects\VisionClustering\plots\evaluation\integration_5\\cifar_res\{k}_{i}.png')

def load_arff(file_name):
    # path = 'path/to/your/file.arff'
    with open(file_name, 'r') as file:
        data = arff.load(file)
    data_values = np.array(data['data'])
    # data, meta = arff.loadarff(file_name)
    # reshaped_data = np.array([item for sublist in data for item in sublist]).reshape(len(data), -1)
    return data_values

def fix_arff_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.lower().startswith('@attribute'):
                # Replace tabs with spaces and reduce multiple spaces to a single space
                fixed_line = ' '.join(line.split())
                outfile.write(fixed_line + '\n')
            else:
                outfile.write(line)


def is_space_delimited(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip header lines until we find the @data section
            if line.strip().lower() == "@data":
                break

        # Check a few lines in the @data section
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue

            # If a line contains spaces but not commas, it's likely space-delimited
            if ' ' in line and ',' not in line:
                return True

            # Check only a few lines
            break

    return False

def convert_arff_delimiters(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        data_section = False
        for line in infile:
            # Check if we are in the data section
            if line.strip().lower() == "@data":
                data_section = True
                outfile.write(line)
                continue

            # If in data section, replace spaces with commas
            if data_section:
                # Split the line by whitespace and then join with commas
                # This handles multiple spaces and trims the line as well
                converted_line = ','.join(line.split())
                outfile.write(converted_line + '\n')
            else:
                outfile.write(line)

def get_curve_problem(file_name):
    image = plt.imread(file_name)
    xx = image[:, :, 0]
    blacks = np.where(xx == 0)
    points = np.column_stack((blacks[0], blacks[1]))
    targets = []
    for p in points:
        for t in targets:
            if np.linalg.norm(p-t) < 5:
                break
        else:
            targets.append(p)

    print()
    with open('benchmark/curve/fan.txt', "w+") as fw:
        for i in range(len(targets)):
            fw.write(f"{targets[i][0] / 10} {targets[i][1] / 10} {0}\n")


def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise ValueError('')


def load_data(filename, dataname):
    if filename.endswith('txt'):
        data = []
        labels = []
        if dataname == 'USPS':
            with open(filename, 'r+') as fr:
                for line in fr:
                    value = line.split(",")
                    value = np.float32(np.array(value))
                    data.append(value)
            data = np.array(data)
        else:
            with open(filename, 'r+') as fr:
                for line in fr:
                    value = line.split(",")
                    value = np.float32(np.array(value))
                    data.append(value[:-1])
                    labels.append(value[-1])

            data = np.array(data), np.array(labels)
    elif filename.endswith('png'):
        data = load_image(filename)
    elif filename.endswith('mnist'):
        data = load_mnist()

    elif filename.endswith('arff'):
        fix_arff_file(filename, f"{dataname}_fixed.arff")
        output_file = f"{dataname}_comma.arff"
        if is_space_delimited(filename):
            print(f"{dataname} is space-delimited. Converting to comma-delimited format...")
            convert_arff_delimiters(f"{filename}_fixed.arff", output_file)
            data_file_to_load = output_file
        else:
            print(f"{dataname} is already comma-delimited.")
            data_file_to_load = f"{filename}_fixed.arff"
        data = load_arff(data_file_to_load)
        if dataname == 'dermatology':
            data = np.delete(data, [33, 34, 35, 36, 262, 263, 264, 265], axis=0)
        if dataname == 'vowel':
            data = np.delete(data, [0, 1, 2], axis=1)
        if dataname == 'wdbc':
            data = np.delete(data, [0], axis=1)
            first_column = data[:, 0].reshape(-1, 1)
            rest_columns = data[:, 1:]
            data = np.concatenate((rest_columns, first_column), axis=1)
        if dataname == 'wine':
            first_column = data[:, 0].reshape(-1, 1)
            rest_columns = data[:, 1:]
            data = np.concatenate((rest_columns, first_column), axis=1)
        if dataname == 'yeast':
            data = np.delete(data, [0], axis=1)

        # assume the arff data have the same structure, the last column is the label in the text form
        encoder = LabelEncoder()
        data[:, -1] = encoder.fit_transform(data[:, -1])
        data = data.astype(float)
        # Convert byte strings to float
        # if isinstance(data[:, -1], bytes) or data[:, -1].dtype == np.dtype('S4') or data[:, -1].dtype == np.dtype('S32')\
        #     or data[:, -1].dtype == np.dtype('S8') or data[:, -1].dtype == np.dtype('S9'):
        #     print("true")
        #     vectorized_conversion = np.vectorize(bytes_to_float)
        #     float_data = vectorized_conversion(data)
        #     data = float_data
        np.savetxt(f'ground-truth-labels/{dataname}.txt', data[:, -1], fmt='%s')
        data = data[:, :-1], data[:, -1]
    elif filename.endswith('h5'):
        data = load_h5(filename)
    else:
        df = pd.read_excel(
            filename,
            engine='openpyxl',
        )
        data = df.values[:, :2], df.values[:, -1], 'xlxs'
    return data

def load_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    return X_tr, y_tr, X_te, y_te


# Define a function to decode bytes and convert to float
def bytes_to_float(b):
    return float(b.decode('utf-8'))

if __name__ == '__main__':
    data = load_data('data/aggregation.txt', 'agg')[0]
    fig = plot_data(data)
    fig.savefig('agg.png', bbox_inches='tight')