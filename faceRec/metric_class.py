import matplotlib.pyplot as plt
import itertools
import numpy as np
from tqdm import tqdm
import os
from . import faceRec
import time
from functools import wraps
import math


class LazyProperty:
    """
    Decorator-Class to facilitate computer operation by storing the results of calculations with the same objects
    """
    def __init__(self, func):
        """
        Constructor to initialize the attributes of the class

        :param: func
            function we work with
        """
        self.func = func

    def __get__(self, instance, cls):
        """Main descriptor in this class"""
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


class Matrix:
    """Class to work with matrix"""

    def __init__(self, TN: int, FP: int, FN: int, TP: int):
        """
        Constructor to initialize the attributes of the class.

        :param: TN[int]
            the number of correctly predicted negative values.
        :param: FP[int]
            the number of incorrectly predicted positive values.
        :param: FN[int]
            the number of incorrectly predicted negative values.
        :param: TP[int]
            the number of correctly predicted positive values.
        """
        self._matrix = np.array([[TN, FP], [FN, TP]])

    @classmethod
    def base(cls):
        """
        Another constructor of the class. Feel Matrix with (0,0,0,0).
        """
        return Matrix(0, 0, 0, 0)

    def __str__(self) -> str:
        """
        Method to correctly printing Matrix object.

        :return: str
        """
        return f"{self._matrix[0][0]} {self._matrix[0][1]} {self._matrix[1][0]} {self._matrix[1][1]}"

    def __getitem__(self, index: int) -> list:
        """
        Method to correctly getting element of Matrix object.

        :param: index[int]
            index of element
        """
        return self._matrix[index]

    def plot_confusion_matrix(self, classes=("No", "Yes"),
                              title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.

        :param: classes[tuple[str]], optional
            names of positive and negative classes.
        :param: title
            title of confusion matrix.
        """
        plt.imshow(self._matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = self._matrix.max() / 2.
        for i, j in itertools.product(range(self._matrix.shape[0]), range(self._matrix.shape[1])):
            plt.text(j, i, self._matrix[i, j],
                     horizontalalignment="center",
                     color="white" if self._matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @LazyProperty
    def precision(self) -> float:
        """
        Calculate precision. Use LazyProperty.

        :return: float
        """
        result = float(self._matrix[1][1] / float(self._matrix[1][1] + self._matrix[0][1]))
        return 0 if math.isnan(result) else result

    @LazyProperty
    def recall(self):
        """
        Calculate recall. Use LazyProperty.

        :return: float
        """
        result = float(self._matrix[1][1] / float(self._matrix[1][1] + self._matrix[1][0]))
        return 0 if math.isnan(result) else result

    @LazyProperty
    def f_measure(self):
        """
        Calculate F-measure. Use LazyProperty.

        :return: float
        """
        try:
            result = 2 * ((self.recall * self.precision) / (self.recall + self.precision))
            return result
        except ZeroDivisionError:
            return 0


def timer(func):
    """
    decorator measuring function execution time.

    :param: func
        function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper


class Metric:
    """
    Class for measuring model metrics
    """

    def __init__(self, name_of_project: str, train_photos: list, persons: list, matrix=None, dataSet="main_dataSet_31_people"):
        """
        Constructor to initialize the attributes of the class.

        :param: name_of_project[str]
            name of project where we're testing our model
        """
        self.name_of_project = name_of_project
        self.face_recognizer = faceRec.FaceRec(self.name_of_project)
        self.persons = persons
        self.train_photos = train_photos
        self.matrix = matrix
        self.dataSet = dataSet

    @timer
    def _running_dataset(self) -> [Matrix]:
        """
        Main method in this class. Calculate matrix(TP, TN...).
        """

        dirs = os.listdir(self.dataSet)

        list_of_dop_im = [[0 for _ in range(len(dirs))] for _ in range(len(dirs))]

        for i, dir in enumerate(dirs):
            image_names = os.listdir(fr"{self.dataSet}\{dir}")
            for n in range(len(image_names)):
                index = n % len(dirs)
                if index == i:
                    continue
                list_of_dop_im[i][index] += 1

        matrix_list = [Matrix.base() for _ in range(len(dirs))]

        for i, dir in enumerate(tqdm(dirs)):
            images = os.listdir(fr"{self.dataSet}\{dir}")

            for file_name in images:

                if file_name in self.train_photos:
                    continue

                name = self.face_recognizer.predict_photo(fr"{self.dataSet}\{dir}\{file_name}",
                                                          save=False, threshold=0.5, csm_image=False)
                print(name, dir)
                if name == dir:
                    for n, lst in enumerate(list_of_dop_im):
                        if n == i:
                            matrix_list[n][1][1] += 1
                        elif lst[i] != 0:
                            matrix_list[n][0][0] += 1
                            lst[i] -= 1
                elif name != dir:
                    for n, lst in enumerate(list_of_dop_im):
                        if n == i:
                            matrix_list[n][1][0] += 1
                        elif lst[i] != 0:

                            if name == dirs[n]:
                                matrix_list[n][0][1] += 1
                            else:
                                matrix_list[n][0][0] += 1
                            lst[i] -= 1
        return matrix_list

    def checking_matrix(self):
        """
        Check class's matrix. If it is None calls _running_dataset
        """
        if self.matrix is None:
            data = self._running_dataset()
            self.matrix = data

    def __getattr__(self, item) -> [float]:
        """
        calculate metrics(f-measure).
        """
        self.checking_matrix()

        measure_list = []

        for matrix in self.matrix:
            measure_list.append(getattr(matrix, item))
        return measure_list

