import imgaug.augmenters as iaa
import cv2
import numpy as np
import os
from . import faceRec
import itertools
from tqdm import tqdm
import shutil



def aug_name(dir_name: str, path: str = os.path.dirname(os.path.abspath(__file__))):
    """
    Creates 33 augmented images for each image from the directory, passed by the dir_name parameter and saves them in
    the same directory under the name augmented_{name_of_image}_{unic_index}.
    :param: dir_name [str]
        Name of directory.
    :param: path [str], optional
        Full path to dir, passed by dir_name parameter, is needed if directory,
        you want work with is not in directory faceRec
        by default os.path.dirname(os.path.abspath(__file__)).
    """

    dataPath = os.path.join(path, dir_name)
    image_paths = [os.path.join(dataPath, f) for f in os.listdir(dataPath)]

    augmentations = [
        iaa.PerspectiveTransform(scale=0.15, random_state=False),
        iaa.ScaleX(1.35),
        iaa.ScaleX(0.8),
        iaa.ScaleY(1.35),
        iaa.ScaleY(0.8),
        iaa.Affine(rotate=45),
        iaa.Affine(rotate=-45),
        iaa.Fliplr(1),
        iaa.Flipud(1),
        iaa.GaussianBlur(2.5)
    ]

    add_bright = [
        iaa.Multiply(0.5),
        iaa.Multiply(1.5)
    ]

    for path in image_paths:
        image = cv2.imread(path)

        augmented_images = []
        for i, augmentation in enumerate(augmentations):
            augmented_image = np.copy(image)
            augmented_image = augmentation(image=augmented_image)
            augmented_images.append(augmented_image)
        augmented_images.append(image)
        res_augmented_images = augmented_images.copy()
        for img in augmented_images:
            for aug in add_bright:
                aug_img = np.copy(img)
                aug_img = aug(image=aug_img)
                res_augmented_images.append(aug_img)

        for i, img in enumerate(res_augmented_images):
            img_name = os.path.join(dataPath,
                                    ''.join(["augmented_", os.path.basename(path)[:4], '_', str(i + 1), ".jpg"]))
            cv2.imwrite(img_name, img)


def aug_dataset(name_of_project: str):
    """
    Applies the aug_name function to every non-augmented user of the name_of_project project.
    :param: name_of_project[str]
        name of project.
    """
    db_path = os.path.join("trainers", name_of_project, "database.db")
    ds_path = os.path.join("dataSets", name_of_project)
    for id in os.listdir(ds_path):
        aug_name(id, os.path.join("dataSets", name_of_project))


def train_dataset(number_of_people: int = 1, dataset: str = "main_dataSet_31_people"):
    """
    Trains the model on the number_of_people of people 
    from each folder in the submitted dataset.
    
    :param:number_of_people[int], optional
        number of people from each folder.
        By default, 1
    
    :param:dataset[str], optional
        dataset on which we are training model.
        By default, "main_dataSet_31_people"
    """
    
    face_rec = faceRec.FaceRec("test")

    for person in tqdm(os.listdir(dataset)):
        for h in itertools.islice(sorted(os.listdir(f"{dataset}/{person}")), None, number_of_people):
            image = cv2.imread(f"{dataset}/{person}/{h}")
            face_rec.new_user(person, image)

    face_rec.fit()


def full_destroy():
    """
    destroy all projects.
    """
    shutil.rmtree("dataSets")
    shutil.rmtree("trainers")
