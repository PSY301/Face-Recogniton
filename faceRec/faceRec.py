import cv2
import numpy
from . import utils
from . import database
from . import dataset
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
from sklearn.svm import SVC
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pickle
from torchvision import datasets
from torch.utils.data import DataLoader
workers = 0 if os.name == 'nt' else 4


class FaceRec:
    """
    The main face recognition class.
    """
    def __init__(self, name_of_project: str, people: dict = None):
        """
        Constructor to initialize the attributes of the class. Creates new project name_of_project or recreate old.
        :param: name_of_project[str]
            name of project.
        :param: people[dict], optional
            dict of people of project name_of_project
            keys: names of peoples
            values: image objects
            by default None.
        """
        self.name_of_project = name_of_project
        self.people = people
        self.trainer = SVC(kernel='linear', probability=True)
        self.detector = MTCNN(
                           image_size=160, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True
                           )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        if utils.check_project(self.name_of_project):
            with open(os.path.join("FaceRec", "trainers", name_of_project, "trainer.pickle"), "rb") as f:
                self.trainer = pickle.load(f)
            self.db = database.DBWorker(
                os.path.join(os.path.abspath(os.path.join("faceRec", "trainers")), self.name_of_project, "database.db"))
        else:
            os.mkdir(os.path.join(os.path.abspath(os.path.join("faceRec", "dataSets")), self.name_of_project))
            os.mkdir(os.path.join(os.path.abspath(os.path.join("faceRec", "trainers")), self.name_of_project))
            self.db = database.DBWorker(
                os.path.join(os.path.abspath(os.path.join("faceRec", "trainers")), self.name_of_project, "database.db"))
            self.db.create()
        if self.people is not None:
            for name, img in self.people.items():
                self.new_user(name, img)

    def new_user(self, name: str, filename: str):
        """
        Add new user to project self.name_of_project.
        :param: name[str]
            name of new user.
        :param: filename[str]
            path to image.
        """

        if name not in self.db.get_names():
            real_id = self.db.new_row(name)

            os.mkdir(os.path.join(os.path.abspath(os.path.join("faceRec", "dataSets")), self.name_of_project,
                                  str(real_id)))
            im_name = "0.jpg"
        else:
            real_id = self.db.get_id(name)
            print(real_id)
            im_names_bef = sorted(os.listdir(os.path.join(os.path.abspath(os.path.join("faceRec", "dataSets")), self.name_of_project, str(real_id))))
            last_im_name_bef = im_names_bef[len(im_names_bef)]
            mask = re.compile(r"\d+")
            im_name = int(mask.findall(last_im_name_bef)[0]) + 1

        img = cv2.imread(filename)
        cv2.imwrite(os.path.join(os.path.abspath(os.path.join("faceRec", "dataSets")), self.name_of_project,
                                 str(real_id), im_name), img)
        dataset.aug_name(str(real_id), os.path.join(os.path.abspath(os.path.join("faceRec", "dataSets")),
                         self.name_of_project))

    def create_dataset_loader(self) -> (datasets.folder.ImageFolder, torch.utils.data.dataloader.DataLoader):
        print("Create dataloader")
        dataset = datasets.ImageFolder(os.path.join("faceRec", "dataSets", self.name_of_project))
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=lambda x: x[0], num_workers=workers)
        print(type(dataset), type(loader))
        print("done")
        return dataset, loader

    def extract_face(self, filename: str = None, image_pixels: np.ndarray = None, required_size: tuple = (160, 160)) -> (np.ndarray, tuple[int]):
        """
        extract faces in image.
        
        :param: filename[str], optional
            path to image.
            By default, None
        :param: image_pixels[numpy.ndarray], optional
            image pixels. If image_pixels is None and filename is None 
            function doesn't work.
            By default, None
        :param: required_size[tuple[int]], optional
            size of result image.
            By default, (160, 160)
        :return:
            boxes of faces and images of faces.
        """
        if filename is not None:
            image = Image.open(filename)
            image = image.convert('RGB')
            pixels = np.asarray(image)
        elif image_pixels is not None:
            pixels = image_pixels
        else:
            raise TypeError("filename is None and image_pixels is None")
        results = self.detector.detect(pixels)
        if results[0] is None:
            return None, None
        x1, y1, width, height = map(int, results[0][0])
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        box_dimensions = (x1, y1, width, height)
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array, box_dimensions

    def get_embedding(self, image_pixels: np.ndarray = None, filename: str = None) -> torch.tensor:
        """
        get embeddings of face from image.

        :param: filename[str], optional
            path to image.
            By default, None
        :param: image_pixels[numpy.ndarray], optional
            image pixels. If image_pixels is None and filename is None
            function doesn't work.
            By default, None
        :return: torch.Tensor.
        """
        if image_pixels is not None:
            img = image_pixels
        elif filename is not None:
            image = Image.open(filename)
            image = image.convert('RGB')
            img = np.asarray(image)
        else:
            raise TypeError("filename is None and image_pixels is None")
        x_aligned = self.detector(img)
        if x_aligned is None:
            return None
        x_aligned = x_aligned.clone().detach()
        return self.resnet(x_aligned.unsqueeze(0))

    def create_dataset(self) -> (list[np.ndarray], list[str]):
        """
        Creates dataset(X train, Y train).

        :return: list of embeddings of images.
        """
        dataset, loader = self.create_dataset_loader()

        X = []
        Y = []
        print("Get embeddings")
        for x, y in tqdm(loader):
            emd = self.get_embedding(image_pixels=x)
            if emd is not None:
                X.append(emd.detach().numpy()[0])
                Y.append(dataset.idx_to_class[y])
        print("done")
        return X, Y

    def fit(self):
        """
        fit model for project and save it in trainer.pickle.
        """
        print("Creating dataset for training")
        X_train, Y_train = self.create_dataset()
        print("Creating model")
        print("Fitting model")
        self.trainer.fit(X_train, Y_train)
        print("Model is ready!")
        print("Saving model")
        with open(os.path.join("FaceRec", "trainers", self.name_of_project, "trainer.pickle"), "wb") as file:
            pickle.dump(self.trainer, file)
        print("Done!")

    def predict_photo(self, filename: str = None, image_pixels: np.ndarray = None, threshold: float = 0.8,
                      save: bool = False, show: bool = True, csm_image: bool = True) -> str:
        """
        Rectangles the faces in the picture along the path to photo and adds an inscription with the username that it
predicts, but if the percentage of predictability is less than threshold, the "Unknown user" label is added.
Then saves it as a new image in the same directory under the name predicted_{name_of_image}.
        :param: path_to_photo[str]
            path to photo.
        :param: threshold[int], optional
            the threshold of similarity of the image
            by default 50.
        :param: save[bool], optional
            should function save image with predicted face?
            By default, True.
        :param: show[bool], optional
            should function show image with predicted face?
            By default, False.
        :return: str
            name of user who was predicted.
        """

        if image_pixels is None:
            image = cv2.imread(filename)
        else:
            image = image_pixels
        faces, box = self.extract_face(filename, image_pixels)#, required_size=(1080, 1080))
        if faces is None:
            return None, None, None
        embedding = self.get_embedding(image_pixels=image)
        if embedding is None:
            return "Unknown person"
        prob = np.max(self.trainer.predict_proba(embedding.detach().numpy()))
        label = self.trainer.predict(embedding.detach().numpy())[0]
        print(prob)
        if prob < threshold:
            label = "Unknown person"
        else:
            label = self.db.get_profile(label)
            label = label[0]

        if save or show or csm_image:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (225, 0, 0), 2)
            cv2.putText(image, label + str(round(prob, 3)), (box[0], box[3]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255))

        if save:
            cv2.imwrite(os.path.join(os.path.dirname(filename), f"predicted_{os.path.basename(filename)}"), image)
        if show:
            cv2.imshow("predicted image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return label

    def predict_video(self, threshold: int = 0.8):
        """
        Receives an image from your device's webcam, rectangles the face on it and adds an inscription with the username
        that it predicts, but if the percentage of predictability is less than threshold, the "Unknown user" label is added.
        Then, shows the image in a separate window. All subsequent acquired images are displayed in the same window,
        composing a video.
        :param: threshold[int], optional
            the threshold of similarity of the image
            by default 0.8.
        """
        cam = cv2.VideoCapture(0)
        while cv2.waitKey(1) < 0:
            ret, im = cam.read()
            if not ret:
                continue
            name = self.predict_photo(image_pixels=im, threshold=0.6, save=False)
            cv2.imshow("predicted_photo", im)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
