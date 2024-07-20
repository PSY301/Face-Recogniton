# Face Recognition
## introduction
It is a face recognition library that can accurately predict from 3 or less initial photos. I will explain how the model learns and predict.
### Model training
1. The program creates folders dataSets and trainers. In the future, the dataSets folder will store training images of people, and the trainers folder will contain the database for determining the name of the predicted person by id and will contain the trained model.
2. The program creates in the folders dataSets and trainers a project (folder) in which you will work.
3. The program receives training images with names of people on them, creates folders for each person, copies training images there and augments each photo in each folder. The name of the folder is the id of the person. To determine the name of a person by his id, the program creates a database.
4. The program detects faces in each training image using facenet, obtains their embeddings and, using the SVD algorithm, trains a model on them and stores it.
### Face recognition 
1. The program gets an image with an as yet unidentified face.
2. The program gets the faces in the image and rectangles them.
3. The model recognizes a person's face in the image, gets that person's name from the database and writes it at the bottom of the box.
4. The program shows or saves the finished image in the same folder.
## Installing 
To install the library you need to type this command in the command line:
```python
pip install face_rec
```
This library use site-packages like: opencv-python, scikit-learn, facenet_pytorch, torch and etc. to install them download requirements.txt file and install this packages by command:
```python
pip install requirements.txt
```
## Example of working
1. Import library.
2. Create your project.
3. Add images of people to train model.
4. Train model.
```python
# Import library.
import face_rec.faceRec

# Create your project.
project = face_rec.faceRec.faceRec("Your_name_of_project")

# Add images of people to train model.
project.new_user("name_of_user", "path_to_image")

# Train model.
project.fit()

# Use model to predict photo.
project.predict_photo("path_to_predicted_image")

# OR predict video from your device's webcam.
project.predict_video()
```

It is all you need to know to begin to work with this library. More info you can find in doc-strings.
