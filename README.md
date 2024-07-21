# Face Recognition
## introduction
It is a face recognition library that can accurately predict from 3 or less initial photos. The number of people to detect can be increased by changing the classification algorithm. The method of detection as well as obtaining face tokens is based on facenet model. The tokens are subsequently categorized using svd.
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
This is an example of basic project creation, with adding a new user, training a model on it and then using it:
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
