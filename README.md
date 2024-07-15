# Face Recognition
To begin with, you need to install this library.
```python
pip install face_rec
```
## instruction of working
1. Import library.
```python
import face_rec.faceRec
```
2. Create your project.
```python
project = face_rec.faceRec.faceRec("Your_name_of_project")
```
3. Add images of people to train model.
```python
project.new_user("name_of_user", "path_to_image")
```
4. Train model.
```python
project.fit()
```
Good job! You train your model. But what can you do with it?
* predict photo.
```python
project.predict_photo("path_to_predicted_image")
```
Function will show you image with rectangled face and label with name that predict model in the bottom of it.
* predict_video.
```python
project.predict_video()
```
function will predict faces from video on your device's webcam.

It is all you need to know to begin to work with this library. More info you can find in doc-strings.
