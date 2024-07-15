from .faceRec import *
import os

if not os.path.exists(fr"{os.path.abspath(r'faceRec')}\dataSets"):
    os.mkdir(fr"{os.path.abspath(r'faceRec')}\dataSets")

if not os.path.exists(fr"{os.path.abspath(r'faceRec')}\trainers"):
    os.mkdir(fr"{os.path.abspath(r'faceRec')}\trainers")
