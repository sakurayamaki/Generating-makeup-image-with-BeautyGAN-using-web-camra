BeautyGAN on webcamera
===

> This is a demo of generating makeup image from the image gotten from web camera.

#### Setup:
1. Run `pip install -r requirements.txt`
2. Download and extract shape predictor 68 landmarks at this [link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
3. Place the extracted predictor in the root directory of this project.
4. Find some images with faces, tiled if possible.

#### Run:
```bash
python main.py
```

#### Process log:
```
Which make up style do you choose?
Input the style number.(1~5)
Input output file name.
Doyou want to get your face?(yes/no)
What is your input name?
```

##### Dependencies:
- [dlib](http://dlib.net/)
- [opencv-python](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
