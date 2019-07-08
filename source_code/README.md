BeautyGAN on webcamera
===

> This is a demo of generating makeup image from the image gotten from web camera.

### Setup:
1. Download or clone source code from github.
2. Download train data from this [url](https://drive.google.com/file/d/17Nm2hR3Hz4U0Yh6M93wfc2A2wQ_pE-hP/view?usp=sharing)
3. Unzip the downloaded zip file (named checkpoints_ori) at the same derectory of README.
4. Run `pip install -r requirements.txt` in directry named "source_code"  
5. Run `python main.py` in same directry.  
6. Then you get some question as follow. Please answer it following instruction.
7. Finally you get the both image & video with BeautyGAN with the error as below.  
  `AttributeError: 'NoneType' object has no attribute 'shape'`  
  But it doesn't matter for the result. Don't worry.

## Makeup stype:
 - 6 kind of makeup style  
 
![images_all](https://user-images.githubusercontent.com/20176579/59491600-ab90d380-8eb9-11e9-9b2d-f54534598fd2.png)

### Demo:
 - Makeup style(Chose image 1)  
 ![images1](https://user-images.githubusercontent.com/20176579/59495678-6ae98800-8ec2-11e9-842f-5200b9f93199.jpg)
 
 - Original video  
 ![test](https://user-images.githubusercontent.com/20176579/59502783-3e893800-8ed1-11e9-9206-688c72e498c8.gif)
 
 - Genarated with BeautyGAN video  
 ![images1](https://user-images.githubusercontent.com/20176579/59502797-4812a000-8ed1-11e9-87af-186a03a70f46.gif)
 
##### Dependencies:
- [BeautyGAN](http://liusi-group.com/pdf/BeautyGAN-camera-ready_2.pdf)  
- [dlib](http://dlib.net/)  
- [opencv-python](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)
