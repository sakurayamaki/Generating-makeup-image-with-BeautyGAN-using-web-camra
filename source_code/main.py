import os
import sys
import time
import numpy as np
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import cv2, dlib, argparse
from utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from matplotlib import pyplot as plt
#%matplotlib inline

import pylab
import imageio

from beautyGAN_main import BeautyGAN
import tensorflow as tf

from imutils.face_utils import rect_to_bb
import imutils
from helpers import FACIAL_LANDMARKS_68_IDXS
from helpers import FACIAL_LANDMARKS_5_IDXS
from helpers import shape_to_np

tf.reset_default_graph()


class FaceDetector():
    
    def __init__(self, scale=1):
        
        self.scale = scale
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
        self.faceAligher = FaceAligner(self.predictor, desiredLeftEye=(0.32, 0.32),desiredFaceWidth=256)
        
    def get_face(self, img):
        
        height, width = img.shape[:2]
        s_height, s_width = height//self.scale, width//self.scale
        img = cv2.resize(img, (s_width, s_height))
        
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(img_g, 1)
        
        for rect in dets:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect)
            faceAligned, m = self.faceAligher.align(img, img_g, rect)
            
        return faceAligned, m
    
#modified from https://github.com/jrosebr1/imutils/blob/master/imutils/face_utils/facealigner.py
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        
        #simple hack ;)
        if (len(shape)==68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
            
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output, M

def get_makeup_image(input_src, input_tgt, image_name):
    
    img_tgt = cv2.imread(input_tgt )
    img_tgt = imutils.resize(img_tgt, width=800)
    tgt_face, m = detector.get_face(img_tgt)
    tgt = cv2.cvtColor(tgt_face, cv2.COLOR_BGR2RGB)
    tgt = np.expand_dims(tgt,0)/ 127.5 -1
    
    img_src = cv2.imread(input_src )
    img_src = imutils.resize(img_src, width=800)
    src_face, m = detector.get_face(img_src )
    src = cv2.cvtColor(src_face, cv2.COLOR_BGR2RGB)
    src = np.expand_dims(src,0)/ 127.5 -1

    height, width, ch = img_src.shape
    size = (width,height)
    
    _, output = beautyGan.predict(src,tgt) # src:non makeup target: makup

    output = (output + 1) * 127.5

    output = np.uint8(np.squeeze(output))


    im = cv2.invertAffineTransform(m)
    output = cv2.warpAffine(output, im, size,flags=cv2.INTER_CUBIC)

    lip_mask, eye_mask, face_mask = beautyGan.get_mask(src_face, detector.detector, detector.predictor)

    lip_mask = lip_mask.astype(np.uint8)
    face_mask = face_mask.astype(np.uint8)
    eye_mask = eye_mask.astype(np.uint8)

    kernel = np.ones((3,3),np.uint8)
    face_mask = cv2.erode(face_mask,kernel,iterations = 5)
    # face_mask = cv2.erode(face_mask,kernel,iterations = 3)

    mask = np.clip(lip_mask * 3.0+ 0.0*eye_mask + face_mask * 0.0 , 0, 255)


    mask = mask.astype(np.uint8)

    face_mask = cv2.warpAffine(mask, im, size,flags=cv2.INTER_CUBIC)
    face_mask = cv2.blur(face_mask,(10,10))


    foreground = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    background = img_src
    alpha = np.stack([face_mask, face_mask, face_mask],-1)

    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255 


    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    outImage = cv2.add(foreground, background)

#    outImage = cv2.cvtColor(np.uint8(outImage), cv2.COLOR_BGR2RGB)

    cv2.imwrite('output/' + image_name + '.jpg', outImage)
    
    return outImage  


def get_makeup_video(input_tgt, filename, video_name):
    
    img_tgt = cv2.imread(input_tgt )
    tgt_face, m = detector.get_face(img_tgt )
    tgt = cv2.cvtColor(tgt_face, cv2.COLOR_BGR2RGB)

    tgt = np.expand_dims(tgt,0)/ 127.5 -1

    # out.release()
    cv2.destroyAllWindows()
    
    size = (800, 450)
    frame_width,frame_height  = (800, 450)
    
    cap = cv2.VideoCapture(filename)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output/'+video_name+'.mp4', fourcc, 10, (frame_width, frame_height))


    while(cap.isOpened()):

        ret, frame = cap.read()
        start = time.clock()
        if ret == True:
            # Display the resulting frame
            ret, frame = cap.read()
            
#           print(type(frame))
            if(type(frame) == "NoneType"):
                break

            img_src  = imutils.resize(frame, width=800)

            height, width, ch = frame.shape

            size = (width,height)

            src_face, mm = detector.get_face(img_src)

            src = cv2.cvtColor(src_face, cv2.COLOR_BGR2RGB)

            src = np.expand_dims(src,0)/ 127.5 - 1

            start = time.clock()

            _, output = beautyGan.predict(src,tgt) # src:non makeup target: makup
            end = time.clock()

            output = (output + 1) * 127.5
            output = np.uint8(np.squeeze(output))

            im = cv2.invertAffineTransform(mm)
            output = cv2.warpAffine(output, im, size,flags=cv2.INTER_CUBIC)

            lip_mask, eye_mask, face_mask = beautyGan.get_mask(src_face, detector.detector, detector.predictor)

            lip_mask = lip_mask.astype(np.uint8)
            face_mask = face_mask.astype(np.uint8)
            eye_mask = eye_mask.astype(np.uint8)

            kernel = np.ones((3,3),np.uint8)
            face_mask = cv2.erode(face_mask,kernel,iterations = 5)
            # face_mask = cv2.erode(face_mask,kernel,iterations = 3)

            mask = np.clip(lip_mask * 5.0+ 0.0*eye_mask + face_mask * 0.0 , 0, 255)


            mask = mask.astype(np.uint8)

            face_mask = cv2.warpAffine(mask, im, size,flags=cv2.INTER_CUBIC)
            face_mask = cv2.blur(face_mask,(10,10))

            foreground = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            background = img_src
            alpha = np.stack([face_mask, face_mask, face_mask],-1)

            print(alpha.shape)
            print(background.shape)

            foreground = foreground.astype(float)
            background = background.astype(float)

            # Normalize the alpha mask to keep intensity between 0 and 1
            alpha = alpha.astype(float)/255 

            foreground = cv2.multiply(alpha, foreground)
            background = cv2.multiply(1.0 - alpha, background)

            outImage = np.uint8(cv2.add(foreground, background))

#            outImage = cv2.cvtColor(np.uint8(outImage), cv2.COLOR_BGR2RGB)

            out.write(outImage)
            end = time.clock()
            print(end-start)

            # Break the loop
        else: 
            break
            cap.release()


    out.release()
    cv2.destroyAllWindows() 



    # plt.figure(figsize=(20,10))
    # plt.imshow(frame)
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    print('1. Which make up style do you choose?')
    num = input('   Input the style number.')
    input_name = input('2. Input the name of input image.')
    output_name = input('3. Input the name of output image.')
    flag = input('3. Do you want to get your face?(yes/no)')
    if(flag=="no"):
        tf.reset_default_graph()
        beautyGan = BeautyGAN(prediction=True , check_dir='checkpoints_ori')
        detector = FaceDetector()
        
        
        input_tgt = './makeup/images'+ str(num) + '.jpg'
        input_src = 'output/'+ str(input_name) +'.jpg'
        image_name = str(output_name)
        
        outImage = get_makeup_image(input_src, input_tgt, image_name)
    
        input_tgt = './makeup/images'+ str(num) + '.jpg'
        filename = 'output/'+ str(input_name) +'.mp4'
        video_name = output_name
        
        get_makeup_video(input_tgt, filename, video_name)

    else:
        print('Then let me get your face.(1~5)')
        print('Explanation:')
        print('Touch "s" when you choose your face.')
        print('Touch "q" when you finish session.')
        print('You can get the results in output directry.')

        permission = input('Are you OK? (yes)')
        if(permission == 'yes'):
            cap = cv2.VideoCapture(0)
            fps = 30

            # 録画する動画のフレームサイズ（webカメラと同じにする）
            frame_width,frame_height  = (800, 450)
            # 出力する動画ファイルの設定
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter('output/'+ input_name +'.mp4', fourcc, 10, (frame_width, frame_height))
            
            
            while (cap.isOpened()):
                ret, frame = cap.read()
                frame = cv2.resize(frame, dsize=(frame_width, frame_height))
                # 画面表示
                cv2.imshow('frame',frame)
             
                # 書き込み
                video.write(frame)
                
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    cv2.imwrite('output/'+input_name+'.jpg', frame)
                # キー入力待機
                if key == ord('q'):
                    break

            # 終了処理
            cap.release()
            video.release()
            cv2.destroyAllWindows()


            tf.reset_default_graph()
            beautyGan = BeautyGAN(prediction=True , check_dir='checkpoints_ori')
            detector = FaceDetector()

            input_tgt = './makeup/images'+ str(num) + '.jpg'
            input_src = 'output/'+ str(input_name) +'.jpg'
            image_name = output_name

            outImage = get_makeup_image(input_src, input_tgt, image_name)


            input_tgt = '/makeup/images'+ str(num) + '.jpg'
            filename = 'output/'+ str(input_name)+'.mp4'
            video_name = output_name
            get_makeup_video(input_tgt, filename, video_name)
