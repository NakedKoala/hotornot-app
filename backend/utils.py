import cv2 
import boto3 

def img_preparation(img):

    img = cv2.resize(img,(350,350))
    img = img / 255.0
    img_array = img.reshape(1,350,350,3)
    return img_array

def detect_face(img, face_classifier):

    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_classifier.detectMultiScale(gray_scale_img , 1.3, 5)
   

def face_close_crop(img, faces_coord, bounding_offset=15):
    
    x, y, w, h = faces_coord[0]
    x_start = y - bounding_offset  if y - bounding_offset >= 0  else y
    x_end = y+h+bounding_offset  if y+h+bounding_offset <= img.shape[0]  else y+h
    y_start =  x - bounding_offset if x - bounding_offset >= 0 else x
    y_end = x+w+bounding_offset  if x+w+bounding_offset <= img.shape[1] else x+w
    
    face_crop = img[x_start:x_end, y_start:y_end, :]
    file_name =  'face_crop_tmp.jpg'
    cv2.imwrite(file_name, face_crop)
    return face_crop

def upload_crop_to_s3(s3, file_name="face_crop_tmp.jpg", bucket_name="hotornot-bucket"):

    s3.meta.client.upload_file('./'+ file_name, bucket_name, file_name, ExtraArgs={'ACL':'public-read'})
    bucket_location = boto3.client('s3').get_bucket_location(Bucket=bucket_name)
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
        bucket_location['LocationConstraint'],
        bucket_name,
        file_name)
    return object_url