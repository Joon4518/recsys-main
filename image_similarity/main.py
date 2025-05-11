import logging
import argparse
import tensorflow as tf
import boto3
import numpy as np
import os

from PIL import Image
from src.preprocessing import Similarity
from src import load_model, cassandra_api
import boto3

boto3.setup_default_session(
    aws_access_key_id='id',
    aws_secret_access_key='key',
    region_name='us-east-1'
)
#############################

        # load all user_id
        # run double for loop to calculate profile similarity
        # end of each for loop write similarity results to cassandra db

        # keyspaces
        # "newnyup@gmail.com"
        # "tlaznd@0801"

#############################
def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    input_image = tf.cast(image, tf.float32)
    
    return input_image

def resize(input_image, height, width):
    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    
    return input_image

def normalize(input_image):
    input_image = (input_image / 127.5) - 1
    
    return input_image

def load_image(user_id, image_path):
    #id = tf.strings.split(image_file, "/")[-1]
    #id = tf.strings.split(id, ".")[0]
    input_image = load(image_path)
    input_image = resize(input_image, 224, 224)
    input_image = normalize(input_image)
    
    return (user_id, input_image)

def image_download(image_path_dict):

    s3_path = image_path_dict["image_path"]
    local_path = 'images/'+image_path_dict["user_name"] +'.jpg'  # 로컬에 저장할 파일 경로


    # Parse S3 path to get bucket and object key
    s3_components = s3_path.replace('s3://', '').split('/')
    bucket_name = s3_components[0]
    object_key = '/'.join(s3_components[1:])

    # Download image from S3 and save locally
    s3 = boto3.resource('s3', region_name='us-east-1')  # Replace 'us-east-1' with your desired region
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(object_key)

    object.download_file(local_path)

def get_args():
    parser = argparse.ArgumentParser(description="Train the kc-electra model")

    # 위치 기반 인자
    parser.add_argument(
        "is_cpu", type=bool, help="Set CPU, default: True"
    )

    # 리스트 타입의 인자
    parser.add_argument(
        "cassandra_address", nargs='+', help="Set cassandra address"
    )

    # 정수 타입의 인자
    parser.add_argument(
        "cassandra_port", type=int,  default=9142, help="Set cassandra port"
    )

    parser.add_argument(
        "batch_size", type=int, default=1, help="Set batch size, default: 1"
    )

    parser.add_argument(
        "nearest_neighbors", type=int, default=10, help="Set num of nearest neighbors, default: 10"
    )

    parser.add_argument(
        "top_k", type=int, default=20, help="Set top k recommendation number, default: 20"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Setting Cassandra DB, Address:{args.cassandra_address}, Port:{args.cassandra_port}")

    cassandra_obj = cassandra_api.Cassandra()

    device = '/CPU:0' if args.is_cpu else '/GPU:0'
    logging.info(f"Using device {device}")

    all_user_id_image = cassandra_obj.load_all_user()
    print(all_user_id_image)
    #result_dicts = [{"user_name": row.user_name, "image_path":row.image_path} for row in all_user_id_image]

    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')


    res = []
    for r in all_user_id_image:
        
        local_path = image_download(r)
        res.append((r['user_name'], 'images/'+r["user_name"] +'.jpg'))
        print(r['user_name'] + " "+'images/'+r["user_name"] +'.jpg')
     


     
    print("-------------------------이미지 불러오기 테스트----------------------")
    print("---------res-----")
    print(res)
    print("---------res-----")

    #image_dataset = tf.dataset.experimental.from_list(res)
    #image_dataset = tf.dataset.experimental.from_list(all_user_id_image)
    # image_dataset = image_dataset.map(
    #     load_image,
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    #image_dataset = tf.dataset.experimental.from_list(res)
    # res = [('test', 'images/test.jpg'), ('test2', 'images/test2.jpg')]
    # image_dataset = tf.data.Dataset.from_tensor_slices(res)
    # print("testing dataset------ ")
    # print(image_dataset)

    # image_dataset = tf.data.Dataset.from_tensor_slices(res)

 
    # for image_data in image_dataset:
    #     image_dataset_list
    image_dataset_temp = []
    for user_id, image_path in res:
        image_dataset_temp.append(load_image(user_id,image_path))
    
    # user_id와 이미지 텐서를 별도의 리스트로 분리
    user_ids = [user_id for user_id, _ in image_dataset_temp]
    image_tensors = [image for _, image in image_dataset_temp]

    # 별도의 데이터셋 생성
    user_id_dataset = tf.data.Dataset.from_tensor_slices(user_ids)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_tensors)

    # 두 데이터셋을 결합
    final_dataset = tf.data.Dataset.zip((user_id_dataset, image_dataset))

    # image_dataset = image_dataset.map(
    #     load_image,
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )

    # print(image_dataset)
    print(final_dataset)
    VGG = load_model.VGG((224, 224, 3))

    with tf.device(device):
        # feature_vector = final_dataset.map(
        #     lambda user_id, image: (user_id, VGG(None,image)),
        #     num_parallel_calls=tf.data.AUTOTUNE)
        # final_dataset = tf.data.Dataset.zip((user_id_dataset, image_dataset))
        # feature_vector = [[VGG(tensor_image), tensor_user_id] for tensor_user_id, tensor_image in final_dataset]
        
        def apply_model(user_id, image):
            # 이미지에 배치 차원 추가
            image = tf.expand_dims(image, axis=0)
            # 모델 적용
            processed_image = VGG(image)
            # 배치 차원 제거
            processed_image = tf.squeeze(processed_image, axis=0)
            return user_id, processed_image
            
        # 데이터셋에 모델 적용
        feature_vectors = final_dataset.map(
            apply_model, num_parallel_calls=tf.data.AUTOTUNE
        )

        
        
        print("Feature vectors created:", feature_vectors)

        # prep = Similarity(
        #     n_components=args.n_components,
        #     batch_size=args.batch,
        # )


        feature_vectors_np = []
        for _, feature_vector in feature_vectors:
            feature_vectors_np.append(feature_vector.numpy())

        # NumPy 배열로 변환
        feature_vectors_np = np.array(feature_vectors_np)



      
        


        
        # print(type(feature_vectors_np))  
        # print(feature_vectors_np.shape)
        prep = Similarity()
        # prep.fit_ipca(feature_vectors_np)

        # print("IPCA fitting completed")

        # batch_knn = prep.make_batch_knn(feature_vectors_np, args.nearest_neighbors)
        batch_knn = prep.make_batch_knn(feature_vectors, 5)

        print("Batch KNN created")
        batch_knn = (10,10)
        result = prep.fit_knn(batch_knn, feature_vectors, 10)

        print(result)
        

        


