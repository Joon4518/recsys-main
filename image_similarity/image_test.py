import tensorflow as tf

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



if __name__ == "__main__":

    image = load('images/test.jpg')
    print(image)


    res = [('test', 'images/test.jpg'), ('test2', 'images/test2.jpg')]
    image_dataset = tf.data.Dataset.from_tensor_slices(res)
    image_dataset_temp = []
    for user_id, image_path in res:
        image_dataset_temp.append(load_image(user_id,image_path))

    print(image_dataset_temp)

    # image_dataset = image_dataset.map(
    #     load_image,
    #     num_parallel_calls=tf.data.AUTOTUNE
    # )