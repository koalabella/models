import tensorflow as tf
import yaml
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(example):
    #trainning data variables

    height = 600 # Image height
    width = 800 # Image width
    filename = example['path'] # Filename of the image. Empty if image is not from file
    #filename = filename.encode()

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image_data = filename.encode() # Encoded image bytes
    image_format = 'jpg'.encode() # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                 # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['boxes']:
        xmins.append(float(box['xmin'] / width))
        xmaxs.append(float((box['xmax'] + box['x_width']) / width))
        ymins.append(float(box['ymin'] / height))
        ymaxs.append(float((box['ymax'] + box['y_height']) / width))
        classes_text.append(box['class'].encode())
        classes.append(int(LABEL_DICT[box['class']]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # read in udacity dataset to examples variable
    input_yaml = "/home/ubuntu/hdd/test_data/train.yaml"
    imgs = yaml.load(open(input_yaml, 'rb').read())
    examples = []
    for i in range(len(imgs)):
        img_path = imgs[i]['path']
        img_path = os.path.abspath(os.path.join(os.path.dirname(input_yaml), img_path))
        examples.append({'path': img_path})

    counter = 0
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

        if counter % 10 == 0:
            print("Percent done", (counter/len_examples)*100)
        counter += 1

    writer.close()


if __name__ == '__main__':
    tf.app.run()
