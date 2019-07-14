import uuid

from cross_validation import img_train_test_split
from datasets import download_images_from_url_file, cleanup_images

normal_urls_file = 'normal_urls.txt'
abnormal_urls_file = 'abnormal_urls.txt'

img_source_dir = f'raw_data_{uuid.uuid4().hex[:8]}'
normal_images_dir = f'{img_source_dir}/normal'
abnormal_images_dir = f'{img_source_dir}/abnormal'

test_size = 0.2

# Download normal images from google
download_images_from_url_file(normal_urls_file, normal_images_dir)

# Download abnormal images from google
download_images_from_url_file(abnormal_urls_file, abnormal_images_dir)


# Delete corrupt files
cleanup_images(normal_images_dir)
cleanup_images(abnormal_images_dir)

# # Split into training, validation and testing
img_train_test_split(img_source_dir, test_size, train_dir='train', test_dir='validation_and_test')
img_train_test_split('data/validation_and_test', 0.5, train_dir='validation', test_dir='test')


