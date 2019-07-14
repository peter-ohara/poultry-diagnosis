# import the necessary packages
import argparse
import os
import warnings

import requests
from PIL import Image
from imutils import paths


def download_images_from_url_file(urls_file, output_dir):
    # grab the list of URLs from the input file, then initialize the
    # total number of images downloaded thus far

    if not (isinstance(urls_file, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(urls_file):
        raise OSError('img_source_dir does not exist')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = open(urls_file).read().strip().split("\n")
    total = 0

    # loop the URLs
    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)

            # save the image to disk
            p = os.path.sep.join([output_dir, f"{str(total).zfill(8)}.jpg"])
            f = open(p, "wb")
            f.write(r.content)
            f.close()

            # update the counter
            print("[INFO] downloaded: {}".format(p))
            total += 1

        # handle if any exceptions are thrown during the download process
        except (IOError, SyntaxError) as e:
            print(f"[INFO] Error downloading '{p}'... Skipping... {type(e).__name__}: {e}")


def cleanup_images(output_dir):
    if not (isinstance(output_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(output_dir):
        raise OSError('img_source_dir does not exist')

    # loop over the image paths in output_dir
    for imagePath in paths.list_images(output_dir):
        try:
            with warnings.catch_warnings(record=True) as _w:

                # Cause all warnings to be treated as errors
                warnings.simplefilter("error")

                # open and verify image file
                img = Image.open(imagePath)
                img.verify()

                # Open and convert image to jpg if it's not already jpg
                img = Image.open(imagePath)
                if img.format != 'JPEG':
                    print(f"Converting '{imagePath}' to .jpg")

                    # Convert image
                    img = img.convert('RGB')

                    # Save jpg file
                    imagePathSansExt = imagePath.split('.')[0]
                    img.save(f"{imagePathSansExt}.jpg")

                    # Delete original file
                    os.remove(imagePath)

        # If any of the above fails, the file is corrupt so delete it
        except (IOError, SyntaxError) as e:
            # Delete the image
            print(f"Deleting '{imagePath}'... {type(e).__name__}: {e}")
            os.remove(imagePath)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--urls", required=True,
                    help="path to file containing image URLs")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output directory of images")
    args = vars(ap.parse_args())

    download_images_from_url_file(args["urls"], args["output"])
    cleanup_images(args["output"])
