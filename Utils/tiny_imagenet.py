import os
import urllib
import zipfile


def download_dataset():
  print('Beginning dataset download with urllib2')
  url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
  path = "%s/imagenet.zip" % os.getcwd()
  urllib.request.urlretrieve(url, path)
  print("Dataset downloaded")


def unzip_data():
  path_to_zip_file = "%s/imagenet.zip" % os.getcwd()
  directory_to_extract_to = "./Models/data"
  if not os.path.exists(directory_to_extract_to):
    os.makedirs(directory_to_extract_to)
  print("Extracting zip file: %s" % path_to_zip_file)
  with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)
  print("Extracted at: %s" % directory_to_extract_to)


def format_val():
  val_dir = ".Models/data/imagenet/val"
  print("Formatting: %s" % val_dir)
  val_annotations = "%s/val_annotations.txt" % val_dir
  val_dict = {}
  with open(val_annotations, 'r') as f:
    for line in f:
      line = line.strip().split()
      assert(len(line) == 6)
      wnind = line[1]
      img_name = line[0]
      boxes = '\t'.join(line[2:])
      if wnind not in val_dict:
          val_dict[wnind] = []
      entries = val_dict[wnind]
      entries.append((img_name, boxes))
  assert(len(val_dict) == 200)
  for wnind, entries in val_dict.items():
    val_wnind_dir = "%s/%s" % (val_dir, wnind)
    val_images_dir = "%s/images" % val_dir
    val_wnind_images_dir = "%s/images" % val_wnind_dir
    os.mkdir(val_wnind_dir)
    os.mkdir(val_wnind_images_dir)
    wnind_boxes = "%s/%s_boxes.txt" % (val_wnind_dir, wnind)
    f = open(wnind_boxes, "w")
    for img_name, box in entries:
      source = "%s/%s" % (val_images_dir, img_name)
      dst = "%s/%s" % (val_wnind_images_dir, img_name)
      os.system("cp %s %s" % (source, dst))
      f.write("%s\t%s\n" % (img_name, box))
    f.close()
  os.system("rm -rf %s" % val_images_dir)
  print("Cleaning up: %s" % val_images_dir)
  print("Formatting val done")
