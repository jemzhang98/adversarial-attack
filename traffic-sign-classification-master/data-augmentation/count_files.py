import os, os.path
from os.path import join
from os import walk
from shutil import copyfile
from PIL import Image


def countOfFiles():
  TrainOrTest = 'Train'
  for (dirpath, dirnames, filenames) in walk('../../Data/' + TrainOrTest):
    f = []
    for filename in filenames:
      rel_dir = os.path.relpath(dirpath)
      rel_file = os.path.join(rel_dir, filename)
      f.append(rel_file)
    print(dirpath, len(f))

def move_data(src_dir, dest_dir, folder_name, count):
  if not os.path.exists(dest_dir): # if the directory does not exist
    os.makedirs(dest_dir)
    files = os.listdir(src_dir)[:count]
    for file in files:
      copyfile(join(src_dir, file), join(dest_dir, file))

def generate_data (src_dir, dest_dir, folder_name, count):
  files = os.listdir(src_dir)
  if not os.path.exists(dest_dir): # if the directory does not exist
    os.makedirs(dest_dir)
  missing_batch = count//len(files)
  print(folder_name, len(files))
  print('batch', missing_batch)

  if (missing_batch == 0):
    print('one missing')
    print(count)
    for i in range(count): #generate certain number of images
      for file in files:
        if (count > 0): # if still need to generate more images
          print(count)
          img = Image.open(join(src_dir, file))
          rotate90 = img.rotate(90)
          rotate90.save(join(dest_dir, folder_name + '_x' + file[-8:]))
          count -= 1

  elif (missing_batch == 1):
    print('two missing')
    count_to_generate = count - len(files)
    for file in files:
      img = Image.open(join(src_dir, file))
      rotate90 = img.rotate(90)
      rotate90.save(join(dest_dir, folder_name + '_x' + file[-8:]))
    for i in range(count_to_generate): #generate certain number of images
      for file in files:
        if (count_to_generate > 0): # if still need to generate more images
          img = Image.open(join(src_dir, file))
          rotate180 = img.transpose(Image.ROTATE_180)
          rotate180.save(join(dest_dir, folder_name + '_y' + file[-8:]))
          count_to_generate -= 1
  elif (missing_batch == 2):
    print('three missing')
    count_to_generate = count - len(files) * 2
    for file in files:
      img = Image.open(join(src_dir, file))
      rotate90 = img.rotate(90)
      rotate90.save(join(dest_dir, folder_name + '_x' + file[-8:]))
      rotate180 = img.transpose(Image.ROTATE_180)
      rotate180.save(join(dest_dir, folder_name + '_y' + file[-8:]))
    for i in range(count_to_generate):
      for file in files:
        if (count_to_generate > 0): # if still need to generate more images
          img = Image.open(join(src_dir, file))
          rotate270 = img.transpose(Image.ROTATE_270)
          rotate270.save(join(dest_dir, folder_name + '_z' + file[-8:]))
          count_to_generate -= 1


def augment_data(folder_name):
  path = '../../Data/Train/' + folder_name
  threshold = 400
  src = path
  dest = join('../../Augmented_Data/Train/' + folder_name)
  if (len(os.listdir(path)) > threshold):
    move_data(src, dest, folder_name, threshold)
  elif (len(os.listdir(path)) < threshold):
    move_data(src, dest, folder_name, len(os.listdir(path)))
    generate_data(src, dest, folder_name, threshold - len(os.listdir(path)))


for folder in os.listdir('../../Data/Train'):
  augment_data(folder)
# augment_data('w57')
