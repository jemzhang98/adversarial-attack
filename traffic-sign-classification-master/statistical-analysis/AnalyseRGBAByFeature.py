from PIL import Image
from RGBAImage import *
import math
import statistics
from os import walk
import os
import matplotlib.pyplot as plt


def getPixelInfo(image_path):

  im = Image.open(image_path)
  pixel_values = list(im.getdata())

  image = RGBAImage(pixel_values)
  image.setAll()
  maxPixelValues = image.getMaxRGBA()
  minPixelValues = image.getMinRGBA()
  meanPixelValue = image.getMeanRGBA()
  
  #print ({'Maximum Pixel Values': maxPixelValues, 'Minimum PixelValues': minPixelValues, 'Mean Pixel Values': meanPixelValue})
  return (maxPixelValues, minPixelValues, meanPixelValue)


def getAllPng(folderName):
  f = []
  for (dirpath, dirnames, filenames) in walk('./Train/' + folderName):
    for filename in filenames:
      rel_dir = os.path.relpath(dirpath)
      rel_file = os.path.join(rel_dir, filename)
      f.append(rel_file)
  f = f[1:]
  # overallRedMax = []
  # overallGreenMax = []
  # overallBlueMax = []
  # overallRedMin = []
  # overallGreenMin = []
  # overallBlueMin = []
  overallRedMean = []
  overallGreenMean = []
  overallBlueMean = []
  for path in f:
    maximum, minimum, avg = getPixelInfo(path)
    # overallRedMax.append(maximum[0])
    # overallGreenMax.append(maximum[1])
    # overallBlueMax.append(maximum[2])
    # overallRedMin.append(minimum[0])
    # overallGreenMin.append(minimum[1])
    # overallBlueMin.append(minimum[2])
    overallRedMean.append(avg[0])
    overallGreenMean.append(avg[1])
    overallBlueMean.append(avg[2])
  # maxRed = max(overallRedMax)
  # maxGreen = max(overallGreenMax)
  # maxBlue = max(overallBlueMax)
  # minRed = min(overallRedMin)
  # minGreen = min(overallGreenMin)
  # minBlue = min(overallBlueMin)


  # return [[maxRed, maxGreen, maxBlue], [minRed, minGreen, minBlue], [meanRed, meanGreen, meanBlue]]
  return [overallRedMean, overallGreenMean, overallBlueMean]

def main():
  folders = ['w57', 'pl5', 'pl40', 'p5', 'p26', 'p11', 'io', 'pl30', 'pl80', 'pn', 'po', 'p23', 'i4', 'i2', 'i5', 'ip', 'pne', 'pl50', 'pl60']
  meanRedsByFolder = []
  meanGreensByFolder = []
  meanBluesByFolder = []
  for folder in folders:
    means = getAllPng(folder) #pass in Train or Test
    r = means[0]
    g = means[1]
    b = means[2]
    meanRed = statistics.mean(r)
    meanGreen = statistics.mean(g)
    meanBlue = statistics.mean(b)
    picCount = len(r)
    print('Number of Images in Current Folder', picCount)
    meanRedsByFolder.append(meanRed)
    meanGreensByFolder.append(meanGreen)
    meanBluesByFolder.append(meanBlue)
    meanGraysByFolder = []
  for i in range(19):
      meanGraysByFolder.append(statistics.mean([meanRedsByFolder[i], meanBluesByFolder[i], meanGreensByFolder[i]]))
      

  print(len(meanRedsByFolder))
  plt.plot(folders, meanRedsByFolder, color='red', linewidth=0.8, label='red')
  plt.plot(folders, meanGreensByFolder, color='green', linewidth=0.8, label='green')
  plt.plot(folders, meanBluesByFolder, color='blue', linewidth=0.8, label='blue')
  plt.plot(folders, meanGraysByFolder, color='gray', linewidth=0.8, label='gray')
  plt.xlabel('Feature Classes')
  plt.ylabel('Pixel Value')
  plt.title('Pixel Values of Training Data By Features')
  plt.xticks(fontsize=7)
  plt.legend()
  plt.show()



main()

# x = [1,2 , 3]
# y = [2,3 , 4]
# plt.scatter(x, y, label='red', color='red', marker='1', s=30)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('scatter graph')
# plt.legend()
# plt.show()
