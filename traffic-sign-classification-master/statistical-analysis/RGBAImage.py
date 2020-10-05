import statistics

"""
This is a class for analysing RGBA images
"""

class RGBAImage:
  def __init__(self, pixelValue): # pass in list of pixels(tuples)
    self.pixelValue = pixelValue
    self.r = []
    self.g = []
    self.b = []

  def getPixelValue(self):
    return self.pixelValue

  def getRGBA(self, index):
    result = []
    for pixel in self.pixelValue:
      result += [pixel[index]]
    return result
  
  def getRedArray(self):
    return self.getRGBA(0)

  def setRedArray(self):
    self.r = self.getRGBA(0)
  
  def getGreenArray(self):
    return self.getRGBA(1)
  
  def setGreenArray(self):
    self.g = self.getRGBA(1)
  
  def getBlueArray(self):
    return self.getRGBA(2)
  
  def setBlueArray(self):
    self.b = self.getRGBA(2)
  
  def setAll(self):
    self.setRedArray()
    self.setGreenArray()
    self.setBlueArray()

  def getMaxRGBA(self):
    return [max(self.r), max(self.g), max(self.b)]
  
  def getMinRGBA(self):
    return [min(self.r), min(self.g), min(self.b)]

  def getMeanRGBA(self):
    return [statistics.mean(self.r), statistics.mean(self.g), statistics.mean(self.b)]
