import os
from subprocess import Popen, call
import sys
import pdb

def correctPath():
    path = os.getcwd()
    pathlist = path.split('\\')
    path = 'C:'
    for index in range(1,len(pathlist)):
        #if pathlist[index] == "Stanford Courses": path = path + '\\' + '"Stanford Courses"'
        path = path + '\\' + pathlist[index]
    return path

def classify_images():
  path = correctPath()
  path = path + '\\unlabeled\\'
  for filename in os.listdir(path):
    print(path + filename)
    viewer = Popen(['open', path  + filename])
    # Get amount of floating vegetation
    floating = ''.join(input('Amount of floating vegetation? (1-9)\n').split())
    while not floating.isdigit():
      print('Sorry, invalid response. Please try again.')
      floating = ''.join(input('Amount of floating vegetation? (1-9)\n').split())

    # Get amount of emergent vegetation
    emergent = ''.join(input('Amount of emergent vegetation? (1-9)\n').split())
    while not emergent.isdigit():
      print('Sorry, invalid response. Please try again.')
      emergent = ''.join(input('Amount of emergent vegetation? (1-9)\n').split())

    copy_command = 'cp unlabeled/' + filename + ' labeled/' + floating + '-' + emergent + '-' + filename
    call(copy_command.split())

def main():
  print('Welcome to the vegetation classifier!')
  print('After each image appears, enter the percentage of the image covered by floating and emergent vegetation.')
  input("Press enter to begin: ")

  classify_images()


if __name__ == '__main__':
  main()
