import os
from subprocess import Popen, call
import sys
import json

def classify_images():
  for filename in os.listdir('unlabeled'):
    viewer = Popen(['open', 'unlabeled/' + filename])
    # Get amount of floating vegetation
    floating = ''.join(input('Amount of floating vegetation? (1-9)\n').split())
    while not floating.isdigit():
      print('Sorry, invalid response. Please try again.')
      floating = ''.join(input('(1) floating\n(2) side\n(3) placeholder\n(4) another placeholder\n').split())
    
    # Get amount of emergent vegetation
    emergent = ''.join(input('Amount of emergent vegetation? (1-9)\n').split())
    while not emergent.isdigit():
      print('Sorry, invalid response. Please try again.')
      emergent = ''.join(input('(1) floating\n(2) side\n(3) placeholder\n(4) another placeholder\n').split())
  
    copy_command = 'cp unlabeled/' + filename + ' labeled/' + floating + '-' + emergent + '-' + filename
    call(copy_command.split())



def main():
  print('Welcome to the vegetation classifier!')
  print('After each image appears, enter the percentage of the image covered by floating and emergent vegetation.')
  input("Press enter to begin: ")

  classify_images()


if __name__ == '__main__':
  main()

