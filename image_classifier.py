import os
import subprocess
import sys
import json

def classifyImages():
  answers = {}
  for filename in os.listdir('./assets/jpgs/'):
    viewer = subprocess.Popen(['open', './assets/jpgs/' + filename])
    text = ''.join(raw_input('(1) floating\n(2) side\n(3) placeholder\n(4) another placeholder\n').split())
    while not text.isdigit():
      print('Sorry, invalid response. Please try again.')
      text = ''.join(raw_input('(1) floating\n(2) side\n(3) placeholder\n(4) another placeholder\n').split())
    answers[filename] = text
  
  return answers


def main():
  print('Welcome to the vegetation classifier!')
  print('After each image appears, enter a number (or multiple numbers) to classify the type of vegetation.')
  print('The types of vegetation are:')
  print('\t(1) floating\n\t(2) side\n\t(3) placeholder\n\t(4) another placeholder')
  raw_input("Press enter to begin: ")

  labels = classifyImages()

  labels = json.dumps(labels)
  fh = open('labels.json','w')
  fh.write(str(labels))
  fh.close()


if __name__ == '__main__':
  main()

