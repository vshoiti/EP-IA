#!/usr/bin/env python
import sys, re

# TRECHO ABAIXO RETIRADO DE: https://docs.python.org/2/library/htmlparser.html
from HTMLParser import HTMLParser

class MyHTMLParser(HTMLParser, object):

    isCurrentlyWriting = False
    writer = None
    
    def __init__(self, writer):
        super(MyHTMLParser, self).__init__()
        self.writer = writer

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self.isCurrentlyWriting = True

    def handle_endtag(self, tag):
        if tag == "body":
            self.isCurrentlyWriting = False
            self.writer.appendToFile('\n\n')

    def handle_data(self, data):
        if self.isCurrentlyWriting:
            self.writer.appendToFile(data + ' ')
# FIM DO TRECHO RETIRADO DE: https://docs.python.org/2/library/htmlparser.html

class fileWriter:
    def __init__(self, filename):
        self.filename = filename

    def appendToFile(self, data):
        with open(self.filename, 'a') as f:
            f.write(data)
   
# invocado com: python reutersPrep.py [arquivo entrada] [arquivo saida]
if (len(sys.argv) < 3):
    print "missing file names"
    sys.exit()

readFile = sys.argv[1]
writeFile = sys.argv[2]

# TRECHO ABAIXO RETIRADO DE: https://stackoverflow.com/questions/3277503/how-do-i-read-a-file-line-by-line-into-a-list
with open(readFile) as f:
    content = f.readlines() # le as linhas do arquivo e armazena em content

content = [x.strip(' \n') for x in content] # remove espacos e '\n's no inicio e fim de cada string
# FIM DO TRECHO RETIRADO DE: https://stackoverflow.com/questions/3277503/how-do-i-read-a-file-line-by-line-into-a-list

# instantiate the parser and fed it some HTML
parser = MyHTMLParser(fileWriter(writeFile))

for s in content:
    parser.feed(s)
