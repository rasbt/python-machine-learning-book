# Sebastian Raschka, 2015
# convenience function for myself to create nested TOC lists
# use as `python md_toc.py /blank_tocs/ch01.toc`

import sys

ipynb = sys.argv[1]
with open(ipynb, 'r') as f:
    for line in f:
        out_str = ' ' * (len(line) - len(line.lstrip()))
        line = line.strip()
        out_str += '- %s' % line
        print(out_str)
