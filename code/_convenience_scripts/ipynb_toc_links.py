# Sebastian Raschka, 2015
# convenience function for myself to add internal links to IPython toc
# use as `python ipynb_toc_links.py /blank_tocs/ch01.toc`

import sys

ipynb = sys.argv[1]
with open(ipynb, 'r') as f:
    for line in f:
        out_str = ' ' * (len(line) - len(line.lstrip()))
        line = line.strip()
        out_str += '- [%s' % line
        out_str += '](#%s)' % line.replace(' ', '-')
        print(out_str)
