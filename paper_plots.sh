#!/bin/bash

./fancy_plot.py 'SDSS-II SN 19778' SN2010ai -rl --notitle;
./fancy_plot.py SN2007on -rlp -e swift --notitle -m 100 -o figs/SN2007on_short.png;
./fancy_plot.py SN2007on SN2009gf -rl -e swift --notitle;
./fancy_plot.py SN2008hv -rlp -e swift --notitle;