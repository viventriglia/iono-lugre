#!/bin/sh
# Build the standalone GCPM driver used by recompute_fig3.py.
# usage: sh build.sh <copy of PyGCPM's libgcpm source dir (has iri/ xform/ gcpm/ libgcpm.c makefile)> [722|1024]
# The source dir is modified in place: work on a copy of PyGCPM c2c092a.
set -e
SRC=$1; N=${2:-1024}; HERE=$(cd "$(dirname "$0")" && pwd)
cd "$SRC"
# IRI tcon index arrays: 722 overflows with any ig_rz.dat newer than ~2017
sed -i "s/ionoindx([0-9]*),indrz([0-9]*)/ionoindx($N),indrz($N)/" iri/irifun.for
# setLibPath allocates strlen bytes without room for the terminator
sed -i 's/malloc(sizeof(char)\*l)/malloc(l+1)/' libgcpm.c
rm -f iri/*.o xform/*.o gcpm/*.o libgcpm.o
# same flags as PyGCPM's makefiles, without needing make
F="gfortran -w -c -fPIC -fno-automatic"
(cd iri && $F -std=legacy *.for)
(cd xform && $F *.for *.FOR)
(cd gcpm && $F *.for)
gcc -c -fPIC libgcpm.c -o libgcpm.o
gfortran -fno-automatic -o "$HERE/drv_sm$N" "$HERE/drv_sm.c" libgcpm.o iri/*.o xform/*.o gcpm/*.o -lm
echo "built $HERE/drv_sm$N"
