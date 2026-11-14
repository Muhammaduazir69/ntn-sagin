#!/bin/bash
# Standalone build of Paper 6's binary; touches nothing under ns-3-dev and links its existing optimized libraries.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
B=/home/uzair/6g_ntn_ns3/ns-3-dev/build
OUT="$HERE/../bin"
mkdir -p "$OUT"
FLAGS="-DEIGEN_MPL2_ONLY -DHAVE_BOOST -DHAVE_BOOST_UNITS -DHAVE_EIGEN3 -DHAVE_GSL -DHAVE_LIBXML2 -DHAVE_PACKET_H -DHAVE_SQLITE3 -DNS3_BUILD_PROFILE_OPTIMIZED -D__LINUX__ -I$B/include -I/usr/include/eigen3 -I/usr/include/libxml2 -O3 -DNDEBUG -std=c++20 -fPIE -fno-semantic-interposition -Wall -march=native -mtune=native"
LIBS=""
for m in ntn-sagin ntn-constellation core network mobility point-to-point internet applications; do LIBS="$LIBS $B/lib/libns3.43-$m-optimized.so"; done
nice -n 10 g++ $FLAGS -c "$HERE/sagin-store-forward-iot.cc" -o "$OUT/sf.o"
nice -n 10 g++ -o "$OUT/p6-sf-iot.new" "$OUT/sf.o" $LIBS -Wl,-rpath,$B/lib -Wl,-rpath-link,$B/lib
mv -f "$OUT/p6-sf-iot.new" "$OUT/p6-sf-iot"; rm -f "$OUT/sf.o"
sha256sum "$OUT/p6-sf-iot"
