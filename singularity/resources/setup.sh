#!/bin/bash

./waf configure --exp aurora --cpp14=yes --dart /workspace --kdtree /workspace/include --robot_dart /workspace \
--corrade_install_dir /workspace --magnum_integration_install_dir /workspace --magnum_plugins_install_dir /workspace \
--magnum_install_dir /workspace "$@" --robox2d /workspace

./waf --exp aurora -j 6 "$@"
echo 'FINISHED BUILDING. Now fixing name of files'
python -m fix_build --path-folder /git/sferes2/build/exp/aurora/
