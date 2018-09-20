#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line


docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/tmp/rl_toolbox,type=bind araffin/rl-toolbox-cpu\
  bash -c "source activate py35 && cd /tmp/rl_toolbox/ && $cmd_line"
