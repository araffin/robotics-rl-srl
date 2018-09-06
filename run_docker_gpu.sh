#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Excuting in the docker (gpu image):"
echo $cmd_line


docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/tmp/rl_toolbox,type=bind araffin/rl-toolbox\
  bash -c "source activate py35 && cd /tmp/rl_toolbox/ && $cmd_line"
