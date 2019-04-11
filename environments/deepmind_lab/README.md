Installation instructions.

We suppose you have installed SRL-Toolbox using the anaconda environment provided by the file environment.yml.

We follow the installation guidelines provided by the DeepMind Lab developers https://github.com/deepmind/lab/tree/master/python/pip_package

We first install the DM-Lab repo, and then the python package that we build using the DM-Lab repo.

Steps:

1. Follow installation guidelines in SRL-Toolbox README. 
You now have a conda environment which we will use to install DeepMind-Lab.

2. Install bazel. 

3. Install DM-Lab repo.
```
git clone https://github.com/deepmind/lab.git && cd lab
bazel build -c opt python/pip_package:build_pip_package
```

4. Change python.BUILD file by:
```
cc_library(
    name = "python",
    hdrs = glob(["include/python3.5m/*.h","lib/python3.5/site-packages/numpy/core/include/numpy/*.h"]),
    includes = ["include/python3.5m", "lib/python3.5/site-packages/numpy/core/include"],
    visibility = ["//visibility:public"],
    )
```
    
5. Install python DeepMind-Lab package.
```
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
```

Done.