.. _plotting:

Plotting
=========

Plot Learning Curve
~~~~~~~~~~~~~~~~~~~

To plot a learning curve from logs in visdom, you have to pass path to
the experiment log folder:

::

   python -m replay.plots --log-dir /logs/raw_pixels/ppo2/18-03-14_11h04_16/

To aggregate data from different experiments (different seeds) and plot
them (mean + standard error). You have to pass path to rl algorithm log
folder (parent of the experiments log folders):

::

   python -m replay.aggregate_plots --log-dir /logs/raw_pixels/ppo2/ --shape-reward --timesteps --min-x 1000 -o logs/path/to/output_file

Here it plots experiments with reward shaping and that have a minimum of
1000 data points (using timesteps on the x-axis), the plot data will be
saved in the file ``output_file.npz``.

To create a comparison plots from saved plots (.npz files), you need to
pass a path to folder containing .npz files:

::

   python -m replay.compare_plots -i logs/path/to/folder/ --shape-reward --timesteps

Gather Results
~~~~~~~~~~~~~~

Gather results for all experiments of an enviroment. It will report mean
performance for a given budget.

::

   python -m replay.gather_results -i path/to/envdir/ --min-timestep 5000000 --timestep-budget 1000000 2000000 3000000 5000000 --episode-window 100
