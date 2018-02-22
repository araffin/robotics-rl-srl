from pytorch_agents.visualize import visdom_plot, episode_plot
from visdom import Visdom

VISDOM_PORT = 8097
LOG_INTERVAL = 10
LOG_DIR = "logs/"
ALGO = ""
ENV_NAME = ""
PLOT_TITLE = ""
EPISODE_WINDOW = 20
viz = None
n_steps = 0

win, win_smooth, win_episodes = None, None, None


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN) or after n steps (see ACER)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global win, win_smooth, win_episodes, n_steps, viz
    if viz is None:
        viz = Visdom(port=VISDOM_PORT)

    if (n_steps + 1) % LOG_INTERVAL == 0:
        win = visdom_plot(viz, win, LOG_DIR, ENV_NAME, ALGO, bin_size=1, smooth=0, title=PLOT_TITLE)
        win_smooth = visdom_plot(viz, win_smooth, LOG_DIR, ENV_NAME, ALGO, title=PLOT_TITLE + " smoothed")
        win_episodes = episode_plot(viz, win_episodes, LOG_DIR, ENV_NAME, ALGO, window=EPISODE_WINDOW,
                                    title=PLOT_TITLE + " [Episodes]")
    n_steps += 1
    return False
