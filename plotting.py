import matplotlib.pyplot as plt
import numpy as np
def plot_losses(all_results_array):

    # Assuming all_results_array is a numpy array
    all_results_array = np.array(all_results_array)

    fig, ax1 = plt.subplots()

    # Plot the Total Return on the left y-axis
    ax1.plot(all_results_array[:, 2], label="Value Loss", color='b')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Value Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(all_results_array[:, 3], label="Policy Loss", color='r')
    ax2.set_ylabel('Policy Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adding legends to both axes
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()

def plot_rewards(all_results_array):
    # Assuming all_results_array is a numpy array
    all_results_array = np.array(all_results_array)

    fig, ax1 = plt.subplots()

    # Plot the Total Return on the left y-axis
    ax1.plot(all_results_array[:, 0], label="Total Return", color='b')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Return', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(all_results_array[:, 1], label="Max Return", color='r')
    ax2.set_ylabel('Max Return', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adding legends to both axes
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()