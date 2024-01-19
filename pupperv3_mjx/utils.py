from datetime import datetime
import matplotlib.pyplot as plt


def progress(
    num_steps: int,
    metrics: dict,
    times: list,
    x_data: list,
    y_data: list,
    ydataerr: list,
    train_fn,
    min_y: float,
    max_y: float,
):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()
