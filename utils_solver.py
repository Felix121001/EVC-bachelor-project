


def smooth_f0(f0, window_size=5):
    """
    Smooths the F0 contour using a moving average filter.
    
    Parameters:
    f0 (numpy array): The F0 contour array.
    window_size (int): The size of the moving average window. Must be an odd integer.
    
    Returns:
    numpy array: The smoothed F0 contour.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd integer")

    half_window = window_size // 2
    f0_padded = np.pad(f0, (half_window, half_window), mode='reflect')
    f0_smooth = np.convolve(f0_padded, np.ones(window_size) / window_size, mode='valid')

    return f0_smooth




class TrackLosses:
    def __init__(self, loss_names):
        self.loss_names = loss_names
        self.losses = {loss_name: [] for loss_name in loss_names}

    def __getitem__(self, loss_name):
        return self.losses[loss_name]

    def update(self, loss_values):
        for loss_name, loss_value in zip(self.loss_names, loss_values):
            self.losses[loss_name].append(loss_value)

    def reset(self):
        for loss_name in self.loss_names:
            self.losses[loss_name] = []

    def get_last_average(self, loss_name, n=50):
        return np.mean(self.losses[loss_name][-n:])

    def plot(self, save_path=None):
        for loss_name in self.loss_names:
            plt.plot(self.losses[loss_name], label=loss_name)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()


class CustomLRScheduler:
    def __init__(self, optimizer, base_lr, decay_factor, num_steps, min_lr):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.num_steps = num_steps
        self.current_step = 0
        self.min_lr = min_lr

    def step(self):
        self.current_step += 1
        lr = self.base_lr * (self.decay_factor ** (self.current_step / self.num_steps))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = max(
                lr, self.min_lr
            )  # Ensure lr doesn't go below a threshold

    def reset(self):
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base_lr
