






class Computing_Parameters:
    """
    Class gathering all the different parameters that can be chosen when calculating the features of physiological signals.

    + window_length : the epoch width (window size) in seconds, for which features should be calculated
    + window_step_size : the step size for the sliding window in seconds
    + hrv_threshold : the relative portion of IBIs that need to be present in a window for processing. Otherwise, the window is dropped. Adviced : 0.2
    + hrv_clean_data : whether obviously invalid IBIs should be removed before processing. Adviced : True
    + only_on_full_hour_slots : Subtlety, before computing the features, the physiological signals are truncated to be only on full hour slots.
        Example: a session from 10:30 to 14:15 will be truncated as follows, 11:00 to 14:00

    >>> Example : 
    computing_parameters = Computing_Parameters(
        window_length = 3600,
        window_step_size = 3600,
        hrv_threshold = 0.1,
        hrv_clean_data = False,
        only_on_full_hour_slots = True,
    )
    """
    def __init__(self, window_length : int, window_step_size : int, hrv_threshold : float, hrv_clean_data : bool, only_on_full_hour_slots : bool) -> None:
        self.window_length = window_length
        self.window_step_size = window_step_size
        self.hrv_threshold = hrv_threshold
        self.hrv_clean_data = hrv_clean_data
        self.only_on_full_hour_slots = only_on_full_hour_slots
