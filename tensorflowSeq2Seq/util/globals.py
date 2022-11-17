

class Globals:

    __IS_TRAIN  = None
    __TIME      = None
    __EAGER     = False
    __WORKERS   = None
    __GPU       = True

    @staticmethod
    def set_train_flag(flag):
        if Globals.__IS_TRAIN is None:
            Globals.__IS_TRAIN = flag

    @staticmethod
    def is_training():
        return Globals.__IS_TRAIN

    @staticmethod
    def set_time_flag(flag):
        if Globals.__TIME is None:
            Globals.__TIME = flag

    @staticmethod
    def do_timing():
        return Globals.__TIME

    @staticmethod
    def set_eager_flag(flag):
        Globals.__EAGER = flag

    @staticmethod
    def is_eager():
        return Globals.__EAGER

    @staticmethod
    def set_number_of_workers(workers, force=False):
        if Globals.__WORKERS is None or force:
            Globals.__WORKERS = workers

    @staticmethod
    def get_number_of_workers():
        return Globals.__WORKERS

    @staticmethod
    def set_cpu():
        if Globals.__GPU:
            Globals.__GPU = False

    @staticmethod
    def is_gpu():
        return Globals.__GPU

