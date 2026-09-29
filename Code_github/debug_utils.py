import matplotlib.pyplot as plt

DEBUG_PRINT = True
DEBUG_PLOTS = False

def debug_print(*args, **kwargs):
    if DEBUG_PRINT:
        print(*args, **kwargs)

def debug_show():
    if DEBUG_PLOTS:
        plt.show()
    else:
        plt.close()