import threading
from matplotlib import pyplot as plt
import time

def plotter():
    time_buffer = [1,2,3,4,5,6,7,8]
    results = [i*i for i in time_buffer]
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    while True:
        plt.xlabel("time in Seconds")
        plt.ylabel("Squared Error")
        plt.plot(time_buffer,results)
        fig.canvas.draw()
        fig.clear()
        results = [i*1.01 for i in results]
        time.sleep(0.2)

tr = threading.Thread(target=plotter)
tr.start()
