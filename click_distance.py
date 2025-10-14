import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

src_path = "photos/acuro_36.jpg"
img = mpimg.imread(src_path)
fig, ax = plt.subplots()
ax.imshow(img)
points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        ax.plot(x, y, 'ro')
        fig.canvas.draw()
        if len(points) >= 2:
            dist = np.linalg.norm(np.array(points[-1]) - np.array(points[-2]))
            print(f"Distance between point {len(points)-1} and {len(points)}: {dist:.2f} pixels")

def onkey(event):
    if event.key == 'q':
        print("Exiting...")
        plt.close(fig)

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)
plt.title("Click to select points, press 'q' to quit")
plt.show()


