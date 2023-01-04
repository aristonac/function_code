import numpy as np
import matplotlib.pyplot as plt
import math
# import latex
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": False,
    "text.latex.preamble": [r'\usepackage{amsmath}']
    # "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica"
})

w = 100
h = 100
x1,y1 = 0,0


IoU_list = []
IoU_diag_list = []

figure, axis = plt.subplots(1,2)
figure.set_size_inches(7.3,3)

#figure.suptitle('')
axis[0].set_title('Intersection over Union (IoU)')
axis[0].set_xlabel('Normierte Verschiebung in x-Richtung')
axis[0].set_ylabel('Maßzahl ' + r"$IoU = \frac{I}{U}$", color="green")

eq1 = r"$IoU_{k} = \frac{B_{G} \cap B_{k}}{B_{G} \cup B_{k}},$"
eq2 = r"$k = 0, \frac{1}{5}, \dots , \frac{4}{5}, 1$"
eq3 = r"$k \hat{=} $ Verschiebung in y-Richtung"
axis[0].text(1.0, 0.85, eq1, color="green", fontsize=12, horizontalalignment="right")
axis[0].text(1.0, 0.74, eq2, color="green", fontsize=12, horizontalalignment="right")
axis[0].text(1.0, 0.63, eq3, color="green", fontsize=10, horizontalalignment="right")

axis[1].set_title('Euklidische Distanz (dist)')
axis[1].set_xlabel('Normierte Verschiebung in x-Richtung')
axis[1].set_ylabel('Maßzahl ' + r"$IoU = \frac{I}{U}$", color="green")

# Right Y-axis labels
axis[1].text(1.1, 0.5, 'Maßzahl ' + r"$dist = \frac{I_{diag}}{U_{diag}}$", color="blue", horizontalalignment="left", verticalalignment="center", rotation=90)
axis[1].text(1.0, 0.85, r'$dist = \sqrt{(x_{2}-x_{1})^2 + (y_{2}-y_{1})^2}$', color="blue", fontsize=12, horizontalalignment="right")

# es wird sichergestellt, dass die Verschiebung unterhalb der Diagonalen bleibt

coords = range(0,w+1)
normed_coords = [e/w for e in coords]
for j in list(range(0, h+1, (h+1)//5)) + [h]:
  for i in coords:
    x2, y2 = i,i*j/h
    # x2, y2 = 0,h/2
    x2, y2 = (y2, x2) if w*y2 > h*x2 else (x2, y2)

    dx,dy = abs(x2-x1),abs(y2-y1)

    w_intersect = w-dx
    h_intersect = h-dy

    I = w_intersect*h_intersect
    U = 2*w*h-I
    IoU_list += [I/U]

    # ########################
    m = dy/dx if dx > 0 else 0
    
    U_max = 2*math.sqrt(w*w + m*m*w*w)
    U = math.sqrt((w+dx)**2 + (m*(w+dx))**2)
    I = U_max-U
    
    # y_max = h_intersect/w_intersect * max(w,w_intersect) if w_intersect > 0 else 999999999
    # x_max = w_intersect/h_intersect * max(h,h_intersect) if h_intersect > 0 else 999999999
    # max_r = 2*math.sqrt(y_max*y_max+x_max*x_max)-I_dist

    IoU_diag_list += [I/U]

  if j == 0:
    axis[0].plot(normed_coords, IoU_list, color='green', linewidth=.7)
  elif j == h:
    axis[0].plot(normed_coords, IoU_list, color='green', linewidth=.7)
    axis[1].plot(normed_coords, IoU_list, color='green', linewidth=.7)
    axis[1].plot(normed_coords, IoU_diag_list, color='blue', linewidth=.7)
  else:
    # plot von IoU_diag_list wird hier weggelassen, da sie alle übereinander liegen
    axis[0].plot(normed_coords, IoU_list, color='green', linewidth=.5, linestyle='dashed')
    axis[1].plot(normed_coords, IoU_list, color='green', linewidth=.5, linestyle='dashed')
  

  IoU_list = []
  # print(IoU_list)
  IoU_diag_list = []
  #print(IoU_diag_list)



plt.tight_layout()
plt.savefig("IoU-vs-eukdist.png", dpi=1200)
plt.show()
