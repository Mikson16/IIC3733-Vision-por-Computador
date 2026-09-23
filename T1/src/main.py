# imports
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import *

def main():
    # imgs paths
    img_path = os.path.join("..", "imgs")
    # print(os.path.join(img_path, "img1.jpg"))
    img1, img2, img3, img4 = [cv2.imread(os.path.join(img_path, f"img{i}.jpg")) for i in range(1, 5)]
    N, M = img1.shape[:2] # eliminando el canal de color

    # obtener matriz de calibracion de las imagenes
    # definir M, los puntos reales con el punto 1 como el origen, todo en milimetros
    M = np.array([
        [0.0,  0.0,   0.0,   0.0,   570.0, 570.0, 570.0, 570.0],
        [0.0,  294.0, 0.0,   294.0, 0.0,   294.0, 0.0,   294.0],
        [0.0,  0.0,   369.0, 369.0, 369.0, 369.0, 0.0,   0.0],
        [1.0,  1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
    ])
    # lista de puntos
    puntos_list_1 = []
    puntos_list_2 = []
    puntos_list_3 = []
    puntos_list_4 = []

    # Para obtener los puntos manualmente
    # P1, puntos_list_1 = get_calibration(img1, puntos_list_1)
    # P2, puntos_list_2 = get_calibration(img2, puntos_list_2)
    # P3, puntos_list_3 = get_calibration(img3, puntos_list_3)
    # P4, puntos_list_4 = get_calibration(img4, puntos_list_4)

    # Para utilizar los puntos ya obtenidos manualmente
    P1 = np.array([
        [ 1.90495546e-03, -7.02684174e-04, -1.54173974e-04,  4.40461349e-01], 
        [ 1.74486754e-04, -6.32747454e-04, -1.76101539e-03,  8.97766293e-01], 
        [ 4.73323927e-07, -1.77211021e-06, -2.49364647e-07,  1.34363914e-03]
        ])
    P2 = np.array([
        [ 2.26927019e-03, -4.36415121e-04, -1.62449136e-04,  2.03739488e-01],
        [ 3.20280415e-04, -6.89158839e-04, -2.01049096e-03,  9.79019027e-01],
        [ 8.49199339e-07, -1.91627164e-06, -2.85715024e-07,  1.40684900e-03]  
    ])
    P3 = np.array([
        [ 1.71108833e-03, -1.02267908e-03, -8.88822680e-05,  3.66918186e-01],
        [ 6.09066509e-05, -7.27508603e-04, -1.69272489e-03,  9.30248317e-01],
        [ 1.34570461e-07, -1.79961244e-06, -1.51926624e-07,  1.31673976e-03]
    ])
    P4 = np.array([
    [-8.66181198e-04,  1.10850750e-03,  5.39796199e-05, -5.83951246e-01],
    [ 1.34435012e-04,  5.10458149e-04,  1.18268765e-03, -8.11785450e-01],
    [ 3.55838815e-07,  1.23105119e-06,  1.01412492e-07, -1.29111548e-03]
    ])
    puntos_list_1 = [(328, 669), (284, 864), (306, 199), (243, 83), (966, 228), (1262, 162), (946, 617), (1208, 744)]
    puntos_list_2 = [(145, 697), (89, 919), (110, 183), (22, 45), (805, 235), (1071, 178), (792, 613), (1031, 724)]
    puntos_list_3 = [(278, 707), (85, 909), (265, 243), (46, 124), (979, 255), (1248, 156), (964, 691), (1204, 871)]
    puntos_list_4 = [(453, 628), (277, 713), (449, 300), (268, 252), (1007, 284), (1062, 216), (990, 676), (1035, 805)]

    # obtener m1 de la imagen 1
    m1 = get_m1(img1)
    m1 = np.array(m1).reshape(3, 1)
    print(f"El punto rojo m1 es: {m1}")
    # print(f"m1: {m1[0]}, {m1[1]}")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.scatter(m1[0], m1[1], color='red', s=15)
    plt.show()

    # obtener matriz de homografia
    H = homography_with_calibration(P1, P2)
    print(f"Matriz de homografia H = \n{H}")

    # obtener m2 a partir de la matriz de homografia
    m2 = H @ m1
    m2 = m2 / m2[2]  # Normalizar el punto m2 para que la tercera coordenada sea 1
    print(f"Matriz de homografia H = \n{H}")
    # print(f"m1: {m1}")
    print(f"El punto rojo m2 es: {m2}")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.scatter(m2[0], m2[1], color='red', s=15)
    plt.show()

    # obtener m3 a partir de m2
    m3 = get_m3_from_m2(m2, P2, P3, img3, graph=True)
    print(f"El punto rojo m3 es: {m3}")
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.plot(m3[0], m3[1], 'ro', markersize=3)
    plt.show()

    # obtener m4 a partir de m1, m2 y m3
    m4 = get_m4_from_m3(m2, m3, P2, P3, P4, img4, graph=True)

    # Obtener la altura del punto
    Z = reconstruction_3D(m1, m2, m3, m4, P1, P2, P3, P4)
    Z_real = 215.0 # en mm
    print(Z)
    print(f"Diferencia entre Z y Z_real = {abs(Z - Z_real)} mm")
    
    
if __name__ == "__main__":
    main()