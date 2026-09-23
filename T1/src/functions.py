import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

M = np.array([
    [0.0,  0.0,   0.0,   0.0,   570.0, 570.0, 570.0, 570.0],
    [0.0,  294.0, 0.0,   294.0, 0.0,   294.0, 0.0,   294.0],
    [0.0,  0.0,   369.0, 369.0, 369.0, 369.0, 0.0,   0.0],
    [1.0,  1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
])
# Funcion de calibracion de la camara

def calibracion(M,w):

  # Escribe código aquí
  n = M.shape[1]

  X_v = []

  for i in range(n):
    X, Y, Z = M[0, i], M[1, i], M[2, i]
    u, v = w[0, i], w[1, i]

    x_1 = np.array([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
    x_2 = np.array([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    # Apilar
    X_vi = np.vstack([x_1, x_2])

    if i == 0:
      X_v = X_vi
    else:
      X_v = np.vstack([X_v, X_vi])

  U, S, VT = np.linalg.svd(X_v)
  a = VT[-1,:]
  P = a.reshape(3, 4)

  return P
# Callback para mouse
def mouse_callback(event, x, y, flags, param):
    puntos_list, img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        puntos_list.append((x, y))
        
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"{len(puntos_list)}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if puntos_list:
            punto_removido = puntos_list.pop()
            print(f"Punto removido {punto_removido}")
        img[:] = img.copy()
        
        for index, (px, py) in enumerate(puntos_list):
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{index + 1}", (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
# def obtener calibracion de imagen
def get_calibration(img, puntos_list):
    # levantar imagen y mostrarla en una ventana para obtener los puntos de las esquinas
    # una vez seleccionado los puntos apretar q para cerrar la ventana y guardar los puntos
    display_img = img.copy()
    flag = True
    cv2.namedWindow("Seleccionar puntos", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Seleccionar puntos", img.shape[1], img.shape[0])
    cv2.setMouseCallback("Seleccionar puntos", mouse_callback, param=(puntos_list, display_img))

    while flag:
        cv2.imshow("Seleccionar puntos", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            flag = False
        elif key == ord('r'):
            puntos_list.clear()
            display_img = img.copy()
            print("Puntos reiniciados")
    cv2.destroyAllWindows()
    # print(f"Puntos seleccionados: {puntos_list}")
    x, y = zip(*puntos_list)

    plt.figure(figsize=(10, 10))

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray')
    plt.scatter(x, y, color='red', s=15)
    plt.show()
    # print(f"Los puntos que representan la caja son: {puntos_list}")
    # obtener matriz de proyeccion P

    # Definir los vectores de m1
    x_p = np.array([p[0] for p in puntos_list])
    y_p = np.array([p[1] for p in puntos_list])
    # print(f"x_p = {x_p}")
    # print(f"y_p = {y_p}")

    m1 = np.vstack((x_p, y_p, np.ones(len(x_p))))
    # print(f"El punto rojo m1 es: {m1}")

    P = calibracion(M, m1)
    # print(f"Matriz de proyeccion P = \n{P}")
    return (P, puntos_list)

def get_m1_mouse_callback(event, x, y, flags, param):
    state= param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        state['m1'] = np.array([x, y, 1])
        
        cv2.circle(state['display_img'], (x, y), 5, (0, 255, 0), -1)
        cv2.putText(state['display_img'], f"m1", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state['m1'] = None
        state['display_img'][:] = state['img_o'].copy()
        
def get_m1(img):
    """Esta funcion es el unico input del sistema, saca el punto rojo de la caja por medio del mouse y devuelve m1.
    Una vez seleccionado el punto, apretar q, para cerrar la ventana y guardar
    """
    flag = True
    display_img = img.copy()
    state = {'m1': None,
             'display_img': display_img,
             'img_o': img.copy()}
    cv2.namedWindow("Seleccionar punto rojo", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Seleccionar punto rojo", display_img.shape[1], display_img.shape[0])
    cv2.setMouseCallback("Seleccionar punto rojo", get_m1_mouse_callback, param=state)
    while flag:
        cv2.imshow("Seleccionar punto rojo", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            flag = False
        elif key == ord('r'):
            state['m1'] = None
            state['display_img'] = state['img_o'].copy()
            print("Puntos reiniciados")
    cv2.destroyAllWindows()
    return state['m1']
    
    
# Extraido de E02
def homography(X,H):
  N, M, C = X.shape
  Hinv  = np.linalg.inv(H)
  Xp    = np.zeros((N,M,C),np.uint8)

  m = np.ones((N*M,3))
  t = 0
  for i in range(N):
    for j in range(M):
      m[t,0:2] = [j,i]
      t = t+1

  mph = np.dot(Hinv,m.T)  # Transformation m to mp
  mp  = np.divide(mph[0:2,:],mph[2,:])
  mpf = np.trunc(mp).astype(int) #np.fix is deprecated

  ip = mpf[1,:]
  jp = mpf[0,:]
  
  ktj = np.logical_and(jp >= 0,  jp<M)
  kti = np.logical_and(ip >=0, ip < N)
  kt  = np.logical_and(kti, ktj)

  t = 0
  for i in range(N):
    for j in range(M):
      if kt[t]:
        Xp[i,j] = X[ip[t],jp[t]]
      t = t+1
  return Xp

def homography_matrix(m,mp):
  (x ,y ) = m
  (xp,yp) = mp
  n       = len(x)
  A       = np.zeros((2*n,9))
  for i in range(n):
    j = i*2
    A[j  ,:] = [x[i], y[i], 1,     0,     0, 0, -x[i]*xp[i], -y[i]*xp[i], -xp[i]]
    A[j+1,:] = [  0  ,   0  , 0, x[i], y[i], 1, -x[i]*yp[i], -y[i]*yp[i], -yp[i]]
  [U,S,V] = np.linalg.svd(A)
  h       = V[-1,:]
  H       = np.vstack([h[0:3], h[3:6], h[6:9]])
  return H

def homography_with_calibration(P1, P2):
  H = None
  # obtener submatrices de P1 y P2 usando descomposicion RQ Szeliski
  M1 = P1[:, :3] # como P1 y P2 son con el mismo centro optico, la traslacion (t) es 0, K es igual porque es la misma camara y centro optico es 0
  M2 = P2[:, :3] # como P1 y P2 son con el mismo centro optico, la traslacion (t) es 0, K es igual porque es la misma camara y centro optico es 0
  # solo es diferente la rotacion.
  M_intercambiado_1 = M1[::-1, ::-1]
  K_temp, R_temp = np.linalg.qr(M_intercambiado_1.T) #usando la qr de numpy
  
  M_intercambiado_2 = M2[::-1, ::-1]
  K_temp2, R_temp2 = np.linalg.qr(M_intercambiado_2.T) 
  #reconstruir K y R a partir de las submatrices obtenidas
  K_1 = R_temp.T[::-1, ::-1]
  R_r1 = K_temp.T[::-1, ::-1]
  K_2 = R_temp2.T[::-1, ::-1]
  R_r2 = K_temp2.T[::-1, ::-1]

  #correccion de signos
  T1 = np.diag(np.sign(np.diag(K_1)))
  K1 = K_1 @ T1
  R1 = T1 @ R_r1
  
  T2 = np.diag(np.sign(np.diag(K_2)))
  K2 = K_2 @ T2
  R2 = T2 @ R_r2
  
  #normalizacion homogenea
  K1 = K1 / K1[2, 2] # esto fuerza que el ultimo elemento sea 1
  K2 = K2 / K2[2, 2]
  
  # Rotacion relativa
  R = R2 @ R1.T
  
  # armar matriz de homografia
  H = K2 @ R @ np.linalg.inv(K1)
  
  return H
# siendo P2 y P3 las matrices de proyeccion de la imagen 2 y 3 respectivamente
#inspirado en solucion de E4
def get_m3_from_m2(m2, P2, P3, img3, graph=False):
    # obtener el nullspace desde P2, para armar C2
    U_P2, S_P2, Vt_P2 = np.linalg.svd(P2)
    #identificar los numeros que esten debajo de la tolerancia
    null_mask = np.zeros(P2.shape[1], dtype=bool)
    null_mask[len(S_P2): ] = True
    null_mask[: len(S_P2)] = S_P2 < 1e-13
    
    C2 = Vt_P2.T[:, null_mask]
    # print(f"m2 = {m2}")
    
    # proyectar C2 en la imagen 3 para sacar Bc2 antisimetrica
    u = P3 @ C2
    u_anti = np.array([
    [0, -u[2][0], u[1][0]],
    [u[2][0], 0, -u[0][0]],
    [-u[1][0], u[0][0], 0]
    ])
    # print(f"u_anti = {u_anti}")
    
    # armar la matriz pseudoinversa
    pseudo_inv = P2.T @ np.linalg.inv(P2 @ P2.T) 
    # print(f"pseudo_inv = {pseudo_inv}")
    
    # armar matriz fundamental F
    F = u_anti @ P3 @ pseudo_inv
    
    # obtener la recta epipolar l3
    l3 = F @ m2
    l3 = l3 / l3[2] 
    # print(f"l3 = {l3}")
    if graph:
        x_ep, y_ep = graficar_linea_epipolar(img3, l3, m2)
    else:
        x_ep = np.linspace(m2[0]-100, m2[0] + 500, 100)
        y_ep = ((-l3[0, 0]*x_ep - l3[2, 0]) / l3[1, 0])
        
    # por procesamiento de imagen encontrar el elemento rojo a lo largo de la imagen 3, y devolverlo como m3
    m3 = find_red_point_in_epipolar_line(img3, x_ep, y_ep)
    return m3

def graficar_linea_epipolar(img, l, m):
    img_display = img.copy()
    a = l[0, 0]
    b = l[1, 0]
    c = l[2, 0]
    x_m, y_m = m[0], m[1]
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))

    
    x_ep = np.linspace(x_m-100, x_m + 500, 100)
    y_ep = ((-a*x_ep - c) / b)
    # print(f"y_ep = {len(y_ep)}")
    plt.plot(x_ep, y_ep, 'g-', linewidth=2)
    plt.grid(False)
    plt.show()
    return x_ep, y_ep
    
def find_red_point_in_epipolar_line(img, x_ep, y_ep):
    #hacer un algoritmo que busque el primer pixel rojo y el ultimo a lo largo de toda la linea epipolar y devolver el medio
    img = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_low_1 = np.array([0, 100, 50])
    red_high_1 = np.array([10, 255, 255])
    red_low_2 = np.array([170, 100, 50])
    red_high_2 = np.array([180, 255, 255])
    # print(f"x_ep = {x_ep[1]}")
    # print(f"y_ep = {y_ep}")

    pixeles_rojos = []
    h, w = img.shape[:2]
    for x, y in zip(x_ep, y_ep):
        # print(f"x = {x[0]}, y = {y[0]}")
        x = round(x[0])
        y = round(y[0])
        pixel = hsv[y, x]
        
        if (np.all(red_low_1 <= pixel) and np.all(pixel <= red_high_1)) or (np.all(red_low_2 <= pixel) and np.all(pixel <= red_high_2)):
            pixeles_rojos.append((x, y))
            
    first_x, first_y = pixeles_rojos[0]
    last_x, last_y = pixeles_rojos[-1]
    mid_x = (first_x + last_x) // 2
    mid_y = (first_y + last_y) // 2
    return np.array([mid_x, mid_y, 1]).reshape(3, 1)
    
    

def get_F(P1, P2):
    # obtener el nullspace desde P1, para armar C1
    U, S, Vt = np.linalg.svd(P1)
    #identificar los numeros que esten debajo de la tolerancia
    null_mask = np.zeros(P2.shape[1], dtype=bool)
    null_mask[len(S): ] = True
    null_mask[: len(S)] = S < 1e-13
    
    C1 = Vt.T[:, null_mask]
    
    # proyectar C1 en la imagen 2 para sacar Bc1 antisimetrica
    u = P2 @ C1
    u_anti = np.array([
    [0, -u[2][0], u[1][0]],
    [u[2][0], 0, -u[0][0]],
    [-u[1][0], u[0][0], 0]
    ])
    # print(f"u_anti = {u_anti}")
    
    # armar la matriz pseudoinversa
    pseudo_inv = P1.T @ np.linalg.inv(P1 @ P1.T) 
    # print(f"pseudo_inv = {pseudo_inv}")
    
    # armar matriz fundamental F
    F = u_anti @ P2 @ pseudo_inv
    return F
    
def get_m4_from_m3(m2, m3, P2 ,P3, P4, img4, graph=False):
    F24 = get_F(P2, P4)
    F34 = get_F(P3, P4)
    
    # optener rectas epipolares
    l2 = F24 @ m2
    l2 = l2 / l2[2]
    l3 = F34 @ m3
    l3 = l3 / l3[2]
    # print(f"l2 = {l2}")
    # print(f"l3 = {l3}")
    # obtener la interseccion
    m4 = np.cross(l2.flatten(), l3.flatten())
    m4 = m4 / m4[2]
    print(f"El punto rojo m4 es: {m4}")
    if graph:
        x_ep2, y_ep2 = graficar_linea_epipolar(img4, l2, m2)
        x_ep3, y_ep3 = graficar_linea_epipolar(img4, l3, m3)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))
        plt.plot(m4[0], m4[1], 'ro', markersize=3)
        plt.show()
    return m4.reshape(3, 1)
def reconstruction_3D(m1, m2, m3, m4, P1, P2, P3, P4):
    # print(f"m1 = {m1}")
    # print(f"m2 = {m2}")
    # print(f"m3 = {m3}")
    # print(f"m4 = {m4}")
    x_1, y_1 = m1[0][0], m1[1][0]
    x_2, y_2 = m2[0][0], m2[1][0]
    x_3, y_3 = m3[0][0], m3[1][0]
    x_4, y_4 = m4[0][0], m4[1][0]
    # print(f"x_1 = {x_1}, y_1 = {y_1}")
    # print(f"x_2 = {x_2}, y_2 = {y_2}")
    # print(f"x_3 = {x_3}, y_3 = {y_3}")
    # print(f"x_4 = {x_4}, y_4 = {y_4}")
    Q = np.array([
        [P1[2, 0]*x_1 - P1[0, 0], P1[2, 1]*x_1 - P1[0, 1], P1[2, 2]*x_1 - P1[0, 2]],
        [P1[2, 0]*y_1 - P1[1, 0], P1[2, 1]*y_1 - P1[1, 1], P1[2, 2]*y_1 - P1[1, 2]],
        [P2[2, 0]*x_2 - P2[0, 0], P2[2, 1]*x_2 - P2[0, 1], P2[2, 2]*x_2 - P2[0, 2]],
        [P2[2, 0]*y_2 - P2[1, 0], P2[2, 1]*y_2 - P2[1, 1], P2[2, 2]*y_2 - P2[1, 2]],
        [P3[2, 0]*x_3 - P3[0, 0], P3[2, 1]*x_3 - P3[0, 1], P3[2, 2]*x_3 - P3[0, 2]],
        [P3[2, 0]*y_3 - P3[1, 0], P3[2, 1]*y_3 - P3[1, 1], P3[2, 2]*y_3 - P3[1, 2]],
        [P4[2, 0]*x_4 - P4[0, 0], P4[2, 1]*x_4 - P4[0, 1], P4[2, 2]*x_4 - P4[0, 2]],
        [P4[2, 0]*y_4 - P4[1, 0], P4[2, 1]*y_4 - P4[1, 1], P4[2, 2]*y_4 - P4[1, 2]]
    ])
    
    r = np.array([
        [P1[0, 3] - x_1 * P1[2, 3]],
        [P1[1, 3] - y_1 * P1[2, 3]],
        [P2[0, 3] - x_2 * P2[2, 3]],
        [P2[1, 3] - y_2 * P2[2, 3]],
        [P3[0, 3] - x_3 * P3[2, 3]],
        [P3[1, 3] - y_3 * P3[2, 3]],
        [P4[0, 3] - x_4 * P4[2, 3]],
        [P4[1, 3] - y_4 * P4[2, 3]]
    ])
    
    #resolver por minimos cuadrados
    M, _, _, _ = np.linalg.lstsq(Q, r, rcond=None)
    # print(M)
    Z = M[2][0]
    # print(f"Z = {Z}")
    return Z.astype(float)
    
    