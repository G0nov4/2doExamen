# ---------
# Modificar
# ---------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_perdida(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

class RedNeuronal:

  def __init__(self):
    self.p1 = np.random.normal()
    self.p2 = np.random.normal()
    self.p3 = np.random.normal()
    self.p4 = np.random.normal()
    self.p5 = np.random.normal()
    self.p6 = np.random.normal()
    self.p7 = np.random.normal()
    self.p8 = np.random.normal()
    self.p9 = np.random.normal()
    self.p10 = np.random.normal()
    self.p11 = np.random.normal()
    self.p12 = np.random.normal()
    self.p13 = np.random.normal()
    self.p14 = np.random.normal()
    self.p15 = np.random.normal()
    self.p16 = np.random.normal()
    self.p17 = np.random.normal()
    self.p18 = np.random.normal()
    self.p19 = np.random.normal()
    self.p20 = np.random.normal()

    # Bias
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()
    self.b5 = np.random.normal()

  def retroalimentacion(self, x):
    neurona1 = sigmoid(self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1)
    neurona2 = sigmoid(self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3] + self.b2)
    neurona3 = sigmoid(self.p9 * x[0] + self.p10 * x[1] + self.p11 * x[2] + self.p12 * x[3] + self.b3)
    neurona4 = sigmoid(self.p13 * x[0] + self.p14 * x[1] + self.p15 * x[2] + self.p16 * x[3] + self.b4)
    neurona5 = sigmoid(self.p17 * neurona1 + self.p18 * neurona2 + self.p19 * neurona3 + self.p20 * neurona4 + self.b5)
    return neurona5

  def entrenar(self, datos, y_trues):
    tasa_aprendizaje = 0.1
    epocas = 1000 

    for epoca in range(epocas):
	
      for x, y_true in zip(data, y_trues):
        sum_neurona1 = self.p1 * x[0] + self.p2 * x[1] + self.p3 * x[2] + self.p4 * x[3] + self.b1
        neurona1 = sigmoid(sum_neurona1)

        sum_neurona2 = self.p5 * x[0] + self.p6 * x[1] + self.p7 * x[2] + self.p8 * x[3] + self.b2
        neurona2 = sigmoid(sum_neurona2)

        sum_neurona3 = self.p9 * x[0] + self.p10 * x[1] + self.p11 * x[2] + self.p12 * x[3] + self.b3
        neurona3 = sigmoid(sum_neurona3)

        sum_neurona4 = self.p13 * x[0] + self.p14 * x[1] + self.p15 * x[2] + self.p16 * x[3] + self.b4
        neurona4 = sigmoid(sum_neurona4)

        sum_neurona5 = self.p17 * neurona1 + self.p18 * neurona2 + self.p19 * neurona3 + self.p20 * neurona4 + self.b5
        neurona5 = sigmoid(sum_neurona5)
        y_pred = neurona5

        # derivada parcial
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neurona5
        d_ypred_d_p17 = neurona1 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_p18 = neurona2 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_p19 = neurona3 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_p20 = neurona4 * deriv_sigmoid(sum_neurona1)
        d_ypred_d_b5 = deriv_sigmoid(sum_neurona5)

        d_ypred_d_neurona1 = self.p17 * deriv_sigmoid(sum_neurona5)
        d_ypred_d_neurona2 = self.p18 * deriv_sigmoid(sum_neurona5)
        d_ypred_d_neurona3 = self.p19 * deriv_sigmoid(sum_neurona5)
        d_ypred_d_neurona4 = self.p20 * deriv_sigmoid(sum_neurona5)

        # Neurona1
        d_neurona1_d_p1 = x[0] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p2 = x[1] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p3 = x[2] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_p4 = x[3] * deriv_sigmoid(sum_neurona1)
        d_neurona1_d_b1 = deriv_sigmoid(sum_neurona1)

        # Neurona2
        d_neurona2_d_p5 = x[0] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p6 = x[1] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p7 = x[2] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_p8 = x[3] * deriv_sigmoid(sum_neurona2)
        d_neurona2_d_b2 = deriv_sigmoid(sum_neurona2)

        # Neurona3
        d_neurona3_d_p9 = x[0] * deriv_sigmoid(sum_neurona3)
        d_neurona3_d_p10= x[1] * deriv_sigmoid(sum_neurona3)
        d_neurona3_d_p11= x[2] * deriv_sigmoid(sum_neurona3)
        d_neurona3_d_p12= x[3] * deriv_sigmoid(sum_neurona3)
        d_neurona3_d_b3 = deriv_sigmoid(sum_neurona3)

        # Neurona4
        d_neurona4_d_p13 = x[0] * deriv_sigmoid(sum_neurona4)
        d_neurona4_d_p14 = x[1] * deriv_sigmoid(sum_neurona4)
        d_neurona4_d_p15 = x[2] * deriv_sigmoid(sum_neurona4)
        d_neurona4_d_p16 = x[3] * deriv_sigmoid(sum_neurona4)
        d_neurona4_d_b4 = deriv_sigmoid(sum_neurona4)

        # Actualizar
        # Neurona 1
        self.p1 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p1
        self.p2 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p2
        self.p3 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p3
        self.p4 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_p4
        self.b1 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_b1

        # Neurona2
        self.p5 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p5
        self.p6 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p6
        self.p7 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p7
        self.p8 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_p8
        self.b2 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_b2

        # Neurona3
        self.p9 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_p9
        self.p10-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_p10
        self.p11-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_p11
        self.p12-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_p12
        self.b3 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_b3

        # Neurona3
        self.p13-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_p13
        self.p14-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_p14
        self.p15-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_p15
        self.p16-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_p16
        self.b4 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_b4

        # Neurona3
        self.p17 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p17
        self.p18 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p18
        self.p19 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p19
        self.p20 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_p20
        self.b5 -= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_b5

      # perdida por cada epoca
      if epoca % 10 == 0:
        y_preds = np.apply_along_axis(self.retroalimentacion, 1, data)
        perdida = mse_perdida(y_trues, y_preds)
        print("epoca %d perdida: %.3f" % (epoca, perdida))



iris = load_iris()

data = iris.data
y_trues = iris.target

#print(data)
mired = RedNeuronal()
mired.entrenar(data, y_trues) 