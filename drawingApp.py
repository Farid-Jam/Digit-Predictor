import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import pickle

# ========= Load trained weights =========
with open("mnist_weights.pkl", "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

# ========= Neural Network Helpers =========
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forw_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, 0)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forw_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# ========= Drawing App =========
last_x, last_y = None, None

def xy(event):
    """Store starting point for drawing"""
    global last_x, last_y
    last_x, last_y = event.x, event.y

def add_line(event):
    """Draw line as mouse is dragged"""
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y),
                       width=20, fill='black',
                       capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
    last_x, last_y = event.x, event.y

def clear_canvas():
    """Clear the canvas"""
    canvas.delete("all")
    lbl_result.config(text="Prediction: ")

def predict_digit():
    """Capture canvas, preprocess, and predict"""
    # Save canvas to PostScript file
    canvas.postscript(file="canvas.ps", colormode='color')

    # Open and preprocess
    img = Image.open("canvas.ps")
    img = img.convert('L')          # grayscale
    img = ImageOps.invert(img)      # MNIST is white-on-black
    img = img.resize((28, 28))      # shrink to MNIST size

    img = np.array(img) / 255.0
    img = img.reshape(784, 1)

    # Predict
    pred = make_predictions(img, W1, b1, W2, b2)
    print("Prediction:", pred[0])
    lbl_result.config(text=f"Prediction: {pred[0]}")

# ========= Tkinter UI =========
root = tk.Tk()
root.title("Digit Recognizer")

canvas = tk.Canvas(root, width=200, height=200, bg='white')
canvas.pack()

canvas.bind('<Button-1>', xy)
canvas.bind('<B1-Motion>', add_line)

btn_predict = tk.Button(root, text="Predict", command=predict_digit)
btn_predict.pack(pady=5)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(pady=5)

lbl_result = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
lbl_result.pack(pady=10)

root.mainloop()
