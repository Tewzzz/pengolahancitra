import cv2
import numpy as np
from tkinter import filedialog, Tk, Button, Label
from PIL import Image, ImageTk

def load_image():
    global img, img_display
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, 0)  # Membaca citra dalam grayscale
        show_image(img)

def show_image(img_to_show):
    img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)
    img_display.config(image=img_tk)
    img_display.image = img_tk

def apply_thresholding():
    global img
    _, segmented_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # Thresholding sederhana
    show_image(segmented_img)

def apply_kmeans():
    global img
    Z = img.reshape((-1, 1))
    Z = np.float32(Z)
    
    # K-means parameters
    K = 2  # Jumlah cluster
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Konversi hasil k-means ke bentuk citra
    center = np.uint8(center)
    segmented_img = center[label.flatten()]
    segmented_img = segmented_img.reshape((img.shape))
    
    show_image(segmented_img)

# GUI dengan Tkinter
root = Tk()
root.title("Aplikasi Segmentasi Citra")

load_btn = Button(root, text="Muat Gambar", command=load_image)
load_btn.pack()

threshold_btn = Button(root, text="Segmentasi dengan Thresholding", command=apply_thresholding)
threshold_btn.pack()

kmeans_btn = Button(root, text="Segmentasi dengan K-Means", command=apply_kmeans)
kmeans_btn.pack()

img_display = Label(root)
img_display.pack()

root.mainloop()
