# 202231086_Tarissa-Nurhapsari-Laksono_PCD-A_LAPRAK3-4

- TEPI DAN GARIS 
```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import skimage.io
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import data
from skimage.color import rgb2hsv
```
PENJELASAN :  untuk mengimpor semua library yang diperlukan. cv2 adalah OpenCV untuk pemrosesan gambar, matplotlib.pyplot untuk plotting gambar, dan modul-modul lain dari skimage untuk fitur-fitur pengolahan gambar.

```python
image = cv2.imread("2.jpg")
cv2.imshow("gambar parkir",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
PENJELASAN : untuk memuat gambar "2.jpg" menggunakan OpenCV dan menampilkan gambar menggunakan OpenCV dengan jendela yang muncul dan menunggu sampai tombol keyboard ditekan sebelum menutup jendela.

```python
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image,100,150)
```
PENJELASAN : untuk mengonversi gambar ke citra grayscale dan kemudian menerapkan deteksi tepinya menggunakan metode canny.

```python
fig,axs = plt.subplots(1,2,figsize = (10,10))
ax = axs.ravel()

ax[0].imshow(gray,cmap = "gray")
ax[0].set_title("gambar asli")

ax[1].imshow(edges,cmap = "gray")
ax[1].set_title("gambar yg terproses")
```
PENJELASAN : untuk membuat plot dengan dua gambar (asli dan hasil deteksi tepi) dalam satu jendela.

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)
```
PENJELASAN : untuk mengaburkan gambar grayscale untuk mengurangi noise, kemudian menerapkan deteksi tepi Canny lagi. Selanjutnya, menggunakan Transformasi Hough untuk mendeteksi garis-garis dalam gambar

```python
image_line = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_line, (x1, y1), (x2, y2), (0, 255, 0), 2)
```
PENJELASAN : untuk menggambar garis-garis yang dideteksi ke salinan gambar asli menggunakan warna hijau.

```python
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
ax = axs.ravel()

ax[0].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB), cmap='gray')
ax[0].set_title("Gambar Asli")

ax[1].imshow(cv2.cvtColor(image_line, cv2.COLOR_BGR2RGB))
ax[1].set_title("Gambar yang terproses")
```
PENJELASAN :  untuk menampilkan gambar asli dan gambar hasil dengan garis-garis yang ditambahkan.

- EKSTRAKSI FITUR
```python
img = skimage.data.chelsea()
img_hsv = rgb2hsv(img)
```
PENJELASAN : untuk mengambil gambar "chelsea" dari dataset gambar bawaan skimage dan mengonversinya ke ruang warna HSV menggunakan fungsi rgb2hsv dari skimage.

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 10))
ax = axs.ravel()

ax[0].imshow(img)
ax[0].set_title("RGB")

ax[1].imshow(img_hsv, cmap="hsv")
ax[1].set_title("HSV")
```
PENJELASAN : membuat subplot untuk menampilkan gambar asli dalam ruang warna RGB dan gambar yang sudah dikonversi ke HSV.

```python
mean = np.mean(img_hsv.ravel())
std = np.std(img_hsv.ravel())

print(mean, std)
```
PENJELASAN : untuk menghitung rata-rata dan standar deviasi dari nilai piksel dalam saluran V (nilai) dari gambar HSV, yang telah di-flatten untuk dihitung statistiknya.

```python
v_channel = (img_hsv[:, :, 2] * 255).astype('uint8')
glcm = graycomatrix(v_channel, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')
dissimilarity = graycoprops(glcm, 'dissimilarity')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')
ASM = graycoprops(glcm, 'ASM')
```
PENJELASAN : saluran V dari gambar HSV dikonversi ke skala 0-255 dengan mengalikan dengan 255 dan mengubahnya menjadi tipe data uint8 untuk memastikan nilai integer dalam rentang 0-255. Kemudian, GLCM (Gray-Level Co-occurrence Matrix) dibuat untuk saluran V dengan menggunakan jarak 1 pixel, arah 0 derajat (horizontal), dengan 256 level intensitas (0-255). GLCM yang dihasilkan adalah simetris dan dinormalisasi. Setelah itu, dilakukan ekstraksi fitur seperti kontras, disimilaritas, homogenitas, energi, korelasi, dan ASM (Angular Second Moment) dari GLCM untuk menganalisis tekstur gambar.

```python
print("Contrast:", contrast)
print("Dissimilarity:", dissimilarity)
print("Homogeneity:", homogeneity)
print("Energy:", energy)
print("Correlation:", correlation)
print("ASM:", ASM)
```
PENJELASAN : untuk mennampilkan hasil ekstraksi fitur dari GLCM, yang mewakili karakteristik tekstur dari saluran V gambar dalam ruang warna HSV.
