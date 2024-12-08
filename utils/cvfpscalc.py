from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    # Inisialisasi objek dengan panjang buffer untuk menyimpan waktu per frame
    def __init__(self, buffer_len=1):
        # Menyimpan tick awal untuk perhitungan waktu
        self._start_tick = cv.getTickCount()
        # Menghitung frekuensi tick dalam satuan milidetik
        self._freq = 1000.0 / cv.getTickFrequency()
        # Membuat deque untuk menyimpan waktu per frame dengan panjang maksimum buffer_len
        self._difftimes = deque(maxlen=buffer_len)

    # Menghitung FPS berdasarkan waktu antar frame
    def get(self):
        # Mendapatkan tick saat ini
        current_tick = cv.getTickCount()
        # Menghitung selisih waktu antar frame dalam milidetik
        different_time = (current_tick - self._start_tick) * self._freq
        # Memperbarui tick awal untuk frame berikutnya
        self._start_tick = current_tick

        # Menyimpan waktu antar frame dalam deque
        self._difftimes.append(different_time)

        # Menghitung FPS dengan rata-rata waktu antar frame
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        # Membulatkan FPS ke 2 angka desimal
        fps_rounded = round(fps, 2)

        # Mengembalikan nilai FPS yang dibulatkan
        return fps_rounded
