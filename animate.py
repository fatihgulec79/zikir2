import subprocess
import cv2
import numpy as np

W, H, FPS, DUR = 1920, 1080, 30, 60.048
N = int(FPS * DUR)

img = cv2.imread("1794338272717.jpg")
ih, iw = img.shape[:2]
S = 2000.0 / iw  # koordinatlar 2000 px genişlikte tanımlı
# çıktıdan biraz büyük çalış (zoom + kenar payı için)
base = cv2.resize(img, (int(iw * 1.0), int(ih * 1.0)))
sx = iw / 2000.0

# (x merkez, şapka tepe y, taban y, genlik px, faz) - 2000 px genişlik koordinatları
people = [
    (60, 350, 800, 6, 0.0),
    (170, 360, 760, 7, 0.7),
    (440, 310, 740, 9, 1.4),
    (730, 300, 700, 10, 2.1),
    (1000, 310, 670, 10, 2.8),
    (1270, 320, 700, 9, 3.5),
    (1540, 325, 740, 9, 4.2),
    (1860, 305, 600, 8, 4.9),
    (1930, 380, 900, 7, 5.6),
]

yy, xx = np.mgrid[0:ih, 0:iw].astype(np.float32)
X = xx / sx
Y = yy / sx

# Kişi maskeleri (zamandan bağımsız)
person_w = []
for cx, top, bot, amp, ph in people:
    wx = np.exp(-((X - cx) ** 2) / (2 * 130.0**2))
    h = np.clip((bot - Y) / (bot - top), 0, 1) ** 1.4
    person_w.append((wx * h).astype(np.float32))

# Duman maskesi
smoke_cx, smoke_bot, smoke_top = 880, 600, 130
smoke_m = (
    np.exp(-((X - smoke_cx) ** 2) / (2 * 90.0**2))
    * np.clip((smoke_bot - Y) / (smoke_bot - smoke_top), 0, 1)
    * (Y < smoke_bot)
).astype(np.float32)
smoke_m = np.clip(smoke_m * 1.5, 0, 1)

# Pencere ışığı maskesi (parlak bölgeler)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
glow = np.clip((gray - 0.75) / 0.25, 0, 1)
glow = cv2.GaussianBlur(glow, (0, 0), 6)[..., None]

enc = subprocess.Popen(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
     "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", "zikir12.mp3",
     "-t", str(DUR),
     "-vf", "fade=t=in:st=0:d=2,fade=t=out:st=58:d=2,format=yuv420p",
     "-af", "afade=t=in:d=2,afade=t=out:st=58:d=2",
     "-c:v", "libx264", "-preset", "fast", "-crf", "20",
     "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
     "zikir_klip2.mp4"],
    stdin=subprocess.PIPE)

omega = 2 * np.pi * 0.5  # 0.5 Hz ritmik sallanma
for f in range(N):
    t = f / FPS
    dx = np.zeros((ih, iw), np.float32)
    dy = np.zeros((ih, iw), np.float32)
    for (cx, top, bot, amp, ph), m in zip(people, person_w):
        s = np.sin(omega * t + ph)
        dx += m * amp * sx * s
        dy += m * 3 * sx * np.sin(2 * omega * t + ph)
    # duman: yükselirken dalgalanır
    dx += smoke_m * 14 * sx * np.sin(Y / 28.0 - t * 2.2) * np.clip((smoke_bot - Y) / 300, 0, 1.5)
    dy += smoke_m * 6 * sx * np.sin(X / 20.0 + t * 1.3)

    # yavaş zoom + hafif kayma
    z = 1.0 + 0.10 * f / N
    cxp, cyp = iw / 2, ih / 2
    mx = (xx - cxp) / z + cxp + dx
    my = (yy - cyp) / z + cyp + dy
    frame = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # pencere ışığı titreşimi + duman parlaması
    flick = 0.04 * np.sin(t * 1.7) + 0.02 * np.sin(t * 5.3)
    fr = frame.astype(np.float32) * (1 + glow * flick)
    frame = np.clip(fr, 0, 255).astype(np.uint8)

    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    enc.stdin.write(frame.tobytes())

enc.stdin.close()
enc.wait()
