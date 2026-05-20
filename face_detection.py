import cv2

# Load model Haar Cascade bawaan OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Akses webcam
cap = cv2.VideoCapture(0)

print("Tekan Q untuk keluar")

while True:
    # Ambil frame dari webcam
    ret, frame = cap.read()

    if not ret:
        print("Webcam tidak terbaca")
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Gambar bounding box
    for (x, y, w, h) in faces:

        # Kotak wajah
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label
        cv2.putText(
            frame,
            "Wajah Terdeteksi",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Tampilkan hasil
    cv2.imshow("Face Detection", frame)

    # Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam
cap.release()
cv2.destroyAllWindows()