from ultralytics import YOLO
import cv2
import math

# Webcam'den video akışını başlatma
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder; harici bir kamera kullanıyorsanız 1 veya 2 olarak değiştirin

# Modeli yükleme
model = YOLO('best.pt')

# Sınıf isimleri
classnames = [
    'Dangerous',        # 0
    'Helmet',           # 1
    'No-safety-vest',   # 2
    'boots',            # 3
    'glasses',          # 4
    'glove',            # 5
    'no boots',         # 6
    'no glasses',       # 7
    'no glove',         # 8
    'no helmet',        # 9
    'not dangerous',    # 10
    'safety vest'       # 11
]

# Güven eşiğini ayarlama
confidence_threshold = 50

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera bağlantısı kesildi.")
        break

    # Frame boyutunu ayarlama
    frame = cv2.resize(frame, (1280, 768))
    
    # Model ile tahmin yapma
    results = model(frame)

    # Tahmin sonuçlarını işleme
    for result in results:
        for box in result.boxes:
            confidence = box.conf.item() * 100  # Güven skoru yüzdesi
            class_id = int(box.cls[0])
            class_name = classnames[class_id]  # Sınıf adını al

            if confidence > confidence_threshold:  # Güven eşiği
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları al

                # Çerçeve çiz ve metni yerleştir
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name} {confidence:.1f}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
