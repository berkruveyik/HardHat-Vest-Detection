import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('best.pt')

# Video dosyasını oku
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# VideoWriter ile çıkışı kaydedin (isteğe bağlı)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model(frame)

    # Sonuçları ekranda göster ve kaydet
    annotated_frame = results.render()
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    out.write(annotated_frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
