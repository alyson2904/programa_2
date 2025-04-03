import cv2
import torch
from ultralytics import YOLO

# Cargar el modelo YOLO preentrenado (versión ligera para velocidad)Q
model = YOLO("yolov8n.pt")

def detect_persons(frame):
    results = model(frame) # Hacer detecciones
    for result in results:
     for box in result.boxes:
         cls = int(box.cls[0]) # Obtener la clase detectada
         if cls == 0: # Clase 0 en COCO es "persona"
           x1, y1, x2, y2 = map(int, box.xyxy[0])
           conf = box.conf[0].item() # Confianza

           if conf > 0.5: # Solo mostrar detecciones confiables
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Persona {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Captura de video en tiempo real
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_persons(frame)
    cv2.imshow("Detección de Personas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
