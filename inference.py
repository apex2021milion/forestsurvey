from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO('best.pt')
inputs = cv2.imread('DJI_0221.JPG')
inputs = cv2.resize(inputs, (640, 640))
results = model(inputs)

res_plotted = results[0].plot()
print(res_plotted)
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()