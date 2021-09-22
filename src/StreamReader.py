import cv2
import torch

def score_frame(frame, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame)
    labels = results.xyxyn[0][:, -1].cpu().numpy().astype(int)
    cord = results.xyxyn[0][:, :-1].cpu().numpy()
    return labels, cord

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if row[4] < 0.6:
            continue
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        bgr = (0, 255, 0) # color of the box
        classes = model.names # Get the name of label index
        label_font = cv2.FONT_HERSHEY_SIMPLEX #Font for the label.
        cv2.rectangle(frame, \
                      (x1, y1), (x2, y2), \
                       bgr, 2) #Plot the boxes
        cv2.putText(frame,\
                    classes[labels[i]], \
                    (x1, y1), \
                    label_font, 0.9, bgr, 2) #Put a label over box.




# applies the model on the stream and displays the result
def performDetection(stream, model):
    while stream.isOpened():
        ret, frame = stream.read()

        res = score_frame(frame,model)
        plot_boxes(res,frame)

        cv2.imshow('Frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
stream = cv2.VideoCapture(0)

performDetection(stream, model)

stream.release()