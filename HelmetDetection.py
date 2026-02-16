
from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd 
from tkinter import messagebox
import cv2 as cv
import os
from yoloDetection import detectObject, displayImage
# from tf_keras.models import model_from_json  # Lazy load
import yagmail
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ===================== GLOBALS =====================
main = tkinter.Tk()
main.title("Helmet Detection & Triple Riding Detection") 
main.geometry("900x650")
main.config(bg="#D6EAF8")  # Sky blue background

global filename, loaded_model, class_labels, cnn_model, cnn_layer_names, textarea
frame_count = 0 
frame_count_out = 0
pcount = 0
confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416      
option = 0
labels_value = []
current_plate_prediction = None # Cache for the current image

# -------------------- Global Variables for Models --------------------
plate_detecter = None
net = None
cnn_model = None

def load_plate_model():
    global plate_detecter
    if plate_detecter is None:
        print("Loading Number Plate CNN model...")
        from tf_keras.models import model_from_json
        with open('Models/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            plate_detecter = model_from_json(loaded_model_json)
        plate_detecter.load_weights("Models/model_weights.h5")
        print("Number Plate CNN model loaded.")
    
    # Pre-load labels if not already
    global labels_value
    if not labels_value:
        with open("Models/labels.txt", "r") as file:
            for line in file:
                labels_value.append(line.strip())
        print(f"Loaded {len(labels_value)} plate labels.")

def load_yolo_helmet():
    global net
    if net is None:
        print("Loading Helmet Detection YOLO model...")
        modelConfiguration = "Models/yolov3-obj.cfg"
        modelWeights = "Models/yolov3-obj_2400.weights"
        net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        print("Helmet Detection YOLO model loaded.")

def load_yolo_bike():
    global cnn_model, cnn_layer_names, class_labels
    if cnn_model is None:
        print("Loading Motor Bike Detection YOLO model...")
        class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n')
        cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights')
        layer_names = cnn_model.getLayerNames()
        out_layers = cnn_model.getUnconnectedOutLayers()
        if isinstance(out_layers, np.ndarray) and out_layers.ndim > 1:
            cnn_layer_names = [layer_names[i[0] - 1] for i in out_layers]
        else:
            cnn_layer_names = [layer_names[i - 1] for i in out_layers]
        print("Motor Bike Detection YOLO model loaded.")

# ===================== FUNCTIONS =====================
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    if isinstance(out_layers, np.ndarray) and out_layers.ndim > 1:
        return [layersNames[i[0] - 1] for i in out_layers]
    else:
        # For newer OpenCV versions where it returns a 1D array
        return [layersNames[i - 1] for i in out_layers]

def loadLibraries(): 
    # This is now handled by load_yolo_bike
    load_yolo_bike()

def upload():
    global filename, current_plate_prediction
    filename = filedialog.askopenfilename(initialdir="bikes")
    current_plate_prediction = None # Reset cache on new upload

def log_and_email_numberplate(number_plate, alert_type="helmet"):
    excel_file = "detected_numberplates.xlsx"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file, engine='openpyxl')
    else:
        df = pd.DataFrame(columns=["Timestamp", "NumberPlate", "AlertType"])

    new_row = {"Timestamp": timestamp, "NumberPlate": number_plate, "AlertType": alert_type}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False, engine='openpyxl')

    if alert_type == "helmet":
        body = f"No helmet detected — please wear a helmet for your safety.\nRegistration Number: {number_plate}\nTime: {timestamp}"
    elif alert_type == "triple":
        body = f"Triple riding detected! Please follow traffic rules.\nRegistration Number: {number_plate}\nTime: {timestamp}"

    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        receiver_email = os.getenv("RECEIVER_EMAIL")
        subject = "Traffic Alert"
        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(to=receiver_email, subject=subject, contents=body)
        print("Email sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)

def detectBike():
    global option, pcount
    if filename == "":
        messagebox.showwarning("Warning", "Please upload an image first!")
        return
    
    cv.destroyAllWindows() 
    main.config(cursor="wait")
    main.update()
    
    load_yolo_bike()
    
    option = 0
    pcount = 0
    label_colors = (0,255,0)
    try:
        image = cv.imread(filename)
        image_height, image_width = image.shape[:2]
        image, ops, pcount = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, 0)
        
        displayImage(image, 0)
        if ops == 1:
            option = 1
            # Check for Triple Riding (1 bike + 3+ people = pcount >= 4)
            if pcount >= 4:
                textarea.insert(END, "Violation: Triple Riding Detected!\n")
                load_plate_model()
                img = cv.resize(cv.imread(filename), (64,64))
                im2arr = np.array(img).reshape(1,64,64,3)/255.0
                preds = plate_detecter.predict(im2arr, verbose=0)
                detected_plate = str(labels_value[np.argmax(preds)])
                textarea.insert(END, f"Registration Number: {detected_plate}\n\n")
                log_and_email_numberplate(detected_plate, alert_type="triple")
                messagebox.showwarning("Triple Ride Alert", f"Triple riding detected!\nPlate: {detected_plate}")
        else:
            messagebox.showinfo("Info", "No Motorbike or Person detected in the image.")
            
    except Exception as e:
        messagebox.showerror("Error", f"Detection failed: {str(e)}")
    finally:
        main.config(cursor="")

def drawPred(classId, conf, left, top, right, bottom, frame, option):
    global frame_count, textarea
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    
    # Draw bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top_label = max(top, labelSize[1])
    cv.putText(frame, label, (left, top_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    label_name = label.split(':')[0]

    # Helmet detection logic
    if label_name == 'Helmet' and conf > 0.50:
        cv.putText(frame, "Helmet Detected", (left, bottom + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        textarea.insert(END, "Status: Helmet detected.\n\n")
    
    frame_count += 1
    return frame_count

def postprocess(frame, outs, option):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out = 0
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        if isinstance(i, (list, np.ndarray)):
            i = i[0]
        box = boxes[i]
        left, top, width, height = box
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame, option)

def detectHelmet():
    global textarea, frame_count_out
    if filename == "":
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    textarea.delete('1.0', END)
    if option == 1:
        main.config(cursor="wait")
        main.update()
        
        load_yolo_helmet()
        
        frame = cv.imread(filename)
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        
        frame_count_out = 0
        postprocess(frame, outs, 0)
        
        # If no helmet was detected during postprocess (frame_count_out remains 0 or small)
        if frame_count_out == 0:
            cv.putText(frame, "NO HELMET DETECTED", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            textarea.insert(END, "Violation: No Helmet Detected!\n")
            
            global current_plate_prediction
            if current_plate_prediction is None:
                load_plate_model()
                img = cv.resize(cv.imread(filename), (64,64))
                im2arr = np.array(img).reshape(1,64,64,3)/255.0
                preds = plate_detecter.predict(im2arr, verbose=0)
                current_plate_prediction = np.argmax(preds)
            
            detected_plate = str(labels_value[current_plate_prediction])
            textarea.insert(END, f"Registration Number: {detected_plate}\n\n")
            log_and_email_numberplate(detected_plate, alert_type="helmet")
            messagebox.showwarning("Helmet Alert", f"No helmet detected!\nPlate: {detected_plate}")
        
        frame = cv.resize(frame, (800, 500))
        main.config(cursor="")
        displayImage(frame, 0)
    else:
        messagebox.showinfo(
            "Person & Motor bike not detected",
            "Please detect person & motorbike first using 'Detect Motor Bike & Person'"
        )

# ===================== GUI =====================
# Title
title_font = ('Segoe UI', 18, 'bold')
title = Label(main, text="Helmet Detection & Triple Riding Detection using CNN and YOLO",
              bg="#2874A6", fg="white", font=title_font, pady=12)
title.pack(fill="x")

# Buttons
button_frame = Frame(main, bg="#D6EAF8")
button_frame.pack(pady=25)

btn_font = ('Segoe UI', 14, 'bold')

def on_enter(e, b, color):
    b['bg'] = color

def on_leave(e, b, color):
    b['bg'] = color

btn_upload = Button(button_frame, text="Upload Image", command=upload, font=btn_font, bg="#5DADE2", fg="white", width=30)
btn_upload.pack(pady=8)
btn_upload.bind("<Enter>", lambda e: on_enter(e, btn_upload, "#85C1E9"))
btn_upload.bind("<Leave>", lambda e: on_leave(e, btn_upload, "#5DADE2"))

btn_detect_bike = Button(button_frame, text="Detect Motor Bike & Person", command=detectBike, font=btn_font, bg="#3498DB", fg="white", width=30)
btn_detect_bike.pack(pady=8)
btn_detect_bike.bind("<Enter>", lambda e: on_enter(e, btn_detect_bike, "#5DADE2"))
btn_detect_bike.bind("<Leave>", lambda e: on_leave(e, btn_detect_bike, "#3498DB"))

btn_detect_helmet = Button(button_frame, text="Detect Helmet", command=detectHelmet, font=btn_font, bg="#2E86C1", fg="white", width=30)
btn_detect_helmet.pack(pady=8)
btn_detect_helmet.bind("<Enter>", lambda e: on_enter(e, btn_detect_helmet, "#5DADE2"))
btn_detect_helmet.bind("<Leave>", lambda e: on_leave(e, btn_detect_helmet, "#2E86C1"))

btn_exit = Button(button_frame, text="Exit", command=main.destroy, font=btn_font, bg="#CB4335", fg="white", width=30)
btn_exit.pack(pady=8)
btn_exit.bind("<Enter>", lambda e: on_enter(e, btn_exit, "#E74C3C"))
btn_exit.bind("<Leave>", lambda e: on_leave(e, btn_exit, "#CB4335"))

# Output Text Area
output_frame = Frame(main, bg="#D6EAF8")
output_frame.pack()
textarea = Text(output_frame, height=12, width=70, bg="#EBF5FB", fg="#02554C", font=("Consolas", 12), relief="solid", padx=8, pady=8)
textarea.pack(side="left")
scroll = Scrollbar(output_frame, command=textarea.yview)
scroll.pack(side="right", fill="y")
textarea.configure(yscrollcommand=scroll.set)

# Load YOLO libraries
loadLibraries()

# Start GUI loop
main.mainloop()
