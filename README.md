# Helmet & Triple Riding Detection System 🏍️👷‍♂️

A production-grade AI solution for real-time traffic violation detection. This system uses **YOLOv3** for object detection (Motorbikes, Persons, Helmets) and a **CNN-based model** for automated Number Plate Recognition.

## 🌟 Key Features

- **Real-time Violation Detection**: Automatically identifies riders without helmets and motorcycles with 3+ passengers (Triple Riding).
- **Automated Evidence Collection**: Captures the registration number of the violating vehicle using OCR/CNN logic.
- **Instant Alerts**: Sends automated email notifications containing violation details to the authorities or owners.
- **Excel Logging**: Maintains a persistent record of all detected violations with timestamps and vehicle numbers.
- **Performance Optimized**: Built for Python 3.12 with lazy-loading models and non-blocking GUI for a smooth user experience.

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV, YOLOv3
- **Deep Learning**: TensorFlow, tf-keras, CNN
- **UI Framework**: Python Tkinter
- **Language**: Python 3.12.x
- **Data Handling**: Pandas, OpenPyXL
- **Messaging**: Yagmail (SMTP integration)

## 🚀 Installation & Setup

1. **Clone the Project**:
   ```bash
   git clone https://github.com/karthiktatineni/Helmet-tripleride-Detection.git
   cd Helmet-tripleride-Detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   SENDER_EMAIL=your-email@gmail.com
   SENDER_PASSWORD=your-app-password
   RECEIVER_EMAIL=authority-email@gmail.com
   ```

4. **Run the Application**:
   Double-click `run.bat` or run:
   ```bash
   python HelmetDetection.py
   ```

## 📈 System Workflow

1. **Upload Image**: Select a photo from the `bikes` folder or your local storage.
2. **Detect Motor Bike & Person**: The system identifies the vehicle and the number of riders.
3. **Triple Riding Check**: If 3+ people are found on a bike, a violation is logged.
4. **Detect Helmet**: The system crops the rider's head area to check for helmet presence.
5. **Enforcement**: If a violation occurs, the number plate is predicted and an email alert is sent.

## 📂 Project Structure

- `HelmetDetection.py`: Main GUI Application.
- `yoloDetection.py`: Core logic for YOLO object bounding and labeling.
- `Models/`: Contains the weight files (`.weights`), configurations (`.cfg`), and CNN model (`.json`).
- `requirements.txt`: List of necessary Python libraries.
- `detected_numberplates.xlsx`: Automated log of traffic violations.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
**Disclaimer**: This project is intended for educational and smart city planning purposes.
