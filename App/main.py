import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSlider, QFrame, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

class BirdClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐦 Bird Species Classifier")
        self.setFixedSize(900, 700)
        
        # Initialize variables
        self.model = None
        self.current_image_path = None
        self.current_audio_path = None
        
        # Initialize media player
        self.media_player = QMediaPlayer()
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.stateChanged.connect(self.media_state_changed)
        
        # Load model
        self.load_model()
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
    def load_model(self):
        """Load the Keras model"""
        try:
            MODEL_PATH = "Model/bird_classifier.keras"
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
    def setup_ui(self):
        """Setup the main UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title_label = QLabel("🐦 Bird Species Classifier")
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setObjectName("titleLabel")
        main_layout.addWidget(title_label)
        
        # Upload section
        upload_frame = QFrame()
        upload_frame.setObjectName("uploadFrame")
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setContentsMargins(20, 20, 20, 20)
        
        upload_title = QLabel("Upload Image")
        upload_title.setFont(QFont("Arial", 16, QFont.Bold))
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setObjectName("sectionTitle")
        upload_layout.addWidget(upload_title)
        
        self.upload_btn = QPushButton("📁 Browse and Upload Image")
        self.upload_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.upload_btn.setObjectName("uploadButton")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        upload_layout.addWidget(self.upload_btn)
        
        main_layout.addWidget(upload_frame)
        
        # Results section
        results_frame = QFrame()
        results_frame.setObjectName("resultsFrame")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(20, 20, 20, 20)
        
        results_title = QLabel("Classification Results")
        results_title.setFont(QFont("Arial", 16, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setObjectName("sectionTitle")
        results_layout.addWidget(results_title)
        
        # Content area
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Image display area
        self.image_frame = QFrame()
        self.image_frame.setObjectName("imageFrame")
        self.image_frame.setFixedSize(320, 320)
        image_layout = QVBoxLayout(self.image_frame)
        
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("Arial", 12))
        self.image_label.setObjectName("imageLabel")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setScaledContents(True)
        image_layout.addWidget(self.image_label)
        
        content_layout.addWidget(self.image_frame)
        
        # Info and controls area
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_frame.setMaximumWidth(350)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(20)
        
        # Prediction info
        prediction_frame = QFrame()
        prediction_frame.setObjectName("predictionFrame")
        prediction_layout = QVBoxLayout(prediction_frame)
        
        self.species_label = QLabel("Species: -")
        self.species_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.species_label.setObjectName("speciesLabel")
        self.species_label.setWordWrap(True)
        prediction_layout.addWidget(self.species_label)
        
        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.confidence_label.setObjectName("confidenceLabel")
        prediction_layout.addWidget(self.confidence_label)
        
        info_layout.addWidget(prediction_frame)
        
        # Audio controls
        audio_frame = QFrame()
        audio_frame.setObjectName("audioFrame")
        audio_layout = QVBoxLayout(audio_frame)
        
        audio_title = QLabel("🔊 Bird Sound")
        audio_title.setFont(QFont("Arial", 14, QFont.Bold))
        audio_title.setAlignment(Qt.AlignCenter)
        audio_title.setObjectName("audioTitle")
        audio_layout.addWidget(audio_title)
        
        # Play/Pause button
        self.play_btn = QPushButton("▶ Play Audio")
        self.play_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.play_btn.setObjectName("playButton")
        self.play_btn.clicked.connect(self.toggle_audio)
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setEnabled(False)
        audio_layout.addWidget(self.play_btn)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setObjectName("progressSlider")
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderPressed.connect(self.slider_pressed)
        self.progress_slider.sliderReleased.connect(self.slider_released)
        audio_layout.addWidget(self.progress_slider)
        
        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setFont(QFont("Arial", 10))
        self.current_time_label.setObjectName("timeLabel")
        
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setFont(QFont("Arial", 10))
        self.total_time_label.setObjectName("timeLabel")
        
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        audio_layout.addLayout(time_layout)
        
        info_layout.addWidget(audio_frame)
        info_layout.addStretch()
        
        content_layout.addWidget(info_frame)
        results_layout.addLayout(content_layout)
        
        main_layout.addWidget(results_frame)
        
    def apply_styles(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            
            #titleLabel {
                color: #ecf0f1;
                margin: 10px 0px 20px 0px;
            }
            
            #uploadFrame, #resultsFrame {
                background-color: #34495e;
                border: 2px solid #34495e;
                border-radius: 10px;
            }
            
            #sectionTitle {
                color: #ecf0f1;
                margin-bottom: 10px;
            }
            
            #uploadButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                margin: 10px 0px;
            }
            
            #uploadButton:hover {
                background-color: #2980b9;
            }
            
            #uploadButton:pressed {
                background-color: #21618c;
            }
            
            #imageFrame {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 8px;
            }
            
            #imageLabel {
                color: #95a5a6;
                background-color: #2c3e50;
            }
            
            #infoFrame {
                background-color: #34495e;
            }
            
            #predictionFrame {
                background-color: #2c3e50;
                border-radius: 8px;
                padding: 15px;
            }
            
            #speciesLabel {
                color: white;
                text-align: left;
                margin-bottom: 8px;
            }
            
            #confidenceLabel {
                text-align: left;
                color: #27ae60;
            }
            
            #audioFrame {
                background-color: #2c3e50;
                border-radius: 8px;
                padding: 15px;
            }
            
            #audioTitle {
                color: #ecf0f1;
                margin-bottom: 15px;
            }
            
            #playButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 6px;
                margin-bottom: 15px;
            }
            
            #playButton:hover {
                background-color: #229954;
            }
            
            #playButton:pressed {
                background-color: #1e8449;
            }
            
            #playButton:disabled {
                background-color: #566573;
                color: #95a5a6;
            }
            
            #progressSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 6px;
                background: #34495e;
                border-radius: 3px;
            }
            
            #progressSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #2980b9;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -6px 0;
            }
            
            #progressSlider::sub-page:horizontal {
                background: #3498db;
                border-radius: 3px;
            }
            
            #timeLabel {
                color: #95a5a6;
                margin-top: 8px;
            }
        """)
        
    def upload_image(self):
        """Handle image upload and prediction"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp *.gif *.tiff);;All files (*.*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_image(file_path)
            
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Load image with PIL first to handle various formats
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')  # Ensure RGB format
            
            # Save as temporary file for QPixmap
            temp_path = "temp_display.jpg"
            pil_image.save(temp_path, "JPEG")
            
            # Load with QPixmap and scale
            pixmap = QPixmap(temp_path)
            scaled_pixmap = pixmap.scaled(380, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            
    def predict_image(self, image_path):
        """Predict the bird species"""
        if self.model is None:
            QMessageBox.critical(self, "Error", "Model not loaded!")
            return
            
        try:
            predicted_label, confidence = self.predict_and_display(image_path)
            
            # Update labels
            self.species_label.setText(f"Species: {predicted_label}")
            self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
            
            # Get audio path and enable play button
            self.current_audio_path = self.get_bird_audio_path(predicted_label)
            
            if os.path.exists(self.current_audio_path):
                self.play_btn.setEnabled(True)
                self.progress_slider.setEnabled(True)
                self.play_btn.setText("▶ Play Audio")
            else:
                self.play_btn.setEnabled(False)
                self.play_btn.setText("Audio Not Found")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")
            
    def predict_and_display(self, img_path):
        """Predict bird species using the model"""
        class_indices = {'ABBOTTS BABBLER': 0, 'ABBOTTS BOOBY': 1, 'ABYSSINIAN GROUND HORNBILL': 2,
                        'AFRICAN CROWNED CRANE': 3, 'AFRICAN EMERALD CUCKOO': 4, 'AFRICAN FIREFINCH': 5,
                        'AFRICAN OYSTER CATCHER': 6, 'AFRICAN PIED HORNBILL': 7, 'AFRICAN PYGMY GOOSE': 8,
                        'ALBATROSS': 9, 'ALBERTS TOWHEE': 10, 'ALEXANDRINE PARAKEET': 11,
                        'ALPINE CHOUGH': 12, 'ALTAMIRA YELLOWTHROAT': 13, 'AMERICAN AVOCET': 14,
                        'AMERICAN BITTERN': 15, 'AMERICAN COOT': 16, 'AMERICAN FLAMINGO': 17,
                        'AMERICAN GOLDFINCH': 18, 'AMERICAN KESTREL': 19}
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
        # Make prediction
        prediction = self.model.predict(img_array)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index] * 100
        
        # Get label from class_indices
        labels = list(class_indices.keys())
        predicted_label = labels[predicted_index]
        
        return predicted_label, confidence
        
    def get_bird_audio_path(self, predicted_label):
        """Get the audio file path for the predicted bird"""
        audio_path = os.path.join("Birds Sounds", f"{predicted_label}.mp3")
        print(f"Audio Path is: {audio_path}")
        return audio_path
        
    def toggle_audio(self):
        """Toggle audio play/pause"""
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.critical(self, "Error", "Audio file not found!")
            return
            
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        elif self.media_player.state() == QMediaPlayer.PausedState:
            self.media_player.play()
        else:
            # Load and play
            url = QUrl.fromLocalFile(os.path.abspath(self.current_audio_path))
            self.media_player.setMedia(QMediaContent(url))
            self.media_player.play()
            
    def media_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸ Pause")
            self.play_btn.setStyleSheet("""
                #playButton {
                    background-color: #e67e22;
                }
                #playButton:hover {
                    background-color: #d35400;
                }
                #playButton:pressed {
                    background-color: #ba4a00;
                }
            """)
        else:
            self.play_btn.setText("▶ Play")
            self.play_btn.setStyleSheet("")
            
    def position_changed(self, position):
        """Update progress slider when position changes"""
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(position)
            
        # Update current time label
        self.current_time_label.setText(self.format_time(position))
        
    def duration_changed(self, duration):
        """Update slider range when duration is available"""
        self.progress_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))
        
    def slider_pressed(self):
        """Handle slider press"""
        pass
        
    def slider_released(self):
        """Handle slider release - seek to position"""
        self.media_player.setPosition(self.progress_slider.value())
        
    def format_time(self, ms):
        """Format time in MM:SS format"""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}:{seconds:02d}"

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = BirdClassifierGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()