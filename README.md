# Mask Detection Using YOLOv8

This project demonstrates real-time face mask detection using the YOLOv8 model. The model is trained to classify images into three categories:
- **With Mask**
- **Without Mask**
- **Incorrect Mask**

## Features
- **Accurate Mask Classification**: Identifies whether individuals are wearing masks correctly.
- **Streamlit App**: User-friendly interface for uploading and predicting mask usage in images.
- **MASK Detection**: Fast and efficient detection, making it suitable for health monitoring.

## Workflow
1. **Dataset**: Images labeled into three classes: `with mask`, `without mask`, and `incorrect mask`.
2. **Model Training**: YOLOv8 was used for training with custom augmentations.
3. **Prediction Results**: Example predictions can be found below:

![Prediction Example 1](predicted_results/example1.png)
*Prediction: Without Mask*

![Prediction Example 2](predicted_results/example2.png)
*Prediction: Incorrect Mask*

## How to Run the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Upload an image to get predictions.

## Conclusion
This YOLOv8-powered mask detection app ensures high accuracy and speed, making it ideal for real-world deployment in areas like public safety and health monitoring.
