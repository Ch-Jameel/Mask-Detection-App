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

## Main APP View
![Prediction Example 1](Predicted%20Samples/Screenshot%20(267).png)

## APP View when any sample Image Loaded using uploaded image
![Prediction Example 2](Predicted%20Samples/Screenshot%20(268).png)

## When Click Predict then Predicted Image
![Prediction Example 3](Predicted%20Samples/Screenshot%20(269).png)
## Upload another image and get the prediction
![Prediction Example 4](Predicted%20Samples/Screenshot%20(270).png)

## Upload another image and get a prediction
![Prediction Example 5](Predicted%20Samples/Screenshot%20(271).png)


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

## Very Important Note

I use the custom function to handle the dataset formats for YOLO. However, I did not upload the custom.py code file due to my code privacy
