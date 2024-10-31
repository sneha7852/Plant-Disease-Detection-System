# Plant Disease Detection System

Description:
This project aims to provide early detection of common paddy diseases to aid farmers in preventing crop losses and maintaining productivity. By leveraging deep learning and image processing techniques, this model automatically detects diseases such as bacterial leaf blight, brown spot, leaf blast, and leaf smut with high accuracy, eliminating the need for extensive manual monitoring.

Specifications and Features:

Disease Detection Models: Utilizes VGG16, ResNet, Inception v2, and MobileNet for enhanced classification accuracy.
Data Processing: Images of different disease types are stored in a specialized repository, with 80% of data used for training and 20% for testing.
High Precision and Low Error Rate: The system provides accurate classification results with minimized error.
Real-World Application: Designed for use in practical, agricultural settings to assist farmers and agronomists in timely disease intervention.
Tools and Technologies Used:

Deep Learning Framework: TensorFlow/Keras
Pre-trained Models: VGG16, ResNet, Inception v2, MobileNet
Languages: Python
Additional Libraries: OpenCV for image processing, NumPy, Pandas for data handling
Environment: Jupyter Notebook for development and model experimentation

How It Works
The Plant Disease Detection System is a machine learning-based tool that identifies common diseases affecting paddy crops. It combines image processing with deep learning to provide precise and automated disease detection. Here’s a breakdown of the workflow:

1. Data Collection and Preprocessing
Data Collection: A custom dataset of diseased and healthy paddy plant images is used. The dataset includes images of bacterial leaf blight, brown spot, leaf blast, leaf smut, and healthy plants.
Image Preprocessing: Images are standardized and resized to ensure uniformity. Preprocessing steps include scaling, normalization, and augmentation techniques like rotation, flipping, and brightness adjustments to improve model generalization and performance.
2. Model Training
Architecture Selection: Multiple deep learning architectures are explored, including VGG16, ResNet, Inception v2, and MobileNet. These models are selected due to their strengths in handling complex image recognition tasks.
Transfer Learning: Pre-trained versions of these models are fine-tuned using the paddy disease dataset, leveraging prior knowledge to achieve faster and more accurate training.
Training Process: Approximately 80% of the dataset is used for training the models. The data is fed into the models, which learn patterns associated with each disease category by adjusting weights through multiple iterations.
3. Testing and Evaluation
Testing Set: The remaining 20% of data is reserved for testing. This unseen data assesses the model's ability to accurately classify paddy diseases it has not encountered before.
Evaluation Metrics: Accuracy, precision, recall, and F1-score metrics are calculated to measure model performance. The system emphasizes achieving a high picture classification accuracy with a low error rate.
4. Prediction and Deployment
Image Input: Users can input images of paddy leaves directly into the model.
Prediction Output: The model processes the image, identifies disease patterns, and classifies the image into one of the known disease categories or as healthy. The output includes the predicted disease type and a confidence score.
Real-world Use: This tool is intended for practical agricultural applications, enabling quick, accessible, and accurate disease identification. Farmers can use it to assess crop health and intervene promptly, preventing disease spread and minimizing crop loss.
