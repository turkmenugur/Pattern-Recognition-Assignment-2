# Predicting Biological Age Using Gut Microbe Composition  

## Introduction  
This project focuses on solving a machine learning regression problem using a unique dataset: predicting biological age based on gut microbe composition. The project demonstrates the application of machine learning techniques to biological data for predicting a continuous variable, biological age.  
The study indicating this is published in the following paper:  
[Link to study](https://www.sciencedirect.com/science/article/pii/S2589004220303849)  

---

## Regression Problem  

### The Dataset  
- **Dataset Description**:  
  The dataset contains the composition of gut microbes (bacteria and viruses) extracted from stool samples of individuals, along with their biological age.  
- **Dataset URL**:  
  [Download the dataset](https://drive.google.com/file/d/1Lln7SEB3dPz0g6kf3uM3SG-HyRpMKpwX/view?usp=sharing)  

### Data Cleaning  
- Missing values in the dataset were handled by removal.  
- The dataset was cleaned to ensure only non-label features (microorganism composition) were used for training.  

### Data Splitting and Training  
- The dataset was split into **training (80%)** and **testing (20%)** sets to evaluate the model's performance.  
- Features were standardized using `StandardScaler` to ensure all variables had a mean of 0 and a standard deviation of 1.  

---

## Model  

### Algorithm  
**XGBoost Regressor**  

### Model Parameters  
- **Number of estimators**: 200  
- **Learning rate**: 0.1  
- **Maximum depth**: 6  
- **Regularization parameters**:  
  - L1 (α): 1  
  - L2 (λ): 1  
- **Subsampling ratio**: 80%  
- **Column sampling ratio**: 80%  

### Evaluation Metrics  
- **Mean Absolute Error (MAE)**:  
  Measures the average magnitude of errors in predicted ages.  
- **R-squared (R²)**:  
  Indicates the proportion of variance in the biological age explained by the model.
  ### Output
![image](https://github.com/user-attachments/assets/e4bb60b2-86aa-483c-8b94-4fb04dcfb51b)
![image](https://github.com/user-attachments/assets/9056bfae-5efa-4c97-b912-b00e223653f7)
![image](https://github.com/user-attachments/assets/cacaad67-0ade-4458-979d-abfaca1eb204)



