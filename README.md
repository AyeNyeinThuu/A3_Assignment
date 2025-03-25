# Comparison of results from classification reports of both MLflow-registered model and Scikit-learn local model.
![image](https://github.com/user-attachments/assets/ffb2ed3b-8274-4055-8157-b768302114fe)
![image](https://github.com/user-attachments/assets/add1de7e-af49-49c3-8d90-4cb9165c3c45)
As per above screenshots of codes and results:
The results from the Scikit-learn local model and the MLflow-registered model are nearly identical, which confirms that the model was correctly logged to MLflow
and is functioning as expected. Both implementations are being fed the same polynomial-transformed input data, ensuring consistency in preprocessing.
The evaluation metrics—such as precision, recall, f1-score, and accuracy—are closely aligned, with only minimal differences observed at the third or 
fourth decimal place. These slight variations are normal and can be attributed to floating-point rounding and formatting precision. 
Overall, this confirms that both the local and MLflow-deployed versions of the model produce very similar performance.

# Meaning of Support 
"Support" refers to the number of actual instances of each class in the test dataset.
Extract from my classification reports:
Class	   Support
0       	443
1       	430
2       	360
3       	373
This means: 443 test samples belong to class 0, 430 to class 1, 360 to class 2, 373 to class 3. Support is important because it tells us how balanced 
(or imbalanced) our dataset is.For instance, if support for one class is very low, even a small number of errors can impact precision/recall significantly.

# prediction price range through app
![3](https://github.com/user-attachments/assets/c11c3850-470c-4b80-9e81-ed0012cd7e75)
![4](https://github.com/user-attachments/assets/00b93237-d3e6-48dc-b607-e71159711aa1)
![5](https://github.com/user-attachments/assets/8272a2f9-e6eb-4839-8518-8981dd08b938)

# MLflows screenshot
![7](https://github.com/user-attachments/assets/ece95c50-26ac-4c5d-9bbb-aec671cec8c3)
![6](https://github.com/user-attachments/assets/4898667e-f2f4-45c2-a84b-78ef27b525fa)


