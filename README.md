# Predicting Medical Appointment No-Shows: A Machine-Learning Approach to Reduce Missed Visits

**Abstract**

Missed medical appointments (no-shows) increase waiting times, waste clinical capacity, and lead to poorer health outcomes. This project builds a supervised machine learning pipeline to predict whether a scheduled appointment will be missed using routinely collected appointment and patient-level attributes such as age, gender, comorbidities (hypertension, diabetes, alcoholism, handicap), scholarship (health benefit) status, and SMS reminders. The dataset is imbalanced, with roughly 80% attended and 20% missed appointments, so the workflow includes data cleaning, class rebalancing using RandomUnderSampler, and model comparison with tree-based learners (Random Forest, XGBoost). Evaluation is carried out using F1-score, ROC-AUC, and confusion matrices on hold-out test data. The final tuned models reach strong discrimination on the rebalanced data, showing that early identification of likely no-show patients is feasible and can be integrated into reminder or outreach strategies to reduce missed visits.

**Introduction**

Healthcare providers across outpatient and primary-care settings often face the problem of patients not turning up for confirmed appointments. This leads to under-utilization of physicians and facilities, disrupts scheduling, and reduces access for other patients. A predictive model that flags high no-show risk patients a day or two before the appointment can trigger targeted interventions such as additional SMS, phone calls, or overbooking. 

**Objective**

The primary goal of this project is to develop a predictive model that identifies patients who are likely to miss their scheduled medical appointments (classified as “No-show”) based on historical appointment data. The model aims to help healthcare providers proactively reduce no-show rates, improve resource allocation, and enhance patient care through timely reminders or follow-ups.

**Dataset Description**

The dataset was read from a CSV file into a pandas dataframe and contains 110,526 records and 14 columns after filtering out rows with invalid ages (Age = -1). There are no missing values in the data. 
The key variables are:
* PatientId – identifies the patient (dropped before modeling).
* AppointmentID – identifies each appointment (dropped).
* ScheduledDay – date-time when the appointment was created.
* AppointmentDay – date when the patient is expected to visit.
* Age – patient age in years.
* Neighbourhood – clinic location.
* Scholarship – indicates enrolment in a welfare/benefit program.
* Hipertension, Diabetes, Alcoholism, Handcap – comorbidity indicators.
* SMS_received – whether an SMS reminder was sent.
* No-show – target label (Yes/No).

**Exploratory Data Analysis**

<img width="904" height="262" alt="image" src="https://github.com/user-attachments/assets/aa7085e6-b9a9-4f80-9308-fff97ef6a822" />

                                                     Figure 1

From the above figure 1, it can be observed that the dataset is highly imbalanced which is only ~20% of appointments are missed (“Yes”), while the remaining ~80% are attended (“No”). Among the 22,319 patients who missed their appointments, 12,535 (56%) did not receive an SMS reminder and 9,784 (44%) did receive an SMS reminder. A majority of patients who missed their appointments did not receive an SMS. However, a significant portion (44%) still missed their appointments despite receiving a reminder, suggesting that SMS reminders alone are not fully effective in preventing no-shows.

<img width="904" height="286" alt="image" src="https://github.com/user-attachments/assets/7c29f97b-e8d8-4a87-ac3e-7dde212ef64e" />

                                                     Figure 2

From the above figure 2, it can be observed that out of 22,319 no-show patients, 20,889 (93.6%) did not have diabetes and 1,430 (6.4%) had diabetes. Diabetes appears to have no strong influence on missing appointments - the vast majority of no-show patients do not have diabetes. Out of 22,319 no-show patients, 21,642 (97%) had no history of alcoholism and 677 (3%) had alcoholism. Patients with alcoholism form a very small proportion of the no-show group, suggesting that alcoholism is not a key driver of appointment misses in this dataset. Out of 22,319 no-show patients, 18,547 (83%) were not hypertensive and 3,772 (17%) had hypertension. While still a minority, hypertensive patients make up a larger portion of the no-show group compared to diabetic or alcoholic patients. This suggests that hypertension might have a slightly stronger association with the missed appointments.

<img width="904" height="272" alt="image" src="https://github.com/user-attachments/assets/49ed1469-191e-4284-bda5-69a74f2a3e75" />

                                                     Figure 3 

From the above figure 3, it can be observed that the median age of no-show patients appears to be slightly lower than that of patients who showed up. Most no-show patients fall within the 20–50 year age range. Younger patients are slightly more likely to miss appointments than older ones. This suggests that age may influence no-show behavior, possibly due to differing lifestyle priorities or responsibilities. Among the 22,319 patients who missed their appointments, 14,594 (65%) were female and 7,725 (35%) were male. Female patients account for a higher proportion of missed appointments.

**Data Preparation**

Features were created by dropping identifiers and non-predictive time-stamped fields: PatientId, AppointmentID, AppointmentDay, ScheduledDay, Neighbourhood, and the target 'No-show'. The remaining predictors (Gender, Age, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received) were then encoded. Because the original target was imbalanced, RandomUnderSampler was applied with a custom sampling strategy to create a working dataset of 10,000 'Yes' and 8,000 'No' records. This balances the classifier’s exposure to both classes and stabilizes F1-score. Categorical binary columns were also converted from {0,1} to {‘N’, ‘Y’} for compatibility with the one-hot encoder in the pipeline. 

**Modeling Approach**

A scikit-learn/imbalanced-learn pipeline was built with two stages. In the preprocessing stage, StandardScaler was applied to the numeric feature (Age), while categorical features—Gender, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, and SMS_received—were encoded using OneHotEncoder with handle_unknown='ignore' to robustly manage unseen categories. The modeling stage evaluated two classifiers: a RandomForestClassifier and an XGBClassifier, each tuned primarily over n_estimators and max_depth to balance bias–variance and capture non-linear interactions. Hyperparameter selection for both model families used RandomizedSearchCV with 3-fold cross-validation, optimizing the F1-score to better address the class imbalance and prioritize balanced precision–recall performance on the no-show (minority) class.

**Results**

Across models, tree-based learners delivered strong and stable performance on the resampled data. For Random Forest, candidates with deeper trees (≈10 levels) and large ensembles (up to 1000 estimators) produced the best cross-validated F1-scores (≈0.64). On the 12,600/5,400 train/test split, ROC curves showed clear class separation with test AUC closely tracking train AUC, indicating limited overfitting; corresponding confusion matrices reflected more balanced detection of both ‘Yes’ (no-show) and ‘No’ (show) classes under the resampling regime. For XGBoost, shallower models (max_depth = 3–5) with 20–150 trees matched or slightly exceeded Random Forest on F1 (≈0.65 on CV) while converging faster and using fewer parameters. ROC curves for train and test overlapped closely, further supporting good generalization. Taken together, these results suggest that while both approaches are viable, XGBoost is the stronger deployment candidate due to its competitive accuracy, lower complexity, and training efficiency, with Random Forest serving as a robust alternative when model simplicity and bagging-driven stability are preferred.

<img width="904" height="347" alt="image" src="https://github.com/user-attachments/assets/9a1e720a-37fa-4a35-b342-23de86277fad" />

                                                  Figure 4

<img width="904" height="344" alt="image" src="https://github.com/user-attachments/assets/1f9bfb94-ab8a-4288-87d6-20c4ded1c304" />

                                                  Figure 5

<img width="904" height="316" alt="image" src="https://github.com/user-attachments/assets/7f25a26d-99d1-4475-bb2c-edc2ee743860" />

                                                  Figure 6

<img width="904" height="305" alt="image" src="https://github.com/user-attachments/assets/295d2693-e956-425f-b296-f3ada620f75d" />

                                                  Figure 7

**Discussion**

The results show that handling class imbalance is the key to good performance. If I train on the raw data, the model leans toward predicting “No,” which inflates accuracy but misses many true no-shows (“Yes”). After resampling and optimizing on F1, both Random Forest and XGBoost capture useful signals from demographic, clinical, and reminder features, improving recall for the “Yes” class without overfitting. In short, balancing the data and tuning for F1 turns a high-accuracy-but-unhelpful model into one that actually finds the no-shows.

**Conclusion**

This project successfully delivered an end-to-end predictive pipeline for appointment no-shows on real-world data. After cleaning and rebalancing, tree-based ensembles performed consistently: on cross-validation the Random Forest reached F1 ≈ 0.64, while XGBoost matched/slightly exceeded it at ≈ 0.65; on the resampled train/test split (12,600 / 5,400), test ROC curves closely tracked train ROC, indicating limited overfitting, and confusion matrices showed balanced detection of both “Yes” and “No.” These results make operational deployment realistic (e.g., within an appointment-management dashboard) to prioritize reminder calls for high-risk patients, reduce wasted slots, and improve access.



