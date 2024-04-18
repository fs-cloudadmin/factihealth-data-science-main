# Emergency Department Patient Prediction Puzzle

In the heart of a bustling city, a dedicated team of healthcare professionals is on a mission to provide immediate care and support to patients in need. But they face an ongoing and relentless challenge - the unpredictable influx of patients into their Emergency Department (ED). This is not a tale of the past; it's a current and pressing problem that demands a solution.

## Introduction: Addressing the ED's Unpredictable Challenge

The Emergency Department (ED) is a crucial part of the healthcare system, where every second counts. Yet, the team faces an unpredictable and ever-changing stream of patients. The task at hand is to predict the number of patients who will arrive at the ED in the next seven days. The aim is to provide healthcare providers with the necessary tools to optimize resource allocation, staff management, and overall operational efficiency. This, in turn, enhances patient care and ensures the ED's ability to meet the growing demand for immediate medical attention.

## Data Source: A Wealth of Information

To tackle this ongoing challenge, we used the MIMIC-IV-ED database, a remarkable de-identified dataset of ED admissions from the Beth Israel Deaconess Medical Center, spanning from 2011 to 2019. This data treasure trove is not a thing of the past; it's a contemporary source of invaluable insights. It complies with the Health Information Portability and Accountability Act (HIPAA) Safe Harbor provisions, designed for educational initiatives and research studies.

## Data Description: Uncovering Key Insights

We delved deep into the MIMIC-IV-ED database, mining crucial information from key tables:
1. Edstays Table
2. Diagnosis Table
3. Medrecon Table
4. Pyxis Table
5. Triage Table
6. Vitalsign Table

## Modeling: Real-Time Insights

Armed with current data, our healthcare professionals embarked on a journey into the world of predictive modeling, offering real-time solutions:
1. SARIMAX Model
2. Prophet Model

### Metric Comparison:
| Metric                        | SARIMAX | Prophet | Preferred Model |
|-------------------------------|---------|---------|-----------------|
| Mean Absolute Error           | 112.16  | 143.39  | SARIMAX         |
| Mean Squared Error            | 18769.84| 28795.71| SARIMAX         |
| Root Mean Squared Error       | 137.00  | 169.69  | SARIMAX         |
| Mean Absolute Percentage Error| 37.49%  | 34.52%  | Prophet         |

## The Power of Streamlit

To democratize access to these predictive models, the team turned to Streamlit, a powerful framework that transforms data scripts into shareable web applications. This decision not only made the models accessible to healthcare professionals but also facilitated seamless integration into existing hospital workflows.

## Key Performance Indicators: Measuring Success Now

To navigate their journey and gauge success, our healthcare professionals established a set of Key Performance Indicators (KPIs):
1. Patient Counts
2. Stay Duration
3. Bed Utilization
4. Staff Availability
5. Admission to Ward
6. No. of Admissions
7. Wait Time

## Files to Consider:

1. MimicDatasetEDA
2. DataSeasonalityTrendsIntroduction
3. DataSetExploreDailyScaled
4. host_ed.py
5. Data, Model, Report

## Conclusion:

The integration of advanced data science techniques with the intricacies of healthcare, as demonstrated in our patient prediction model for emergency department scenarios using the MIMIC-IV-ED dataset, holds profound implications for real-world applications. Our meticulous data transformation process and the strategic application of cutting-edge time series models, including PROPHET and SARIMAX, mark a pivotal advancement in predicting patient in the Emergency Department.

These models are not mere intellectual exercises; they represent tangible tools that can revolutionize how hospitals operate. By accurately forecasting patient flow, healthcare institutions can proactively allocate resources, optimize staff deployment, and streamline emergency department workflows. The integration of Streamlit further democratizes access to these predictive insights, ensuring that healthcare professionals at all levels can leverage the power of data-driven decision-making.

In the real world, this translates to improved patient outcomes, reduced wait times, and enhanced overall healthcare delivery. Hospitals equipped with these predictive tools gain a strategic edge in managing patient influx, thereby fostering a more efficient and responsive healthcare system. As we navigate the evolving landscape of healthcare, these models stand not just as technological achievements but as beacons of progress, poised to make a meaningful impact on the frontline of patient care.

