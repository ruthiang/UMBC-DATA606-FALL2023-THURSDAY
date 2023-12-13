# Text Classification of Medical Device Recalls 

- Author: Ruth Iang
- Prepared for UMBC Data Science Masters Degree Capstone Class, Fall 2023
- https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth
- Linkedin: https://www.linkedin.com/in/ruth-iang
- Link to your PowerPoint presentation file: https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/blob/main/Docs/DATA%20606%20Capstone%20Project.pdf
- Link to your YouTube video

# Background
This project aims to categorize medical device recalls into three classes based on the severity of their violations of Food and Drug Administration (FDA) laws. Recalls occur when products need correction or removal due to potential risks of injury, deception, or defects, with the goal of safeguarding public health.

The FDA classifies recalls into three categories to assess the level of health hazard posed by the products. Class I indicates a high risk, potentially resulting in serious health complications or even death. Class II involves temporary or reversible medical consequences upon exposure or usage. Class III encompasses situations where product use or presence does not pose severe health risks.

The significance of medical device recalls lies in their role in safeguarding public well-being from potentially harmful devices. This project focuses on automating the classification of FDA devices into their respective categories based on provided recall reasons. Leveraging Natural Language Processing and Machine Learning Algorithms will streamline and expedite the classification process, eliminating the need for manual sorting.

# Data
- Sources: This dataset is from FDA Open source data website.
- Data Size: The dataset is 6.2 MB.
- Data Shape: There are 17 columns and there are 31012 rows.
- Each row represents a medical device that is recalled by the FDA in violation of their laws.
- The dataset has medical devices being recalled by the FDA for the last 11 years, from 2012 to October of 2023.

# Data Dictionary

    1.) FEI Number: Integer64
    2.) Recalling Firm Name: Object, tells us the name of the firms 
    3.) Product Type: Object, tells us what firm the product belongs to
    4.) Product Classification: Object, represents how harmful the product is to the public
    5.) Status: Object, this column shows whether the recall is still going on
    6.) Distribution Pattern: Object, where the product is being distributed
    7.) Recalling Firm City: Object, the city where the product is being recalled 
    8.) Recalling Firm State: Object, the state where the prodcut is being recalled
    9.) Recalling Firm Country: Object, the country where the product is recalled
    10.) Center Classification Date: Datetime64[ns], the date when the product is classified
    11.) Reason for Recall: Object, reason why the product is in recall
    12.) Product Description: Object, describes the product 
    13.) Event ID: Integer64: Object, the id of the product when in recall
    14.) Event Classification: Object, the resulting classification
    15.) Product ID: Integer64, the id representing the product
    16.) Center: Object,shows the center where the devices is being recalled
    17.) Recall Details: Object, the product,and event details along with the history

# Target Variable/Label
Only select columns will undergo analysis or be utilized in the models as not all are crucial to address the core business question. The focal variable will be "Event Classification," serving as the target variable. This column determines the classification of products post-investigation. Classification I denotes the most severe damage to public health, while Classification II signifies moderate health impact. Classification III implies minimal health concerns. The objective is to categorize medical device recalls into these specified classes based on their impact on public health.

# Potential Features/Predictors:
The feature variable for the model is the "Reason for Recall" column. This variable delineates the reasons cited for recalls, which directly correlate with assessing the potential or actual severity of damage to the community caused by the recalled medical devices.

# Exploratory Data Analysis
Data Cleansing: I conducted checks for null and duplicate values within the dataset, subsequently removing any instances found.

Data Preparation: Given that this dataset contains text data, I standardized the feature column, "Reason for Recall," by employing tokenization, converting all text to lowercase, eliminating common stopwords, and performing stemming on the words. This methodology aims to retain only the most pertinent words relevant to the context of the topic for subsequent data preprocessing. Furthermore, this process significantly contributes to enhancing the accuracy of device classification by focusing on crucial textual elements.

Visualizations:
  
<img width="499" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/a65e235f-d44e-4279-94b8-485f311eaa12">
  
The pie chart illustrates a predominant occurrence of Class II classifications within the dataset, indicating that a significant portion of the devices exhibit a moderate impact on health. Conversely, Class III instances are notably scarce, suggesting either a lower count of devices falling into this category or potentially fewer reported complaints associated with devices in this classification. As a result, both Class I and Class III are observed as minority classes in comparison.

<img width="550" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/578125a8-d662-4dc2-bbb7-492d2dd6bdc2">

The WordCloud visually depicts the most frequently occurring words within the dataset. A bilinear approach was utilized to visualize the relationship between two words and their significance within a single layout. Notably, words like "may," "result," "steril," and "compromis" stand out due to their larger font sizes, indicating their higher frequency within the feature variable. In this context, these words are understandably prevalent as they often relate to potential consequences associated with devices.

Additionally, there is a significant presence of terms such as "instruct," "use," "barrier," "contain," and "kit." This makes sense in terms of context, as these terms likely pertain to product packaging containing instructions for device utilization.

# Model Training and Deployment
The text feature underwent transformation using TF-IDF vectorization to enable machine learning comprehension. Addressing class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was applied, particularly beneficial for augmenting samples in minority classes, specifically Class III and Class I.

The first model used to classify the descriptions of the medical devices is Multinomial Naive Bayes. Since our classes cannot be classified accurately by binomial classifiers, multinomial classifiers are the ideal preferred choices. After training the model, it has an accuraacy of .937, A Precision of .937, F1-score of .936,and Recall of .937. The image below also displays the classification for how well the model is performing.

<img width="500" alt="Screen Shot 2023-11-07 at 9 18 50 PM" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/e67241c1-6a92-408b-ac0c-df3faa50b90b">

The confusion matrix heatmap provides a visual representation of the model's predictions across different classes and their accuracy. For instance, the model accurately classified 4,862 recalled medical devices into Class II, predicted 146 instances as Class I when they were actually Class III, and 238 instances as Class II when they were actually Class I. Overall, the model demonstrates proficiency in accurately categorizing the data into their respective classes, showcasing its effectiveness in classification tasks.

<img width="500" alt="Screen Shot 2023-11-07 at 9 21 11 PM" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/628aa88a-6bd6-42e7-ba52-0032d157e90f">

After thorough parameter tuning, the optimal value determined for the Multinomial Naive Bayes model is an alpha value of 0.1, achieving an accuracy of 0.954.Subsequently, employing the Random Forest Classifier as the second model for classification yielded notable results. Post-training, the Random Forest Classifier showcased an accuracy score of .995, a precision score of .937, and a recall of .995. The classification report below provides a detailed breakdown of the model's performance.

  <img width="500" alt="Screen Shot 2023-11-07 at 9 31 12 PM" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/e51aeb4a-5ae3-4860-afa6-9cdda7edb24f">

In addition, the confusion matrix heatmap also shows in greater detail how well the model is doing in classifying all of the devices, displaying both the model's predictions and the actual classifications of the devices.Following parameter tuning to mitigate potential overfitting, the accuracy score is 99.47%, which implies a high proportion of accurate predictions, therby affirming the model's exceptional performance in categorizing the investigated devices.

<img width="500" alt="Screen Shot 2023-11-07 at 9 36 33 PM" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/a76153c5-26d0-4f20-b0ac-a7566099e9a5">

Wrapping up the analysis, a simple neural network using Long Short-Term Memory (LSTM) was deployed to explore its potential for outperforming the previous machine learning models. Post-training, the LSTM model demonstrated an accuracy of .9278. Unfortunately, the LSTM model did not surpass the performance achieved by the machine learning models, indicating that the machine learning algorithms outperformed the LSTM model in this context.

# Conclusions
Using Naive Bayes Models, Random Forest Classifier, and Long Short-Term Memory for categorizing the reasons behind medical device recalls yielded reliable accuracy. Measures to prevent overfitting were employed, including the utilization of SMOTE to rectify the skewed data distribution. Following these meticulous steps for classification, the Random Forest Classifier emerged as the best-performing model, boasting an impressive accuracy score of .995.

This robust model presents a significant advantage in expediting and enhancing the accuracy of future FDA-recalled medical device classifications. The implementation of this model eliminates the need for manual classification, ensuring faster processing and consistently accurate results.

<img width="910" alt="image" src="https://github.com/DATA-606-2023-FALL-THURSDAY/Iang_Ruth/assets/98433448/32ac875c-78fe-4f22-8fe5-6b70a22cc54e">
