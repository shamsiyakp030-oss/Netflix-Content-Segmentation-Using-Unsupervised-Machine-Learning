#  Netflix Content Segmentation Using Unsupervised Machine Learning

##  Overview

This project focuses on segmenting Netflix content using Unsupervised Machine Learning techniques. By applying K-Means Clustering, the project identifies hidden patterns and groups similar titles based on attributes such as ratings, popularity, runtime, vote counts, and release year. The analysis helps uncover meaningful content categories without predefined labels.

---

##  Objectives

- Understand and explore the Netflix dataset
- Perform comprehensive Exploratory Data Analysis (EDA)
- Handle missing values and duplicates
- Detect and analyze outliers
- Scale numerical features for clustering
- Determine the optimal number of clusters
- Apply K-Means Clustering
- Visualize clusters using PCA
- Generate business insights from cluster characteristics

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

##  Exploratory Data Analysis

The EDA phase includes:

- Dataset shape and information
- Statistical summary
- Missing value analysis
- Duplicate record detection
- Histograms and distribution plots
- Outlier detection using Boxplots
- Correlation Heatmap
- Feature relationship analysis

### Key Findings

- Most content falls within average rating and popularity ranges.
- A few titles exhibit exceptionally high popularity and vote counts.
- Recent content dominates the Netflix catalog.
- Several numerical features contain outliers that represent highly successful titles.

---

##  Data Preprocessing

- Missing value treatment
- Duplicate removal
- Feature selection
- Standardization using StandardScaler
- Preparation of data for clustering

---

##  Clustering Methodology

### K-Means Clustering

K-Means clustering was used to group Netflix titles into similar segments based on content characteristics.

### Optimal Cluster Selection

The optimal number of clusters was determined using:

- Elbow Method
- Silhouette Score Analysis

### PCA Visualization

Principal Component Analysis (PCA) was applied to reduce dimensionality and visualize cluster separation in a two-dimensional space.

---

##  Cluster Insights

### Cluster 0
- Recent content
- High popularity
- High audience engagement

### Cluster 1
- Average-rated content
- Moderate popularity
- Standard viewing duration

### Cluster 2
- Premium content
- High ratings
- Strong viewer interest

### Cluster 3
- Niche content
- Lower popularity
- Specialized audience segments

---

##  Results

- Successfully segmented Netflix content into meaningful clusters.
- Identified hidden content patterns and audience preferences.
- Achieved clear cluster separation through PCA visualization.
- Generated actionable insights for content analysis and recommendation systems.

---

##  Business Applications

- Content Recommendation Systems
- Audience Segmentation
- Content Acquisition Strategy
- Personalized User Experience
- Trend Analysis and Content Planning

---

##  Conclusion

This project demonstrates the power of Unsupervised Machine Learning in discovering hidden structures within Netflix content data. Through EDA, preprocessing, clustering, and visualization, meaningful content segments were identified, providing valuable insights for business decision-making and recommendation systems.

---

### ⭐ If you found this project useful, don't forget to star the repository.
