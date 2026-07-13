    import numpy as np

    # ===========================
    # Creating Arrays
    # ===========================

    arr1 = np.array([10, 20, 30, 40, 50])
    print("1D Array:", arr1)

    arr2 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    print("\n2D Array:\n", arr2)

    # ===========================
    # Special Arrays
    # ===========================

    print("\nZeros Array:")
    print(np.zeros((2, 3)))

    print("\nOnes Array:")
    print(np.ones((3, 3)))

    print("\nIdentity Matrix:")
    print(np.eye(4))

    print("\nArray with Range:")
    print(np.arange(1, 11))

    print("\nEven Numbers:")
    print(np.arange(2, 21, 2))

    print("\nLinearly Spaced Values:")
    print(np.linspace(0, 10, 5))

    # ===========================
    # Array Properties
    # ===========================

    print("\nShape:", arr2.shape)
    print("Size:", arr2.size)
    print("Dimensions:", arr2.ndim)
    print("Data Type:", arr2.dtype)

    # ===========================
    # Mathematical Operations
    # ===========================

    print("\nAddition:", arr1 + 5)
    print("Subtraction:", arr1 - 5)
    print("Multiplication:", arr1 * 2)
    print("Division:", arr1 / 2)
    print("Square:", arr1 ** 2)
    print("Square Root:", np.sqrt(arr1))

    # ===========================
    # Statistical Functions
    # ===========================

    print("\nSum:", np.sum(arr1))
    print("Mean:", np.mean(arr1))
    print("Median:", np.median(arr1))
    print("Maximum:", np.max(arr1))
    print("Minimum:", np.min(arr1))
    print("Standard Deviation:", np.std(arr1))
    print("Variance:", np.var(arr1))

# ===========================
# Indexing and Slicing
# ===========================

print("\nFirst Element:", arr1[0])
print("Last Element:", arr1[-1])
print("Elements from Index 1 to 3:", arr1[1:4])

# ===========================
# Reshaping Arrays
# ===========================

a = np.arange(1, 13)
print("\nOriginal Array:")
print(a)

print("\nReshaped to 3x4:")
print(a.reshape(3, 4))

# ===========================
# Array Concatenation
# ===========================

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("\nConcatenated Array:")
print(np.concatenate((a, b)))

# ===========================
# Sorting
# ===========================

arr = np.array([8, 3, 9, 1, 5])

print("\nOriginal:", arr)
print("Sorted:", np.sort(arr))

# ===========================
# Random Numbers
# ===========================

print("\nRandom Integers:")
print(np.random.randint(1, 100, 5))

print("\nRandom Decimal Numbers:")
print(np.random.rand(3))

# ===========================
# Matrix Operations
# ===========================

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("\nMatrix Addition:")
print(A + B)

print("\nMatrix Subtraction:")
print(A - B)

print("\nMatrix Multiplication:")
print(np.dot(A, B))

print("\nTranspose of Matrix A:")
print(A.T)

# ===========================
# Logical Operations
# ===========================

print("\nElements Greater Than 25:")
print(arr1 > 25)

print("\nElements Greater Than 25:")
print(arr1[arr1 > 25])

# ===========================
# Aggregate Operations
# ===========================

matrix = np.array([[2, 4, 6],
                   [8, 10, 12]])

print("\nRow-wise Sum:")
print(np.sum(matrix, axis=1))

print("\nColumn-wise Sum:")
print(np.sum(matrix, axis=0))
# Import Libraries
import pandas as pd
import seaborn as sns

# ===========================
# Load Titanic Dataset
# ===========================
df = sns.load_dataset("titanic")

# ===========================
# Display Data
# ===========================
print("First 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nRandom 5 Rows:")
print(df.sample(5))

# ===========================
# Dataset Information
# ===========================
print("\nShape of Dataset:")
print(df.shape)

print("\nNumber of Rows:")
print(df.shape[0])

print("\nNumber of Columns:")
print(df.shape[1])

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

# ===========================
# Statistical Summary
# ===========================
print("\nStatistical Summary:")
print(df.describe())

print("\nSummary Including Categorical Columns:")
print(df.describe(include='all'))

# ===========================
# Missing Values
# ===========================
print("\nMissing Values:")
print(df.isnull().sum())

# ===========================
# Duplicate Values
# ===========================
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ===========================
# Unique Values
# ===========================
print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nUnique Passenger Classes:")
print(df["class"].unique())

import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris_raw = load_iris()


iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)

# Map numeric targets to taxonomic species names
iris_df['species_id'] = iris_raw.target
species_mapping = {i: name for i, name in enumerate(iris_raw.target_names)}
iris_df['species_name'] = iris_df['species_id'].map(species_mapping)

print("Iris dimensions:", iris_df.shape)
print("Missing entries checking:\n", iris_df.isnull().sum())
print(iris_df.head(3))


# Calculating central tendency across variables
for col in iris_raw.feature_names:
    mean_val = iris_df[col].mean()
    median_val = iris_df[col].median()
    mode_val = iris_df[col].mode()[0]
    print(f"Feature: {col}")
    print(f"  Mean:   {mean_val:.4f} cm")
    print(f"  Median: {median_val:.4f} cm")
    print(f"  Mode:   {mode_val:.4f} cm")
    print("-" * 40)


# Calculating dispersion characteristics
for col in iris_raw.feature_names:
    variance = iris_df[col].var()
    std_dev = iris_df[col].std()
    data_range = iris_df[col].max() - iris_df[col].min()
    q1 = iris_df[col].quantile(0.25)
    q3 = iris_df[col].quantile(0.75)
    print(f"Feature: {col}")
    print(f"  Variance:           {variance:.4f}")
    print(f"  Std Deviation:      {std_dev:.4f}")
    print(f"  Range (Max - Min):  {data_range:.4f}")
    print(f"  IQR (Q3 - Q1):      {q3 - q1:.4f}")
    print("-" * 40)

# Evaluating distribution moments
for col in iris_raw.feature_names:
    skewness = iris_df[col].skew()
    kurtosis = iris_df[col].kurt()
    print(f"Feature: {col}")
    print(f"  Skewness (Asymmetry): {skewness:.4f}")
    print(f"  Kurtosis (Tails):     {kurtosis:.4f}")
    print("-" * 40)

# Segmenting by class species using groupby
grouped_species = iris_df.groupby('species_name')

print("--- Group-Wise Mean Values ---")
print(grouped_species[iris_raw.feature_names].mean())

print("\n--- Group-Wise Standard Deviation Profiles ---")
print(grouped_species[iris_raw.feature_names].std())


aggregated_stats = iris_df.groupby('species_name')['petal length (cm)'].agg(['mean', 'median', 'std', 'skew'])
print("Aggregated Statistics (Petal Length):\n", aggregated_stats)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris Dataset
df = sns.load_dataset("iris")

# Display first five rows
print(df.head())
# Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["sepal_length"], bins=15, kde=True)
plt.title("Histogram of Sepal Length")
plt.show()

# Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x="species", data=df)
plt.title("Species Count")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(y="petal_length", data=df)
plt.title("Box Plot of Petal Length")
plt.show()

# Violin Plot
plt.figure(figsize=(6,4))
sns.violinplot(y="petal_width", data=df)
plt.title("Violin Plot of Petal Width")
plt.show()

# Density Plot
plt.figure(figsize=(6,4))
sns.kdeplot(df["sepal_width"], fill=True)
plt.title("Density Plot of Sepal Width")
plt.show()
# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="sepal_length", y="petal_length",
                hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.show()

# Pair Plot
sns.pairplot(df, hue="species")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Box Plot by Species
plt.figure(figsize=(6,4))
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length by Species")
plt.show()

# Violin Plot by Species
plt.figure(figsize=(6,4))
sns.violinplot(x="species", y="sepal_width", data=df)
plt.title("Sepal Width by Species")
plt.show()

# Bar Plot
plt.figure(figsize=(6,4))
sns.barplot(x="species", y="petal_width", data=df)
plt.title("Average Petal Width by Species")
plt.show()
