# TF-IDF Visualization
WordCloud, Bar Chart, Heatmap, PCA, and Cluster Map visualizations using TF-IDF features.
# Overview

This repository demonstrates how to extract, analyze, and visualize TF–IDF features from unstructured text datasets such as reviews, news articles, and tweets.
Visualization helps interpret important terms, document similarity, and category-level differences in text data.


# Folder Structure
tfidf-visualization/
├── data/
│   └── sample_docs.csv
├── examples/
│   ├── wordcloud_viz.py
│   ├── bar_chart_viz.py
│   ├── heatmap_viz.py
│   ├── pca_viz.py
│   ├── cluster_map_viz.py
│   └── comparative_viz.py
├── requirements.txt
└── README.md

# Installation

Clone or download the repository:

git clone https://github.com/<your-username>/tfidf-visualization.git
cd tfidf-visualization


Set up environment and install dependencies:

python -m venv venv
source venv/bin/activate     # on Windows use: venv\\Scripts\\activate
pip install -r requirements.txt

# Sample Dataset

File: data/sample_docs.csv

text,category
"Machine learning models improve with data","Technology"
"Deep learning outperforms traditional ML in image tasks","Technology"
"Economy grows as market stabilizes","Business"
"Stock prices increase with investor confidence","Business"
"Healthcare sector adopts AI for diagnosis","Health"
"New vaccines improve public health outcomes","Health"

# Running the Examples

Each script under examples/ corresponds to a specific visualization type.

| Script               | Visualization         | Description                                              |
| :------------------- | :-------------------- | :------------------------------------------------------- |
| `wordcloud_viz.py`   | **Word Cloud**        | Displays most important words (size ∝ TF–IDF score).     |
| `bar_chart_viz.py`   | **Bar Chart**         | Plots top 10 terms by average TF–IDF weight.             |
| `heatmap_viz.py`     | **Heatmap**           | Shows TF–IDF weights across documents.                   |
| `pca_viz.py`         | **PCA Scatter Plot**  | Visualizes document similarity in 2D space.              |
| `cluster_map_viz.py` | **Cluster Map**       | Combines clustering and heatmap for topic grouping.      |
| `comparative_viz.py` | **Comparative Chart** | Highlights discriminative vocabulary between categories. |

Run any script, for example:

python examples/wordcloud_viz.py

# Visualization Examples
| Technique            | Output                         |
| -------------------- | ------------------------------ |
| **Word Cloud**       | Highlights high-weight words   |
| **Bar Chart**        | Compares top TF–IDF terms      |
| **Heatmap**          | Visualizes TF–IDF distribution |
| **PCA Plot**         | Shows document clusters        |
| **Cluster Map**      | Groups related documents       |
| **Comparative Plot** | Compares two text categories   |

# Libraries Used

scikit-learn – TF–IDF feature extraction & PCA

wordcloud – Word cloud generation

matplotlib / seaborn – Plotting visualizations

pandas / numpy – Data manipulation

plotly / bokeh (optional) – Interactive visualization extensions
