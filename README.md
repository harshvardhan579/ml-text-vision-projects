# ml-text-vision-projects

This repository contains three applied machine learning projects completed as part of a semester-long graduate-level course. The focus spans both natural language processing and computer vision, showcasing end-to-end solutions including data preprocessing, model development, evaluation, and benchmarking.

## 📁 Project Overviews

### 1. 📨 Spam Classification using Naïve Bayes and Bag-of-Words
- Built a spam detector using the UCI SMS Spam Collection and a Kaggle email dataset.
- Applied text preprocessing and tokenization to construct a Bag-of-Words model.
- Trained a custom Naïve Bayes classifier and benchmarked it against `scikit-learn`'s MultinomialNB with Laplace smoothing.
- Evaluated with accuracy, precision, recall, F1-score, and confusion matrices on both SMS and email datasets.

**Technologies:** Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

### 2. 🌟 Yelp Review Rating Prediction with BERT and Curriculum Learning
- Preprocessed the Yelp Academic dataset and applied Flesch Reading Ease scores to sort reviews by complexity.
- Fine-tuned a BERT-base transformer model using curriculum learning over 1MB, 10MB, and 100MB data subsets.
- Compared the BERT models against TF-IDF + Logistic Regression baselines, evaluating on multi-class classification metrics.
- Visualized performance across models using confusion matrices and bar charts.

**Technologies:** PyTorch, Hugging Face Transformers, Pandas, TextStat, Scikit-learn

### 3. 🖼️ Image Similarity Learning with ConvNeXt and ArcFace Loss
- Leveraged the Tiny ImageNet-200 dataset to train ConvNeXt models using ArcFace loss to learn similarity-aware embeddings.
- Conducted experiments on different angular margin values and tracked training loss curves.
- Used cosine similarity on learned embeddings to classify image pairs as similar or dissimilar.
- Evaluated generalization by testing on the CIFAR-100 dataset.

**Technologies:** PyTorch, Torchvision, ConvNeXt, ArcFace, Matplotlib, Seaborn
