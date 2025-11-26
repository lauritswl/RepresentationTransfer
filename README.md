# Representation Transfer - 
How stable is a semantic concept vector defined in one distribution of texts (dataset) when tested out of distribution (another dataset).
We test:
1. If a sentiment vector defined as negative -> positive, is the same as a negative -> neutral or neutral -> positive.
2. What is the cosine similarity between these vectors, if defined in different datasets.
3. Visualize the how Concept Vector Projection method predicts the sentiment/valence of across datasets. ("Correlation-Confusion Matrix")
4. How stable is mean-of-extremes Concept Vectors for the Arousal dimension of sentiment.
5. An applied use-case describing the relation between Arousal and Valence - Human Rated Analysis vs Concept Projection Analysis.

## How to get the plots:
1. Pull repo
2. Run ntb.py
2.1. Load datasets (Saved in Corpus Dictionary)
2.2. Embed text files (Saved in Corpus Dictionary)
2.3. Set cutoff points for positive/negative classes (currently one SD from Mean) (Saved in Corpus Dictionary)
2.4. Create Concept Vector Classes for analysis (Saved in CV Dictionary)
2.4.1. concept_vector.fit(negative_embeddings, positive_embeddings) # Defines Vector
2.4.2. concept_vector.project(embeddings) # Projects embeddings onto vector

3. Do all the plots and analysis you want :)


*OBS*
The plotting functions are extremely long, as genAI has been used to write them, they are in need of human eyes, before any real conclusions are made.
