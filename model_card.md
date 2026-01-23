# Model Card: Census Income Classification

## Model Details

- **Model Name**: Census Income Classifier
- **Model Version**: 1.0.0
- **Model Type**: Random Forest Classifier
- **Framework**: scikit-learn
- **Developer**: Student / Udacity ML DevOps Nanodegree
- **Date**: January 2026

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Number of Estimators**: 100
- **Max Depth**: 10
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Random State**: 42

### Training Parameters

- **Training/Test Split**: 80/20
- **Cross-Validation**: Not used (simple train-test split)
- **Feature Encoding**: OneHotEncoder for categorical features
- **Label Encoding**: LabelBinarizer for target variable

## Intended Use

### Primary Use Case

This model is designed to predict whether an individual's annual income exceeds $50,000 based on census demographic data. It can be used for:

- Demographic analysis and research
- Understanding income distribution patterns
- Educational purposes demonstrating ML pipeline deployment

### Intended Users

- Data scientists and ML engineers learning deployment practices
- Researchers studying income prediction models
- Organizations performing demographic analysis

### Out-of-Scope Uses

This model should **NOT** be used for:

- Making employment decisions
- Loan approval or denial
- Any high-stakes decisions affecting individuals
- Real-time production systems without proper validation

## Training Data

### Data Source

- **Dataset**: UCI Census Income Dataset (Adult Dataset)
- **Source URL**: https://archive.ics.uci.edu/ml/datasets/adult
- **Original Source**: 1994 Census Bureau database
- **Extraction**: Barry Becker

### Dataset Description

- **Total Samples**: 32,561
- **Training Samples**: ~26,048 (80%)
- **Test Samples**: ~6,513 (20%)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Continuous | Age of the individual |
| workclass | Categorical | Type of employment |
| fnlgt | Continuous | Final weight (census weighting) |
| education | Categorical | Highest education level |
| education-num | Continuous | Years of education |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship in household |
| race | Categorical | Race of individual |
| sex | Categorical | Gender |
| capital-gain | Continuous | Capital gains |
| capital-loss | Continuous | Capital losses |
| hours-per-week | Continuous | Hours worked per week |
| native-country | Categorical | Country of origin |

### Target Variable

- **Name**: salary
- **Classes**: `<=50K`, `>50K`
- **Class Distribution**: Imbalanced (approximately 75% <=50K, 25% >50K)

## Evaluation Data

The evaluation data is a 20% holdout from the original dataset, created using stratified sampling to maintain class distribution.

## Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Precision | 0.XXXX |
| Recall | 0.XXXX |
| F1 Score | 0.XXXX |

*Note: Update these values after training the model.*

### Performance by Slice

Performance metrics were computed for each categorical feature slice. See `slice_output.txt` for detailed results.

#### Example Slice Performance (Education)

| Education Level | Samples | Precision | Recall | F1 |
|-----------------|---------|-----------|--------|-----|
| Bachelors | XXX | 0.XX | 0.XX | 0.XX |
| HS-grad | XXX | 0.XX | 0.XX | 0.XX |
| Masters | XXX | 0.XX | 0.XX | 0.XX |

*Note: Update these values after training the model.*

## Ethical Considerations

### Potential Biases

1. **Historical Bias**: The data is from 1994 and may not reflect current income distributions or societal changes.

2. **Demographic Representation**: 
   - The dataset is predominantly US-based
   - Certain demographic groups may be underrepresented
   - Historical discrimination may be reflected in the data

3. **Protected Attributes**: The model uses features like race, sex, and native-country which are protected attributes. Performance should be carefully monitored across these groups.

### Fairness Considerations

- Model performance varies across demographic slices
- Higher performance on majority groups may indicate bias
- Regular auditing with tools like Aequitas is recommended

### Privacy Considerations

- The training data is publicly available and anonymized
- No personally identifiable information (PII) is used
- The model does not store any input data

## Caveats and Recommendations

### Limitations

1. **Data Age**: The 1994 data does not reflect current economic conditions, inflation, or job market changes.

2. **Binary Classification**: Income is simplified to binary classes; real income is continuous and nuanced.

3. **Geographic Limitation**: Primarily represents US population demographics.

4. **Feature Limitations**: 
   - Does not include important factors like location cost-of-living
   - Education quality is not captured
   - Job experience beyond education years is not included

### Recommendations

1. **Do not use for consequential decisions**: This model is for educational/research purposes only.

2. **Regular monitoring**: Track performance across demographic groups over time.

3. **Update with recent data**: For any real application, train on more recent census data.

4. **Combine with human judgment**: Model predictions should be one factor among many in any analysis.

5. **Validate slice performance**: Before deployment, verify performance on all relevant demographic slices.

## Quantitative Analyses

### Slice Analysis Summary

The model was evaluated on slices of all categorical features:
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

Detailed results are available in `slice_output.txt`.

### Key Findings

*Update this section after training with specific findings about:*
- Which slices perform best/worst
- Any significant disparities across protected groups
- Recommendations based on slice analysis

## References

1. UCI Machine Learning Repository - Adult Dataset
   - https://archive.ics.uci.edu/ml/datasets/adult

2. Model Cards for Model Reporting (Mitchell et al., 2019)
   - https://arxiv.org/pdf/1810.03993.pdf

3. scikit-learn RandomForestClassifier Documentation
   - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html