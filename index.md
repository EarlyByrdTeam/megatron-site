---
layout: default
title: "MEGATRON: Meta-Learning for Acceleration of Novel Technology Adoption"
---
<p style="margin-top:5em;"></p>
# <a id="home"></a>Meta-Learning for Medical Imaging
Welcome to the project site!  

---
## <a id="abstract"></a>Abstract

The need to quickly develop and deploy of novel technologies motivates the needs for a meta-learning framework that make use of vast sources of legacy medical imaging data.

---
## <a id="introduction"></a>Introduction

This project explores **few-shot meta-learning** for object detection in medical imaging datasets.

![Pipeline](figures/learning_rate/mAP50_vs_learning_rate.png){: style="max-width:60%; height:auto; display:block; margin:0 margin-right:1em;" }

---
## <a id="methodology"></a>Methodology

Our approach:

1. Pre-train backbone on conventional datasets.  
2. Meta-train using episodic tasks (few-shot learning).  
3. Fine-tune on target dataset (EIT).  

![Pipeline](figures/box_plots/N7_boxplot.png){: style="max-width:60%; height:auto; display:block; margin:0 margin-right:1em;" }

### <a id="datasets"></a>Datasets

### <a id="data_pipeline"></a>Data Processing Pipeline

### <a id="meta_learn"></a>Meta Learning

#### <a id="meta_learn"></a>Meta Training

#### <a id="meta_learn"></a>Meta Validation

#### <a id="meta_learn"></a>Few-Shot Learning



---
## <a id="experiments"></a>Experiments & Findings

Datasets:

- Breast MRI  
- Chest X-Ray  
- Ultrasound  

| Dataset | Baseline | Meta-Learning | Δ |
|---------|----------|---------------|---|
| MRI     | 72.3%    | **81.4%**     | +9.1 |
| X-Ray   | 68.5%    | **77.2%**     | +8.7 |

![Results Plot](figures/class_distribution_pie_chart.png){: style="max-width:60%; height:auto; display:block; margin:0; margin-right:1em;" }

### Experiment 1
In experiment 1, we found that...

### Experiment 2
In experiment 2, we wanted to investigate the impact of....

### Experiment 3
In experiment 2, we wanted to investigate the impact of....
    
#### Comparison with Standard Transfer Learning

<p style="margin-top:10em;"></p>

---
## <a id="references"></a>References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.  
1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.
1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.
1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.
1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.

<p style="margin-top:7em;"></p>