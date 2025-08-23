---
layout: default
title: "MEGATRON: Meta-Learning for Acceleration of Novel Technology Adoption"
---

# <a id="home"></a>Meta-Learning for Medical Imaging

Welcome to the project site!  
This project explores **few-shot meta-learning** for object detection in medical imaging datasets.

![Pipeline](figures/mAP50_vs_learning_rate.png)

---

## <a id="methodology"></a>Methodology

Our approach:

1. Pre-train backbone on conventional datasets.  
2. Meta-train using episodic tasks (few-shot learning).  
3. Fine-tune on target dataset (EIT).  

![Pipeline](figures/boxplot_metrics_epoch_40.png)

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

![Results Plot](figures/class_distribution_pie_chart_legend.png)

---

## <a id="references"></a>References

1. Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning (MAML).  
2. Hospedales, T. et al. (2022). Meta-Learning in Neural Networks: A Survey.  
3. Dataset references: ChestX-ray14, BreastMRI, UltrasoundNerveSeg.  
