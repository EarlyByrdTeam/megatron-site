---
layout: default
title: "MEGATRON: Meta-learning for Next-Gen Advanced Technology Realization & Acceleration"
---
<p style="margin-top:7em;"></p>
# <a id="home"></a>MEGATRON: Meta-Learning for Next-Generation Advanced Technology Realization & Acceleration

<p style="margin-top:3em;"></p>

<div style="text-align: justify; font-size: 0.85em; margin: 30px auto; max-width: 85%; line-height: 1.5; padding: 0 15px; text-indent: 0;">

<strong>Abstract.</strong> Breast cancer remains the leading cause of cancer-related female mortality worldwide, with current mammography screening methods limited by radiation exposure and accessibility issues. In recent years, Electrical Impedance Tomography (EIT) has shown potential as a cost-effective, non-invasive, and portable alternative for early detection. This new imaging modality can be paired with deep learning to enhance and speed-up its early detection capabilities. However, its point-of-care adoption is hindered by insufficient real-world training data for model development. This study presents MEGATRON, a meta-learning framework designed to leverage abundant conventional medical imaging data to accelerate the development of EIT-based diagnostic tools through few-shot learning capabilities. We develop a comprehensive data processing pipeline that processes conventional medical imaging data from multiple modalities (mammography, ultrasound, MRI, X-ray) across various anatomical regions from GREI open-source repositories. The processed data is then used to train a generalized meta-model, before fine-tuning on a proprietary EIT dataset to enable few-shot object detection. Results from systematic experiments determine the optimal model configurations and hyperparameter settings needed to maximize performance. Our meta-model successfully achieves 88% mean Average Precision (mAP50) on sparse EIT data, demonstrating effective knowledge transfer from conventional imaging modalities to novel EIT data. This work demonstrates the feasibility of using meta-learning to overcome data scarcity limitations in novel medical imaging technologies, with the MEGATRON framework providing a validated, open-source solution for accelerating the clinical deployment of EIT-based early detection tools, and broader applications for emerging diagnostic technologies across multiple domains where real-world data remains limited.

<br><br>
<strong>Keywords:</strong> Meta-learning, Data Reuse, Few-shot Learning, Electrical Impedance Tomography, Breast Cancer Detection
</div>

---
## <a id="introduction"></a>Introduction

<div style="text-align: justify;">

Breast cancer represents one of the most significant global health challenges, accounting for nearly one-third of all new cancer diagnoses in women in the United States as of 2023 (Siegel et al., 2023). Despite substantial advances in detection and treatment methodologies, breast cancer continues to be the primary source of cancer-related female mortality worldwide (Torre et al., 2015). Current screening paradigms, predominantly relying on mammography, have demonstrated a 20% reduction in breast cancer mortality through widespread implementation (Siegel et al., 2023). However, these conventional approaches face substantial limitations including radiation exposure, high costs, patient discomfort, and accessibility barriers that collectively reduce screening compliance and potentially delay critical early-stage detection (Marmot et al., 2013). The economic burden associated with late-stage breast cancer diagnoses is staggering, with treatment costs frequently exceeding $150,000 per patient (Milliman Research, 2017). Given that approximately one in eight women will be diagnosed with breast cancer during their lifetime, yet many avoid regular screenings due to the aforementioned barriers, there exists an urgent need for alternative diagnostic modalities that can provide accurate, accessible, and patient-friendly screening options (Siegel et al., 2023; Myklebust et al., 2009).

<br><br>

Electrical Impedance Tomography (EIT) has emerged as a promising non-invasive imaging modality that offers significant advantages over conventional screening methods. EIT technology provides portable, radiation-free imaging capabilities that can potentially transform point-of-care breast cancer screening, particularly in resource-limited settings or for patients who experience discomfort with traditional mammography procedures (Halter et al., 2008). The fundamental principle underlying EIT involves the measurement of electrical impedance variations within biological tissues, which can reveal pathological changes indicative of malignant processes. Recent developments in EIT technology have demonstrated considerable potential for early breast cancer detection, with several studies showing promising results when combined with advanced signal processing and machine learning algorithms (Polydorides & Lionheart, 2002; Lionheart, 2004). However, the clinical translation and widespread adoption of EIT-based diagnostic tools face a critical bottleneck: the scarcity of sufficient real-world EIT datasets required for training robust machine learning models capable of accurate tumor detection and classification.

<br><br>

The development of effective machine learning algorithms for medical imaging typically requires extensive datasets comprising thousands to millions of labeled examples to achieve clinically acceptable performance levels. Traditional approaches to addressing this requirement in emerging imaging modalities like EIT have relied heavily on synthetic data generation and physics-based simulations (Liu et al., 2018). While synthetic data has proven useful for initial algorithm development, it inherently fails to capture the pathological variability and demographic diversity present in real-world clinical environments, resulting in models that may perform well in idealized laboratory settings but demonstrate limited generalizability to actual patient populations. The challenges associated with synthetic data generation extend beyond mere representational limitations. Creating realistic synthetic EIT datasets requires extensive computational resources, deep understanding of the underlying physics, and careful consideration of noise characteristics that mirror real-world acquisition conditions. Concurrently, vast amounts of conventional medical imaging data from established modalities such as mammography, ultrasound, MRI, and X-ray remain underutilized in the development of novel imaging technologies. These datasets, accumulated over decades of clinical practice and research, represent a rich repository of expertly annotated pathological and normal cases that could potentially accelerate the development of emerging diagnostic modalities if appropriately leveraged.

<br><br>

Meta-learning, often described as "learning to learn," presents a compelling solution to the data scarcity challenge facing novel medical imaging technologies (Finn et al., 2017). This paradigm enables machine learning models to rapidly adapt to new tasks with minimal training data by leveraging knowledge gained from related tasks during a meta-training phase. In the context of medical imaging, meta-learning offers the possibility of training models on abundant conventional datasets and subsequently fine-tuning them for novel modalities with limited available data. The application of meta-learning to medical imaging represents a paradigm shift from traditional single-task learning approaches toward more generalizable, adaptable systems that can extract universal feature representations applicable across different imaging modalities. This approach is particularly relevant for emerging technologies like EIT, where the fundamental task of identifying pathological tissue changes shares commonalities with established imaging modalities, despite differences in acquisition methods and image characteristics.

<br><br>

This study presents MEGATRON: Meta-Learning for Next-Generation Advanced Technology Realization & Acceleration, a comprehensive framework designed to address the data scarcity limitations hindering the clinical deployment of EIT-based breast cancer detection systems. The primary objective of this research is to demonstrate the feasibility and effectiveness of using meta-learning approaches trained on conventional medical imaging datasets to enable accurate few-shot tumor detection on limited EIT data. We present an automated, robust ETL (Extract, Transform, Load) pipeline capable of ingesting and standardizing heterogeneous medical imaging data from multiple open-source repositories, formats, and modalities, while introducing a novel meta-learning framework specifically designed for medical imaging tasks that can effectively transfer knowledge from conventional imaging modalities to emerging technologies like EIT. Our study provides empirical evidence for the effectiveness of cross-modal knowledge transfer between different medical imaging modalities, demonstrating that features learned from conventional imaging can enhance performance on novel imaging tasks. Additionally, we contribute a validated, reproducible open-source framework that can be applied beyond EIT to accelerate the development of various emerging diagnostic technologies facing similar data scarcity challenges, establishing a practical pathway for translating research-stage imaging technologies into clinically deployable tools by leveraging existing data resources.

<br><br>

The implications of this research extend beyond breast cancer detection and EIT technology. The meta-learning framework presented here addresses a fundamental challenge in medical technology development, that is, the need to validate and deploy novel diagnostic tools rapidly while ensuring clinical accuracy and reliability. By demonstrating effective knowledge transfer across imaging modalities, this work opens new avenues for accelerating medical innovation and improving patient outcomes through faster technology translation.
The remainder of this paper presents our methodology, experimental results, and analysis of the MEGATRON framework's performance in enabling few-shot learning for EIT-based breast cancer detection, along with a discussion of broader applications and future research directions.
</div>

---
## <a id="methodology"></a>Methodology

Our approach:

1. Pre-train backbone on conventional datasets.  
2. Meta-train using episodic tasks (few-shot learning).  
3. Fine-tune on target dataset (EIT). 

### <a id="requirements"></a> Requirements
<div style="text-align: justify;">
The table below shows the minimum hardware and software requirements needed for reproducing the results of this research project. For training the model, we used an A100 NVIDIA GPU with PCIe and 40 GB of vRAM. This is the minimum vRAM needed to be able to train a meta-model across all datasets without needing any modifications to the code provided in the Github repository. For a GPU with lower vRAM, the batch size during training may need to be reduced to ensure the model, input data, gradients, optimizer states, and intermediate activations can all fit within the available GPU memory. 
<br><br>
</div>

| Item         | Minimum Required |
|--------------------------------|
| GPU          | CUDA-enabled with compute capability > 3.0 |
| CUDA Toolkit | 11.8             |
| Storage      | 1 TB             |
| vRAM         | 40 GB            |
| Linux Distro | Ubuntu 22.04     |
| Conda        | 25.1.1           |
| Python       | 3.10             |

### <a id="datasets"></a>Datasets

![Results Plot](figures/Data/Data_Summary.png){: style="max-width:100%; height:auto; display:block; margin:0; margin-right:1em;" }

| Name | Pathology | Annotation Type | Classes | Image Format | No. of Samples | Source | DOI |
|------------------|-----------------|----------|--------|------------|-------|------------|-------------------|
| Deepsight-2d-Mammogram | Breast Cancer | Detection | 3 | npz | 161589 | Dataverse | [doi:10.7910/DVN/KXJCIU](https://doi.org/10.7910/DVN/KXJCIU) |
| MRI-Brain-Tumor | Brain Cancer | Detection | 4 | mat, jpg | 5249 | Figshare, Mendeley | [doi:10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427), [doi:10.17632/zp67tkpj2y.1](https://doi.org/10.17632/zp67tkpj2y.1) |
| Breast-Ultrasound | Breast Cancer | Segmentation | 3 | png | 683 | Mendeley | [doi:10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3) |
| CBIS-DDSM-Mammogram | Breast Cancer | Segmentation | 2 | dcm | 3568 | Zenodo | [doi:10.7937/K9/TCIA.2016.7O02S9CY](https://zenodo.org/records/10960991) |
| Advanced-MRI-Breast-Lesions | Breast Cancer | Segmentation | 2 | dcm | 147 | TCIA | [doi:10.7937/C7X1-YN57](https://doi.org/10.7937/C7X1-YN57) |
| Chest-xray | Pneumonia | Detection | 14 | png | 18000 | Zenodo | [doi:10.5281/zenodo.12721389](https://doi.org/10.5281/zenodo.12721389) |
| RSNA-pneumonia | Pneumonia | Detection | 2 | dcm | 29684 | RSNA | [doi:10.1148/ryai.2019180041](https://pubs.rsna.org/doi/10.1148/ryai.2019180041) |
| EIT-Novel-Data (Ours) | Breast Cancer | Detection | 2 | png | 40 | Zenodo | [doi:10.5281/zenodo.17001019](https://doi.org/10.5281/zenodo.17001019) |

<div style="text-align: justify;">

Since the purpose of this study is to develop a meta-model that can specifically adapt to our novel EIT dataset in a few-shot learning scenario, it is essential to identify which datasets share the greatest similarity with the EIT data and would therefore be most likely to produce positive transfer effects during meta-model training. To evaluate dataset similarity, we employed multiple complementary approaches. First, histogram plots were generated from 50 randomly sampled grayscale images per dataset, capturing the distribution of pixel intensities from 1–254, excluding pure black and white values at 0 and 255. We then computed Wasserstein distances between the target EIT-Novel-Data histogram and all other dataset histograms to quantify distributional differences. We also generated t-SNE embeddings to visualize high-dimensional relationships between datasets. 

</div>

<figure style="text-align:center;">
  <img src="figures/Dataset_Similarity/dataset_histograms.png" 
       alt="Histogram"
       style="max-width:100%; height:auto; display:block; margin:0 auto;" />
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure 2: Histogram plots obtained by averaging grayscale histograms of 50 randomly sampled images per dataset.
  </figcaption>
</figure>

<figure style="text-align:left;">

<div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center; align-items:flex-start;">
  <figure style="flex:1; text-align:center; margin:0;">
    <img src="figures/Dataset_Similarity/tSNE-of-CNN-embeddings.png" alt="t-SNE of CNN embeddings" style="max-width:100%; height:auto;"/>
    <figcaption style="margin-top:1.5em;"> (a) t-SNE of CNN embeddings for all datasets.</figcaption>
  </figure>

  <figure style="flex:1; text-align:center; margin:1;">
    <img src="figures/Dataset_Similarity/ws_distance.png" alt="Wasserstein distance" style="max-width:100%; height:auto;"/>
    <figcaption>(b) Wasserstein distance between EIT data and every other dataset.</figcaption>
  </figure>
</div>
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure 3: Comparing similarity between novel EIT data and all other datasets using two similarity metrics: (a) 3D t-SNE plot using 50 randomly sampled images per dataset and (b) Wasserstein distance between average histograms.
  </figcaption>

</figure>

<div style="text-align: justify;">
The grayscale histogram analysis in Figure 2 demonstrates that the EIT-Novel-Data exhibits a distinct, narrow peak centered around grayscale values of approximately 120 to 150, in contrast to the skewed or multi-modal distributions observed in the remaining datasets. For instance, CBIS-DDSM-Mammogram and MRI-Brain-Tumor display complex multi-peaked structures, while datasets such as Breast-Ultrasound and Advanced-MRI-Breast-Lesion are heavily skewed toward darker intensities, and X-ray datasets contain a wide range of pixel intensities as shown by their broader spectrums. The Wasserstein distance comparison in Figure 3b provides a quantitative assessment of these differences, showing that EIT-Novel-Data grayscale distribution aligns most closely with CBIS-DDSM-Mammogram, followed by RSNA-Pneumonia and Chest X-ray, whereas it diverges strongly from Advanced-MRI-Breast-Lesion and MRI-Brain-Tumor. These findings suggest that, despite being derived from a distinct imaging modality, EIT-Novel-Data shares greater grayscale intensity similarity with mammography and X-ray datasets than with MRI or ultrasound datasets.
<br><br>
The t-SNE visualization in Figure 3a reveals that each dataset forms a distinct cluster in the CNN embedding space, highlighting clear modality-specific differences. The EIT-Novel-Data cluster is positioned relatively close to most of the datasets,  suggesting that it shares feature-level similarity across multiple modalities. Hoever, it's more clearly separated from the Advanced-MRI-Breast-Lesion and chest X-ray datasets. This pattern partially aligns with the Wasserstein analysis since EIT is close to mammography in both views, but despite intensity-based similarity to X-ray, the embedding places EIT farther from the X-ray datasets, indicating that relying on grayscale similarity alone is insufficient when determining similarity in representational structures.
<br><br>
We hypothesize that training a meta-model using datasets that are proximally-close to the target dataset in the embedding space will yield the best model performance since meta-learning involves learning to learn from limited data, and proximally-related tasks share similar representational structures, thus providing a better model initialization for rapid adaptation to the target domain. 
</div>

### <a id="data_pipeline"></a>Data Processing Pipeline

<figure style="text-align:center;">
  <img src="figures/Data/Pipeline.png" 
       alt="Data Processing Pipeline"
       style="max-width:100%; height:auto; display:block; margin:0 auto;" />
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure X: Data processing pipeline utilized for standardizing datasets across different repositories, modalities, and image formats before feeding the data to a downstream meta-learning model.
  </figcaption>
</figure>

### <a id="meta_learn"></a>Meta Learning

<div style="text-align: justify;  margin-bottom:2em">
We adopt a meta-learning framework to enable rapid adaptation to novel tasks with limited labeled data. The goal is to learn a set of meta-parameters \(\theta\) that capture shared structure across tasks, facilitating few-shot learning on unseen tasks. In our setup, tasks are drawn from a distribution \(\mathcal{T}\), where each task \(T_i\) corresponds to a distinct dataset. To simulate a few-shot learning scenario, we employ an episodic training paradigm for each task in every meta-epoch. For each task, we sample a fixed number of examples from its training set to form the support set, which drives task-specific adaptation of the base model, and sample a fixed number of examples from the validation set to form the query set, on which the meta-model is evaluated. By using a fixed number of support and query examples per task, we ensure that each episode is standardized, which is particularly beneficial when tasks have varying numbers of samples. This standardization prevents tasks with larger datasets from dominating the meta-training process and allows the meta-model to learn representations that generalize across tasks of different sizes.
</div>

#### <a id="meta_training"></a>Meta Training

<div style="text-align: justify; margin-bottom:2em">
During meta-training, the Reptile algorithm is used to optimize the meta-parameters \(\theta\) of the meta-model. For each meta-epoch, a task \(T_i\) is sampled from \(\mathcal{T}\), and task-specific parameters \(\theta_i\) are initialized from \(\theta\). The task-specific support set is fed to the model, and its parameters are updated via \(K\) steps of gradient descent using the task-specific loss \(\mathcal{L}_{T_i}\) with learning rate \(\alpha\), producing adapted parameters \(\theta_i^{(K)}\). The meta-parameters are then updated in the direction of the task-adapted parameters according to
$$\theta\gets\theta+\beta(\theta_i^{(k)}−\theta),$$
where \(\beta\) is the meta learning rate. This procedure is repeated across tasks and meta-epochs, gradually biasing \(\theta\) toward regions of the parameter space that enable rapid adaptation across the full distribution of tasks. Once training is complete, we'll have generated a meta-model capable of fast multi-task adaptation. 
</div> 

<figure style="text-align:center;">
  <img src="figures/Algo/meta-train.png" 
       alt="Algorithm"
       style="max-width:80%; height:auto; display:block; margin:0 auto;" />
  <figcaption style="margin-top:0.0em; font-style:italic;">
    Figure X: Meta-Training Algorithm
  </figcaption>
</figure>

#### <a id="meta_validation"></a>Meta Validation

<div style="text-align: justify; margin-bottom:2em">
To monitor generalization, we evaluate the meta-model on held-out task-specific query sets that were not used during meta-training. Prior to evaluation on a particular task, the meta-parameters are fine-tuned on the given task’s support set using 5 gradient descent steps. Skipping this step would mean the model is effectively being evaluated in a zero-shot setting for that task. Afterwards, the meta-model’s performance on the particular task is assessed by performing a prediction on its query set. This procedure provides a realistic estimate of few-shot adaptation performance by simulating the scenario where the model adapts to a new task with only limited support examples before being evaluated.
</div>

<figure style="text-align:center;">
  <img src="figures/Algo/meta-val.png" 
       alt="Algorithm"
       style="max-width:80%; height:auto; display:block; margin:0 auto;" />
  <figcaption style="margin-top:0.0em; font-style:italic;">
    Figure X: Meta-Validation Algorithm
  </figcaption>
</figure>

#### <a id="few_shot_learning"></a>Few-Shot Learning
<div style="text-align: justify; margin-bottom:2em">
After meta-training, the learned meta-parameters \(\theta\) can be quickly adapted to novel tasks with very few labeled examples. Given a new task \(T_{\text{new}}\), the meta-parameters \(\theta\) are fine-tuned on the support set of \(T_{\text{new}}\) using a small number of gradient steps. The adapted parameters are then used to make predictions on the task’s query set. This approach enables rapid generalization to unseen tasks with minimal supervision, leveraging the shared knowledge encoded in the meta-parameters.
<br><br>
One can go a step further and incorporate the new task into the meta-model by including it as an additional task during meta-training. This can be achieved with ease because the meta-learning algorithm is designed to extend to new tasks without needing to retrain the model from scratch. Incorporating the new task nudges the meta-parameters toward an optimal minima that can quickly adapt to tasks that share similarities with the new task. This strengthens the model's long term adaptability as more tasks are introduced in the future, improves robustness by reducing the need for extensive fine-tuning, shortens deployment timelines, and ensures the model can continuously accommodate process or domain shifts without sacrificing performance on existing tasks.
</div> 

---
## <a id="experiments"></a>Experiments & Findings

| Dataset | Baseline | Meta-Learning | Δ |
|---------|----------|---------------|---|
| MRI     | 72.3%    | **81.4%**     | +9.1 |
| X-Ray   | 68.5%    | **77.2%**     | +8.7 |

### Experiment 1 - Optimal Meta-Learning Rate
<div style="display:flex; flex-wrap:wrap; gap:1em;">
  <img src="figures/learning_rate/mAP50_curves_vs_epoch.png" alt="Learning Rate" style="max-width:49%; height:auto;"/>
  <img src="figures/learning_rate/mAP50_vs_learning_rate.png" alt="Learning Rate" style="max-width:49%; height:auto;"/>
</div>

### Experiment 2 - Generalizability and Task Dependence
<div style="text-align: justify;">
The purpose of this experiment was to determine how the meta-model's generalization performance varies with number and diversity of tasks. Figure X reveals that the meta-model performance follows an inverted U-shaped curve when scaling from 2 to 7 tasks, with peak performance achieved at 4-5 tasks. The chart shows that mAP50 increases from 0.66 when trained on 2 tasks to a maximum of 0.69 for 5 tasks before declining to 0.49 for 7 tasks.
<br><br>
The model's performance trajectory supports the hypothesis that meta-model generalization benefits from increased task diversity up to an optimal point. The initial improvement going from 2 to 5 tasks suggests that exposure to more diverse medical imaging tasks enhances the meta-learner's ability to quickly adapt to new detection problems by learning more generalizable feature representations. Beyond 5 tasks, the meta-model's generalizability begins to deteriorate with the inclusion of out-of-distribution chest X-ray tasks. This is due to task interference introducing negative transfer effects that counter the benefits of diversity. The performance degradation is most pronounced in precision, suggesting that discriminative capabilities become compromised when the meta-learner attempts to accommodate too many disparate visual domains and detection requirements simultaneously.
</div>

<figure style="text-align:center;">
  <img src="figures/Bar_Chart/bar_chart_metrics_epoch_40.png" 
       alt="Histogram"
       style="max-width:60%; height:auto; display:block; margin:0 auto;" />
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure X: Impact on meta-model performance (mAP50, precision, recall) as more tasks are added during meta-training. Each bar represents the average value over all 7 tasks.
  </figcaption>
</figure>

<figure style="text-align:left;">

  <div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center;">
    <img src="figures/box_plots/N2_boxplot.png" alt="Box Plot N2" style="max-width:32%; height:auto;"/>
    <img src="figures/box_plots/N3_boxplot.png" alt="Box Plot N3" style="max-width:32%; height:auto;"/>
    <img src="figures/box_plots/N4_boxplot.png" alt="Box Plot N4" style="max-width:32%; height:auto;"/>
  </div>

  <div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center;">
    <img src="figures/box_plots/N5_boxplot.png" alt="Box Plot N5" style="max-width:32%; height:auto;"/>
    <img src="figures/box_plots/N6_boxplot.png" alt="Box Plot N6" style="max-width:32%; height:auto;"/>
    <img src="figures/box_plots/N7_boxplot.png" alt="Box Plot N7" style="max-width:32%; height:auto;"/>
  </div>

  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure X: Box plots for various performance metrics including mean average precision at IOU 50%, recall, precision and F1 score for meta-models trained on N=2 to N=7 tasks.
  </figcaption>

</figure>

<div style="text-align: justify;">
The boxplots in Figure X provide additional granularity in the model behavior, and illustrate how different task characteristics contribute to this meta-learning dynamic. The inclusion of diverse modalities like mammography, MRI, ultrasound, and chest X-ray exposes the meta-learner to varied visual patterns, tumor sizes and boundaries, grayscale intensities, and imaging artifacts. It is interesting to see how the model behaves on tasks with varying levels of difficulty. For instance, the model maintains consistently high performance on MRI-Brain-Tumor (~0.85-0.90 mAP50) due to its high quality annotations and sharp tumor contrast. But, the model exhibits poor performance on the challenging DeepSight-2d-Mammogram dataset (~0.2-0.3 mAP50) due to its low image contrast. However, the overall model performance is still relatively high which means this challenging dataset still provides a valuable learning signal for the meta-model. Further, the performance on the Advanced-MRI-Breast-Lesion dataset is adversely affected when trained alongside RSNA-Pneumonia, with the value dropping from 0.7 to 0.25, and later recovering to 0.55 when Chest X-ray is added to the training schedule. This drop is probably because the Advanced-MRI-Breast-Lesion dataset is a small dataset that contributes a weak learning signal to the model, while the rise is probably because the Chest X-ray data may introduce regularization effects that help stabilize training.
<br><br>
These findings indicate an optimal selection of tasks with sufficient heterogeneity to improve generalization without causing destructive interference to the meta-model's learned parameters.
</div>

### Experiment 3 - Multi-task Meta-Learning Performance
In experiment 2, we wanted to investigate the impact of....

<figure style="text-align:left;">

<div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center; align-items:flex-start;">
  <figure style="flex:1; text-align:center; margin:0;">
    <img src="figures/Spider_Plots/number_of_FT_epochs/n7_ft5.png" alt="Meta-Model with 5 fine-tuning epochs" style="max-width:100%; height:auto;"/>
    <figcaption> (a) Meta-Model with 5 fine-tuning epochs.</figcaption>
  </figure>

  <figure style="flex:1; text-align:center; margin:0;">
    <img src="figures/Spider_Plots/number_of_FT_epochs/n7_ft20.png" alt="Meta-Model with 20 fine-tuning epochs" style="max-width:100%; height:auto;"/>
    <figcaption>(b) Meta-Model with 20 fine-tuning epochs.</figcaption>
  </figure>
</div>
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure X: Multi-Task Meta-Model performance across all 7 tasks with (a) 5 fine-tuning epochs and (b) 20 fine-tuning epochs during the meta-validation step.
  </figcaption>

</figure>

### Experiment 4 - Model Size and Warmup Schedule
In experiment 2, we wanted to investigate the impact of....
These findings underscore meta-learning's fundamental strength in development of novel imaging applications or in resource-constrained settings. In settings where collecting vast amounts of data is impractical, one can leverage highly efficient meta-models which preserve knowledge across diverse diagnostic tasks to close the gap.

<figure style="text-align:left;">

<div style="display:flex; flex-wrap:wrap; gap:3em; justify-content:center; align-items:flex-start;">
  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Spider_Plots/small_vs_large_model/v8n_spider.png" alt="v8n_spider" style="max-width:100%; height:auto;"/>
    <figcaption>(a) Small model</figcaption>
  </figure>

  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Spider_Plots/small_vs_large_model/v8n_warmup_spider.png" alt="v8n_warmup_spider" 
    style="max-width:100%; height:auto;"/>
    <figcaption>(b)  Small model with 3 warmup epochs</figcaption>
  </figure>
</div>

<div style="display:flex; flex-wrap:wrap; gap:3em; margin:2em; justify-content:center; align-items:flex-start;">
  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Spider_Plots/small_vs_large_model/v8s_spider.png" alt="v8s_spider" style="max-width:100%; height:auto;"/>
    <figcaption>(a) Large Model </figcaption>
  </figure>

  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Spider_Plots/small_vs_large_model/v8s_warmup_spider.png" alt="v8s_warmup_spider" 
    style="max-width:100%; height:auto;"/>
    <figcaption>(b) Large model with 3 warmup epochs</figcaption>
  </figure>
</div>

<figcaption style="margin-top:0.5em; font-style:italic;">
  Figure X: Spider plots showing how different base learner sizes and warmup schedules affects the multi-task performance of the meta-model trained on all 7 tasks. The plots show the mAP50 values evaluated on the validation set for each task.

</figcaption>
</figure>
    
### Experiment 5 - Comparison with Incremental Transfer Learning

We performed standard transfer learning on three tasks in an incremental manner to benchmark the performance of our meta-model against a conventionally fine-tuned model. By incremental transfer learning, we refer to the process in which the base model is fine-tuned on one task, saved, and subsequently used as the initialization for transfer learning on the next task. This procedure was carried out in two ways, namely with and without freezing model layers between tasks, resulting in two models. Their performance on each task’s validation set evaluated at intervals of 10 training epochs is shown below.

<div style="text-align: justify;">
</div>

<figure style="text-align:left;">

  <div style="gap:1em; text-align:center;">
    <figure style="text-align:center; margin:0 0 1em 0;">
        <img src="figures/Benchmark_CBIS_MRI_DS/benchmark_CBIS_MRI_DS_freeze.png" alt="benchmark_CBIS_MRI_DS_freeze" style="width:100%; height:auto;"/>
        <figcaption>(a) Incremental Transfer Learning without freezing layers between tasks.</figcaption>
    </figure>
    <figure style="text-align:center; margin:0;">
        <img src="figures/Benchmark_CBIS_MRI_DS/benchmark_CBIS_MRI_DS_no_freeze.png" alt="benchmark_CBIS_MRI_DS_no_freeze0" style="width:100%; height:auto;"/>
        <figcaption>(b) Incremental Transfer Learning with 10 frozen layers during task 2 training, and 15 frozen layers during task 3 training.</figcaption>
    </figure>
  </div>

  <figcaption style="margin-top:0.5em; font-style:italic;">
  Figure 7: Training two models on three tasks using incremental transfer learning where by (a) has no frozen layers between tasks and (b) uses 10 and 15 frozen layers after the addition of the second and third task, respectively.
  </figcaption>

</figure>

<div style="text-align: justify;">
A comparison between meta-learning and incremental transfer learning reveals several critical insights about multi-task medical imaging performance. Both approaches demonstrate consistent relative dataset difficulty rankings, with MRI-Brain-Tumor achieving the highest performance of 0.94 with meta-learning and 0.9 with transfer learning, and DeepSight-2d-Mammogram showing the poorest results of 0.54 with meta-learning and 0.3 with transfer learning. Notably, CBIS-DDSM-Mammogram exhibits remarkably similar performance between approaches (~0.76 vs ~0.75), suggesting this dataset represents an optimal difficulty level for both learning paradigms.
<br><br>
The incremental transfer learning results reveal severe limitations inherent to this approach. The line plots demonstrate clear evidence of catastrophic forgetting, where previously learned tasks experience immediate and severe performance degradation to near-zero mAP50 when the model begins learning subsequent tasks. These results show that traditional fine-tuning approaches cannot maintain multi-task competency in medical imaging domains, and are especially unfit for use cases that require new datasets to be integrated in the training pipeline.
<br><br>
In contrast, the meta-learning approach successfully avoids catastrophic forgetting entirely, maintaining functional performance across all seven tasks simultaneously as shown in Figure X. This multi-task retention ability is particularly valuable in scenarios where process changes and domain variability introduce new tasks that must be incorporated into the training and deployment pipeline, while still preserving high performance on previously learned tasks
</div>

<p style="margin-top:3em;"></p>
### Experiment 6 - Few Shot Learning on novel EIT data
#### Few Shot Learning with N=30 samples

<figure style="text-align:left;">

<div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center; align-items:flex-start;">
  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Benchmark_EIT/mAP50_vs_epoch_benchmark.png" alt="Benchmark EIT" style="max-width:100%; height:auto;"/>
    <figcaption>(a) Baseline Transfer Learning</figcaption>
  </figure>

  <figure style="flex:1; text-align:center;  margin:0;">
    <img src="figures/Benchmark_EIT/mAP50_vs_epoch_meta.png" alt="Meta EIT" 
    style="max-width:100%; height:auto;"/>
    <figcaption>(b) Finetuning a Meta-Model</figcaption>
  </figure>
</div>
  <figcaption style="margin-top:0.5em; font-style:italic;">
    Figure 8: Comparing model performance on the EIT validation set by (a) finetuning YOLO by transfer learning on EIT data for 300 epochs and (b) finetuning a meta-model (n=5) on EIT data for 50 meta-epochs, whereby each meta-epoch consists of 10 training epochs and 5 finetuning epochs on EIT data only. This means 20 meta-epochs is computationally equivalent to 300 epochs used during baseline transfer learning.

  </figcaption>
</figure>

<p style="margin-top:1em;"></p>

The Meta-Model trained on all five in-distribution tasks retains a better initialization for the task of fine-tuning on the novel EIT dataset, as indicated by the higher mAP50 after the first meta-epoch in Figure (a). In contrast,

<p style="margin-top:3em;"></p>

#### Very Few Shot Learning with N=15 samples

<div style="display:flex; flex-wrap:wrap; gap:1em; justify-content:center; align-items:flex-start;">

  <figure style="flex:1; text-align:center; margin:0;">
    <img src="figures/Benchmark_EIT/EIT_small/FT50_Train50.png" alt="FT50 Train50" style="max-width:100%; height:auto;"/>
    <figcaption>(a) Fine-tuning (50 epochs)</figcaption>
  </figure>

  <figure style="flex:1; text-align:center; margin:0;">
    <img src="figures/Benchmark_EIT/EIT_small/FT70_Train50.png" alt="FT70 Train50" style="max-width:100%; height:auto;"/>
    <figcaption>(b) Fine-tuning (70 epochs)</figcaption>
  </figure>

</div>

## <a id="conclusion"></a>Conclusion



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