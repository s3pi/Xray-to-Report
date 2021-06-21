# Hierarchical X-Ray Report Generation via Pathology tags and Multi Head Attention

Abstract: Examining radiology images, such as X-Ray images as accu-
rately as possible, forms a crucial step in providing the best healthcare
facilities. However, this requires high expertise and clinical experience.
Even for experienced radiologists, this is a time-consuming task. Hence,
the automated generation of accurate radiology reports from chest X-
Ray images is gaining popularity. However, compared to other image
captioning tasks, where coherence is the key criterion, we need coher-
ence and high accuracy in detecting medical anomalies and information
in the medical domain. That is, the report must be easy to read and con-
vey medical facts accurately. To achieve this, we propose a deep neural
network. Given a set of Chest X-Ray images of the patient, the proposed
network predicts the medical tags and generates a readable radiology
report. For generating the report and tags, the proposed network learns
to extract salient features of the image from a deep CNN and generates
tag embeddings for each patient's X-Ray images. We use transformers
for learning self and cross attention. We encode the image and tag fea-
tures with self-attention to get a ner representation. Use both the above
features in cross attention with the input sequence to generate the re-
port's Findings. Then, cross attention is applied between the generated
Findings and the input sequence to generate the report's Impressions. For
evaluating the proposed network, we use a publicly available dataset. The
performance indicates that we can generate a readable radiology report,
with a relatively higher BLEU score over SOTA.

##### Accepted in ACCV 2020, Kyoto, Japan.

##### Paper: https://openaccess.thecvf.com/content/ACCV2020/html/Srinivasan_Hierarchical_XRay_Report_Generation_via_Pathology_tags_and_Multi_Head_ACCV_2020_paper.html

##### Tensorflow #numpy #scipy 

##### https://youtu.be/e0p9Q4WKB1o
