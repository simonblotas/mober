import pandas as pd


mappings = [
    ["Ovarian Epithelial Tumor", "Ovarian serous cystadenocarcinoma"],
    ["Acute Myeloid Leukemia", "Acute Myeloid Leukemia"],
    ["Colorectal Adenocarcinoma", "Colon adenocarcinoma or Rectum adenocarcinoma"],
    ["Melanoma", "Skin Cutaneous Melanoma"],
    ["Bladder Urothelial Carcinoma", "Bladder Urothelial Carcinoma"],
    [
        "Non-Small Cell Lung Cancer",
        "Lung adenocarcinoma or Lung squamous cell carcinoma",
    ],
    [
        "Renal Cell Carcinoma",
        "Kidney renal clear cell carcinoma or Kidney renal papillary cell carcinoma",
    ],
    ["Invasive Breast Carcinoma", "Breast invasive carcinoma"],
    ["B-Lymphoblastic Leukemia/Lymphoma", "Acute Myeloid Leukemia"],
    ["Pancreatic Adenocarcinoma", "Pancreatic adenocarcinoma"],
    ["Non-Hodgkin Lymphoma", "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma"],
    ["Meningothelial Tumor", "Brain Lower Grade Glioma"],
    ["Diffuse Glioma", "Glioblastoma multiforme or Brain Lower Grade Glioma"],
    ["Sarcoma, NOS", "Sarcoma"],
    ["Ewing Sarcoma", "Sarcoma"],
    ["Non-Cancerous", "Controls"],
    [
        "Esophagogastric Adenocarcinoma",
        "Esophageal carcinoma or Stomach adenocarcinoma",
    ],
    ["Rhabdomyosarcoma", "Sarcoma"],
    ["Fibrosarcoma", "Sarcoma"],
    ["Embryonal Tumor", "Miscellaneous"],
    ["Well-Differentiated Thyroid Cancer", "Thyroid carcinoma"],
    ["Hodgkin Lymphoma", "Miscellaneous"],
    ["Myeloproliferative Neoplasms", "Chronic Myelogenous Leukemia"],
    ["Neuroblastoma", "Miscellaneous"],
    ["Osteosarcoma", "Sarcoma"],
    ["Pleural Mesothelioma", "Mesothelioma"],
    ["Prostate Adenocarcinoma", "Prostate adenocarcinoma"],
    ["Rhabdoid Cancer", "Miscellaneous"],
    ["T-Lymphoblastic Leukemia/Lymphoma", "Acute Myeloid Leukemia"],
    ["Adenosquamous Carcinoma of the Pancreas", "Pancreatic adenocarcinoma"],
    ["Lung Neuroendocrine Tumor", "Lung adenocarcinoma"],
    ["Intracholecystic Papillary Neoplasm", "Miscellaneous"],
    ["Leiomyosarcoma", "Sarcoma"],
    ["Head and Neck Squamous Cell Carcinoma", "Head and Neck squamous cell carcinoma"],
    ["Endometrial Carcinoma", "Uterine Corpus Endometrial Carcinoma"],
    ["Anaplastic Thyroid Cancer", "Thyroid carcinoma"],
    ["Prostate Small Cell Carcinoma", "Prostate adenocarcinoma"],
    ["Ampullary Carcinoma", "Miscellaneous"],
    ["Intraductal Papillary Neoplasm of the Bile Duct", "Cholangiocarcinoma"],
    ["Hepatocellular Carcinoma", "Liver hepatocellular carcinoma"],
    ["Esophageal Squamous Cell Carcinoma", "Esophageal carcinoma"],
    ["Pancreatic Neuroendocrine Tumor", "Pancreatic adenocarcinoma"],
    ["Breast Ductal Carcinoma In Situ", "Breast invasive carcinoma"],
    ["Chondrosarcoma", "Sarcoma"],
    ["Uterine Sarcoma/Mesenchymal", "Uterine Carcinosarcoma"],
    ["Poorly Differentiated Thyroid Cancer", "Thyroid carcinoma"],
    ["Myelodysplastic Syndromes", "Acute Myeloid Leukemia"],
    [
        "Cervical Squamous Cell Carcinoma",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    ],
    ["Small Bowel Cancer", "Miscellaneous"],
    ["Hepatoblastoma", "Miscellaneous"],
    [
        "Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma",
        "Sarcoma",
    ],
    ["Bladder Squamous Cell Carcinoma", "Bladder Urothelial Carcinoma"],
    ["Urethral Cancer", "Miscellaneous"],
    [
        "Cervical Adenocarcinoma",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    ],
    ["Merkel Cell Carcinoma", "Miscellaneous"],
    ["Synovial Sarcoma", "Sarcoma"],
    ["Retinoblastoma", "Miscellaneous"],
    ["Medullary Thyroid Cancer", "Thyroid carcinoma"],
    ["Cutaneous Squamous Cell Carcinoma", "Skin Cutaneous Melanoma"],
    [
        "Mixed Cervical Carcinoma",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    ],
    ["Ovarian Germ Cell Tumor", "Ovarian serous cystadenocarcinoma"],
    ["Squamous Cell Carcinoma of the Vulva/Vagina", "Miscellaneous"],
    ["Adrenocortical Carcinoma", "Adrenocortical carcinoma"],
    ["Epithelioid Sarcoma", "Sarcoma"],
    ["Ocular Melanoma", "Uveal Melanoma"],
    ["Gestational Trophoblastic Disease", "Miscellaneous"],
    ["Liposarcoma", "Sarcoma"],
    ["Non-Seminomatous Germ Cell Tumor", "Testicular Germ Cell Tumors"],
    ["Breast Neoplasm, NOS", "Breast invasive carcinoma"],
    [
        "Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma",
        "Liver hepatocellular carcinoma and Cholangiocarcinoma",
    ],
    [
        "Glassy Cell Carcinoma of the Cervix",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    ],
    ["Ovarian Cancer, Other", "Ovarian serous cystadenocarcinoma"],
    ["Mucosal Melanoma of the Vulva/Vagina", "Miscellaneous"],
    ["Nerve Sheath Tumor", "Miscellaneous"],
    ["Sex Cord Stromal Tumor", "Miscellaneous"],
    ["Extra Gonadal Germ Cell Tumor", "Testicular Germ Cell Tumors"],
    [
        "Small Cell Carcinoma of the Cervix",
        "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    ],
    ["Chordoma", "Sarcoma"],
    ["Head and Neck Carcinoma, Other", "Head and Neck squamous cell carcinoma"],
    ["Salivary Carcinoma", "Head and Neck squamous cell carcinoma"],
    ["Acute Leukemias of Ambiguous Lineage", "Acute Myeloid Leukemia"],
    ["Hereditary Spherocytosis", "Miscellaneous"],
    ["Gastrointestinal Stromal Tumor", "Sarcoma"],
]

# Create a panda data frame with the two columns OncotreePrimaryDisease and TCGA_project 'disease'
df_list = pd.DataFrame(mappings, columns=["OncotreePrimaryDisease", "TCGA_project"])

# send everything to lower case
df_list["OncotreePrimaryDisease"] = df_list["OncotreePrimaryDisease"].str.lower()
df_list["TCGA_project"] = df_list["TCGA_project"].str.lower()

# Create a dictionary out of the previous panda data frame first colum is OncotreePrimaryDisease and second TCGA_project 'disease'
primary = pd.Series(
    df_list["TCGA_project"].values, index=df_list["OncotreePrimaryDisease"].values
).to_dict()

double_mapped = list(
    df_list[df_list["TCGA_project"].str.contains(" or")][
        "OncotreePrimaryDisease"
    ].unique()
)

sub_type = {key: None for key in double_mapped}

sub_type["colorectal adenocarcinoma"] = {
    "Colon Adenocarcinoma": "Colon adenocarcinoma",
    "Rectal Adenocarcinoma": "Rectum adenocarcinoma",
    "Mucinous Adenocarcinoma": "Colon adenocarcinoma or Rectum adenocarcinoma",
    "Colorectal Adenocarcinoma": "Colon adenocarcinoma or Rectum adenocarcinoma",
    "Mucinous Adenocarcinoma of the Colon and Rectum": "Colon adenocarcinoma or Rectum adenocarcinoma",
}

sub_type["non-small cell lung cancer"] = {
    "Lung Adenocarcinoma": "Lung adenocarcinoma",
    "Large Cell Lung Carcinoma": "Lung adenocarcinoma",
    "Non-Small Cell Lung Cancer": "Lung adenocarcinoma or Lung squamous cell carcinoma",
    "Lung Squamous Cell Carcinoma": "Lung squamous cell carcinoma",
    "Lung Adenosquamous Carcinoma": "Lung adenocarcinoma or Lung squamous cell carcinoma",
    "Giant Cell Carcinoma of the Lung": "Lung adenocarcinoma",
    "Mucoepidermoid Carcinoma of the Lung": "Lung adenocarcinoma",
    "Poorly Differentiated Non-Small Cell Lung Cancer": "Lung adenocarcinoma or Lung squamous cell carcinoma",
}

sub_type["renal cell carcinoma"] = {
    "Renal Clear Cell Carcinoma": "Kidney renal clear cell carcinoma",
    "Renal Papillary Cell Carcinoma": "Kidney renal papillary cell carcinoma",
    "Renal Cell Carcinoma": "Kidney renal clear cell carcinoma or Kidney renal papillary cell carcinoma",
    "Renal Medullary Carcinoma": "Miscellaneous",
    "Papillary Renal Cell Carcinoma": "Kidney renal papillary cell carcinoma",
}

sub_type["diffuse glioma"] = {
    "Glioblastoma": "Glioblastoma multiforme",
    "Astrocytoma": "Brain Lower Grade Glioma",
    "Oligodendroglioma": "Brain Lower Grade Glioma",
    "Gliosarcoma": "Glioblastoma multiforme",
    "Anaplastic Astrocytoma": "Brain Lower Grade Glioma",
    "Diffuse Intrinsic Pontine Glioma": "Brain Lower Grade Glioma",
}

sub_type["esophagogastric adenocarcinoma"] = {
    "Esophageal Adenocarcinoma": "Esophageal carcinoma",
    "Stomach Adenocarcinoma": "Stomach adenocarcinoma",
    "Esophageal Carcinoma": "Esophageal carcinoma",
    "Small Cell Carcinoma of the Stomach": "Stomach adenocarcinoma",
    "Tubular Stomach Adenocarcinoma": "Stomach adenocarcinoma",
    "Diffuse Type Stomach Adenocarcinoma": "Stomach adenocarcinoma",
    "Adenosquamous Carcinoma of the Stomach": "Stomach adenocarcinoma",
    "Signet Ring Cell Carcinoma of the Stomach": "Stomach adenocarcinoma",
    "Mucinous Stomach Adenocarcinoma": "Stomach adenocarcinoma",
    # "Mucinous Adenocarcinoma of the Stomach": "Stomach adenocarcinoma",
    "Adenocarcinoma of the Gastroesophageal Junction": "Esophageal carcinoma or Stomach adenocarcinoma",
}

# Send all keys and values to lower case
for key in sub_type.keys():
    sub_type[key] = {
        k.lower(): v.lower() for k, v in sub_type[key].items() if v is not None
    }

sub_type = {k.lower(): v for k, v in sub_type.items()}


def tcga_mapper(x, sub_x, primary, sub_type):
    if x in sub_type.keys():
        return sub_type[x][sub_x]
    elif x in primary.keys():
        return primary[x]
    else:
        return "miscellaneous"  # Means something like mixed disease
