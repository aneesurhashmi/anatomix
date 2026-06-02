import json
import os
import re
from typing import Dict, Optional, List, Sequence
import stanza
import torchvision.transforms.v2 as transforms
from .box_ops import box_xyxy_to_cxcywh
import torch


# TODO: Move to utils
class NormalizeBBoxes(object):
    def __init__(self, divisor=1024):
        self.divisor = divisor

    def __call__(self, sample, boxes):
        if boxes is not None:
            if len(boxes) == 0:
                return sample, boxes
            cxcywh = box_xyxy_to_cxcywh(boxes)
            boxes = cxcywh / self.divisor
        return sample, boxes


def get_transforms(type="chest_imagenome"):
    if type == "chest_imagenome":
        # Define transformations
        return transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                NormalizeBBoxes(divisor=1024),
            ]
        )
    elif type == "no_aug":
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                NormalizeBBoxes(divisor=1024),
            ]
        )
    else:
        raise f"Transforms for type = {type} dot found. "


def load_names_and_labels():
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]

    # sorted names of the anatomical objects (MUST BE IN THIS ORDER AFTER TRAINING)
    OBJECT_NAMES = sorted(
        list(
            json.load(
                open(os.path.join(BASE_DIR, "configs/chest_imagenome_objects.json"))
            ).values()
        )
    )
    NAME_TO_LABEL = dict(zip(OBJECT_NAMES, range(len(OBJECT_NAMES))))
    LABEL_TO_NAME = {v: k for k, v in NAME_TO_LABEL.items()}

    # sorted class names for mimic (image labels)
    MIMIC_CLASSES = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    return OBJECT_NAMES, NAME_TO_LABEL, LABEL_TO_NAME, MIMIC_CLASSES


def get_default_context(only_embeddings=False):
    OBJECT_NAMES, NAME_TO_LABEL, LABEL_TO_NAME, MIMIC_CLASSES = load_names_and_labels()

    if only_embeddings:
        context = "Image: <image_start>\n"
        context += (
            "\n".join(f"emb:<emb><obj_{sidx}></emb>" for sidx in range(36)) + "\n"
        )
        return context

    context = "Image: <image_start>\nLikely findings:\n"
    context += (
        "\n".join(
            f"emb:<emb><obj_{sidx}></emb> box:<box>__BOX{sidx}__</box> {OBJECT_NAMES[sidx]}"
            for sidx in range(36)
        )
        + "\n"
    )
    return context


def section_text(text):
    """Splits text into sections.
    Assumes text is in a radiology report format, e.g.:
        COMPARISON:  Chest radiograph dated XYZ.
        IMPRESSION:  ABC...
    Given text like this, it will output text from each section,
    where the section type is determined by the all caps header.
    Returns a three element tuple:
        sections - list containing the text of each section
        section_names - a normalized version of the section name
        section_idx - list of start indices of the text in the section
    """
    p_section = re.compile(r"\n ([A-Z ()/,-]+):\s", re.DOTALL)

    sections = list()
    section_names = list()
    section_idx = list()

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0 : s.start(1)])
        section_names.append("preamble")
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find("\n")
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)

    else:
        sections.append(text)
        section_names.append("full report")
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    # remove empty sections
    # this handles when the report starts with a finding-like statement
    #  .. but this statement is not a section, more like a report title
    #  e.g. p10/p10103318/s57408307
    #    CHEST, PA LATERAL:
    #
    #    INDICATION:   This is the actual section ....
    # it also helps when there are multiple findings sections
    # usually one is empty
    for i in reversed(range(len(section_names))):
        if section_names[i] in ("impression", "findings"):
            if sections[i].strip() == "":
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    if ("impression" not in section_names) & ("findings" not in section_names):
        # create a new section for the final paragraph
        if "\n \n" in sections[-1]:
            sections.append("\n \n".join(sections[-1].split("\n \n")[1:]))
            sections[-2] = sections[-2].split("\n \n")[0]
            section_names.append("last_paragraph")
            section_idx.append(section_idx[-1] + len(sections[-2]))

    # sections = [remove_spaces(i) for i in sections] # remove spaces and new lines
    return sections, section_names, section_idx


def remove_spaces(text):
    text = re.sub(" +", " ", text.replace("\n", "")).replace("\n", "")
    text = ". ".join(i for i in text.split(". "))
    return text


def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    frequent_sections = {
        "preamble": "preamble",  # 227885
        "impression": "impression",  # 187759
        "comparison": "comparison",  # 154647
        "indication": "indication",  # 153730
        "findings": "findings",  # 149842
        "examination": "examination",  # 94094
        "technique": "technique",  # 81402
        "history": "history",  # 45624
        "comparisons": "comparison",  # 8686
        "clinical history": "history",  # 7121
        "reason for examination": "indication",  # 5845
        "notification": "notification",  # 5749
        "reason for exam": "indication",  # 4430
        "clinical information": "history",  # 4024
        "exam": "examination",  # 3907
        "clinical indication": "indication",  # 1945
        "conclusion": "impression",  # 1802
        "chest, two views": "findings",  # 1735
        "recommendation(s)": "recommendations",  # 1700
        "type of examination": "examination",  # 1678
        "reference exam": "comparison",  # 347
        "patient history": "history",  # 251
        "addendum": "addendum",  # 183
        "comparison exam": "comparison",  # 163
        "date": "date",  # 108
        "comment": "comment",  # 88
        "findings and impression": "impression",  # 87
        "wet read": "wet read",  # 83
        "comparison film": "comparison",  # 79
        "recommendations": "recommendations",  # 72
        "findings/impression": "impression",  # 47
        "pfi": "history",
        "recommendation": "recommendations",
        "wetread": "wet read",
        "ndication": "impression",  # 1
        "impresson": "impression",  # 2
        "imprression": "impression",  # 1
        "imoression": "impression",  # 1
        "impressoin": "impression",  # 1
        "imprssion": "impression",  # 1
        "impresion": "impression",  # 1
        "imperssion": "impression",  # 1
        "mpression": "impression",  # 1
        "impession": "impression",  # 3
        "findings/ impression": "impression",  # ,1
        "finding": "findings",  # ,8
        "findins": "findings",
        "findindgs": "findings",  # ,1
        "findgings": "findings",  # ,1
        "findngs": "findings",  # ,1
        "findnings": "findings",  # ,1
        "finidngs": "findings",  # ,2
        "idication": "indication",  # ,1
        "reference findings": "findings",  # ,1
        "comparision": "comparison",  # ,2
        "comparsion": "comparison",  # ,1
        "comparrison": "comparison",  # ,1
        "comparisions": "comparison",  # ,1
    }

    p_findings = [
        "chest",
        "portable",
        "pa and lateral",
        "lateral and pa",
        "ap and lateral",
        "lateral and ap",
        "frontal and",
        "two views",
        "frontal view",
        "pa view",
        "ap view",
        "one view",
        "lateral view",
        "bone window",
        "frontal upright",
        "frontal semi-upright",
        "ribs",
        "pa and lat",
    ]
    p_findings = re.compile("({})".format("|".join(p_findings)))

    main_sections = ["impression", "findings", "history", "comparison", "addendum"]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = "findings"

        # if it looks like it is describing the entire study
        # it's equivalent to findings
        # group similar phrasings for impression

    return section_names


def custom_mimic_cxr_rules():
    custom_section_names = {
        "s50913680": "recommendations",  # files/p11/p11851243/s50913680.txt
        "s59363654": "examination",  # files/p12/p12128253/s59363654.txt
        "s59279892": "technique",  # files/p13/p13150370/s59279892.txt
        "s59768032": "recommendations",  # files/p13/p13249077/s59768032.txt
        "s57936451": "indication",  # files/p14/p14325424/s57936451.txt
        "s50058765": "indication",  # files/p14/p14731346/s50058765.txt
        "s53356173": "examination",  # files/p15/p15898350/s53356173.txt
        "s53202765": "technique",  # files/p16/p16076182/s53202765.txt
        "s50808053": "technique",  # files/p16/p16631485/s50808053.txt
        "s51966317": "indication",  # files/p10/p10817099/s51966317.txt
        "s50743547": "examination",  # files/p11/p11388341/s50743547.txt
        "s56451190": "note",  # files/p11/p11842879/s56451190.txt
        "s59067458": "recommendations",  # files/p11/p11984647/s59067458.txt
        "s59215320": "examination",  # files/p12/p12408912/s59215320.txt
        "s55124749": "indication",  # files/p12/p12428492/s55124749.txt
        "s54365831": "indication",  # files/p13/p13876470/s54365831.txt
        "s59087630": "recommendations",  # files/p14/p14267880/s59087630.txt
        "s58157373": "recommendations",  # files/p15/p15032392/s58157373.txt
        "s56482935": "recommendations",  # files/p15/p15388421/s56482935.txt
        "s58375018": "recommendations",  # files/p15/p15505556/s58375018.txt
        "s54654948": "indication",  # files/p17/p17090359/s54654948.txt
        "s55157853": "examination",  # files/p18/p18975498/s55157853.txt
        "s51491012": "history",  # files/p19/p19314266/s51491012.txt
    }

    custom_indices = {
        "s50525523": [201, 349],  # files/p10/p10602608/s50525523.txt
        "s57564132": [233, 554],  # files/p10/p10637168/s57564132.txt
        "s59982525": [313, 717],  # files/p11/p11989982/s59982525.txt
        "s53488209": [149, 475],  # files/p12/p12458657/s53488209.txt
        "s54875119": [234, 988],  # files/p13/p13687044/s54875119.txt
        "s50196495": [59, 399],  # files/p13/p13894879/s50196495.txt
        "s56579911": [59, 218],  # files/p15/p15394326/s56579911.txt
        "s52648681": [292, 631],  # files/p15/p15666238/s52648681.txt
        "s59889364": [172, 453],  # files/p15/p15835529/s59889364.txt
        "s53514462": [73, 377],  # files/p16/p16297706/s53514462.txt
        "s59505494": [59, 450],  # files/p16/p16730991/s59505494.txt
        "s53182247": [59, 412],  # files/p16/p16770442/s53182247.txt
        "s51410602": [47, 320],  # files/p17/p17069955/s51410602.txt
        "s56412866": [522, 822],  # files/p17/p17612000/s56412866.txt
        "s54986978": [59, 306],  # files/p17/p17912487/s54986978.txt
        "s59003148": [262, 505],  # files/p17/p17916384/s59003148.txt
        "s57150433": [61, 394],  # files/p18/p18335791/s57150433.txt
        "s56760320": [219, 457],  # files/p18/p18418794/s56760320.txt
        "s59562049": [158, 348],  # files/p18/p18502016/s59562049.txt
        "s52674888": [145, 296],  # files/p19/p19381919/s52674888.txt
        "s55258338": [192, 568],  # files/p13/p13719117/s55258338.txt
        "s59330497": [140, 655],  # files/p15/p15479218/s59330497.txt
        "s52119491": [179, 454],  # files/p17/p17959278/s52119491.txt
        # below have no findings at all in the entire report
        "s58235663": [0, 0],  # files/p11/p11573679/s58235663.txt
        "s50798377": [0, 0],  # files/p12/p12632853/s50798377.txt
        "s54168089": [0, 0],  # files/p14/p14463099/s54168089.txt
        "s53071062": [0, 0],  # files/p15/p15774521/s53071062.txt
        "s56724958": [0, 0],  # files/p16/p16175671/s56724958.txt
        "s54231141": [0, 0],  # files/p16/p16312859/s54231141.txt
        "s53607029": [0, 0],  # files/p17/p17603668/s53607029.txt
        "s52035334": [0, 0],  # files/p19/p19349312/s52035334.txt
    }

    return custom_section_names, custom_indices


"""
Some pre-processing is taken from https://github.com/ttanida/rgrg
"""


def split_into_sections(full_text_report: str, study: str) -> Optional[Dict[str, str]]:
    # exclude these special cases
    custom_section_names, custom_indices = custom_mimic_cxr_rules()
    if study in custom_indices or study in custom_section_names:
        return None

    sections, section_names, section_idx = section_text(full_text_report)
    sections_by_name = {}
    for section, name in zip(sections, section_names):
        sections_by_name[name] = section

    return sections_by_name


PATTERN_REPLACE_MULTILINES = re.compile(r"(?:[\t ]*(?:\r\n|\n)+)+", flags=re.MULTILINE)
PATTERN_REPLACE_MULTISPACES = re.compile(r"[\t ]+")
PATTERN_REPLACE_MULTIDOT = re.compile(r"(\. ?)+")
SUBSTRINGS_TO_REMOVE = "WET READ VERSION|WET READ|UPRIGHT PORTABLE AP CHEST RADIOGRAPH:|UPRIGHT AP VIEW OF THE CHEST:|UPRIGHT AP AND LATERAL VIEWS OF THE CHEST:|TECHNOLOGIST'S NOTE:|TECHNIQUE:|SUPINE PORTABLE RADIOGRAPH:|SUPINE PORTABLE CHEST RADIOGRAPHS:|SUPINE PORTABLE CHEST RADIOGRAPH:|SUPINE PORTABLE AP CHEST RADIOGRAPH:|SUPINE FRONTAL CHEST RADIOGRAPH:|SUPINE CHEST RADIOGRAPH:|SUPINE AP VIEW OF THE CHEST:|SINGLE SUPINE PORTABLE VIEW OF THE CHEST:|SINGLE SEMI-ERECT AP PORTABLE VIEW OF THE CHEST:|SINGLE PORTABLE UPRIGHT CHEST RADIOGRAPH:|SINGLE PORTABLE CHEST RADIOGRAPH:|SINGLE PORTABLE AP CHEST RADIOGRAPH:|SINGLE FRONTAL VIEW OF THE CHEST:|SINGLE FRONTAL PORTABLE VIEW OF THE CHEST:|SINGLE AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|SINGLE AP UPRIGHT CHEST RADIOGRAPH:|SINGLE AP PORTABLE CHEST RADIOGRAPH:|SEMIERECT PORTABLE RADIOGRAPH OF THE CHEST:|SEMIERECT AP VIEW OF THE CHEST:|SEMI-UPRIGHT PORTABLE RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT PORTABLE CHEST RADIOGRAPH:|SEMI-UPRIGHT PORTABLE AP RADIOGRAPH OF THE CHEST:|SEMI-UPRIGHT AP VIEW OF THE CHEST:|SEMI-ERECT PORTABLE FRONTAL CHEST RADIOGRAPH:|SEMI-ERECT PORTABLE CHEST:|SEMI-ERECT PORTABLE CHEST RADIOGRAPH:|REPORT:|PORTABLES SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE UPRIGHT FRONTAL VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE CHEST:|PORTABLE UPRIGHT AP VIEW OF THE ABDOMEN:|PORTABLE SUPINE FRONTAL VIEW OF THE CHEST:|PORTABLE SUPINE FRONTAL CHEST RADIOGRAPH:|PORTABLE SUPINE CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT RADIOGRAPH:|PORTABLE SEMI-UPRIGHT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT CHEST RADIOGRAPH:|PORTABLE SEMI-UPRIGHT AP CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPHS:|PORTABLE SEMI-ERECT FRONTAL CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT CHEST RADIOGRAPH:|PORTABLE SEMI-ERECT AP AND PA CHEST RADIOGRAPH:|PORTABLE FRONTAL VIEW OF THE CHEST:|PORTABLE FRONTAL CHEST RADIOGRAPH:|PORTABLE ERECT RADIOGRAPH:|PORTABLE CHEST RADIOGRAPH:|PORTABLE AP VIEW OF THE CHEST:|PORTABLE AP UPRIGHT CHEST RADIOGRAPH:|PORTABLE AP CHEST RADIOGRAPH:|PA AND LATERAL VIEWS OF THE CHEST:|PA AND LATERAL CHEST RADIOGRAPHS:|PA AND LATERAL CHEST RADIOGRAPH:|PA AND LAT CHEST RADIOGRAPH:|PA AND AP CHEST RADIOGRAPH:|NOTIFICATION:|IMPRESSON:|IMPRESSION: AP CHEST:|IMPRESSION: AP|IMPRESSION:|IMPRESSION AP|IMPRESSION|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPRIGHT PORTABLE CHEST:|FRONTAL UPPER ABDOMINAL RADIOGRAPH, TWO IMAGES:|FRONTAL SUPINE PORTABLE CHEST:|FRONTAL SEMI-UPRIGHT PORTABLE CHEST:|FRONTAL RADIOGRAPH OF THE CHEST:|FRONTAL PORTABLE SUPINE CHEST:|FRONTAL PORTABLE CHEST:|FRONTAL PORTABLE CHEST RADIOGRAPH:|FRONTAL LATERAL VIEWS CHEST:|FRONTAL LATERAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPHS:|FRONTAL CHEST RADIOGRAPH:|FRONTAL CHEST RADIOGRAPH WITH THE PATIENT IN SUPINE AND UPRIGHT POSITIONS:|FRONTAL AND LATERAL VIEWS OF THE CHEST:|FRONTAL AND LATERAL FRONTAL CHEST RADIOGRAPH:|FRONTAL AND LATERAL CHEST RADIOGRAPHS:|FRONTAL AND LATERAL CHEST RADIOGRAPH:|FRONTAL|FINIDNGS:|FINDNGS:|FINDINGS:|FINDINGS/IMPRESSION:|FINDINGS AND IMPRESSION:|FINDINGS|FINDING:|FINAL REPORT FINDINGS:|FINAL REPORT EXAMINATION:|FINAL REPORT|FINAL ADDENDUM ADDENDUM:|FINAL ADDENDUM ADDENDUM|FINAL ADDENDUM \*\*\*\*\*\*\*\*\*\*ADDENDUM\*\*\*\*\*\*\*\*\*\*\*|FINAL ADDENDUM|EXAMINATION: DX CHEST PORT LINE/TUBE PLCMT 1 EXAM|CONCLUSION:|COMPARISONS:|COMPARISON:|COMPARISON.|CHEST:|CHEST/ABDOMEN RADIOGRAPHS:|CHEST, TWO VIEWS:|CHEST, SINGLE AP PORTABLE VIEW:|CHEST, PA AND LATERAL:|CHEST, AP:|CHEST, AP UPRIGHT:|CHEST, AP UPRIGHT AND LATERAL:|CHEST, AP SUPINE:|CHEST, AP SEMI-UPRIGHT:|CHEST, AP PORTABLE, UPRIGHT:|CHEST, AP AND LATERAL:|CHEST SUPINE:|CHEST RADIOGRAPH:|CHEST PA AND LATERAL RADIOGRAPH:|CHEST AP:|BEDSIDE UPRIGHT FRONTAL CHEST RADIOGRAPH:|AP:|AP,|AP VIEW OF THE CHEST:|AP UPRIGHT PORTABLE CHEST RADIOGRAPH:|AP UPRIGHT CHEST RADIOGRAPH:|AP UPRIGHT AND LATERAL CHEST RADIOGRAPHS:|AP PORTABLE SUPINE CHEST RADIOGRAPH:|AP PORTABLE CHEST RADIOGRAPH:|AP FRONTAL CHEST RADIOGRAPH:|AP CHEST:|AP CHEST RADIOGRAPH:|AP AND LATERAL VIEWS OF THE CHEST:|AP AND LATERAL CHEST RADIOGRAPHS:|AP AND LATERAL CHEST RADIOGRAPH:|5. |4. |3. |2. |1. |#1 |#2 |#3 |#4 |#5 "


def clean_section_text(text: str) -> str:
    text = remove_wet_read(text)
    text = re.sub(SUBSTRINGS_TO_REMOVE, "", text, flags=re.DOTALL)
    text = PATTERN_REPLACE_MULTISPACES.sub(" ", text).strip()
    text = remove_spaces(text)
    return text


def remove_wet_read(text):
    """Removes substring like 'WET READ: ___ ___ 8:19 AM' that is irrelevant."""
    # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
    index_slices_to_remove = []
    for index in range(len(text)):
        if text[index : index + 8] == "WET READ":
            # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
            for curr_index in range(index + 8, len(text)):
                # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                # if the next WET READ substring is encountered
                if (
                    text[curr_index : curr_index + 2] in ["AM", "PM"]
                    or text[curr_index : curr_index + 8] == "WET READ"
                ):
                    break

                # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                if text[curr_index : curr_index + 2] in ["AM", "PM"]:
                    index_slices_to_remove.append((index, curr_index + 2))

    # remove the slices in reversed order, such that the correct index order is preserved
    for indices_tuple in reversed(index_slices_to_remove):
        start_index, end_index = indices_tuple
        text = text[:start_index] + text[end_index:]

    return text


def clean_sentence(sentence: str) -> str:
    # remove newlines or multiple newlines (replace by space)
    sentence = PATTERN_REPLACE_MULTILINES.sub("\n", sentence)
    sentence = sentence.replace("\n", " ")
    # merge multiple spaces into one
    sentence = PATTERN_REPLACE_MULTISPACES.sub(" ", sentence)
    # remove multiple dots (replace by one dot)
    sentence = PATTERN_REPLACE_MULTIDOT.sub(".", sentence)
    sentence = PATTERN_REPLACE_MULTISPACES.sub(" ", sentence).strip()
    # capitalize first letter
    if len(sentence) > 0:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def remove_duplicate_sentences(sentences: List[str]) -> List[str]:
    return list(dict.fromkeys(sentences))


class ReportProcessor:
    def __init__(
        self,
        lang: str = "en",
        min_words: int = 2,
        section_names: Sequence[str] = ("findings", "impression"),
    ):
        stanza.download(lang)
        # self.stanza_tokenizer = stanza.Pipeline(lang, processors='tokenize', use_gpu=False)
        self.min_words = min_words
        self.section_names = section_names

    def __call__(self, report_full_text: str, study: str) -> Optional[List[str]]:
        sections = split_into_sections(report_full_text, study)
        if sections is None:
            # log.warning(f"Ignoring study {study}.")
            return {}
        if not any(key in sections for key in self.section_names):
            # log.warning(f"Ignoring study {study} because it does not contain any of the following sections: {self.section_names}. Available sections: {sections.keys()}.")
            return {}
        relevant_sections = [
            sections[key] for key in self.section_names if key in sections
        ]
        relevant_sections = {
            key: clean_section_text(value)
            for key, value in sections.items()
            if key in self.section_names
        }
        return relevant_sections
        # relevant_sentences: List[str] = [sentence for section in relevant_sections for sentence in self._process_section(section)]

        # if len(relevant_sentences) == 0:
        #     log.warning(f"Ignoring study {study} because it does not contain any sentences in the relevant sections (or only too short or removed sentences).")
        #     return None
        # return relevant_sentences

    # def _process_section(self, section_txt: str) -> List[str]:
    #     section_txt = clean_section_text(section_txt)
    #     # doc: Document = self.stanza_tokenizer(section_txt)

    #     # sentences = [sent.text for sent in doc.sentences]
    #     # sentences = [clean_sentence(sentence) for sentence in sentences]
    #     # sentences = [sentence for sentence in sentences if len(sentence.split()) >= self.min_words]
    #     section_txt = remove_duplicate_sentences(section_txt)

    #     return section_txt
