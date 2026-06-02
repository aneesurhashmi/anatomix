import random
from tqdm import tqdm
import os
import re
import copy

from src.data.scenegraph_data import SceneGraphData
from src.util.data_utils import ReportProcessor
from src.util.data_utils import load_names_and_labels

OBJECT_NAMES, NAME_TO_LABEL, LABEL_TO_NAME, MIMIC_CLASSES = load_names_and_labels()


def get_anatomy_instruciton_prompt_template(
    organ_name_list, boxes_str, n_templates=2
):
    questions_variations = [
        "Where is the {} located in this Chest X-ray?",
        "Can you point out the {}'s position on the image?",
        "What's the location of the {} in the X-ray?",
        "Identify where the {} is on this Chest X-ray, please.",
        "Where exactly is the {} found on this image?",
        "Could you specify where to find the {} on this X-ray?",
        "Highlight the {}'s area on the image.",
        "Show me the {}'s location on this CXR.",
        "Where should I look to find the {} in this image?",
        "Can you locate the {} on this X-ray for me?",
        "Please point to the {} on this Chest X-ray.",
        "Indicate the position of the {} on this image.",
        "Describe the location of the {} on the X-ray.",
        "Where on this image is the {} located?",
        "Point out the exact location of the {} in the Chest X-ray.",
        "How can I identify the {} on this image?",
        "Where is the {} situated in this CXR?",
        "Can you highlight the {} on this image?",
        "Indicate where the {} is found on this X-ray.",
        "Describe where to find the {} on this Chest X-ray.",
    ]

    answer_variations = [
        "The <ref>{}</ref> is located at the coordinates {} on the image.",
        "You'll find the <ref>{}</ref> at {} in the X-ray.",
        "The <ref>{}</ref> can be seen at {} on the Chest X-ray.",
        "The location of the <ref>{}</ref> is at {} on the image.",
        "For the <ref>{}</ref>, the coordinates are {} on the X-ray.",
        "The <ref>{}</ref> is situated at {} in the image.",
        "On the Chest X-ray, the <ref>{}</ref> is located at {}.",
        "The <ref>{}</ref> appears at the coordinates {} on the image.",
        "In the X-ray, the <ref>{}</ref> is identifiable at {}.",
        "The location for the <ref>{}</ref> is marked at {} on the Chest X-ray.",
        "The <ref>{}</ref> is positioned at {} on the image.",
        "The area occupied by the <ref>{}</ref> is at {} in the X-ray.",
        "On the image, you can find the <ref>{}</ref> at {}.",
        "The <ref>{}</ref>'s  location is at {} on the Chest X-ray.",
        "In terms of coordinates, the <ref>{}</ref> is found at {} on the image.",
        "Regarding the <ref>{}</ref>, it is located at {} on the X-ray.",
        "The <ref>{}</ref> specifically is at {} on the Chest X-ray.",
        "Concerning the <ref>{}</ref>, you will find it at {} in the image.",
        "The <ref>{}</ref> is at {} on the X-ray.",
        "For identifying the <ref>{}</ref>, look at {} on the Chest X-ray.",
    ]

    # if n_samples==1:
    #     questions = []
    #     answers = []
    #     for org_idx, organ_name in enumerate(organ_name_list):
    #         q = random.choice(questions_variations).format(organ_name)
    #         a = random.choice(answer_variations).format(organ_name, boxes_str[org_idx])

    #         questions.append(q)
    #         answers.append(a)
    #     return questions, answers

    questions = []
    answers = []
    for org_idx, organ_name in enumerate(organ_name_list):
        q = [i.format(organ_name) for i in random.sample(questions_variations, k = n_templates)]
        a = [i.format(organ_name, boxes_str[org_idx]) for i in random.sample(answer_variations, k = n_templates)]

        questions.append(q)
        answers.append(a)

    # instruction = {"question": question, "answer": answer}
    return questions, answers


def get_report_gen_prompt_template(sample_one=False):
    templates = [
        "Generate report:\n",
        "Write a comprehensive report:\n",
        "Write findings and impressions:\n",
        "Generate findings and impressions:\n",
    ]
    if sample_one:
        return random.choice(templates)

    return templates


class ReportANDAnatomyGroundingProcessor(SceneGraphData):
    def __init__(
        self,
        system_prompt="",
        return_image=False,
        report_sections=("findings", "impression"),
        drop_if_section_missing=False,
        load_reports=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.system_prompt = system_prompt
        self.object_names = OBJECT_NAMES
        self.return_image = return_image
        self.df["report_path"] = self.df.apply(
            lambda a: os.path.join(
                kwargs["reports_dir"],
                f"p{str(a['subject_id'])[:2]}/p{str(a['subject_id'])}/s{str(a['study_id'])}.txt",
            ),
            axis=1,
        )
        self.report_paths = self.df.set_index("dicom_id")["report_path"].to_dict()
        self.df.drop(columns=["report_path"], inplace=True)

        # for anatomy grounding
        self.anatomy_box_placeholders = [
            f"<box>__BOX{idx}__</box>" for idx in range(len(OBJECT_NAMES))
        ]

        # for report gen
        self.target = "report"
        self.report_sections = report_sections
        self.drop_if_section_missing = drop_if_section_missing
        self.report_processor = False
        if load_reports:
            self.report_processor = True
            self.report_processor = ReportProcessor(section_names=report_sections)
            self.load_all_reports()  # load and preprocess all reports

    def load_all_reports(self):
        reports = {}
        for sg in tqdm(self.sg_list, total=len(self.sg_list), desc="loading report..."):
            report_path = self.report_paths[sg["image_id"]]
            report = self.load_and_process_reports(report_path, sg["study_id"])
            # if len(report)> 0:
            reports[sg["image_id"]] = report
        self.reports = reports

    def extract_sections(self, report_text):
        # Normalize text
        text = report_text.upper().strip()

        # Define regex for different section headers
        findings_pattern = r"(FINDINGS:)(.*?)(?=(IMPRESSION:|CONCLUSION:|$))"
        impression_pattern = r"(IMPRESSION:|CONCLUSION:)(.*)"

        findings_match = re.search(findings_pattern, text, re.DOTALL)
        impression_match = re.search(impression_pattern, text, re.DOTALL)

        findings = findings_match.group(2).strip() if findings_match else None
        impressions = impression_match.group(2).strip() if impression_match else None

        if findings is not None:
            findings = re.sub(" +", " ", findings.replace("\n", "")).replace("\n", "")
            findings = ". ".join(i.capitalize() for i in findings.split(". "))
        else:
            findings = ""

        if impressions is not None:
            impressions = re.sub(" +", " ", impressions.replace("\n", "")).replace(
                "\n", ""
            )
            impressions = ". ".join(i.capitalize() for i in impressions.split(". "))
        else:
            impressions = ""
        return findings, impressions

    def load_and_process_reports(self, path, study):
        with open(path) as f:
            report = f.read()

        # findings, impression = self.extract_sections(report)
        sections = self.report_processor(report, study)
        findings = sections.get("findings")
        impression = sections.get("impression")

        if self.drop_if_section_missing:
            if not findings or not impression:
                return ""

        # findings = "" if not findings else findings
        # impression = "" if not impression else impression
        out = f"Findings:\n{findings}\n\n" if findings else ""
        out += f"Impression:\n{impression}" if impression else ""

        return out

    def to_chat_template(
        self,
        sentences,
        res,
        instruction,
        content_only=False,
    ):
        context = "Image: <image_start>\nLikely findings:\n"
        context += (
            "\n".join(
                f"emb:<emb><obj_{sidx}></emb> box:<box>__BOX{sidx}__</box> {sentence}"
                for sidx, sentence in enumerate(sentences)
            )
            + "\n"
        )

        if content_only:
            return context
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": {"context": context, "instruction": instruction},
            },
            {"role": "assistant", "content": res},
        ]

    def create_anatomy_grounding_data(self, idx, content_only=False, obj_per_image = 2):
        sg = self.sg_list[idx]
        object_items = self.get_anatomical_objects(sg)
        sentences = object_items["findings"]

        # instructions, responses = get_anatomy_instruciton_prompt_template(
        #     self.object_names, self.anatomy_box_placeholders
        # )

        # select a subset of objects per image (otherwise the dataset becomes too big!!)
        curr_obj, curr_box_placeholders = zip(*random.sample(list(zip(self.object_names, self.anatomy_box_placeholders)), k=obj_per_image))
        instructions, responses = get_anatomy_instruciton_prompt_template(
            curr_obj, curr_box_placeholders
        )

        all_sgs = []
        for i in range(len(instructions)):
            messages = self.to_chat_template(
                sentences=sentences,
                res=responses[i],
                instruction=instructions[i],
                content_only=content_only,
            )
            new_sg = copy.deepcopy(sg)
            new_sg["messages"] = messages
            new_sg["bboxes"] = object_items["bboxes"].tolist()
            new_sg["split"] = self.split
            new_sg["dataset"] = "MIMIC-CXR"
            new_sg["task"] = "Anatomy Grounding"
            new_sg.pop("objects")
            all_sgs.append(new_sg)

        return all_sgs

    def create_report_gen_data(self, idx, content_only=False):
        instruction = get_report_gen_prompt_template()
        sg = self.sg_list[idx]

        object_items = self.get_anatomical_objects(sg)
        sentences = object_items["findings"]
        report = self.reports[sg["image_id"]]

        messages = self.to_chat_template(
            sentences=sentences,
            res=[report],
            instruction=instruction,
            content_only=content_only,
        )

        sg.pop("objects")
        new_sg = sg
        new_sg["messages"] = messages
        new_sg["bboxes"] = object_items["bboxes"].tolist()
        new_sg["split"] = self.split
        new_sg["dataset"] = "MIMIC-CXR"

        if len(self.report_sections) == 1:
            new_sg["task"] = f"{self.report_sections[0].capitalize()} Generation"
        else:
            new_sg["task"] = "Report Generation"

        return new_sg
