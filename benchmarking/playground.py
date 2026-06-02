import os
import pandas as pd
import matplotlib.pyplot as plt


metrics = [ 
    # 'radgraph_partial',
    # 'radgraph_complete',
    'bleu',
    'bertscore',
    'rouge1',
    # 'rouge2',
    # 'rougeL',
    # 'srr_bert_weighted_f1',
    # 'srr_bert_weighted_precision',
    # 'srr_bert_weighted_recall',
    'meteor',
    'radgraph_simple',
    'chexbert-5_micro avg_f1-score',
    'chexbert-all_micro avg_f1-score',
    # 'chexbert-5_macro avg_f1-score',
    # 'chexbert-all_macro avg_f1-score',
    'chexbert-5_weighted_f1',
    'chexbert-all_weighted_f1'
    ]



output_dir = "/path/to/data/experiments/benchmarking"
# dataset_name = "mimic_report_gen"
# dataset_name = "mimic_impression_gen"
# dataset_name = "mimic_findings_gen"

os.makedirs('./results/', exist_ok=True)
for dataset_name in ["mimic_report_gen", "mimic_impression_gen", "mimic_findings_gen"]:
    results_filenames = sorted([i for i in os.listdir(os.path.join(output_dir, dataset_name)) if "result" in i])

    all_data = {}

    for filename in results_filenames:
        print(filename)
        if "48000" in filename or "88000" in filename or "0000" in filename:
            continue
        if "mymodel" in filename:
            # if "_default_context" not in filename and "sentence" not in filename:
            if "sentence" not in filename:
                continue
        modelname = "_".join(filename.split("_")[:-1])
        filepath = os.path.join(output_dir, dataset_name, filename)
        df = pd.read_csv(filepath)
        data = df[metrics].describe().loc['mean'].to_dict()

        all_data[modelname] = data


    all_df = pd.DataFrame(all_data)
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot grouped bar chart
    all_df.plot(kind="bar", ax=ax, width=0.8)

    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", label_type="edge", padding=2, rotation=90, fontsize=10)

    plt.title(f"{dataset_name} Benchmark")
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.legend(title="Models")
    plt.grid()
    plt.savefig(f"./results/{dataset_name}.png")

