from flask import Flask, jsonify, render_template, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import json
from matplotlib.patches import Patch

app = Flask(__name__)

#loading main data
def load_data(file_path='data/allQuestData_all.csv'):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return str(e)

#loading subject metadata
def load_metadata(file_path='data/subject_data.csv'):
    try:
        metadata = pd.read_csv(file_path)
        return metadata
    except Exception as e:
        return str(e)

#section for subjects with the lowest thresholds
def lowest_avg_threshold(df):
    exclude = ['cn', 'zh', 'jb', 'cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    # exclude = ['gc']
    df_filtered = df[~df['subject'].isin(exclude)].copy()

    if "subject" not in df.columns or "threshold" not in df.columns:
        return {"error": "Missing required columns: 'subject' and 'threshold'"}
    
    #relevant subjects are grouped
    df_avg = df_filtered.groupby("subject")["threshold"].mean().reset_index()
    
    #sort average
    df_sorted = df_avg.sort_values(by="threshold", ascending=True)
    
    #pick the top 5
    lowest_thresholds = df_sorted.head(5).reset_index(drop=True)
    lowest_thresholds.index += 1

    return lowest_thresholds

#section for average threshold of all subjects, overall and best 2 of 3
def overall_thresholds(df):
    if "subject" not in df.columns or "threshold" not in df.columns or "stimulus" not in df.columns or "nStim" not in df.columns or "noise" not in df.columns:
        return {"error": "Missing required columns."}
    
    #overall average per subject, stimulus, nStim, and noise
    overall_avg = df.groupby(["subject", "stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    #calculating SEM for overall
    overall_sem = overall_avg.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    overall_sem["sem_threshold"] = overall_sem["std_threshold"] / np.sqrt(overall_sem["count"])

    #best 2 out of 3 by selecting the lowest 2 values per group
    best_2_of_3 = (
        df.groupby(["subject", "stimulus", "nStim", "noise"])["threshold"]
        .apply(lambda x: x.nsmallest(2).mean())  
        .reset_index()
    )

    #calculating SEM for best 2 of 3
    best_2_sem = best_2_of_3.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    best_2_sem["sem_threshold"] = best_2_sem["std_threshold"] / np.sqrt(best_2_sem["count"])
  
    #final averages across subjects for each condition
    overall_avg_cond = overall_avg.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()
    best_2_avg_cond = best_2_of_3.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    return overall_avg_cond, best_2_avg_cond, overall_sem, best_2_sem

#section for averaging thresholds per subject
def avg_per_subject(df, subject):
    if "subject" not in df.columns or "threshold" not in df.columns or "stimulus" not in df.columns or "nStim" not in df.columns or "noise" not in df.columns:
        return {"error": "Missing required columns."}
    
    df_subject = df[df["subject"] == subject]
    
    if df_subject.empty:
        return {"error": f"No data found for subject {subject}"}

    #calculate average per condition
    avg_threshold = df_subject.groupby(["stimulus", "nStim", "noise"])["threshold"].mean().reset_index()

    #calculating SEM for 3 out of 3
    subject_sem = df_subject.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    subject_sem["sem_threshold"] = subject_sem["std_threshold"] / np.sqrt(subject_sem["count"])
    
    #best 2 out of 3
    best_2_of_3 = (
        df_subject.groupby(["stimulus", "nStim", "noise"])["threshold"]
        .apply(lambda x: x.nsmallest(2).mean())  
        .reset_index()
    )

    #best 2 out of 3 for SEM
    best_2_of_3_sem = (
        df_subject.groupby(["stimulus", "nStim", "noise"])["threshold"]
        .apply(lambda x: x.nsmallest(2))  
        .reset_index()
    )
    best_2_of_3_sem = best_2_of_3_sem.rename(columns={"threshold": "threshold_best2"})

    #calculating SEM
    subject_best_2_sem = best_2_of_3_sem.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold_best2", "mean"),
        std_threshold=("threshold_best2", "std"),
        count=("threshold_best2", "count")
    ).reset_index()
    subject_best_2_sem["sem_threshold"] = subject_best_2_sem["std_threshold"] / np.sqrt(subject_best_2_sem["count"])
    
    return avg_threshold, best_2_of_3, subject_sem, subject_best_2_sem

#handles when a subject is selected to load it, working with java in html
@app.route("/subject/<subject>")
def subject(subject):
    df = load_data()
    
    if isinstance(df, str):  
        return jsonify({"error": "Error loading data"}), 500
    
    subject_avg, subject_best_2_avg, subject_sem, subject_best_2_sem = avg_per_subject(df, subject)
    
    if isinstance(subject_avg, dict):  
        return jsonify(subject_avg), 404  # Return error if subject not found

    subject_avg_fixed = subject_avg[subject_avg['noise'] == 'fixed']
    subject_avg_variable = subject_avg[subject_avg['noise'] == 'variable']
    subject_best_2_fixed = subject_best_2_avg[subject_best_2_avg['noise'] == 'fixed']
    subject_best_2_variable = subject_best_2_avg[subject_best_2_avg['noise'] == 'variable']

    subject_sem_fixed = subject_sem[subject_sem['noise'] == 'fixed']
    subject_sem_variable = subject_sem[subject_sem['noise'] == 'variable']
    subject_best_2_sem_fixed = subject_best_2_sem[subject_best_2_sem['noise'] == 'fixed']
    subject_best_2_sem_variable = subject_best_2_sem[subject_best_2_sem['noise'] == 'variable']

    return jsonify({
        "title": subject,
        "subject_avg_title": f"Thresholds for {subject.upper()}",
        "subject_best_2_title": f"Best 2 of 3 Thresholds for {subject.upper()}",
        "subject_avg": subject_avg.to_dict(orient="records"),
        "subject_best_2_avg": subject_best_2_avg.to_dict(orient="records"),

        "subject_sem": subject_sem.to_dict(orient="records"),
        "subject_best_2_sem": subject_best_2_sem.to_dict(orient="records"),

        "subject_avg_fixed": subject_avg_fixed.to_dict(orient="records"),
        "subject_avg_variable": subject_avg_variable.to_dict(orient="records"),
        "subject_best_2_fixed": subject_best_2_fixed.to_dict(orient="records"),
        "subject_best_2_variable": subject_best_2_variable.to_dict(orient="records"),

        "subject_sem_fixed": subject_sem_fixed.to_dict(orient="records"),
        "subject_sem_variable": subject_sem_variable.to_dict(orient="records"),
        "subject_best_2_sem_fixed": subject_best_2_sem_fixed.to_dict(orient="records"),
        "subject_best_2_sem_variable": subject_best_2_sem_variable.to_dict(orient="records"),

        "subject_avg_html": subject_avg.set_index(pd.Index(range(1, len(subject_avg) + 1))).to_html(classes='table table-stripped'),
        "subject_best_2_html": subject_best_2_avg.set_index(pd.Index(range(1, len(subject_best_2_avg) + 1))).to_html(classes='table table-stripped'),

        "subject_sem_html": subject_sem.to_html(classes='table table-stripped'),
        "subject_best_2_sem_html": subject_best_2_sem.to_html(classes='table table-stripped'),
    })

#section for handling acuity thresholds and tables
def acuity(metadata, df):
    exclude = ['cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    # exclude = ['gc']
    #filter and clean metadata
    metadata_filtered = metadata[~metadata['subject'].isin(exclude)].copy()
    metadata_filtered["visual_acuity_base"] = metadata_filtered["visual acuity"].str.extract(r"(20/\d+)").fillna("20/20")
    
    #categorize by acuity (numerically parsed)
    metadata_filtered['acuity_category'] = metadata_filtered['visual_acuity_base'].apply(
        lambda x: 'Lower than 20/20' if int(x.split('/')[1]) < 20 else '20/20 or higher'
    )

    #merge full data with subject-level acuity categories
    df_filtered = df[~df['subject'].isin(exclude)]
    merged = pd.merge(df_filtered, metadata_filtered[['subject', 'acuity_category']], on='subject', how='inner')

    #group by acuity + condition (stimulus, nStim, noise)
    #lower
    lower_avg_threshold = merged[merged['acuity_category'] == 'Lower than 20/20'] \
        .groupby(['stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #calculating SEM for lower
    lower = merged[merged['acuity_category'] == 'Lower than 20/20'] 

    lower_sem_threshold = lower.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    lower_sem_threshold["sem_threshold"] = lower_sem_threshold["std_threshold"] / np.sqrt(lower_sem_threshold["count"])
    
    #higher
    higher_avg_threshold = merged[merged['acuity_category'] == '20/20 or higher'] \
        .groupby(['stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #calculating SEM for lower
    higher = merged[merged['acuity_category'] == '20/20 or higher']
    
    higher_sem_threshold = higher.groupby(["stimulus", "nStim", "noise"]).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    higher_sem_threshold["sem_threshold"] = higher_sem_threshold["std_threshold"] / np.sqrt(higher_sem_threshold["count"])

    #count number of subjects per category
    acuity_counts = metadata_filtered['acuity_category'].value_counts().reset_index()
    acuity_counts.columns = ['Acuity Category', 'Subject Count']
    acuity_counts.index += 1

    return acuity_counts, lower_avg_threshold, higher_avg_threshold, lower_sem_threshold, higher_sem_threshold

#section for handedness-based thresholds and tables
def handedness(df, metadata):
    #exclude the subjects
    exclude = ['cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    # exclude = ['gc']
    metadata_filtered = metadata[~metadata['subject'].isin(exclude)].copy()
    df_filtered = df[~df['subject'].isin(exclude)].copy()

    #merge metadata and data for handedness information
    merged = pd.merge(df_filtered, metadata_filtered[['subject', 'handedness']], on='subject', how='inner')
    
    #left handed
    left_handed = merged[merged['handedness'] == 'Left']
    left_avg_threshold  = left_handed.groupby(['handedness', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #calculating SEM for left handed
    left_sem = left_handed.groupby(['handedness', 'stimulus', 'nStim', 'noise']).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    left_sem["sem_threshold"] = left_sem["std_threshold"] / np.sqrt(left_sem["count"])
    
    #right handed
    right_handed = merged[merged['handedness'] == 'Right']
    subject_averages = right_handed.groupby('subject')['threshold'].mean().reset_index()
    lowest_two_subjects = subject_averages.nsmallest(2, 'threshold')['subject']

    #filter right-handed data for those two subjects
    right_lowest = right_handed[right_handed['subject'].isin(lowest_two_subjects)]

    #average their thresholds per condition
    right_avg_threshold = right_lowest.groupby(['handedness', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #calculating SEM for right handed
    right_sem = right_lowest.groupby(['handedness', 'stimulus', 'nStim', 'noise']).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    right_sem["sem_threshold"] = right_sem["std_threshold"] / np.sqrt(right_sem["count"])

    #handedness table count
    handedness_counts = metadata[~metadata['subject'].isin(exclude)]['handedness'].value_counts().reset_index()
    handedness_counts.columns = ['Handedness', 'Subject Count']
    handedness_counts.index += 1

    return left_avg_threshold, right_avg_threshold, handedness_counts, left_sem, right_sem

#section for gender-based thresholds and tables
def gender(df, metadata):
    #exclude the subjects
    exclude = ['cc', 'co', 'db', 'ec', 'gc', 'jf', 'jm', 'kf']
    # exclude = ['gc']
    metadata_filtered = metadata[~metadata['subject'].isin(exclude)].copy()
    df_filtered = df[~df['subject'].isin(exclude)].copy()

    #merge metadata and data for handedness information
    merged = pd.merge(df_filtered, metadata_filtered[['subject', 'gender']], on='subject', how='inner')
    
    #female
    female = merged[merged['gender'] == 'F']
    female_avg_threshold  = female.groupby(['gender', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #SEM for female
    female_sem = female.groupby(['gender', 'stimulus', 'nStim', 'noise']).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    female_sem["sem_threshold"] = female_sem["std_threshold"] / np.sqrt(female_sem["count"])

    #right handed
    male = merged[merged['gender'] == 'M']
    male_avg_threshold = male.groupby(['gender', 'stimulus', 'nStim', 'noise'])['threshold'].mean().reset_index()

    #SEM for male
    male_sem = male.groupby(['gender', 'stimulus', 'nStim', 'noise']).agg(
        mean_threshold=("threshold", "mean"),
        std_threshold=("threshold", "std"),
        count=("threshold", "count")
    ).reset_index()
    male_sem["sem_threshold"] = male_sem["std_threshold"] / np.sqrt(male_sem["count"])

    #handedness table count
    gender_counts = metadata[~metadata['subject'].isin(exclude)]['gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Subject Count']
    gender_counts.index += 1

    return female_avg_threshold, male_avg_threshold, gender_counts, female_sem, male_sem

def uncertainty(df):
    exclude_subj_fixed = ["th", "eh", "sj", "eo", "cc", "xr"]
    df_fixed = df[~df['subject'].isin(exclude_subj_fixed)]

    fixed_noise = df_fixed[df_fixed['noise'] == 'fixed']

    exclude_subj_variable = ["gc", "eh"]
    df_variable = df[~df['subject'].isin(exclude_subj_variable)]

    variable_noise = df_variable[df_variable['noise'] == 'variable']

    fixed_subject_avg = (
        fixed_noise
        .groupby(['subject', 'stimulus'])['threshold']
        .mean()
        .reset_index()
        .pivot(index='subject', columns='stimulus', values='threshold')
        .reset_index()
    )
    fixed_subject_avg = fixed_subject_avg.dropna(subset=['grating', 'texture'])

    variable_subject_avg = (
        variable_noise
        .groupby(['subject', 'stimulus'])['threshold']
        .mean()
        .reset_index()
        .pivot(index='subject', columns='stimulus', values='threshold')
        .reset_index()
    )
    variable_subject_avg = variable_subject_avg.dropna(subset=['grating', 'texture'])

    subject_colors = {
        "jf": "gold", "cc": "aquamarine", "co": "bisque", "jm": "blue",
        "db": "brown", "ec": "burlywood", "eo": "cadetblue", "gc": "chartreuse",
        "kf": "chocolate", "eh": "coral", "na": "pink", "kc": "cyan",
        "lp": "darkgoldenrod", "th": "darkorange", "jc": "darkorchid",
        "gv": "darkseagreen", "cw": "darkslategray", "mr": "deeppink",
        "an": "deepskyblue", "sj": "firebrick", "ve": "azure",
        "xr": "slateblue", "jb": "slategray", "zh": "plum", "cn": "turquoise"
    }

    return fixed_subject_avg, variable_subject_avg, subject_colors

def plot(subject_avg, subject_colors):
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in subject_avg.iterrows():
        subj = row['subject']
        color = subject_colors.get(subj, 'black')
        ax.scatter(row['texture'], row['grating'], color=color, marker='o', s=100, label=subj)
        ax.scatter(row['texture'], row['texture'], color=color, marker='^', s=100)

    lims = [
        min([subject_avg[['texture', 'grating']].min().min(), 0]),
        max([subject_avg[['texture', 'grating']].max().max()])
    ]
    
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Texture')
    ax.set_ylabel('Grating')
    ax.set_title('Grating vs Texture Thresholds')
    ax.set_xlim(lims[0] * 0.95, lims[1] * 1.05)
    ax.set_ylim(lims[0] * 0.95, lims[1] * 1.05)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')


@app.route("/dashboard")
def dashboard():
    df = load_data()
    metadata = load_metadata()

    #handling errors
    if isinstance(df, str) or isinstance(metadata, str):
        return jsonify({"error": "Error loading data"}), 500
    
    #getting the lowest 5 subjects formula
    lowest_thresholds = lowest_avg_threshold(df)

    fixed_subject_avg, variable_subject_avg, subject_colors = uncertainty(df)

    plot_fixed_uncertainty = plot(fixed_subject_avg, subject_colors)
    plot_variable_uncertainty = plot(variable_subject_avg, subject_colors)

    #getting overall and best 2 of 3 formula
    overall_avg_cond, best_2_avg_cond, overall_sem, best_2_sem = overall_thresholds(df)
    overall_avg_fixed = overall_avg_cond[overall_avg_cond['noise'] == 'fixed']
    overall_avg_variable = overall_avg_cond[overall_avg_cond['noise'] == 'variable']
    best_2_avg_fixed = best_2_avg_cond[best_2_avg_cond['noise'] == 'fixed']
    best_2_avg_variable = best_2_avg_cond[best_2_avg_cond['noise'] == 'variable']

    overall_sem_fixed = overall_sem[overall_sem['noise'] == 'fixed']
    overall_sem_variable = overall_sem[overall_sem['noise'] == 'variable']
    best_2_sem_fixed = best_2_sem[best_2_sem['noise'] == 'fixed']
    best_2_sem_variable = best_2_sem[best_2_sem['noise'] == 'variable']

    #getting acuity formula
    acuity_counts, lower_avg_threshold, higher_avg_threshold, lower_sem_threshold, higher_sem_threshold = acuity(metadata, df)
    #low acuity
    lower_avg_threshold_fixed = lower_avg_threshold[lower_avg_threshold['noise'] == 'fixed']
    lower_avg_threshold_variable = lower_avg_threshold[lower_avg_threshold['noise'] == 'variable']
    lower_sem_threshold_fixed = lower_sem_threshold[lower_sem_threshold['noise'] == 'fixed']
    lower_sem_threshold_variable = lower_sem_threshold[lower_sem_threshold['noise'] == 'variable']

    #high acuity
    higher_avg_threshold_fixed = higher_avg_threshold[higher_avg_threshold['noise'] == 'fixed']
    higher_avg_threshold_variable = higher_avg_threshold[higher_avg_threshold['noise'] == 'variable']
    higher_sem_threshold_fixed = higher_sem_threshold[higher_sem_threshold['noise'] == 'fixed']
    higher_sem_threshold_variable = higher_sem_threshold[higher_sem_threshold['noise'] == 'variable']

    #getting handedness formula
    left_avg_threshold, right_avg_threshold, handedness_counts, left_sem, right_sem = handedness(df, metadata)

    #left handed
    left_avg_threshold_fixed = left_avg_threshold[left_avg_threshold['noise'] == 'fixed']
    left_avg_threshold_variable = left_avg_threshold[left_avg_threshold['noise'] == 'variable']
    left_sem_fixed = left_sem[left_sem['noise'] == 'fixed']
    left_sem_variable = left_sem[left_sem['noise'] == 'variable']

    #right handed
    right_avg_threshold_fixed = right_avg_threshold[right_avg_threshold['noise'] == 'fixed']
    right_avg_threshold_variable = right_avg_threshold[right_avg_threshold['noise'] == 'variable']
    right_sem_fixed = right_sem[right_sem['noise'] == 'fixed']
    right_sem_variable = right_sem[right_sem['noise'] == 'variable']

    #getting gender formula
    female_avg_threshold, male_avg_threshold, gender_counts, female_sem, male_sem = gender(df, metadata)

    #female
    female_avg_threshold_fixed = female_avg_threshold[female_avg_threshold['noise'] == 'fixed']
    female_avg_threshold_variable = female_avg_threshold[female_avg_threshold['noise'] == 'variable']
    female_sem_fixed = female_sem[female_sem['noise'] == 'fixed']
    female_sem_variable = female_sem[female_sem['noise'] == 'variable']

    #male
    male_avg_threshold_fixed = male_avg_threshold[male_avg_threshold['noise'] == 'fixed']
    male_avg_threshold_variable = male_avg_threshold[male_avg_threshold['noise'] == 'variable']
    male_sem_fixed = male_sem[male_sem['noise'] == 'fixed']
    male_sem_variable = male_sem[male_sem['noise'] == 'variable']

    #generate tables for gender, handedness, age group, visual acuity
    gender_table = metadata["gender"].value_counts().reset_index()
    gender_table.columns = ["Gender", "Count"]
    gender_table.index += 1
    
    handedness_table = metadata["handedness"].value_counts().reset_index()
    handedness_table.columns = ["Handedness", "Count"]
    handedness_table.index += 1
    
    #extract only the base visual acuity values (removing +/- adjustments)
    metadata["visual_acuity_base"] = metadata["visual acuity"].str.extract(r"(20/\d+)").fillna("20/20")

    #sorting visual acuity
    acuity_order = ["20/12", "20/16", "20/20", "20/25", "20/30", "20/35", "20/40"]
    metadata["visual_acuity_base"] = pd.Categorical(metadata["visual_acuity_base"], categories=acuity_order, ordered=True)

    #count occurrences and sort
    acuity_table = metadata["visual_acuity_base"].value_counts().sort_index().reset_index()
    acuity_table.columns = ["Visual Acuity", "Count"]
    acuity_table.index += 1

    #categorize age into bins
    bins = [0, 20, 30, 40, 50, 60, 70, 100]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    metadata["age_group"] = pd.cut(metadata["age"], bins=bins, labels=labels, right=False)

    #ensure correct order by making age_group categorical with an explicit order
    metadata["age_group"] = pd.Categorical(metadata["age_group"], categories=labels, ordered=True)

    #count occurrences and sort by the categorical order
    age_table = metadata["age_group"].value_counts().sort_index().reset_index()
    age_table.columns = ["Age Group", "Count"]
    age_table.index += 1

    subjects = df["subject"].unique().tolist()

    return render_template("dashboard.html",
                            subjects=subjects,
                            plot_fixed_uncertainty=plot_fixed_uncertainty,
                            plot_variable_uncertainty=plot_variable_uncertainty,

                            dataframes=dict(
                                # "Top 5 Subjects with Lowest Average Thresholds"
                                lowest_thresholds=lowest_thresholds.to_dict(orient="records"),
                                
                                # "Tables"
                                gender_table=gender_table.to_dict(orient="records"),
                                
                                handedness_table=handedness_table.to_dict(orient="records"),
                                age_table=age_table.to_dict(orient="records"),
                                acuity_table=acuity_table.to_dict(orient="records"),

                                # "Counts" *for after exclusion
                                acuity_counts=acuity_counts.to_dict(orient="records"),
                                handedness_counts=handedness_counts.to_dict(orient="records"),
                                gender_counts=gender_counts.to_dict(orient="records"),

                                # "Overall Average Thresholds"
                                overall_avg = overall_avg_cond.to_dict(orient="records"),
                                overall_avg_fixed = overall_avg_fixed.to_dict(orient="records"),
                                overall_avg_variable = overall_avg_variable.to_dict(orient="records"),

                                # "Best 2 of 3 Average Thresholds"
                                best_2_avg = best_2_avg_cond.to_dict(orient="records"),
                                best_2_avg_fixed = best_2_avg_fixed.to_dict(orient="records"),
                                best_2_avg_variable = best_2_avg_variable.to_dict(orient="records"),

                                # "Overall Average SEM Thresholds"
                                overall_sem = overall_sem.to_dict(orient="records"),
                                overall_sem_fixed = overall_sem_fixed.to_dict(orient="records"),
                                overall_sem_variable = overall_sem_variable.to_dict(orient="records"),

                                # "Best 2 of 3 Average SEM Thresholds"
                                best_2_sem = best_2_sem.to_dict(orient="records"),
                                best_2_sem_fixed = best_2_sem_fixed.to_dict(orient="records"),
                                best_2_sem_variable = best_2_sem_variable.to_dict(orient="records"),

                                # "Thresholds for Lower(Better) Acuity Subjects"
                                lower_avg_threshold = lower_avg_threshold.to_dict(orient="records"),                                
                                lower_avg_threshold_fixed = lower_avg_threshold_fixed.to_dict(orient="records"),
                                lower_avg_threshold_variable = lower_avg_threshold_variable.to_dict(orient="records"),
                                
                                # "Thresholds for Higher(Worse) Acuity Subjects"
                                higher_avg_threshold = higher_avg_threshold.to_dict(orient="records"),                                
                                higher_avg_threshold_fixed = higher_avg_threshold_fixed.to_dict(orient="records"),
                                higher_avg_threshold_variable = higher_avg_threshold_variable.to_dict(orient="records"),

                                # "Thresholds for Lower(Better) SEM Acuity Subjects"
                                lower_sem_threshold = lower_sem_threshold.to_dict(orient="records"),                                
                                lower_sem_threshold_fixed = lower_sem_threshold_fixed.to_dict(orient="records"),
                                lower_sem_threshold_variable = lower_sem_threshold_variable.to_dict(orient="records"),
                                
                                # "Thresholds for Higher(Worse) SEM Acuity Subjects"
                                higher_sem_threshold = higher_sem_threshold.to_dict(orient="records"),                                
                                higher_sem_threshold_fixed = higher_sem_threshold_fixed.to_dict(orient="records"),
                                higher_sem_threshold_variable = higher_sem_threshold_variable.to_dict(orient="records"),

                                # "Average Thresholds for Left-Handed Subjects (2)"
                                left_avg_threshold = left_avg_threshold.to_dict(orient="records"),
                                left_avg_threshold_fixed = left_avg_threshold_fixed.to_dict(orient="records"),
                                left_avg_threshold_variable = left_avg_threshold_variable.to_dict(orient="records"),
                                
                                # "Average Thresholds for Right-Handed Subjects (Best 2)"
                                right_avg_threshold = right_avg_threshold.to_dict(orient="records"),
                                right_avg_threshold_fixed = right_avg_threshold_fixed.to_dict(orient="records"),
                                right_avg_threshold_variable = right_avg_threshold_variable.to_dict(orient="records"),

                                # "SEM Thresholds for Left-Handed Subjects (2)"
                                left_sem = left_sem.to_dict(orient="records"),
                                left_sem_fixed = left_sem_fixed.to_dict(orient="records"),
                                left_sem_variable = left_sem_variable.to_dict(orient="records"),
                                
                                # "SEM Thresholds for Right-Handed Subjects (Best 2)"
                                right_sem = right_sem.to_dict(orient="records"),
                                right_sem_fixed = right_sem_fixed.to_dict(orient="records"),
                                right_sem_variable = right_sem_variable.to_dict(orient="records"),

                                # "Average Thresholds for Female Subjects"
                                female_avg_threshold = female_avg_threshold.to_dict(orient="records"),
                                female_avg_threshold_fixed = female_avg_threshold_fixed.to_dict(orient="records"),
                                female_avg_threshold_variable = female_avg_threshold_variable.to_dict(orient="records"),
                                
                                # "Average Thresholds for Male Subjects"
                                male_avg_threshold = male_avg_threshold.to_dict(orient="records"),
                                male_avg_threshold_fixed = male_avg_threshold_fixed.to_dict(orient="records"),
                                male_avg_threshold_variable = male_avg_threshold_variable.to_dict(orient="records"),

                                # "SEM Thresholds for Female Subjects"
                                female_sem = female_sem.to_dict(orient="records"),
                                female_sem_fixed = female_sem_fixed.to_dict(orient="records"),
                                female_sem_variable = female_sem_variable.to_dict(orient="records"),
                                
                                # "SEM Thresholds for Male Subjects"
                                male_sem = male_sem.to_dict(orient="records"),
                                male_sem_fixed = male_sem_fixed.to_dict(orient="records"),
                                male_sem_variable = male_sem_variable.to_dict(orient="records"),

                            ),

                            tables=dict(
                                overall_avg=overall_avg_cond.set_index(pd.Index(range(1, len(overall_avg_cond) + 1))).to_html(classes='table'),
                                best_2_avg=best_2_avg_cond.to_html(classes='table'),
                                lowest_thresholds=lowest_thresholds.to_html(classes='table'),
                                lower_avg_threshold=lower_avg_threshold.set_index(pd.Index(range(1, len(lower_avg_threshold) + 1))).to_html(classes='table'),
                                higher_avg_threshold=higher_avg_threshold.set_index(pd.Index(range(1, len(higher_avg_threshold) + 1))).to_html(classes='table'),
                                left_avg_threshold=left_avg_threshold.set_index(pd.Index(range(1, len(left_avg_threshold) + 1))).to_html(classes='table'),
                                right_avg_threshold=right_avg_threshold.set_index(pd.Index(range(1, len(right_avg_threshold) + 1))).to_html(classes='table'),
                                female_avg_threshold=female_avg_threshold.set_index(pd.Index(range(1, len(female_avg_threshold) + 1))).to_html(classes='table'),
                                male_avg_threshold=male_avg_threshold.set_index(pd.Index(range(1, len(male_avg_threshold) + 1))).to_html(classes='table'),
                                gender_table=gender_table.to_html(classes='table'),
                                handedness_table=handedness_table.to_html(classes='table'),
                                age_table=age_table.to_html(classes='table'),
                                acuity_table=acuity_table.to_html(classes='table'),
                                acuity_counts=acuity_counts.to_html(classes='table'),
                                handedness_counts=handedness_counts.to_html(classes='table'),
                                gender_counts=gender_counts.to_html(classes='table'),
                            )
                        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

