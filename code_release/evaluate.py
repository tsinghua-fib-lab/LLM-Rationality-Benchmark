import openai
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import re
import time
import pickle
import copy
import json
import requests
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
filepath='./'
plotpath='./'
assert os.path.exists(filepath)

def evaluate_psychology():
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]

    df=pd.read_excel(os.path.join(filepath,'survey_analysis.xlsx'))
    # fill na
    df['Concepts'].fillna(method='ffill', inplace=True)
    df['Scale'].fillna(method='ffill', inplace=True)
    # df['Scale Elements'].fillna(method='ffill', inplace=True)
    df['Higher score & Ratonality'].fillna(method='ffill', inplace=True)

    df=df[df.apply(lambda x:x['Show']==1,axis=1)]

    ## Psychology+Theoretical

    selected_scales=['self reflection and insight scale (SRIS)',]
    selected_elements=['Self-reflection','Insight',]

    # df1=df[df.apply(lambda x:x['Total/Average/Level'] not in ['Total_score','Level','Correct_number'],axis=1)]
    df1=df[df.apply(lambda x: x['Scale'] in selected_scales,axis=1)]
    df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']

    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1 = dfplot.copy()
    dfplot

    plt.figure(figsize=(10, 1.5))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.rcParams['font.family'] = 'sans-serif'  # 选择一个好看的字体
    plt.rcParams['font.sans-serif'] = ['Arial']  # 例如 Arial


    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)

    plt.yticks(fontsize=16)
    # plt.title('Psychology-Theoretical')
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'psy_the_orig.pdf'), bbox_inches='tight')
    plt.show()

    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 1.5))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
    # plt.title('Psychology-Theoretical')
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'psy_the_norm.pdf'), bbox_inches='tight')
    plt.show()

    ## Psychology+Practical

    selected_scales=['Emotion Regulation Questionnaire (ERQ)',
                'Need for Cognition',]
    selected_elements=['Reappraisal Items','Suppression Items','Need for Cognition']

    df1=df[df.apply(lambda x:x['Show']==1,axis=1)]
    # df1=df[df.apply(lambda x:x['Total/Average/Level'] not in ['Total_score','Level','Correct_number'],axis=1)]
    df1=df1[df1.apply(lambda x: x['Scale'] in selected_scales,axis=1)]
    df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1 = dfplot.copy()
    dfplot

    plt.figure(figsize=(10, 2))
    heatmap=sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'psy_pra_orig.pdf'), bbox_inches='tight')
    plt.show()

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()
    # dfplot

    plt.figure(figsize=(10, 2))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'psy_pra_norm.pdf'), bbox_inches='tight')
    plt.show()


def evaluate_cognitive_science():
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]

    df=pd.read_excel(os.path.join(filepath,'survey_analysis.xlsx'))
    # fill na
    df['Concepts'].fillna(method='ffill', inplace=True)
    df['Scale'].fillna(method='ffill', inplace=True)
    # df['Scale Elements'].fillna(method='ffill', inplace=True)
    df['Higher score & Ratonality'].fillna(method='ffill', inplace=True)

    df=df[df.apply(lambda x:x['Show']==1,axis=1)]

    ## Cognitive and Behavioral Sciences+Theoretical

    selected_scales=['Rationality-Experimental Inventory (REI)',
                'Cognitive Reflection Test',
                'Letter Sets Test',
                    'Logiqa 2.0',
                    'Causal Reasoning',
                    'Wason Selection Task',
                    'Defeasible Reasoning',
                'Scientific Reasoning Scale',
                'Critical Thinking Disposition Scale',
                    'Actively Open-Minded Thinking Scale'
                    ]
    # selected_elements=['Rationality (overall)','Experimentality (overall)','Rational answer','Overall','overall',
    #             'odd','even']

    # df1=df[df.apply(lambda x:x['Total/Average/Level'] not in ['Total_score','Level','Correct_number'],axis=1)]
    df1=df[df.apply(lambda x:x['Show']==1,axis=1)]
    df1=df1[df1.apply(lambda x: x['Scale'] in selected_scales,axis=1)]
    # df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1 = dfplot.copy()
    dfplot

    plt.figure(figsize=(10, 6))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'cog_the_orig.pdf'), bbox_inches='tight')
    plt.show()

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 6))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'cog_the_norm.pdf'), bbox_inches='tight')
    plt.show()

    ## Cognitive and Behavioral Sciences+Practical

    selected_scales=['Belief Bias in Syllogistic Reasoning',
                    'Bias Blind Spot',
                    'Hindsight Bias',
                    'Illusion of Control',
                    
                    'Regret Aversion']
    selected_elements=['Outcome bias','Regret Aversion']

    df1=df[df.apply(lambda x:x['Show']==1,axis=1)]
    df1=df1[df1.apply(lambda x:x['Scale'] in selected_scales,axis=1)]
    df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1=dfplot.copy()
    dfplot

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'cog_pra_orig.pdf'), bbox_inches='tight')
    plt.show()

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'cog_pra_norm.pdf'), bbox_inches='tight')
    plt.show()


def evaluate_decision_making():
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]

    df=pd.read_excel(os.path.join(filepath,'survey_analysis.xlsx'))
    # fill na
    df['Concepts'].fillna(method='ffill', inplace=True)
    df['Scale'].fillna(method='ffill', inplace=True)
    # df['Scale Elements'].fillna(method='ffill', inplace=True)
    df['Higher score & Ratonality'].fillna(method='ffill', inplace=True)

    df=df[df.apply(lambda x:x['Show']==1,axis=1)]


    ## Decision Making & Practical

    selected_scales=['General Decision-Making Style',
                'Availability Heuristics',
                'Base-Rate Neglect (Statistical)',
                'Base-Rate Neglect (Causal)',
                    'Better-Than-Average Effect',
                    'Confirmation Bias',
                'Conjunction Fallacy',
                    'Covariation Detection',
                    'Denominator Neglect (or Ratio Bias)',
                'Framing Effect (Risk and Attribute)',
                'Probabilistic Matching',
                    'Outcome bias',]

    selected_elements=['Rational','Avoidant','Dependent','Intuitive','Spontaneous',
                'Availability Heuristics',
                'Base-Rate Neglect (Statistical)',
                    'Base-Rate Neglect (Causal)',
                    'Conjunction Fallacy','Overall',
                    'Probabilistic Matching',
                    ]

    df1=df[df.apply(lambda x:x['Show']==1,axis=1)]
    df1=df1[df1.apply(lambda x:x['Scale'] in selected_scales,axis=1)]
    df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1 = dfplot.copy()

    dfplot

    plt.figure(figsize=(10, 8))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'dec_orig.pdf'), bbox_inches='tight')
    plt.show()

    # plot difference with human
    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'dec_norm.pdf'), bbox_inches='tight')
    plt.show()








def evaluate_economics():
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]

    df=pd.read_excel(os.path.join(filepath,'survey_analysis.xlsx'))
    # fill na
    df['Concepts'].fillna(method='ffill', inplace=True)
    df['Scale'].fillna(method='ffill', inplace=True)
    # df['Scale Elements'].fillna(method='ffill', inplace=True)
    df['Higher score & Ratonality'].fillna(method='ffill', inplace=True)

    df=df[df.apply(lambda x:x['Show']==1,axis=1)]

    ## Economics+Overall

    selected_scales=[
                'Overconfidence',
                ]

    tmp=[
                'Risk-Taking',
                'Risk-Perception',
                'Risk Propensity Scale',
        'Temporal Discounting',
        'Endowment Effect',
        "Gambler's Fallacy",
        'Loss Aversion',
        'Mental Accounting',
        'Regression to the Mean',
                'Sunk Cost Fallacy',
                ]
    selected_scales.extend(tmp)


    # df1=df[df.apply(lambda x:x['Total/Average/Level'] not in ['Total_score','Level','Correct_number'],axis=1)]
    df1=df[df.apply(lambda x:x['Show']==1,axis=1)]
    df1=df1[df1.apply(lambda x: x['Scale'] in selected_scales,axis=1)]
    # df1

    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]
    dfplot_1 = dfplot.copy()

    dfplot

    plt.figure(figsize=(10, 10))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
        
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'econ_orig.pdf'), bbox_inches='tight')
    plt.show()

    # plot difference with human
    # df1['name']=df1['Scale']+'\n'+df1['Scale Elements']
    reverse_names=df1[df1['Higher score & Ratonality']=='Oppose Rationality'].loc[:,'name']

    dfplot=df1.drop(columns=['Concepts','Scale','Scale Elements','Higher score & Ratonality','Total/Average/Level'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        return (row-m) / (M-m)

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    average_row = dfplot.mean().round(3)
    dfplot.loc['Overall',:] = average_row

    dfplot=dfplot[col_order]

    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)
    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 10))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'econ_norm.pdf'), bbox_inches='tight')
    plt.show()








def evaluate_game_theory():
    
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    # 'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]
    df=pd.read_excel(os.path.join(filepath,'game_results.xlsx'))
    df['Game'].fillna(method='ffill', inplace=True)
    # individual

    selected_games=['SecPriAuc','BeaCon','OnePD','FinRepPD','OnePG','FinRepPG']
    # selected_vers=['fin']

    df1=df[df.apply(lambda x:x['Game'] in selected_games and x['Show']==1,axis=1)]
    df1

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score']) ###############
    # dfplot = dfplot.apply(normalize_row, axis=1)

    # reverse
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row

    dfplot_1 = dfplot.copy()

    dfplot

    plt.figure(figsize=(10, 4))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'game_orig.pdf'), bbox_inches='tight')
    plt.show()

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score']) ###############
    # dfplot = dfplot.apply(normalize_row, axis=1)

    # reverse
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row
    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)

    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 4))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5)

    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'game_norm.pdf'), bbox_inches='tight')
    plt.show()





def evaluate_collective_rationality():
    col_order=['Human',
    'gpt-4o',
    'gpt-4',
    'gpt-3.5',
    'deepseek-v2.5',
    # 'bard',
    'text-bison-001',
    'text-davinci-003',
    'text-davinci-002',
    'claude-instant',
    'qwen-72b', 
    'qwen-32b', 
    'openchat-13b',
    'wizardlm-13b',
    'vicuna-13b',
    'llama2-13b',
    'oasst-12b',
    'qwen-7b',
    'vicuna-7b',
    'llama2-7b',
    'chatglm2-6b',
    ]
    df=pd.read_excel(os.path.join(filepath,'game_results.xlsx'))
    df['Game'].fillna(method='ffill', inplace=True)

    # social game

    selected_games=['InfRepPD','FinRepSH','FinRepBoS','FinRepME']
    selected_vers=['fin']

    # df1=df[df.apply(lambda x:x['Game'] in selected_games and x['ver'] in selected_vers,axis=1)]
    df1=df[df.apply(lambda x:x['Game'] in selected_games and x['Show']==1,axis=1)]
    # df1=df1.loc[selected_games]
    df1

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
    #     M=row.max()
    #     m,M=row.min(),row.max()
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row

    dfplot_1 = dfplot.copy()

    dfplot

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'coop_orig.pdf'), bbox_inches='tight')
    plt.show()

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
    #     M=row.max()
    #     m,M=row.min(),row.max()
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row
    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)

    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 3))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'coop_norm.pdf'), bbox_inches='tight')
    plt.show()

    # Wisdom of crowds

    selected_games=['MMLU','MATH']
    selected_vers=['fin']

    # df1=df[df.apply(lambda x:x['Game'] in selected_games and x['ver'] in selected_vers,axis=1)]
    df1=df[df.apply(lambda x:x['Game'] in selected_games and x['Show']==1,axis=1)]
    # df1=df1.loc[selected_games]
    df1

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
    #     M=row.max()
    #     m,M=row.min(),row.max()
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row

    dfplot_1 = dfplot.copy()

    dfplot

    plt.figure(figsize=(10, 2))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=0, vmax=1, annot_kws={"size": 10},cbar_kws={"pad": 0.02})

    # # 添加红色边框在第一列
    # for _, j in enumerate(range(len(dfplot.columns))):
    #     if j == 0:  # 检查是否是1st列
    #         plt.gca().add_patch(plt.Rectangle((j, 0), 1, len(dfplot.index), fill=False, edgecolor='red', lw=3))
    # # 将第一列的x轴刻度标签设置为红色
    # xticks_labels = plt.gca().get_xticklabels()
    # if xticks_labels:
    #     xticks_labels[0].set_color('red')
        
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(rotation=0, fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'wisd_orig.pdf'), bbox_inches='tight')
    plt.show()

    df1['name']=df1['Game']
    reverse_names=df1[df1['Reverse']==1].loc[:,'name']

    dfplot=df1.drop(columns=['Game','ver','Metric','Reverse'])

    dfplot.set_index('name', inplace=True)
    dfplot=dfplot.astype(float)

    def normalize_row(row):
        m= row.loc['Min_score']
        M = row.loc['Max_score']
    #     M=row.max()
    #     m,M=row.min(),row.max()
        norm=(row-m) / (M-m)
        return norm

    dfplot = dfplot.apply(normalize_row, axis=1)
    dfplot=dfplot.drop(columns=['Max_score','Min_score'])
    dfplot.loc[reverse_names]=1-dfplot.loc[reverse_names]

    dfplot=dfplot[col_order]

    average_row = dfplot.mean().round(2)
    dfplot.loc['Overall',:] = average_row
    dfplot = dfplot.apply(lambda x:x-x['Human'], axis=1).round(2)

    dfplot_2 = dfplot.copy()

    plt.figure(figsize=(10, 2))
    sns.heatmap(dfplot, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-0.5, vmax=0.5, annot_kws={"size": 10},cbar_kws={"pad": 0.02})
        
    plt.xticks(rotation=60, ha='right', fontsize=16)
    plt.tick_params(axis='x', pad=0)  # 调整tick的距离
    plt.yticks(rotation=0, fontsize=16)

    ax = plt.gca()
    import matplotlib.transforms as mtrans
    trans = mtrans.Affine2D().translate(10, 0)
    for t in ax.get_xticklabels():
        t.set_transform(t.get_transform()+trans)
        
    plt.ylabel('')
    plt.xlabel('')
    plt.savefig(os.path.join(plotpath, 'wisd_norm.pdf'), bbox_inches='tight')
    plt.show()