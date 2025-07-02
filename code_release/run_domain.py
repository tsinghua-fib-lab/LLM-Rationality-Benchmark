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
from LLM_setup import *

filepath=r'./records/'
assert os.path.exists(filepath)

def run_psychology(model):
    # self reflection scale
    prompt="""
    Please act like a participant in this survey. For the following statements, please respond to each statement by selecting a number from 1 to 6. This number should best represent your opinion on a 6-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = somewhat disagree, 4 = somewhat agree, 5 = agree, 6 = strongly agree).
    1. I don’t often think about my thoughts
    2. I rarely spend time in self-reflection
    3. I frequently examine my feelings
    4. I don’t really think about why I behave in the way that I do
    5. I frequently take time to reflect on my thoughts
    6. I often think about the way I feel about things
    7. I am not really interested in analyzing my behavior
    8. It is important to me to evaluate the things that I do
    9. I am very interested in examining what I think about
    10. It is important to me to try to understand what my feelings mean
    11. I have a definite need to understand the way that my mind works
    12. It is important to me to be able to understand how my thoughts arise 
    Please tell me your choice for statement 1-12, respectively. 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Insight Scale
    prompt="""
    Please act like a participant in this survey. For the following statements, please respond to each statement by selecting a number from 1 to 6. This number should best represent your opinion on a 6-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = somewhat disagree, 4 = somewhat agree, 5 = agree, 6 = strongly agree).
    1. I am usually aware of my thoughts
    2. I’m often confused about the way that I really feel about things
    3. I usually have a very clear idea about why I’ve behaved in a certain way
    4. I’m often aware that I’m having a feeling, but I often don’t quite know what it is
    5. My behavior often puzzles me
    6. Thinking about my thoughts makes me more confused
    7. Often I find it difficult to make sense of the way I feel about things
    8. I usually know why I feel the way I do 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Risk-Taking 1-10
    prompt="""
    Please act like a participant in this survey. For each of the following statements, please indicate the likelihood that you would engage in the described activity or behavior if you were to find yourself in that situation. Select a number from 1 to 7 that best represent your opinion on a 7-point Likert scale (1 = Extremely Unlikely, 2 = Moderately Unlikely, 3 = Somewhat Unlikely, 4 = Not Sure, 5 = Somewhat Likely, 6 = Moderately Likely, 7 = Extremely Likely).
    1. Admitting that your tastes are different from those of a friend.
    2. Going camping in the wilderness.
    3. Betting a day’s income at the horse races.
    4. Investing 10% of your annual income in a moderate growth mutual fund.
    5. Drinking heavily at a social function.
    6. Taking some questionable deductions on your income tax return.
    7. Disagreeing with an authority figure on a major issue.
    8. Betting a day’s income at a high-stake poker game.
    9. Having an affair with a married man/woman.
    10. Passing off somebody else’s work as your own.
    11. Going down a ski run that is beyond your ability.
    12. Investing 5% of your annual income in a very speculative stock.
    13. Going whitewater rafting at high water in the spring.
    14. Betting a day’s income on the outcome of a sporting event.
    15. Engaging in unprotected sex.
    16. Revealing a friend’s secret to someone else.
    17. Driving a car without wearing a seat belt.
    18. Investing 10% of your annual income in a new business venture.
    19. Taking a skydiving class.
    20. Riding a motorcycle without a helmet.
    21. Choosing a career that you truly enjoy over a more secure one.
    22. Speaking your mind about an unpopular issue in a meeting at work.
    23. Sunbathing without sunscreen.
    24. Bungee jumping off a tall bridge. 
    25. Piloting a small plane.
    26. Walking home alone at night in an unsafe area of town.
    27. Moving to a city far away from your extended family.
    28. Starting a new career in your mid-thirties.
    29. Leaving your young children alone at home while running an errand.
    30. Not returning a wallet you found that contains $200. 
    Please tell me your choice for statement 1-30.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Risk-Perception 1-10
    prompt="""
    Please act like a participant in this survey. People often see some risk in situations that contain uncertainty about what the outcome or consequences will be and for which there is the possibility of negative consequences. However, riskiness is a very personal and intuitive notion, and we are interested in your gut level assessment of how risky each situation or behavior is. For each of the following statements, please indicate how risky you perceive each situation on a 7-point Likert scale. Select a number from 1 to 7, where 1 being 'Not at all Risky', 2 being 'Slightly Risky', 3 being 'Somewhat Risky', 4 being 'Moderately Risky', 5 being 'Risky', 6 being 'Very Risky', and 7 being 'Extremely Risky'.
    1. Admitting that your tastes are different from those of a friend.
    2. Going camping in the wilderness.
    3. Betting a day’s income at the horse races.
    4. Investing 10% of your annual income in a moderate growth mutual fund.
    5. Drinking heavily at a social function.
    6. Taking some questionable deductions on your income tax return.
    7. Disagreeing with an authority figure on a major issue.
    8. Betting a day’s income at a high-stake poker game.
    9. Having an affair with a married man/woman.
    10. Passing off somebody else’s work as your own.
    11. Going down a ski run that is beyond your ability.
    12. Investing 5% of your annual income in a very speculative stock.
    13. Going whitewater rafting at high water in the spring.
    14. Betting a day’s income on the outcome of a sporting event.
    15. Engaging in unprotected sex.
    16. Revealing a friend’s secret to someone else.
    17. Driving a car without wearing a seat belt.
    18. Investing 10% of your annual income in a new business venture.
    19. Taking a skydiving class.
    20. Riding a motorcycle without a helmet.
    21. Choosing a career that you truly enjoy over a more secure one.
    22. Speaking your mind about an unpopular issue in a meeting at work.
    23. Sunbathing without sunscreen.
    24. Bungee jumping off a tall bridge. 
    25. Piloting a small plane.
    26. Walking home alone at night in an unsafe area of town.
    27. Moving to a city far away from your extended family.
    28. Starting a new career in your mid-thirties.
    29. Leaving your young children alone at home while running an errand.
    30. Not returning a wallet you found that contains $200. 
    Please tell me your choice for statement 1-30, respectively.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Risk Propensity Scale
    prompt="""
    Please act like a participant in this survey. For each of the following statements, please select a number from 1 to 7 that best reflects your opinion using a 7-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = somewhat disagree, 4 = neutral, 5 = somewhat agree, 6 = agree, 7 = strongly agree).
    1. I always try to avoid situations involving a risk of getting into trouble.
    2. I always play it safe even when it means occasionally losing out on a good opportunity.
    3. I am a cautious person who generally avoids risks.
    4. I am rather bold and fearless in my actions
    5. I am generally cautious when trying something new.
    Please tell me your choice for statement 1-5, respectively.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Emotion Regulation Questionnaire (ERQ)
    prompt="""
    Please act like a participant in this survey. We would like to ask you some questions about your emotional life, in particular, how you control (that is, regulate and manage) your emotions. The questions below involve two distinct aspects of your emotional life. One is your emotional experience, or what you feel like inside. The other is your emotional expression, or how you show your emotions in the way you talk, gesture, or behave. Although some of the following questions may seem similar to one another, they differ in important ways. For each statement below, please select a number from 1 to 7 that best reflects your opinion on a 7-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = somewhat disagree, 4 = neutral, 5 = somewhat agree, 6 = agree, 7 = strongly agree).
    1. When I want to feel more positive emotion (such as joy or amusement), I change what I’m thinking about. 
    2. I keep my emotions to myself. 
    3. When I want to feel less negative emotion (such as sadness or anger), I change what I’m thinking about. 
    4. When I am feeling positive emotions, I am careful not to express them.
    5. When I’m faced with a stressful situation, I make myself think about it in a way that helps me stay calm. 
    6. I control my emotions by not expressing them.
    7. When I want to feel more positive emotion, I change the way I’m thinking about the situation. 
    8. I control my emotions by changing the way I think about the situation I’m in.
    9. When I am feeling negative emotions, I make sure not to express them.
    10. When I want to feel less negative emotion, I change the way I’m thinking about the situation. 
    Please tell me your choice for statement 1-10, respectively.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)


    # Need for Cognition 
    prompt="""
    Please act like a participant in this survey. For each statement below, please select a number from 1 to 5 that best reflects your opinion on a 5-point Likert scale (1 = extremely uncharacteristic of me, 2 = uncharacteristic of me, 3 = neutral, 4 = characteristic of me, 5 = extremely characteristic of me).
    1. I would prefer complex to simple problems.
    2. I like to have the responsibility of handling a situation that requires a lot of thinking.
    3. Thinking is not my idea of fun.
    4. I would rather do something that requires little thought than something that is sure to challenge my thinking abilities.
    5. I try to anticipate and avoid situations where there is likely chance I will have to think in depth about something.
    6. I find satisfaction in deliberating hard and for long hours.
    7. I only think as hard as I have to. 
    8. I prefer to think about small, daily projects to long-term ones.
    9. I like tasks that require little thought once I've learned them.
    10. The idea of relying on thought to make my way to the top appeals to me.
    11. I really enjoy a task that involves coming up with new solutions to problems.
    12. Learning new ways to think doesn't excite me very much.
    13. I prefer my life to be filled with puzzles that I must solve.
    14. The notion of thinking abstractly is appealing to me.
    15. I would prefer a task that is intellectual, difficult, and important to one that is somewhat important but does not require much thought.
    16. I feel relief rather than satisfaction after completing a task that required a lot of mental effort.
    17. It's enough for me that something gets the job done; I don't care how or why it works.
    18. I usually end up deliberating about issues even when they do not affect me personally.
    Please tell me your choice for statement 1-18.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)



def run_cognitive_science(model):
    # Rationality inventory (REI-R)
    prompt="""
    Please act like a participant in this survey. For each statement below, please select a number from 1 to 5 that best reflects your opinion on a 5-point Likert scale (1 = completely false, 2 = false, 3 = neither true nor false, 4 = true, 5 = completely true).
    1. I try to avoid situations that require thinking in depth about something.
    2. I'm not that good at figuring out complicated problems.
    3. I enjoy intellectual challenges.
    4. I am not very good at solving problems that require careful logical analysis.
    5. I don't like to have to do a lot of thinking.
    6. I enjoy solving problems that require hard thinking.
    7. Thinking is not my idea of an enjoyable activity.
    8. I am not a very analytical thinker.
    9. Reasoning things out carefully is not one of my strong points.
    10. I prefer complex problems to simple problems.
    11. Thinking hard and for a long time about something gives me little satisfaction.
    12. I don't reason well under pressure.
    13. I am much better at figuring things out logically than most people.
    14. I have a logical mind.
    15. I enjoy thinking in abstract terms.
    16. I have no problem thinking things through carefully.
    17. Using logic usually works well for me in figuring out problems in my life.
    18. Knowing the answer without having to understand the reasoning behind it is good enough for me.
    19. I usually have clear, explainable reasons for my decisions.
    20. Learning new ways to think would be very appealing to me.
    Please tell me your choice for statement 1-20.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Rationality inventory Part2
    prompt="""
    Please act like a participant in this survey. For each statement below, please select a number from 1 to 5 that best reflects your opinion on a 5-point Likert scale (1 = completely false, 2 = false, 3 = neither true nor false, 4 = true, 5 = completely true).
    21. I like to rely on my intuitive impressions.
    22. I don't have a very good sense of intuition.
    23. Using my gut feelings usually works well for me in figuring out problems in my life.
    24. I believe in trusting my hunches.
    25. Intuition can be a very useful way to solve problems.
    26. I often go by my instincts when deciding on a course of action.
    27. I trust my initial feelings about people.
    28. When it comes to trusting people, I can usually rely on my gut feelings.
    29. If I were to rely on my gut feelings, I would often make mistakes.
    30. I don't like situations in which I have to rely on intuition.
    31. I think there are times when one should rely on one's intuition.
    32. I think it is foolish to make important decisions based on feelings.
    33. I don't think it is a good idea to rely on one's intuition for important decisions.
    34. I generally don't depend on my feelings to help me make decisions.
    35. I hardly ever go wrong when I listen to my deepest gut feelings to find an answer.
    36. I would not want to depend on anyone who described himself or herself as intuitive.
    37. My snap judgments are probably not as good as most people’s.
    38. I tend to use my heart as a guide for my actions.
    39. I can usually feel when a person is right or wrong, even if I can't explain how I know.
    40. I suspect my hunches are inaccurate as often as they are accurate.
    Please tell me your choice for statement 21-40.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Cognitive Reflection Test 
    prompt="""
    Please act like a participant in this survey. Please respond to the following questions by filling in the blank with your answer. Please provide only the blank answer, and do not include any additional text or explanations in your responses.
    (1)  A bat and a ball cost $1.10 in total. The bat costs a dollar more than the ball. How much does the ball cost? ____ cents
    (2)  If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? ____ minutes 
    (3)  In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake? ____ days 
    (4)  If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 days, how long would it take them to drink one barrel of water together? _____ days 
    (5)  Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are in the class? ______ students
    (6)  A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it finally for $90. How much has he made? _____ dollars
    (7)  Simon decided to invest $8,000 in the stock market one day early in 2008. Six months after he invested, on July 17, the stocks he had purchased were down 50%. 
    Fortunately for Simon, from July 17 to October 17, the stocks he had purchased went up 75%.  At this point, Simon has:  
    a. broken even in the stock market
    b. is ahead of where he began
    c. has lost money
    (choose one of a,b,c)
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    ####
    questions="""
    1. Please select the set of letters that is different: QPPQ, HGHH, TTTU, DDDE, MLMM.
    a. QPPQ
    b. HGHH 
    c. TTTU 
    d. DDDE 
    e. MLMM

    2. Please select the set of letters that is different: BCDE, FGHI, JKLM, PRST, VWXY.
    a. BCDE
    b. FGHI
    c. JKLM
    d. PRST
    e. VWXY

    3. Please select the set of letters that is different: BVZC, FVZG, JVZK, PWXQ, SVZT.
    a. BVZC
    b. FVZG
    c. JVZK
    d. PWXQ
    e. SVZT

    4. Please select the set of letters that is different: BCEF, FGIJ, STWX, CDFG, PQST.
    a. BCEF
    b. FGIJ
    c. STWX
    d. CDFG
    e. PQST

    5. Please select the set of letters that is different: BCCB, GFFG, LMML, QRRQ, WXXW.
    a. BCCB
    b. GFFG
    c. LMML
    d. QRRQ
    e. WXXW

    6. Please select the set of letters that is different: AAPP, CCRR, QQBB, EETT, DDSS.
    a. AAPP
    b. CCRR
    c. QQBB
    d. EETT
    e. DDSS

    7. Please select the set of letters that is different: ABDC, EGFH, IJLK, OPRQ, UVXW
    a. ABDC
    b. EGFH
    c. IJLK
    d. OPRQ
    e. UVXW

    8. Please select the set of letters that is different: CERT, KMTV, FHXZ, BODQ, HJPR.
    a. CERT
    b. KMTV
    c. FHXZ
    d. BODQ
    e. HJPR

    9. Please select the set of letters that is different: PABQ, SEFT, VIJW, COPD, FUZG.
    a. PABQ
    b. SEFT
    c. VIJW
    d. COPD
    e. FUZG

    10. Please select the set of letters that is different: CFCR, JCVC, CGCS, CLXC, KCWC.
    a. CFCR
    b. JCVC
    c. CGCS
    d. CLXC
    e. KCWC

    11. Please select the set of letters that is different: XDBK,TNLL, VEGV, PFCC, ZAGZ.
    a. XDBK
    b. TNLL
    c. VEGV
    d. PFCC
    e. ZAGZ

    12. Please select the set of letters that is different: CAEZ, CEIZ, CIOZ, CGVZ, CAUZ.
    a. CAEZ
    b. CEIZ
    c. CIOZ
    d. CGVZ
    e. CAUZ

    13. Please select the set of letters that is different: VEBT, XGDV, ZIFX, KXVH, MZXJ.
    a. VEBT
    b. XGDV
    c. ZIFX
    d. KXVH
    e. MZXJ

    14. Please select the set of letters that is different: AFBG, EJFK, GKHM, PSQT, RWSX.
    a. AFBG
    b. EJFK
    c. GKHM
    d. PSQT
    e. RWSX

    15. Please select the set of letters that is different: KGDB, DFIM, KIFB, HJMQ, LHEC.
    a. KGDB
    b. DFIM
    c. KIFB
    d. HJMQ
    e. LHEC

    16. Please select the set of letters that is different: ABCX, EFGX, IJKX, OPQX, UVWZ.
    a. ABCX
    b. EFGX
    c. IJKX
    d. OPQX
    e. UVWZ

    17. Please select the set of letters that is different: LNLV, DTFL, CLNL, HRLL, LLWS.
    a. LNLV
    b. DTFL
    c. CLNL
    d. HRLL
    e. LLWS

    18. Please select the set of letters that is different: ABCE, EFGI, IJKM, OPQT, UVWY.
    a. ABCE
    b. EFGI
    c. IJKM
    d. OPQT
    e. UVWY

    19. Please select the set of letters that is different: GFFG, DCCD, STTS, RQQR, MLLM.
    a. GFFG
    b. DCCD
    c. STTS
    d. RQQR
    e. MLLM

    20. Please select the set of letters that is different: DCDD, HGHH, MMLM, QQQR, WWVW.
    a. DCDD
    b. HGHH
    c. MMLM
    d. QQQR
    e. WWVW

    21. Please select the set of letters that is different: FEDC, MKJI, DCBA, HGFE, JIHG.
    a. FEDC
    b. MKJI
    c. DCBA
    d. HGFE
    e. JIHG

    22. Please select the set of letters that is different: BDBB, BFDB, BHBB, BBJB, BBLB.
    a. BDBB
    b. BFDB
    c. BHBB
    d. BBJB
    e. BBLB

    23. Please select the set of letters that is different: BDCE, FHGI, JLKM, PRQS, TVWU.
    a. BDCE
    b. FHGI
    c. JLKM
    d. PRQS
    e. TVWU

    24. Please select the set of letters that is different: BDEF, FHIJ, HJKL, NPQR, SVWX.
    a. BDEF
    b. FHIJ
    c. HJKL
    d. NPQR
    e. SVWX

    25. Please select the set of letters that is different: NABQ, PEFS, RIJV, GOPK, CUWG.
    a. NABQ
    b. PEFS
    c. RIJV
    d. GOPK
    e. CUWG

    26. Please select the set of letters that is different: DEGF, KLHJ, NOQP, PQSR, TURS.
    a. DEGF
    b. KLHJ
    c. NOQP
    d. PQSR
    e. TURS

    27. Please select the set of letters that is different: AOUI, CTZR, JHTN, PBRL, RTVH.
    a. AOUI
    b. CTZR
    c. JHTN
    d. PBRL
    e. RTVH

    28. Please select the set of letters that is different: BEPW, HJTX, KNRZ, KOSV, WRPM.
    a. BEPW
    b. HJTX
    c. KNRZ
    d. KOSV
    e. WRPM

    29. Please select the set of letters that is different: RRBR, QQAR, FTEF, JXIJ, SSCS.
    a. RRBR
    b. QQAR
    c. FTEF
    d. JXIJ
    e. SSCS

    30. Please select the set of letters that is different: QIFB, CGIJ, BCOR, ZRED, JIFC.
    a. QIFB
    b. CGIJ
    c. BCOR
    d. ZRED
    e. JIFC
    """

    # Letter Sets Test
    prompt0="""
    Please act like a participant in this survey. Each problem in this survey has five sets of letters with four letters in each set. Four of the sets of letters are alike in some way. You are to find the rule that makes these four sets alike. The fifth letter set is different from them and will not fit this rule. Please select the set of letters that is different.

    Note: The rules will not be based on the sounds of sets of letters, the shapes of letters, or whether letter combinations form words or parts of words.

    Examples: 
    Example A. Please select the set of letters that is different: NOPQ, DEFL, ABCD, HIJK, UVWX.
    a. NOPQ
    b. DEFL
    c. ABCD
    d. HIJK
    e. UVWX
    In Example A, four of the sets have letters in alphabetical order. Therefore, the answer for this question is b.

    Example B. Please select the set of letters that is different: NLIK, PLIK, QLIK, THIK, VLIK.
    a. NLIK
    b. PLIK
    c. QLIK
    d. THIK
    e. VLIK
    In Example B, four of the sets contain the letter L. Therefore, the answer for this question is d.

    <question>
    Tell me your answer of this problem.
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # V2 Defeasible Reasoning
    questions="""
    1. Hittas are usually not waffs. All of the hittas are oxers. Oxers are usually waffs. Jukk is a hitta. Is Jukk a waff?
    a. Neither Jukk being a waff nor he not being a waff is a more reasonable answer than the other. 
    b. It is more likely that Jukk is a waff than that he is not a waff.
    c. It is more likely that Jukk is not a waff than that he is a waff.

    2. Wiflons are usually not kiglers. Wiflons are usually brindops. All of the brindops are kiglers. Floxxi is a wiflon. Is Floxxi a kigler?
    a. It is more likely that Floxxi is a kigler than that he is not a kigler.
    b. Neither Floxxi being a kigler nor he not being a kigler is a more reasonable answer than the other. 
    c. It is more likely that Floxxi is not a kigler than that he is a kigler.

    3. Zugs are usually not vlogs. Zugs are usually storps. Storps are usually vlogs.Duss is a zug.Is Duss a vlog?
    a. It is more likely that Duss is a vlog than that he is not a vlog.
    b. It is more likely that Duss is not a vlog than that he is a vlog.
    c. Neither Duss being a vlog nor he not being a vlog is a more reasonable answer than the other. 

    4. Humnols are usually not crerks. All of the posders are twerbers. Twerbers are usually crerks. Vouncy is a humnol.Vouncy is a posder. Is Vouncy a crerk?
    a. Neither Vouncy being a crerk nor he not being a crerk is a more reasonable answer than the other. 
    b. It is more likely that Vouncy is a crerk than that he is not a crerk.
    c. It is more likely that Vouncy is not a crerk than that he is a crerk.

    5. Arkons are usually not gakks. Jaggas are usually wollers. All of the wollers are gakks. Fertha is an arkon.Fertha is a jagga. Is Fertha a gakk?
    a. It is more likely that Fertha is a gakk than that he is not a gakk.
    b. It is more likely that Fertha is not a gakk than that he is a gakk.
    c. Neither Fertha being a gakk nor he not being a gakk is a more reasonable answer than the other. 

    6. Voltners are usually not zillos. Kikkas are usually crolders. Crolders are usually zillos. Grolli is a voltner. Grolli is a kikka. Is Grolli a kikka?
    a. It is more likely that Grolli is a zillo than that he is not a zillo.
    b. Neither Grolli being a zillo nor he not being a zillo is a more reasonable answer than the other. 
    c. It is more likely that Grolli is not a zillo than that he is a zillo.
    """
    prompt0="""
    Please act like a participant in this survey. Consider the following scenarios set on an imaginary planet inhabited by species with strange-looking names. Given this information, please select the best option for the following questions:
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # Scientific Reasoning Scale
    questions="""
    1. In a taste test, a researcher puts Brand A coffee in a cup with white tape on it and Brand B coffee in an identical cup with black tape on it. A lab assistant gives tasters one of the cups, while the researcher watches their facial expressions. Based on this information, please assess whether this statement is true or false: The lab assistant should not watch the cups being filled.
    2. A researcher finds that American states with larger parks have fewer endangered species. Based on this information, please assess whether this statement is true or false: These data show that increasing the size of American state parks will reduce the number of endangered species.
    3. A researcher has subjects put together a jigsaw puzzle either in a cold room with a loud radio or in a warm room with no radio. Subjects solve the puzzle more quickly in the warm room with no radio. Based on this information, please assess whether this statement is true or false: The scientist cannot tell if the radio caused subjects to solve the puzzle more slowly.
    4. An education researcher wants to measure the general math ability of a sample of high-performing math students. All the students have taken classes in geometry and pre-calculus. Based on this information, please assess whether this statement is true or false: The education researcher can measure general math ability by giving the students a geometry test.
    5. Two scientists test an anti-acne cream on teenagers with acne. Scientist A wants to give the cream to all the teenagers in the study. Scientist B wants to give the cream to half the teenagers and give a cream without anti-acne ingredients to the other half. Based on this information, please assess whether this statement is true or false: Both ways of testing the cream are equally good.
    6. A researcher has a group of subjects play a competitive game. Each subject’s goal is to make money by buying and selling tokens. Subjects are paid a flat fee for participating in the experiment. Based on this information, please assess whether this statement is true or false: The researcher can confidently state that the behavior in the experiment reflects real-life buying and selling behavior.
    7. A randomly selected sample of Americans is surveyed about disease A before and after a 6-month media campaign about the disease. Mid-way through the media campaign, a famous celebrity dies of Disease A. The survey data indicate that knowledge of Disease A is higher after the campaign. Based on this information, please assess whether this statement is true or false: The media campaign may not have increased knowledge of Disease A.
    8. Subjects in an experiment must press a button whenever a blue dot flashes on their computer screen. At first, the task is easy for subjects. But as they continue to perform the task, they make more and more errors. Based on this information, please assess whether this statement is true or false: The blue dot must flash more quickly as the task progresses.
    9. Researchers want to see whether a health intervention helps school children to lose weight. School children are sorted into either an intervention or control group. Based on this information, please assess whether this statement is true or false: The researchers should assign the overweight children to the intervention group.
    10. A researcher develops a new method for measuring the surface tension of liquids. This method is more consistent than the old method. Based on this information, please assess whether this statement is true or false: The new method must also be more accurate than the old method.
    11. Two researchers are developing a survey to measure consumers’ feelings about customer service. Researcher A wants customers to rate their agreement with the statement “I am satisfied with customer service” on a 5-point scale. Researcher B wants customers to rate customer service on a 5-point scale, where 1 = not dissatisfied at all and 5 = highly dissatisfied.  Based on this information, please assess whether this statement is true or false: These questions are equally good for measuring how consumers feel about customer service.
    """.strip()
    len(questions.split('\n'))


    prompt0="""
    Please act like a participant in this survey. Please evaluate the statements provided and respond with either "True" or "False".
    <question>
    Tell me your answer (True or False) of this problem.
    """
    for q in questions.split('\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break


    # V2 Wason Selection Task
    questions=f"""
    1. Please act like a participant in this survey. You are presented with four cards, each labeled with A, D, 3, and 7 on one side of the card, respectively. These cards have information on both sides. On one side of a card is a letter, and on the other side is a number. 
    Here is a rule: If there is an A on one side of the card, then there is a 3 on the other side of the card. 
    Select the cards that you need to turn over to determine whether or not the cards are violating the rule:
    a. the card labeled with "A"
    b. the card labeled with "D"
    c. the card labeled with "3"
    d. the card labeled with "7"

    2. Please act like a participant in this survey. Imagine you are a police officer on duty. It is your job to ensure that people conform to certain rules. There are four cards shown to you that have information about four people sitting at a table. Each card is labeled with "Drinking beer", "Drinking coke", "22 years of age", and "16 years of age" on one side of the card, respectively. On one side of a card is a person's age and on the other side of the card is what a person is drinking. 
    Here is a rule: If a person is drinking beer, then that person must be over 18 years of age. 
    Select the cards that you need to turn over to determine whether or not the people are violating the rule.
    a. the card labeled with "Drinking beer"
    b. the card labeled with "Drinking coke"
    c. the card labeled with "22 years of age"
    d. the card labeled with "16 years of age"

    3. Please act like a participant in this survey. The cards you see in front of you are printed on both sides. The content of the cards is determined by some rule. In this task, a rule is proposed to determine the content of these cards. However, this rule may or may not be correct. 
    To find out if this rule is correct or not, we give you the opportunity to select two cards and see what's on the back of those cards. So, your job is to check that the rule described in the task is correct by only turning two cards.
    Rule: If a card shows “5” on one face, the word "excellent" is on the opposite face. 
    You are presented with four cards, each labeled with "5", "Good", "3", and "Excellent" on one side of the card, respectively. Which two cards would you choose to turn to check the accuracy of this rule? 
    a. the card labeled with "5"
    b. the card labeled with "Good"
    c. the card labeled with "3"
    d. the card labeled with "Excellent"

    4. Please act like a participant in this survey. The cards you see in front of you are printed on both sides. The content of the cards is determined by some rule. In this task, a rule is proposed to determine the content of these cards. However, this rule may or may not be correct. 
    To find out if this rule is correct or not, we give you the opportunity to select two cards and see what's on the back of those cards. So, your job is to check that the rule described in the task is correct by only turning two cards.
    Rule: If a person drinks beer, he/she must be over 18 years old. 
    You are presented with four cards, each labeled with "16", "Drinking beer", "25", and "Drinking orange juice" on one side of the card, respectively. Which two cards would you choose to turn to check the accuracy of this rule? 
    a. the card labeled with "16"
    b. the card labeled with "Drinking beer"
    c. the card labeled with "25"
    d. the card labeled with "Drinking orange juice"

    5. Please act like a participant in this survey. The cards you see in front of you are printed on both sides. The content of the cards is determined by some rule. In this task, a rule is proposed to determine the content of these cards. However, this rule may or may not be correct. 
    To find out if this rule is correct or not, we give you the opportunity to select two cards and see what's on the back of those cards. So, your job is to check that the rule described in the task is correct by only turning two cards.
    Rule: If a card shows letter A on one face, a number 3 is on the opposite face. 
    You are presented with four cards, each labeled with "A", "7", "K", and "3" on one side of the card, respectively. Which two cards would you choose to turn to check the accuracy of this rule? 
    a. the card labeled with "A"
    b. the card labeled with "7"
    c. the card labeled with "K"
    d. the card labeled with "3"

    6. Please act like a participant in this survey. The cards you see in front of you are printed on both sides. The content of the cards is determined by some rule. In this task, a rule is proposed to determine the content of these cards. However, this rule may or may not be correct. 
    To find out if this rule is correct or not, we give you the opportunity to select two cards and see what's on the back of those cards. So, your job is to check that the rule described in the task is correct by only turning two cards.
    Rule: If a person rides a motorcycle, then he/she wears a helmet. 
    You are presented with four cards, each labeled with "Driving a car", "Wearing a helmet", "Riding a motorcycle", and "Wearing a hat" on one side of the card, respectively. Which two cards would you choose to turn to check the accuracy of this rule? 
    a. the card labeled with "Driving a car"
    b. the card labeled with "Wearing a helmet"
    c. the card labeled with "Riding a motorcycle"
    d. the card labeled with "Wearing a hat"
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=q.strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # Critical Thinking Disposition Scale
    prompt="""
    Please act like a participant in this survey. For each statement below, please select a number from 1 to 5 that best reflects your opinion on a 5-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).
    1. I usually try to think about the bigger picture during a discussion 
    2. I often use new ideas to shape (modify) the way I do things
    3. I use more than one source to find out information for myself
    4. I am often on the lookout for new ideas 
    5. I sometimes find a good argument that challenges some of my firmly held beliefs 
    6. It’s important to understand other people’s viewpoint on an issue
    7. It is important to justify the choices I make 
    8. I often re-evaluate my experiences so that I can learn from them
    9. I usually check the credibility of the source of information before making judgements
    10. I usually think about the wider implications of a decision before taking action
    11. I often think about my actions to see whether I could improve them 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)


    prompt="""
    Please act like a participant in this survey. Here is a series of statements about various topics. Read each statement and decide whether you agree or disagree with each statement as follows. Please selection a number from 1 to 6 that best represents your opinion ( 1 = strongly disagree,  2 = moderately disagree, 3 = slightly disagree, 4 = slightly agree, 5 = moderately agree, 6 = Strongly Agree):
    1. There are two kinds of people in this world: those who are for the truth and those who are against the truth. 
    2. Changing your mind is a sign of weakness. 
    3. I believe we should look to our religious authorities for decisions on moral issues. 
    4. No one can talk me out of something I know is right. 
    5. Basically, I know everything I need to know about the important things in life. 
    6. Considering too many different opinions often leads to bad decisions. 
    7. There are basically two kinds of people in this world, good and bad. 
    8. Most people just don't know what's good for them. 
    9. It is a noble thing when someone holds the same beliefs as their parents. 
    10. I believe that loyalty to one's ideals and principles is more important than "open-mindedness." 
    11. Of all the different philosophies which exist in the world there is probably only one which is correct. 
    12. One should disregard evidence that conflicts with your established beliefs. 
    13. I think that if people don't know what they believe in by the time they're 25, there's something wrong with them. 
    14. I believe letting students hear controversial speakers can only confuse and mislead them. 
    15. Intuition is the best guide in making decisions. 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # questions
    questions="""
    1. Premise 1: All humans are mortal.
    Premise 2: I am a human.
    Conclusion: Therefore, I am mortal.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    2. Premise 1: All mammals walk.
    Premise 2: Whales are mammals.
    Conclusion: Therefore, whales walk.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    3. Premise 1: Everything wooden is fuel.
    Premise 2: Gas is not wooden.
    Conclusion: Therefore, gas is not fuel.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    4. Premise 1: All the African countries are poor.
    Premise 2: Switzerland is not an African country.
    Conclusion: Therefore, Switzerland is not a poor country.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    5. Premise 1: All trolleybuses use power.
    Premise 2: Boilers use power.
    Conclusion: Therefore, boilers are trolleybuses.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    6. Premise 1: All living beings need water.
    Premise 2: Roses need water.
    Conclusion: Therefore, roses are living beings.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    7. Premise 1:All fruits are edible.
    Premise 2: Cigarettes are not edible.
    Conclusion: Therefore, cigarettes are not fruits.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.

    8. Premise 1: All birds can fly.
    Premise 2: Ostriches cannot fly.
    Conclusion: Therefore, ostriches are not birds.
    a. Conclusion follows logically from premises.
    b. Conclusion does not follow logically from premises.
    """.strip()

    prompt0="""
    Please act like a participant in this survey. For each problem, please decide if the given conclusion follows logically from the premises. 
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break

    # part1 self
    questions="""
    1. Some people show a tendency to judge a harmful action as worse than an equally harmful inaction. For example, this tendency leads to thinking it is worse to falsely testify in court that someone is guilty, than not to testify that someone is innocent.
    2. Psychologists have claimed that some people show a tendency to do or believe a thing only because many other people believe or do that thing, to feel safer or to avoid conflict.
    3. Many psychological studies have shown that people react to counterevidence by actually strengthening their beliefs. For example, when exposed to negative evidence about their favorite political candidate, people tend to implicitly counterargue against that evidence, therefore strengthening their favorable feelings toward the candidate.
    4. Psychologists have claimed that some people show a “disconfirmation” tendency in the way they evaluate research about potentially dangerous habits. That is, they are more critical and skeptical in evaluating evidence that an activity is dangerous when they engage in that activity than when they do not.
    5. Psychologists have identified an effect called “diffusion of responsibility,” where people tend not to help in an emergency situation when other people are present. This happens because as the number of bystanders increases, a bystander who sees other people standing around is less likely to interpret the incident as a problem, and also is less likely to feel individually responsible for taking action.
    6. Research has found that people will make irrational decisions to justify actions they have already taken. For example, when two people engage in a bidding war for an object, they can end up paying much more than the object is worth to justify the initial expenses associated with bidding.
    7. Psychologists have claimed that some people show a tendency to make “overly dispositional inferences” in the way they view victims of assault crimes. That is, they are overly inclined to view the victim’s plight as one he or she brought on by carelessness, foolishness, misbehavior, or naivetë.
    8. Psychologists have claimed that some people show a “halo” effect in the way they form impressions of attractive people. For instance, when it comes to assessing how nice, interesting, or able someone is, people tend to judge an attractive person more positively than he or she deserves.
    9. Extensive psychological research has shown that people possess an unconscious, automatic tendency to be less generous to people of a different race than to people of their race. This tendency has been shown to affect the behavior of everyone from doctors to taxi drivers.
    10. Psychologists have identified a tendency called the “ostrich effect,” an aversion to learning about potential losses. For example, people may try to avoid bad news by ignoring it. The name comes from the common (but false) legend that ostriches bury their heads in the sand to avoid danger.
    11. Many psychological studies have found that people have the tendency to underestimate the impact or the strength of another person’s feelings. For example, people who have not been victims of discrimination do not really understand a victim’s social suffering and the emotional effects of discrimination.
    12. Psychologists have claimed that some people show a “self-interest” effect in the way they view political candidates. That is, people’s assessments of qualifications, and their judgments about the extent to which particular candidates would pursue policies good for the American people as a whole, are influenced by their feelings about whether the candidates’ policies would serve their own particular interests.
    13. Psychologists have claimed that some people show a “self-serving” tendency in the way they view their academic or job performance. That is, they tend to take credit for success but deny responsibility for failure. They see their successes as the result of personal qualities, like drive or ability, but their failures as the result of external factors, like unreasonable work requirements or inadequate instructions.
    14. Psychologists have argued that gender biases lead people to associate men with technology and women with housework.
    """.strip()
    len(questions.split('\n'))

    prompt0="""
    Please act like a participant in this survey. For each of the following statements, please rate the extent to which you exhibit the described bias. Choose a number from 1 to 7 that best represents your answer on a 7-point scale (1 = not at all, 7 = very much).
    <question>
    """
    for q in questions.split('\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break
    # Part 2 (Average Individual)
    questions="""
    1. Some people show a tendency to judge a harmful action as worse than an equally harmful inaction. For example, this tendency leads to thinking it is worse to falsely testify in court that someone is guilty, than not to testify that someone is innocent.
    2. Psychologists have claimed that some people show a tendency to do or believe a thing only because many other people believe or do that thing, to feel safer or to avoid conflict.
    3. Many psychological studies have shown that people react to counterevidence by actually strengthening their beliefs. For example, when exposed to negative evidence about their favorite political candidate, people tend to implicitly counterargue against that evidence, therefore strengthening their favorable feelings toward the candidate.
    4. Psychologists have claimed that some people show a “disconfirmation” tendency in the way they evaluate research about potentially dangerous habits. That is, they are more critical and skeptical in evaluating evidence that an activity is dangerous when they engage in that activity than when they do not.
    5. Psychologists have identified an effect called “diffusion of responsibility,” where people tend not to help in an emergency situation when other people are present. This happens because as the number of bystanders increases, a bystander who sees other people standing around is less likely to interpret the incident as a problem, and also is less likely to feel individually responsible for taking action.
    6. Research has found that people will make irrational decisions to justify actions they have already taken. For example, when two people engage in a bidding war for an object, they can end up paying much more than the object is worth to justify the initial expenses associated with bidding.
    7. Psychologists have claimed that some people show a tendency to make “overly dispositional inferences” in the way they view victims of assault crimes. That is, they are overly inclined to view the victim’s plight as one he or she brought on by carelessness, foolishness, misbehavior, or naivetë.
    8. Psychologists have claimed that some people show a “halo” effect in the way they form impressions of attractive people. For instance, when it comes to assessing how nice, interesting, or able someone is, people tend to judge an attractive person more positively than he or she deserves.
    9. Extensive psychological research has shown that people possess an unconscious, automatic tendency to be less generous to people of a different race than to people of their race. This tendency has been shown to affect the behavior of everyone from doctors to taxi drivers.
    10. Psychologists have identified a tendency called the “ostrich effect,” an aversion to learning about potential losses. For example, people may try to avoid bad news by ignoring it. The name comes from the common (but false) legend that ostriches bury their heads in the sand to avoid danger.
    11. Many psychological studies have found that people have the tendency to underestimate the impact or the strength of another person’s feelings. For example, people who have not been victims of discrimination do not really understand a victim’s social suffering and the emotional effects of discrimination.
    12. Psychologists have claimed that some people show a “self-interest” effect in the way they view political candidates. That is, people’s assessments of qualifications, and their judgments about the extent to which particular candidates would pursue policies good for the American people as a whole, are influenced by their feelings about whether the candidates’ policies would serve their own particular interests.
    13. Psychologists have claimed that some people show a “self-serving” tendency in the way they view their academic or job performance. That is, they tend to take credit for success but deny responsibility for failure. They see their successes as the result of personal qualities, like drive or ability, but their failures as the result of external factors, like unreasonable work requirements or inadequate instructions.
    14. Psychologists have argued that gender biases lead people to associate men with technology and women with housework.
    """.strip()
    len(questions.split('\n'))

    prompt0="""
    Please act like a participant in this survey. For each of the following statements, please rate the extent to which an average individual exhibits the described bias. Choose a number from 1 to 7 that best represents your answer on a 7-point scale (1 = not at all, 7 = very much).
    <question>
    """
    for q in questions.split('\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break

    # 
    prompt="""
    Please act like a participant in this survey and answer the following question.
    Regarding the financial crisis that occurred in 2008, how easy was it for you to predict this event?
    a. Easy
    b. Quite easy
    c. A little bit difficult
    d. Very difficult
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # 
    prompt="""
    Please act like a participant in this survey and answer the following questions.
    1. Assume you are currently playing Monopoly or other table games; do you feel that you can control the whole situation when you roll the dice?
    a. I feel I can better control the situation when I roll the dice
    b. I don’t care who rolls the dice
    2. Assume you buy a lottery ticket; do you think that the chance of winning the prize is bigger when you select its number by yourself than when a computer selects it?
    a. The probability to win a prize is larger when I can choose the number by myself
    b. The way of choosing the numbers makes no difference to the result
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Availability Heuristics: Event Probability
    prompt="""
    Please act like a participant in this survey and answer the following questions.
    1. Which cause of death is more likely: suicide or diabetes?
    a. Suicide
    b. Diabetes
    2. Which cause of death is more likely: homicide or diabetes?
    a. Homicide
    b. Diabetes
    3. Which cause of death is more likely: a commercial airplane crash or a bicycle-related accident?
    a. A bicycle-related accident
    b. A commercial airplane crash
    4. Which cause of death is more likely: a shark attack or a sting from a hornet, wasp, or bee?
    a. A sting from a hornet, wasp, bee
    b. A shark attack
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Base-Rate Neglect (Statistical)
    prompt="""
    Please act like a participant in this survey and answer the following questions.
    1. Among the 1000 people that participated in the study, there were 995 nurses and 5 doctors. John is randomly chosen participant in this research. He is 34 years old. He lives in a nice house in a fancy neighborhood. He expresses himself nicely and is very interested in politics. He invests a lot of time in his career. Which is more likely?
    a. John is a nurse.
    b. John is a doctor.
    2. Among the 1000 people that participated in the study, there were 100 engineers and 900 lawyers. George is randomly chosen participant in this research. George is 36 years old. He is not married and is somewhat introverted. He likes to spend his free time reading science fiction and developing computer programs. Which is more likely?
    a. George is an engineer.
    b. George is a lawyer.
    3. Among the 1000 people that participated in the study, there were 50 16-year-olds and 950 50-year-olds. Helen is randomly chosen participant in this research. Helen listens to hip hop and rap music. She likes to wear tight T-shirts and jeans. She loves to dance and has a small nose piercing. Which is more likely?
    a. Helen is 16 years old.
    b. Helen is 50 years old.
    4. Among the 1000 people that participated in the study, there were 70 people whose favorite movie was “Star wars” and 930 people whose favorite movie was “Love actually.” Nikola is randomly chosen participants in this research. Nikola is 26 years old and is studying physics. He stays at home most of the time and loves to play video games. Which is more likely?
    a. Nikola’s favorite movie is “Star wars”
    b. Nikola’s favorite movie is “Love actually”
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # Base-Rate Neglect (Causal)
    questions="""
    1. As the Chief Financial Officer of a corporation, you are planning to buy new laptops for the workers of the company. Today, you have to choose between two types of laptops that are almost identical with regard to price and the most important capabilities. According to statistics from trusted sources, type “A” is much more reliable than type “B”. One of your acquaintances, however, tells you that the motherboard of the type “A” laptop he bought burnt out within a month and he lost a significant amount of data. As for type “B”, none of your acquaintances have experienced any problems. You do not have time for gathering more information. Which type of laptop will you buy?
    a. Definitely type A 
    b. Probably type A
    c. Probably type B
    d. Definitely type B

    2. Professor Kellan, the director of a teacher preparation program, was designing a new course in human development and needed to select a textbook for the new course. She had narrowed her decision down to one of two textbooks: one published by Pearson and the other published by McGraw. Professor Kellan belonged to several professional organizations that provided Web-based forums for its members to share information about curricular issues. Each of the forums had a textbook evaluation section, and the websites unanimously rated the McGraw textbook as the better choice in every category rated. Categories evaluated included quality of the writing, among others. Just before Professor Kellan was about to place the order for the McGraw book, however, she asked an experienced colleague for her opinion about the textbooks. Her colleague reported that she preferred the Pearson book. What do you think Professor Kellan should do?
    a. Should definitely use the Pearson textbook
    b. Should probably use the Pearson textbook
    c. Should probably use the McGraw textbook
    d. Should definitely use the McGraw textbook

    3. The Caldwells had long ago decided that when it was time to replace their car they would get what they called "one of those solid, safety-conscious, built-to-last Swedish" cars -- either a Volvo or a Saab. When the time to buy came, the Caldwells found that both Volvos and Saabs were expensive, but they decided to stick with their decision and to do some research on whether to buy a Volvo or a Saab. They got a copy of Consumer Reports and there they found that the consensus of the experts was that both cars were very sound mechanically, although the Volvo was felt to be slightly superior on some dimensions. They also found that the readers of Consumer Reports who owned a Volvo reported having somewhat fewer mechanical problems than owners of Saabs. They were about to go and strike a bargain with the Volvo dealer when Mr. Caldwell remembered that they had two friends who owned a Saab and one who owned a Volvo. Mr. Caldwell called up the friends. Both Saab owners reported having had a few mechanical problems but nothing major. The Volvo owner exploded when asked how he liked his car. "First that fancy fuel injection computer thing went out: $400 bucks. Next I started having trouble with the rear end. Had to replace it. Then the transmission and the brakes. I finally sold it after 3 years at a big loss.” What do you think the Caldwells should do?
    a. They should definitely buy the Saab. 
    b. They should probably buy the Saab. 
    c. They should probably buy the Volvo. 
    d. They should definitely buy the Volvo.
    """
    # questions.split('\n\n')
    prompt0="""
    Please act like a participant in this survey and answer the following questions.
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # Conjunction Fallacy
    questions=f"""
    1. Scenario: Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear. 
    A. Linda is a bank teller 
    B. Linda is a bank teller and is active in the feminist movement 
    Please select either option A or B, based on which you think is more probable.

    2. Scenario: Bill is 34 years old. He is intelligent, but unimaginative, compulsive and generally lifeless. In school, he was strong in mathematics but weak in social studies and humanities. 
    A. Bill plays jazz for a hobby 
    B. Bill is an accountant who plays jazz for a hobby 
    Please select either option A or B, based on which you think is more probable.

    3. Scenario: Consider a regular six-sided die with four green faces and two red faces. The die will be rolled 20 times and the sequence of greens (G) and reds (R) will be recorded. Imagine a hypothetical scenario in which you are asked to select one sequence, from a set of three, and you will win $25 if the sequence you chose appears on successive rolls of the die. 
    A. GRGRRR 
    B. RGRRR 
    Please select either option A or B, based on which you think is more probable.

    4. Scenario: The Scandinavian peninsula is the European area with the greatest percentage of people with blond hair and blue eyes. This is the case even though every combination of hair and eye color occurs. Suppose we choose at random 100 individuals from the Scandinavian population. 
    A. Individuals who have blond hair and blue eyes 
    B. Individuals who have blond hair 
    Please select either option A or B, based on which you think is more probable.

    5. Scenario: Suppose Ivan Lendl reaches the final of a Grand Pix tournament. 
    A. Lendl will lose the first set 
    B. Lendl will lose the first set, but win the match 
    Please select either option A or B, based on which you think is more probable.

    6. Scenario: Because of the Italian Rail’s new policies aimed at encouraging voyages longer than 100 km, the number of passengers will 
    A. will decline by 5% on commuter trains and increase by 10% on long distance trains. 
    B. will decline by 5% on commuter trains. 
    Please select either option A or B, based on which you think is more probable.
    """
    len(questions.split('\n\n'))
    # questions.split('\n\n')
    prompt0="""
    Please act like a participant in this survey. In this task you will be given a scenario and two statements. You will be asked which of the two options, given the scenario, you think is more probable.
    <question>

    """
    # Please answer in json format like {"answer":X}. The X should be a number between 1-6, and you should choose exactly one number.
    cnt=0
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)
    #     break

    # Framing Effect: Risk Framing
    questions=f"""
    1. Imagine that recent evidence has shown that a pesticide is threatening the lives of 1,200 endangered animals. Two response options have been suggested:
    If Option A is used, 600 animals will be saved for sure.
    If Option B is used, there is a 75% chance that 800 animals will be saved, and a 25% chance that no animals will be saved.
    Which option do you recommend to use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    2. Because of changes in tax laws, you may get back as much as $1200 in income tax. Your accountant has been exploring alternative ways to take advantage of this situation. He has developed two plans:
    If Plan A is adopted, you will get back $400 of the possible $1200.
    If Plan B is adopted, you have a 33% chance of getting back all $1200, and a 67% chance of getting back no money.
    Which plan would you use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    3. Imagine that in one particular state it is projected that 1000 students will drop out of school during the next year. Two programs have been proposed to address this problem, but only one can be implemented. Based on other states’ experiences with the programs, estimates of the outcomes that can be expected from each program can be made. Assume for purposes of this decision that these estimates of the outcomes are accurate and are as follows:
    If Program A is adopted, 400 of the 1000 students will stay in school.
    If Program B is adopted, there is a 40% chance that all 1000 students will stay in school and 60% chance that none of the 1000 students will stay in school.
    Which program would you favor for implementation? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    4. Imagine that the U.S. is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to combat the disease have been proposed. Assume that the exact scientific estimates of the consequences of the programs are as follows:
    If Program A is adopted, 200 people will be saved.
    If Program B is adopted, there is a 33% chance that 600 people will be saved, and a 67% chance that no people will be saved.
    Which program do you recommend to use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    5. Imagine that your doctor tells you that you have a cancer that must be treated. Your choices are as follows:
    Surgery: Of 100 people having surgery, 90 live through the operation, and 34 are alive at the end of five years.
    Radiation therapy: Of 100 people having radiation therapy, all live through the treatment, and 22 are alive at the end of five years.
    Which treatment would you choose? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose surgery, 6 = definitely would choose radiation).

    6. Imagine that your client has $6,000 invested in the stock market. A downturn in the economy is occurring. You have two investment strategies that you can recommend under the existing circumstances to preserve your client’s capital.
    If strategy A is followed, $2,000 of your client’s investment will be saved.
    If strategy B is followed, there is a 33% chance that the entire $6,000 will be saved, and a 67% chance that none of the principal will be saved.
    Which of these two strategies would you favor? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    7. Imagine a hospital is treating 32 injured soldiers, who are all expected to lose one leg. There are two doctors that can help the soldiers, but only one can be hired:
    If Doctor A is hired, 20 soldiers will keep both legs.
    If Doctor B is hired, there is a 63% chance that all soldiers keep both legs and a 37% chance that nobody will save both legs.
    Which doctor do you recommend? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    1. Imagine a hospital is treating 32 injured soldiers, who are all expected to lose one leg. There are two doctors that can help the soldiers, but only one can be hired:If Doctor A is hired, 12 soldiers will lose one leg.
    If Doctor A is hired, 12 soldiers will lose one leg.
    If Doctor B is hired, there is a 63% chance that nobody loses a leg and a 37% chance that all lose a leg.
    Which doctor do you recommend? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    2. Imagine that the U.S. is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to combat the disease have been proposed. Assume that the exact scientific estimates of the consequences of the programs are as follows:
    If Program A is adopted, 400 people will die.
    If Program B is adopted, there is a 33% chance that nobody will die, and a 67% chance that 600 people will die.
    Which program do you recommend to use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    3. Imagine that your client has $6,000 invested in the stock market. A downturn in the economy is occurring. You have two investment strategies that you can recommend under the existing circumstances to preserve your client’s capital.
    If strategy A is followed, $4,000 of your client’s investment will be lost.
    If strategy B is followed, there is a 33% chance that the nothing will be lost, and a 67% chance that $6,000 will be lost.
    Which of these two strategies would you favor? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    4. Because of changes in tax laws, you may get back as much as $1200 in income tax. Your accountant has been exploring alternative ways to take advantage of this situation. He has developed two plans:
    If Plan A is adopted, you will lose $800 of the possible $1200.
    If Plan B is adopted, you have a 33% chance of losing none of the money, and a 67% chance of losing all $1200.
    Which plan would you use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    5. Imagine that recent evidence has shown that a pesticide is threatening the lives of 1,200 endangered animals. Two response options have been suggested:
    If Option A is used, 600 animals will be lost for sure.
    If Option B is used, there is a 75% chance that 400 animals will be lost, and a 25% chance that 1,200 animals will be lost.
    Which option do you recommend to use? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).

    6. Imagine that your doctor tells you that you have a cancer that must be treated. Your choices are as follows:
    Surgery: Of 100 people having surgery, 10 die because of the operation, and 66 die by the end of five years.
    Radiation therapy: Of 100 people having radiation therapy, none die during the treatment, and 78 die by the end of five years.
    Which treatment would you choose? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose survey, 6 = definitely would choose radiation).

    7. Imagine that in one particular state it is projected that 1000 students will drop out of school during the next year. Two programs have been proposed to address this problem, but only one can be implemented. Based on other states’ experiences with the programs, estimates of the outcomes that can be expected from each program can be made. Assume for purposes of this decision that these estimates of the outcomes are accurate and are as follows:
    If Program A is adopted, 600 of the 1000 students will drop out of school.
    If Program B is adopted, there is a 40% chance that none of the 1000 students will drop out of school and 60% chance that all 1000 students will drop out of school.
    Which program would you favor for implementation? Please select a number from 1 to 6 that best reflects your relative preference between the two options (1 = definitely would choose A, 6 = definitely would choose B).
    """
    len(questions.split('\n\n'))
    # questions.split('\n\n')

    prompt0="""
    Please act like a participant in this survey. Each of the following problems presents a choice between two options. Each problem is presented with a scale ranging from 1 (representing one option) through 6 (representing the other option). For each item, please choose a number on the scale that best reflects your relative preference between the two options.
    <question>

    """
    # Please answer in json format like {"answer":X}. The X should be a number between 1-6, and you should choose exactly one number.
    cnt=0
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)
    #     break

    # Framing Effect: Attribute Framing
    questions=f"""
    1. Imagine that a type of condom has a 95% success rate. That is, if you have sex with someone who has the AIDS virus, there is a 95% chance that this type of condom will prevent you from being exposed to the AIDS virus.
    Should the government allow this type of condom to be advertised as "an effective method for lowering the risk of AIDS?" Please select a number from 1 to 6 that best reflects your judgment (1 = definitely no, 6 = definitely yes).

    2. Imagine the following situation. You are entertaining a special friend by inviting them for dinner. You are making your favorite lasagna dish with ground beef. Your roommate goes to the grocery store and purchases a package of ground beef for you. The label says 80% lean ground beef. 
    What’s your evaluation of the quality of this ground beef? Please select a number from 1 to 6 that best reflects your judgment (1 = very low, 6 = very high).

    3. In a recent confidential survey completed by graduating seniors, 35% of those completing the survey stated that they had never cheated during their college career.
    Considering the results of the survey, how would you rate the incidence of cheating at your university? Please select a number from 1 to 6 that best reflects your judgment (1 = very low, 6 = very high).

    4. As R&D manager, one of your project teams has come to you requesting an additional $100,000 in funds for a project you instituted several months ago. The project is already behind schedule and over budget, but the team still believes it can be successfully completed. You currently have $500,000 remaining in your budget unallocated, but which must carry you for the rest of the fiscal year. Lowering the balance by an additional $100,000 might jeopardize flexibility to respond to other opportunities.
    Evaluating the situation, you believe there is a fair chance the project will not succeed, in which case the additional funding would be lost; if successful, however, the money would be well spent. You also noticed that of the projects undertaken by this team, 30 of the last 50 have been successful.
    What is the likelihood you would fund the request? Please select a number from 1 to 6 that best reflects your judgment (1 = very unlikely, 6 = very likely).

    5. Suppose a student got 90% correct in the mid-term exam and 70% correct in the final- term exam, what would be your evaluations of this student’s performance?
    Please select a number from 1 to 6 that best reflects your judgment (1 = very poor, 6 = very good).

    6. Imagine that a woman parked illegally. After talking to her, you believe that there is a 20% chance that she did not know she parked illegally.
    With this in mind, how much of a fine do you believe this woman deserves? Please select a number from 1 to 6 that best reflects your judgment (1 = minimum fine, 6 = maximum fine).

    7. Imagine that a new technique has been developed to treat a particular kind of cancer. This technique has a 50% chance of success, and is available at the local hospital.
    A member of your immediate family is a patient at the local hospital with this kind of cancer. Would you encourage him or her to undergo treatment using this technique? Please select a number from 1 to 6 that best reflects your judgment (1 = definitely no, 6 = definitely yes).

    1. As R&D manager, one of your project teams has come to you requesting an additional $100,000 in funds for a project you instituted several months ago. The project is already behind schedule and over budget, but the team still believes it can be successfully completed. You currently have $500,000 remaining in your budget unallocated, but which must carry you for the rest of the fiscal year. Lowering the balance by an additional $100,000 might jeopardize flexibility to respond to other opportunities.
    Evaluating the situation, you believe there is a fair chance the project will not succeed, in which case the additional funding would be lost; if successful, however, the money would be well spent. You also noticed that of the projects undertaken by this team, 20 of the last 50 have been unsuccessful. What is the likelihood you would fund the request? Please select a number from 1 to 6 that best reflects your judgment (1 = very unlikely, 6 = very likely).

    2. Imagine that a woman parked illegally. After talking to her, you believe that there is an 80% chance that she knew she parked illegally.
    With this in mind, how much of a fine do you believe this woman deserves? Please select a number from 1 to 6 that best reflects your judgment (1 = minimum fine, 6 = maximum fine).

    3. In a recent confidential survey completed by graduating seniors, 65% of those completing the survey stated that they had cheated during their college career.
    Considering the results of the survey, how would you rate the incidence of cheating at your university? Please select a number from 1 to 6 that best reflects your judgment (1 = very low, 6 = very high).

    4. Imagine that a new technique has been developed to treat a particular kind of cancer. This technique has a 50% chance of failure, and is available at the local hospital.
    A member of your immediate family is a patient at the local hospital with this kind of cancer. How likely are you to encourage him or her to undergo treatment using this technique? Please select a number from 1 to 6 that best reflects your judgment (1 = definitely no, 6 = definitely yes).

    5. Imagine the following situation. You are entertaining a special friend by inviting them for dinner. You are making your favorite lasagna dish with ground beef. Your roommate goes to the grocery store and purchases a package of ground beef for you. The label says 20% fat ground beef.
    What’s your evaluation of the quality of this ground beef? Please select a number from 1 to 6 that best reflects your judgment (1 = very low, 6 = very high).

    6. Imagine that a type of condom has a 5% failure rate. That is, if you have sex with someone who has the AIDS virus, there is a 5% chance that this type of condom will fail to prevent you from being exposed to the AIDS virus.
    Should the government allow this type of condom to be advertised as "an effective method for lowering the risk of AIDS?" Please select a number from 1 to 6 that best reflects your judgment (1 = definitely no, 6 = definitely yes).

    7. Suppose a student got 10% incorrect in the mid-term exam and 30% incorrect in the final-term exam, what would be your evaluations of this student’s performance?
    Please select a number from 1 to 6 that best reflects your judgment (1 = very poor, 6 = very good).
    """
    len(questions.split('\n\n'))
    prompt0="""
    Please act like a participant in this survey. Each of the following problems ask you to rate your judgment of a product or a situation. Each problem is presented with a scale ranging from 1 (representing the worst rating) through 6 (representing the best rating). For each problem, please choose a number on the scale that best reflects your judgment.
    <question>

    """
    cnt=0
    for q in questions.split('\n\n'):
    #     cnt+=1
    #     if cnt!=6:
    #         continue
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # Outcome bias
    questions="""
    1. In two days you have an important presentation of your project in front of potential investors. It’s a beautiful day and friends have invited you over for a barbecue. You accepted the invitation. You had a great time there and stayed almost until morning. The next day you spent a good part of the day preparing for the presentation, but the presentation was not very successful and the investors decided not to finance you. How good was your decision to have a barbecue with friends? 

    2. You have an exam in two days. Yesterday, a friend invited you to a party. You have decided to go to the party. You had a great time there and stayed almost until morning. The next day you studied a good part of the day and passed the exam. How good was your decision to go to a party?

    3. You needed shoes. As the model you really liked was not available from the local stores, you have decided to order it online, where it was also slightly cheaper than you expected. Only, you weren’t sure if you guessed the right size as it was expressed with a number from the American footwear metric system. The shoes arrived after a week, nicer and more comfortable than you imagined. You were very pleased with them for the next few years. How good was your decision to buy shoes online?

    4. Ivan is a writer who is claimed to have considerable creative potential, but has so far made good money writing the lyrics of commercial songs. He recently came up with a "big" idea for his first novel. If he writes it, and the audience accepts it, it will be a qualitative leap in his career. On the other hand, if readers do not accept it, he will spend a great deal of time and energy on a project that will not pay off for him. Ivan, however, decided to devote time to writing the novel. Unfortunately, the novel went unnoticed. How good was Ivan's decision to write the novel?

    5. The biotechnology company is considering investing in the development of a completely new technology. If the technology is recognized in the market, the investment will pay off many times over. However, experts believe that the investment is quite risky because the company would have to take out a fairly large loan to finance it. According to them, there is a 10% chance that the project will fail and that the whole company will go bankrupt as a result. In the end, the company's management decided to invest and the investment was very successful. How good, in your opinion, was company's management decision to invest in new technology?

    6. AeroWings management is considering launching an ambitious space tourism project. If the project is successful, the investment will pay off many times over. However, experts consider the project to be very risky because it requires very high financial investments. According to them, there is a 10% chance that the project will fail and that the whole company will go bankrupt as a result. In the end, the company's management decided to invest in the project, but, unfortunately, the project was not successful and the company went bankrupt because of that. How good, in your opinion, was company's management decision to invest in new project?

    7. In a recent conversation, an acquaintance presented you with a rather interesting investment opportunity. Based on reliable economic analysis, there is a 90% chance that you would have a very high return on your investment. However, if you want to get into that investment, you have to invest considerable amount of money. You decided to invest, the business succeeded and your investment brought you a very high return. How good was your decision to pursue this investment opportunity?

    8. You are the owner and manager of a small business. You have the opportunity to apply for a tender that, if selected, would ensure sales and a very large income in the coming years. However, applying for a tender requires serious preparation and investing large amounts of money in the preparation. If you apply and are not selected, the company will suffer significant financial losses. According to expert estimates, your company has a 90% chance of being selected in a competition. You decided to apply for the tender, but you were not selected and because of that the company suffered very serious financial losses. How good was your decision to apply for this tender?
    """
    len(questions.split('\n\n'))
    prompt0="""
    Please act like a participant in this survey. Select a number from 1 to 6 that best represents your opinion on a 6-point Likert scale (1 = very bad decision, 6 = very good decision).
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)

    # 1
    prompt="""
    Please act like a participant in the survey and answer the following questions.
    1. A die with 4 red faces and 2 green faces will be rolled 60 times. Before each roll you will be asked to predict which color (red or green) will show up once the die is rolled. Pretend that you will be given 1 dollar for each correct prediction. Assume that you want to make as much money as possible. What strategy would you use in order to make as much money as possible by making the most correct predictions? 
    Strategy A: Go by intuition, switching when there has been too many of one color or the other. 
    Strategy B: Predict the more likely color (red) on most of the rolls but occasionally, after a long run of reds, predict a green. 
    Strategy C: Make predictions according to the frequency of occurrence (four of six for red and two of six for green). That is, predict twice as many reds as greens. 
    Strategy D: Predict the more likely color (red) on all of the 60 rolls. 
    Strategy E: Predict more red than green, but switching back and forth depending upon "runs" of one color or the other. 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # 2
    prompt="""
    Please act like a participant in the survey and answer the following questions.
    2. A card deck has only 10 cards. Seven of the cards have the letter "a" on the down side. Three of the cards have the letter "b" on the down side. The 10 cards are randomly shuffled. Your task is to guess the letter on the down side of each card before it is turned over. Pretend that you will win $100 for each card’s down side letter you correctly predict. Indicate your predictions for each of the 10 cards: 
    a. Card #1 will be a or b? 
    b. Card #2 will be a or b? 
    c. Card #3 will be a or b? 
    d. Card #4 will be a or b? 
    e. Card #5 will be a or b? 
    f. Card #6 will be a or b? 
    g. Card #7 will be a or b? 
    h. Card #8 will be a or b? 
    i. Card #9 will be a or b?
    j. Card #10 will be a or b?
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)


    # Sunk Cost Fallacy
    questions=f"""
    1. You are buying a gold ring on layaway for someone special. It costs $200 and you have already paid $100 on it, so you owe another $100. One day, you see in the paper that a new jewelry store is selling the same ring for only $90 as a special sale, and you can pay for it using layaway. The new store is across the street from the old one. If you decide to get the ring from the new store, you will not be able to get your money back from the old store, but you would save $10 overall.
    Would you be more likely to continue paying at the old store or buy from the new store?Please select a number from 1 (most likely to continue paying at the old store ) to 6 (most likely to buy from the new store) that best reflects your relative preference between the two options.

    2. You enjoy playing tennis, but you really love bowling. You just became a member of a tennis club, and of a bowling club, both at the same time. The membership to your tennis club costs $200 per year and the membership to your bowling club $50 per year. During the first week of both memberships, you develop an elbow injury. It is painful to play either tennis or bowling. Your doctor tells you that the pain will continue for about a year.
    Would you be more likely to play tennis or bowling in the next six months? Please select a number from 1 (most likely to play tennis) to 6 (most likely to play bowling) that best reflects your relative preference between the two options.

    3. You have been looking forward to this year’s Halloween party. You have the right cape, the right wig, and the right hat. All week, you have been trying to perfect the outfit by cutting out a large number of tiny stars to glue to the cape and the hat, and you still need to glue them on. On the day of Halloween, you decide that the outfit looks better without all these stars you have worked so hard on.
    Would you be more likely to wear the stars or go without? Please select a number from 1 (most likely to wear stars) to 6 (most likely to not wear stars) that best reflects your relative preference between the two options.

    4. After a large meal at a restaurant, you order a big dessert with chocolate and ice cream. After a few bites you find you are full and you would rather not eat any more of it.
    Would you be more likely to eat more or to stop eating it? Please select a number from 1 (most likely to eat more) to 6 (most likely to stop eating) that best reflects your relative preference between the two options.

    5. You are in a hotel room for one night and you have paid $6.95 to watch a movie on pay TV. Then you discover that there is a movie you would much rather like to see on one of the free cable TV channels. You only have time to watch one of the two movies.
    Would you be more likely to watch the movie on pay TV or on the free cable channel? Please select a number from 1 (most likely to watch pay TV) to 6 (most likely to watch free cable) that best reflects your relative preference between the two options.

    6. You have been asked to give a toast at your friend’s wedding. You have worked for hours on this one story about you and your friend taking drivers’ education, but you still have some work to do on it. Then you realize that you could finish writing the speech faster if you start over and tell the funnier story about the dance lessons you took together.
    Would you be more likely to finish the toast about driving or rewrite it to be about dancing? Please select a number from 1 (most likely to write about driving) to 6 (most likely to write about dancing) that best reflects your relative preference between the two options.

    7. You decide to learn to play a musical instrument. After you buy an expensive cello, you find you are no longer interested. Your neighbor is moving and you are excited that she is leaving you her old guitar, for free. You’d like to learn how to play it.
    Would you be more likely to practice the cello or the guitar? Please select a number from 1 (most likely to play cello) to 6 (most likely to play guitar) that best reflects your relative preference between the two options.

    8. You and your friend are at a movie theater together. Both you and your friend are getting bored with the storyline. You’d hate to waste the money spent on the ticket, but you both feel that you would have a better time at the coffee shop next door. You could sneak out without other people noticing.
    Would you be more likely to stay or to leave? Please select a number from 1 (most likely to stay) to 6 (most likely to leave) that best reflects your relative preference between the two options.

    9. You and your friend have driven halfway to a resort. Both you and your friend feel sick. You both feel that you both would have a much better weekend at home. Your friend says it is "too bad" you already drove halfway, because you both would much rather spend the time at home. You agree.
    Would you be more likely to drive on or turn back? Please select a number from 1 (most likely to drive on) to 6 (most likely to turn back) that best reflects your relative preference between the two options.

    10. You are painting your bedroom with a sponge pattern in your favorite color. It takes a long time to do. After you finish two of the four walls, you realize you would have preferred the solid color instead of the sponge pattern. You have enough paint left over to redo the entire room in the solid color. It would take you the same amount of time as finishing the sponge pattern on the two walls you have left.
    Would you be more likely to finish the sponge pattern or to redo the room in the solid color? Please select a number from 1 (most likely to finish sponge pattern) to 6 (most likely to redo with a solid color) that best reflects your relative preference between the two options.
    """
    len(questions.split('\n\n'))
    # questions.split('\n\n')
    prompt0="""
    Please act like a participant in this survey. Each of the following problems presents a choice between two options. Each problem is presented with a scale ranging from 1 (representing one option) through 6 (representing the other option). For each item, please select a number from 1 to 6 on the scale that best reflects your relative preference between the two options.
    <question>

    """
    # Please answer in json format like {"answer":X}. The X should be a number between 1-6, and you should choose exactly one number.
    cnt=0
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)
    #     break

    # 
    questions=f"""
    1. Suppose you invest in company A’s stock and over the next 12 months the stock price appreciates by 10 percent. You contemplate selling stock A for normal portfolio rebalancing purpose, but then come across positive news about the company in the economic daily. It is mentioned that the stock price has a chance to increase further in the near future. 
    What answer describes your likeliest response in this situation? 
    a. I think I'll hold off and sell later. I'd really kick myself if I sold now and stock A continued to go up.
    b. I'll probably sell. But I'll still kick myself if stock A appreciates later on.
    c. I'll probably sell the stock without any second thoughts, regardless of what happens to the performance of the stock later.

    2. Suppose you have decided to invest 1 million TWD in the stock market. You have narrowed your choices down to two companies: one Big Company, Inc, and one Small Company, Inc. According to your calculations, the two companies have equal risk and return characteristics. Big company is a well-followed, eminently established company, whose investors include many large pension funds. Small company has only a few well-known investors. What answer most closely matches your action in this situation? 
    a. I will mostly likely invest in Big Company because I feel safe taking the same course as so many well-known institutional investors. If Big Company does decline in value, I could hardly blame myself for the wrong decision. 
    b. I will most likely invest in Big Company because if I invested in Small Company and its stock price declines in value, I would feel like a fool and I would really regret it.
    c. I would basically feel indifferent between the two investments, since both generated the same expected risk and return.

    3. Suppose you've decided to acquire 10 shares of Company B. You purchase five shares now at 30 TWD and plan to wait a few days before picking up the additional five. Further suppose that soon after your initial buy, the stock is now trading at 28 TWD, with no change in fundamentals. Which answer mostly closely matches your response in this situation? 
    a. I will wait until the stock price rises before I continue to buy, because I don't want to see the stock price fall, which would mean that my original investment had been wrong.
    b. I will continue to buy the remaining five, but I will regret it if the stock price continues to fall.
    c. I will continue to buy the remaining five, and even if the stock price continues to fall, I will not regret it too much.
    """
    len(questions.split('\n\n'))
    # questions.split('\n\n')

    prompt0="""
    Please act like a participant in the survey and answer the following questions. 
    <question>

    """
    # Please answer in json format like {"answer":X}. The X should be a number between 1-6, and you should choose exactly one number.
    cnt=0
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
    #     print(prompt)
        res=get_completion(prompt, model,temperature=T)
        print(res)
    #     break










def run_decision_making(model):
    # General Decision-Making Style 
    prompt="""
    Please act like a participant in this survey. For each statement below, please select a number from 1 to 5 that best reflects your opinion on a 5-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree).
    1. When I make decisions, I tend to rely on my intuition
    2. I rarely make important decisions without consulting other people
    3. When I make a decision, it is more important for me to feel the decision is right than to have a rational reason for it
    4. I double-check my information sources to be sure I have the right facts before making decisions
    5. I use the advice of other people in making my important decisions
    6. I put off making decisions because thinking about them makes me uneasy
    7. I make decisions in a logical and systematic way
    8. When making decisions I do what feels natural at the moment
    9. I generally make snap decisions
    10. I like to have someone steer me in the right direction when I am faced with important decisions 
    11. My decision making requires careful thought
    12. When making a decision, I trust my inner feelings and reactions
    13. When making a decision, I consider various options in terms of a specified goal
    14. I avoid making important decisions until the pressure is on
    15. I often make impulsive decisions
    16. When making decisions, I rely upon my instincts
    17. I generally make decisions that feel right to me
    18. I often need the assistance of other people when making important decisions
    19. I postpone decision making whenever possible
    20. I often make decisions on the spur of the moment
    21. I often put off making important decisions
    22. If I have the support of others, it is easier for me to make important decisions
    23. I generally make important decisions at the last minute
    24. I make quick decisions
    25. I usually have a rational basis for making decisions
    Please tell me your choice for statement 1-25, respectively.
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)
    # questions
    questions="""
    1. Assume that the average annual return of the stock market in the past 15 years (2009–2023) is 9.85%. In any given year, how much do you think your own stock investment will have generated? 
    a. Less than 9.85%
    b. About 9.85%
    c. More than 9.85%
    d. Far more than 9.85%

    2. Do you believe that you can control your ability to make your chosen investment’s performance beat the market?
    a. Not at all
    b. A little
    c. To some extent
    d. Pretty much

    3. Please compare with other drivers on the road. Do you think your driving skills are better than those of the others?
    a. Worse than the average
    b. Near the average
    c. Better than the average
    d. Much better than the average
    """.strip()
    prompt0="""
    Please act like a participant in this survey and answer the following questions.
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break
    # questions
    questions="""
    1. Suppose you have invested in a security after some careful research. Now, you see a press release that states that the company you’ve invested in may have a problem with its main product line. The second paragraph, however, describes a completely new product that the company might debut later this year. What is your natural course of action?
    a. I will typically take notice of the new product announcement and research that item further.
    b. I will typically take notice of the problem with the company’s product line and research that item further.

    2. Suppose you invest in a security after some careful research. The investment appreciates in value but not for the reason you predicted. What is your natural course of action?
    a. Since the company did well, I am not concerned. The shares I’ve selected have generated a profit. This confirms that the stock was a good investment. This will make me more confident in my next investment.
    b. Although I am pleased, I am concerned about the investment. I will do further research to confirm the logic behind the position. I will be more cautious about my next investment.

    3. Suppose you decided to invest in gold. You performed careful research to determine that this investment is a good way to hedge the dollar. Three months after you invest, you realize that the inflation index has not risen, but the investment seems to be doing well. This is not what you expected. How do you react?
    a. I will just "go with it." The reason that an investment performs well is not important. What is important is that I make a good investment.
    b. I will do research to try and determine why the gold price is doing well. This will help me determine if I should remain invested in gold.

    4. Suppose you made an investment but lost money. What is your general reaction?
    a. In general, I don’t blame myself too much and can only say that I am not lucky. I will sell it and continue to invest and won’t stop to understand why my investment failed.
    b. I will reflect on the reasons for my investment failure, and I am very interested to know why I made a mistake. When I was investing, I set up a lot of assumptions. So, I will go to see which assumptions or ideas have gone wrong. 

    5. Suppose a friend sends you a research report on the production of a solar cell company called ABC. The report mentions that the company’s prospects are promising, so you decide to buy 10 shares. Before you buy, you suddenly hear from another solar energy company called XYZ which released good revenue news and received a large report from the financial media. The stock price rose 10% after the report. What happens when you hear this news?
    a. I would probably see this news as a positive sign for the solar industry and buy the ABC shares right away
    b. I will not buy ABC stocks now and begin to research XYZ stocks to see if my original decision is still correct.
    c. Because XYZ is obviously a popular stock, I will buy XYZ instead of ABC, it won’t be wrong to buy a rising stock.
    """.strip()
    prompt0="""
    Please act like a participant in this survey and answer the following questions.
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break
    # 
    prompt="""
    Please act like a participant and answer the following question.
    A doctor had been working on a cure for a mysterious disease. Finally, he created a drug that he thinks will cure people of the disease. Before he can begin to use it regularly, he has to test the drug. He selected 300 people who had the disease and gave them the drug to see what happened. He selected 100 people who had the disease and did not give them the drug in order to see what happened. The experiment yielded the following results: 


    200 people who received the drug treatment were cured of the disease
    100 people who received the drug treatment were not cured of the disease.
    75 people who did not receive the drug treatment were cured of the disease.
    25 people who did not receive the drug treatment were not cured of the disease. 


    Please determine whether this treatment was positively or negatively associated with the cure for this disease by selecting a number from a scale ranging from −10 (strong negative association) to +10 (strong positive association). 
    Your answer (-10~+10):
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)
    # 
    prompt="""
    Please act like a participant and answer the following question.
    Assume that you are presented with two trays of black and white marbles: a large tray that contains 100 marbles and a small tray that contains 10 marbles. The marbles are spread in a single layer on each tray. You must draw out one marble (without peeking, of course) from either tray. If you draw a black marble, you win $2. Consider a condition in which the small tray contains 1 black marble and 9 white marbles, and the large tray contains 8 black marbles and 92 white marbles. [A drawing of two trays with their corresponding numbers of marbles arranged neatly in 10-marble rows appeared above the previous sentence.] From which tray would you prefer to select a marble in a real situation? 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)






def run_economics(model):
    # questions
    questions="""
    1. How does one still call an instant camera?
    a. Canon camera 
    b. Polaroid camera 
    c. Minolta camera
    2. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    3. Where do flounders mostly live? 
    a. In coral reef 
    b. Dug on the ground 
    c. In the reed
    4. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    5. What does the rollmops consist of?
    a. Herring filet 
    b. Pork 
    c. Salmon filet
    6. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    7. Which land does the Nobel Prize winner in Literature Gabriel García Márquez
    come from?
    a. Colombia 
    b. Spain 
    c. Venezuela
    8. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    9. Which style movement does anacreontics belong to?
    a. Rococo 
    b. Romanticism 
    c. Realism
    10. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    11. What is a hot chili sauce?
    a. Tabasco 
    b. Curacao 
    c. Macao
    12. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    13. How many letters does the Russian alphabet consist of?
    a. 40 
    b. 33 
    c. 26
    14. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    15. "Tosca" is an opera from ...?
    a. G. Puccini 
    b. G. Verdi 
    c. A. Vivald
    16. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    17. What is the name of the Greek Goddess of wisdom?
    a. Pallas Athena 
    b. Nike 
    c. Penelope
    18. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    19. Which is the most abundant metal on the Earth?
    a. Iron 
    b. Aluminum 
    c. Copper
    20. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    21. How does one call an unknowing person?
    a. Ignatius 
    b. Ignorant 
    c. Ideologue
    22. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    23. Who flew for the first time with an airship around the Eiffel Tower?
    a. Santos-Dumont 
    b. Count Zeppelin 
    c. Saint-Exupéry
    24. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    25. How is the snow shelter of Eskimos called?
    a. Wigwam 
    b. Igloo 
    c. Tipi
    26. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    27. Which enterprise does Bill Gates belong to?
    a. Intel 
    b. Microsoft 
    c. Dell Computers
    28. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    29. How is the fasting month in Islam called?
    a. Sharia 
    b. Ramadan 
    c. Imam
    30. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    31. Which language does the concept "Fata Morgana" come from?
    a. Italian
    b. Arabic 
    c. Swahili
    32. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    33. How many days does a hen need to incubate an egg?
    a. 21 days
    b. 14 days 
    c. 28 days
    34. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.

    35. What is ascorbic acid?
    a. Apple vinegar 
    b. Vitamin C 
    c. Vitamin A
    36. How confident are you that your answer is correct? Please provide your answer as a percentage, ranging from 0 to 100%.
    """.strip()
    prompt0="""
    Please act like a participant in this survey and answer the following questions.
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break
    # 
    prompt="""
    Please act like a participant in this survey and answer the following questions.
    1. Now, imagine you have a choice between the following two options:
    Option A: A lottery with a 50% chance of winning $80 and a 50% chance of losing $50. 
    Option B: Zero dollars.
    Which option would you choose?
    a. Option A
    b. Option B
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)
    # 
    prompt="""
    Please act like a participant in this survey and answer the following questions.
    2. Now, imagine you have a choice between the following two options:
    Option A: Play the lottery from the previous question (50% chance of winning 80%, 50% chance of losing $50) six times.
    Option B: Zero dollars.
    Which option would you choose?
    a. Option A
    b. Option B
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # questions
    questions="""
    1. Do you prefer $3,400 this month or $3,800 next month?
    a. I strongly prefer $3,400 this month.
    b. I slightly prefer $3,400 this month.
    c. I prefer $3,400 this month.
    d. I prefer $3,800 next month.
    e. I slightly prefer $3,800 next month.
    f. I strongly prefer $3,800 next month.

    2. Do you prefer $100 now or $140 next year?
    a. I strongly prefer $100 this month.
    b. I slightly prefer $100 this month.
    c. I prefer $100 this month.
    d. I prefer $140 next month.
    e. I slightly prefer $140 next month.
    f. I strongly prefer $140 next month.

    3. Do you prefer $100 now or $1,100 in 10 years?
    a. I strongly prefer $100 now.
    b. I slightly prefer $100 now. 
    c. I prefer $100 now. 
    d. I prefer $1,100 in 10 years. 
    e. I slightly prefer $1,100 in 10 years. 
    f. I strongly prefer $1,100 in 10 years.

    4. Do you prefer $9 now or $100 in 10 years?
    a. I strongly prefer $9 now. 
    b. I slightly prefer $9 now.
    c. I prefer $9 now. 
    d. I prefer $100 in 10 years. 
    e. I slightly prefer $100 in 10 years. 
    f. I strongly prefer $100 in 10 years.

    5. Do you prefer $40 immediately or $1,000 in 10 years?
    a. I strongly prefer $40 immediately. 
    b. I slightly prefer $40 immediately. 
    c. I prefer $40 immediately. 
    d. I prefer $1,000 in 10 years. 
    e. I slightly prefer $1,000 in 10 years. 
    f. I strongly prefer $1,000 in 10 years.

    """.strip()
    len(questions.split('\n\n'))

    prompt0="""
    Please act like a participant in this survey. For each question below, please select the response that best aligns with your preference.
    <question>
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break

    # questions
    questions="""
    1. Which of the following set of lottery numbers has the greatest probability of being selected as the winning combination?
    a. 1,2,3,4,5,6
    b. 8, 18, 3, 55, 32, 28
    c. Each of the above has an equal probability of being selected

    2. Which gives you the best chance of winning the jackpot on a slot machine?
    a. Playing a slot machine that has not had a jackpot in over a month.
    b. Playing a slot machine that had a jackpot an hour ago.
    c. Your chances of winning the jackpot are the same on both machines.

    3. How lucky are you? If 10 people’s names were put into a hat and one name drawn for a prize, how likely is it that your name would be chosen?
    a. About the same likelihood as everyone else 
    b. Less likely than other people 
    c. More likely than other people 

    4. If you were to buy a lottery ticket, which would be the best place to buy it from?
    a. A place that has sold many previous winning tickets
    b. A place that has sold few previous winning tickets 
    c. One place is as good as another

    5. A positive attitude or doing good deeds increases your likelihood of winning money when gambling. 
    a. Disagree 
    b. Agree 

    6. A gambler goes to the casino and wins 75% of the time. How many times has he or she likely gone to the casino?
    a. 4 times 
    b. 100 times 
    c. It is just as likely that he has gone either 4 or 100 times 

    7. You go to a casino with $100 hoping to double your money. Which strategy gives you the best chance of doing this? 
    a. Betting all your money on a single bet 
    b. Betting small amounts of money on several different 
    bets 
    c. Either strategy gives you an equal chance of doubling your money. 

    8. Which game can you consistently win money at if you use the right strategy?
    a. Slot machines 
    b. Roulette 
    c. Bingo (Baccarat for Asian populations) 
    d. None of the above 

    9. Your chances of winning a lottery are better if you are able to choose your own numbers.
    a. Disagree 
    b. Agree

    10. You have flipped a coin and correctly guessed ‘heads’ 5 times in a row. What are the odds that heads will come up on the next flip. Would you say…
    a. 50% 
    b. More than 50% 
    c. Less than 50% 
    """.strip()
    len(questions.split('\n\n'))

    prompt0="""
    Please act like a participant in this survey and answer the following questions.
    <question>
    Your answer:
    """
    for q in questions.split('\n\n'):
        print('-------------------------')
        prompt=prompt0.replace('<question>',q.strip()).strip()
        res=get_completion(prompt, model,temperature=T)
    #     print(prompt)
        print(res)
    #     break

    # 
    prompt="""
    Please act like a participant and answer the following question.
    After the first 2 weeks of the major league baseball season, newspapers begin to print the top 10 batting averages. Typically, after 2 weeks, the leading batter often has an average of about .450. However, no batter in major league history has ever averaged .450 at the end of the season. Why do you think this is? Circle one: 
    a. When a batter is known to be hitting for a high average, pitchers bear down more when they pitch to him. 
    b. Pitchers tend to get better over the course of a season, as they get more in shape. As pitchers improve, they are more likely to strike out batters, so batters’ averages go down. 
    c. A player’s high average at the beginning of the season may be just luck. The longer season provides a more realistic test of a batter’s skill.
    d. A batter who has such a hot streak at the beginning of the season is under a lot of stress to maintain his performance record. Such stress adversely affects his playing. 
    e. When a batter is known to be hitting for a high average, he stops getting good pitches to hit. Instead, pitchers “play the corners” of the plate because they don’t mind walking him. 
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)

    # WTA
    prompt="""
    Please act like a participant in this study. You currently own a university coffee mug, which is priced at $6 in the local university bookstore. You have the option of selling it and receiving money for it. 
    For each listed price, please indicate whether you would wish to keep the mug or sell it at that price.
    1. Would you sell or keep the mug if the offering price is $0?
    a. Sell
    b. Keep
    2. Would you sell or keep the mug if the offering price is $0.5?
    a. Sell
    b. Keep
    3. Would you sell or keep the mug if the offering price is $1?
    a. Sell
    b. Keep
    4. Would you sell or keep the mug if the offering price is $1.5?
    a. Sell
    b. Keep
    5. Would you sell or keep the mug if the offering price is $2?
    a. Sell
    b. Keep
    6. Would you sell or keep the mug if the offering price is $2.5?
    a. Sell
    b. Keep
    7. Would you sell or keep the mug if the offering price is $3?
    a. Sell
    b. Keep
    8. Would you sell or keep the mug if the offering price is $3.5?
    a. Sell
    b. Keep
    9. Would you sell or keep the mug if the offering price is $4?
    a. Sell
    b. Keep
    10. Would you sell or keep the mug if the offering price is $4.5?
        a. Sell
        b. Keep
    11. Would you sell or keep the mug if the offering price is $5?
        a. Sell
        b. Keep
    12. Would you sell or keep the mug if the offering price is $5.5?
        a. Sell
        b. Keep
    13. Would you sell or keep the mug if the offering price is $6?
        a. Sell
        b. Keep
    14. Would you sell or keep the mug if the offering price is $6.5?
        a. Sell
        b. Keep
    15. Would you sell or keep the mug if the offering price is $7?
        a. Sell
        b. Keep
    16. Would you sell or keep the mug if the offering price is $7.5?
        a. Sell
        b. Keep
    17. Would you sell or keep the mug if the offering price is $8?
        a. Sell
        b. Keep
    18. Would you sell or keep the mug if the offering price is $8.5?
        a. Sell
        b. Keep
    19. Would you sell or keep the mug if the offering price is $9?
        a. Sell
        b. Keep
    20. Would you sell or keep the mug if the offering price is $9.5?
        a. Sell
        b. Keep
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)
    # WTP 
    prompt="""
    Please act like a participant in this study. You currently do not own a university coffee mug, which is priced at $6 in the local university bookstore. You have the option of buying one by paying money for it. 
    For each listed price, please indicate whether you would wish to buy the mug or not at that price.

    1. Would you buy the mug if the selling price is $0?
    a. Buy
    b. Not Buy
    2. Would you buy the mug if the selling price is $0.5?
    a. Buy
    b. Not Buy
    3. Would you buy the mug if the selling price is $1?
    a. Buy
    b. Not Buy
    4. Would you buy the mug if the selling price is $1.5?
    a. Buy
    b. Not Buy
    5. Would you buy the mug if the selling price is $2?
    a. Buy
    b. Not Buy
    6. Would you buy the mug if the selling price is $2.5?
    a. Buy
    b. Not Buy
    7. Would you buy the mug if the selling price is $3?
    a. Buy
    b. Not Buy
    8. Would you buy the mug if the selling price is $3.5?
    a. Buy
    b. Not Buy
    9. Would you buy the mug if the selling price is $4?
    a. Buy
    b. Not Buy
    10. Would you buy the mug if the selling price is $4.5?
        a. Buy
        b. Not Buy
    11. Would you buy the mug if the selling price is $5?
        a. Buy
        b. Not Buy
    12. Would you buy the mug if the selling price is $5.5?
        a. Buy
        b. Not Buy
    13. Would you buy the mug if the selling price is $6?
        a. Buy
        b. Not Buy
    14. Would you buy the mug if the selling price is $6.5?
        a. Buy
        b. Not Buy
    15. Would you buy the mug if the selling price is $7?
        a. Buy
        b. Not Buy
    16. Would you buy the mug if the selling price is $7.5?
        a. Buy
        b. Not Buy
    17. Would you buy the mug if the selling price is $8?
        a. Buy
        b. Not Buy
    18. Would you buy the mug if the selling price is $8.5?
        a. Buy
        b. Not Buy
    19. Would you buy the mug if the selling price is $9?
        a. Buy
        b. Not Buy
    20. Would you buy the mug if the selling price is $9.5?
        a. Buy
        b. Not Buy
    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    print(res)
    # 
    prompt="""
    1. Suppose you are at a big store, where you intend to purchase an oven. The model you’ve selected is priced at is 2000 TWD (New Taiwan Dollar), and you are about to pay. However, at the last minute, you notice an advertisement flyer featuring the same oven, at a price of 1700 TWD. The discount offer is only valid for today. You’ll need to drive 10 min to buy it in a competing store. Would you like to take the bus to the other store to take advantage of the lower price? 
    a. Yes
    b. No

    2. Now suppose you are in the same store, this time to buy a refrigerator. The refrigerator you want costs 30,000 TWD, and you are willing to pay. While you are waiting, you strike up a conversation with another store patron, who reveals that she has seen the same refrigerator available for 29,700 TWD at a competing local store about 10 min drive away. Will you drive to the other store to obtain the lower price? 
    a. Yes
    b. No


    """
    prompt=prompt.strip()
    res=get_completion(prompt, model,temperature=T)
    # res=get_completion(prompt, 'gpt-4',temperature=T)
    print(res)










def run_game_theory(model):
    filepath=r'./records/'

    ######### 1. second price auction
    def correct_json(s):
        prompt=f"""
    The following string delimited by triple backticks is in json format, but there are some mistakes, and I cannot directly convert it to json by json.loads(). For example, there may be missing comma or quotes. Please help me correct the mistakes, and output the string in valid json format. Please only output the corrected string, and do not output any other things.
    ```
    {s}
    ```
        """.strip()
        response = get_completion(prompt)
        return response

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  
            end_index = response.rfind('}')  
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{') 
            end_index = response.find('}') 
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  
                    end_index = corrected_json.rfind('}')  
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                assert 0
        return out

    def get_action(bot,model,display=False):
        message=f"""
    Now let's start the auction. Tell me how you think and the bid you would like to place. Please answer in json format with keys 'thought' and 'bid'. For example, {{"thought": "xxx","bid": x}}. 
    """.strip()
    #     The 'bid' should be a number.
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        action=out['bid']
        
        reason=out['thought']
        try:
            action=float(action)
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason
    nsess=10

    max_try=50
    display_result=0

    system_message_0 = """
    Please act as a human bidder in an auction. You are participating in an auction with another bidder. There is only one item, and your private value of the item is {} points. You do not know the private value of the other bidder. 
    You and the other bidder will simultaneously place a bid (can be 0 or any positive number). The bidder who places the higher bid will get the item (the other will not), and only need to pay a number of points equaling to the second-highest bid among all the bids.
    If you get the item, your payoff is calculated as: payoff = your value of the item – the number of points you pay
    If you do not get the item, your payoff is zero.
    Your goal is to maximize your own payoff. You should first think carefully about what to do, and then make a decision.
    """.strip()

    for value in range(10,101,10):
        system_message=system_message_0.format(value)
        
        for sess in range(1,1+nsess):
            print(f'==============value={value},sess{sess}==============')

            recordfile=os.path.join(filepath,f'records_{expname}_value{value}_sess{sess}.pkl')
            if os.path.exists(recordfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                print('existed!')
                continue

            starttime=time.time()
            for _ in range(max_try):
                try:
                    bot=ChatApp(system_message=system_message)
                    action,reason=get_action(bot,model=model,display=display_result)
                    records=[{'action':action,'reason':reason}]
                    print('time:',time.time()-starttime)
                    # save file
                    recordfile=os.path.join(filepath,f'records_{expname}_value{value}_sess{sess}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)
    # calculate payoffdeviation from NE
    nsess=10
    nround=10
    round=nround

    result=[]
    for value in range(10,101,10):
        for sess in range(1,nsess+1):
            recordfile=os.path.join(filepath,f'records_{expname}_value{value}_sess{sess}.pkl')
            with open(recordfile, 'rb') as f:
                records= pickle.load(f)
            for record in records:
                tmp=record['action']
    #             print(tmp)
                result.append(abs(tmp-value)/value)
            
    dev=np.mean(result)
    print(dev)


    ######### 2.beauty contest
    max_number=30

    system_message = f"""
    Please act as a human player. You are playing a game with another human player (called the opponent) for only one round. You and the opponent will simultaneously choose a number between 0 and {max_number} (including 0 and {max_number}). The winner is the player whose number is closest to two thirds of the average of all chosen numbers. The winner will earn 100 points, and the other player will earn nothing. If there is a tie, both players will earn 50 points.
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then make a decision.
    """.strip()
    def correct_json(s):
        prompt=f"""
    The following string delimited by triple backticks is in json format, but there are some mistakes, and I cannot directly convert it to json by json.loads(). For example, there may be missing comma or quotes. Please help me correct the mistakes, and output the string in valid json format. Please only output the corrected string, and do not output any other things.
    ```
    {s}
    ```
        """.strip()
        response = get_completion(prompt,model=model)
        return response
    s="""{
    "thought": "aaa"
    "action": J
    }"""
    # print(correct_json(s))


    summarize_action=f"""Given the following paragraph delimited by triple backticks:
    ```
    <out>
    ```
    Please summarize how he thought, and the number he chose in the first person from above paragraph in json format with keys 'thought' and 'number'. The 'number' should be between 0 and 100 (including 0 and 100).
    """.strip()
    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                out=json.loads(out)
    #             print('No json found in:',response)
    #             assert 0
        return out


    def get_action(bot,model,display=False):
        message=f"""
    Now let's start the game. Tell me how you think and the number you would like to choose. Please answer in json format with keys 'thought' and 'number'. The 'number' should be between 0 and {max_number} (including 0 and {max_number}).
    """.strip()
    #     message=f"""
    # Now let's start the game. Tell me how you think and the number you would like to choose.
    # """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        action=out['number']
        
        reason=out['thought']
        try:
            action=float(action)
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason

    nsess=10

    expname='llama13b'

    max_try=50
    display_result=0

    system_message_0 = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for only one round. You and the opponent will simultaneously choose a number between 0 and {} (including 0 and {}). The winner is the player whose number is closest to two thirds of the average of all chosen numbers. The winner will earn 100 points, and the other player will earn nothing. If there is a tie, both players will earn 50 points.
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then make a decision.
    """.strip()
    for max_number in range(10,101,10):
        system_message=system_message_0.format(max_number,max_number)
    #     print(system_message)
        
        for sess in range(1,1+nsess):
            print(f'==============max_number={max_number},sess{sess}==============')

            recordfile=os.path.join(filepath,f'records_{expname}_max{max_number}_sess{sess}.pkl')
            if os.path.exists(recordfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                print('existed!')
                continue

            starttime=time.time()
            for _ in range(max_try):
                try:
                    bot=ChatApp(system_message=system_message)
                    action,reason=get_action(bot,model=model,display=display_result)
                    records=[{'action':action,'reason':reason}]
                    print('time:',time.time()-starttime)
                    # save file
                    recordfile=os.path.join(filepath,f'records_{expname}_max{max_number}_sess{sess}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)   
    # payoffdeviation from NE
    nsess=10
    nround=10
    round=nround


    result=[]
    for max_number in range(10,101,10):
        for sess in range(1,nsess+1):
            recordfile=os.path.join(filepath,f'records_{expname}_max{max_number}_sess{sess}.pkl')
            with open(recordfile, 'rb') as f:
                records= pickle.load(f)
            for record in records:
                tmp=record['action']
    #             print(tmp)
                result.append(tmp/max_number)
            
    print(len(result),np.mean(result))
    dev=np.mean(result)
    print(dev)     

    ######### 3. One-shot prisoner’s dilemma
    system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for only one round. You and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then choose one of the two actions: F or J.
    """.strip()


    aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points.'
    ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points.'
    ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points.'
    bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points.'
    payoff=[aa,ab,ba,bb]
    payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]

    tmp_system_message=system_message.format(payoff_str=payoff_str)
    print(tmp_system_message)
    def correct_json(s):
        tmp=s.split('"')
        newlist=[]
        for x in tmp:
            striped=x.strip()
            if striped=='': # 缺少逗号
                newlist.append(',')
            elif striped[0]==':' and striped!=':':
                newlist.extend([':',striped[1:-1].strip(),'}'])
            else:
                newlist.append(striped)
        newstr='"'.join(newlist)
        try:
            a=json.loads(newstr)
        except:
            prompt=f"""
    The following string delimited by triple backticks is in json format, but there are some mistakes, and I cannot directly convert it to json by json.loads(). For example, there may be missing comma or quotes. Please help me correct the mistakes, and output the string in valid json format. Please only output the corrected string, and do not output any other things.
    ```
    {s}
    ```
            """.strip()
            response = gpt_completion(prompt)
            return response
        return newstr

    s="""{
    "thought": "aaa"
    "action": J
    }"""
    # print(correct_json(s))

    summarize_action=f"""Given the following paragraph delimited by triple backticks:
    ```
    <out>
    ```
    Please summarize how he thought and the action he choose (F or J) in the first person from above paragraph in json format with keys 'thought' and 'action'. The 'action' should be F or J.
    """.strip()
    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                print('Summarize json:',out)
                out=json.loads(out,strict=0)
        return out

    def get_action(bot,model,display=False):
        message=f"""
    Now let's start the game. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. For example, {{"thought": "xxx","action": x}}. The 'action' should be F or J. 
    """.strip()
    #     For example, {{"thought": "xxx","action": x}}. The 'action' should be F or J. 
    #     message=f"""
    # Now let's start the game. Tell me how you think and the action you would like to choose.
    # """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        try:
            action=out['action']
            reason=out['thought']
        except:
            print('key error:',res)
            prompt=summarize_action.replace('<out>',res)
            out=gpt_completion(prompt)
            out=parse_response(out)
            print('corrected:',out)
            action=out['action']
            reason=out['thought']
            
        try:
            assert action in ['F','J']
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason
    nsess=50

    expname='llama13b'


    max_try=50
    display_result=0

    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for only one round. You and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then choose one of the two actions: F or J.
    """.strip()

        aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points.'
        ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points.'
        ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points.'
        bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points.'
        payoff=[aa,ab,ba,bb]
        
        ######################## init players and records
        N=2
        players=[]
        payoff=[aa,ab,ba,bb]
        for _ in range(N):
            random.shuffle(payoff)
            payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]
            tmp_system_message=system_message.format(payoff_str=payoff_str)
        #     print(tmp_system_message)
            bot=ChatApp(system_message=tmp_system_message)
            players.append(bot)
        records=[]
        for _ in range(N):    
            df = pd.DataFrame(columns=['Round','Your choice','Co-player choice','Earnings','Reason of choice'])
            records.append(df)

        ######################## start game
        # if this sess done, continue
        recordfile=os.path.join(filepath,f'records_{expname}_sess{sess}.pkl')
        playerfile=os.path.join(filepath,f'players_{expname}_sess{sess}.pkl')
        if os.path.exists(recordfile) and os.path.exists(playerfile):
            with open(recordfile, 'rb') as f:
                records = pickle.load(f)
            with open(playerfile, 'rb') as f:
                players = pickle.load(f)
            print('existed!')
            continue
        starttime=time.time()
        for _ in range(max_try):
            try:
                # play
                tmp_records=[]
                for i in tqdm(range(N)):
                    bot=players[i]
                    record=records[i]
                    action,reason=get_action(bot,model=model,display=display_result)
                    tmp_records.append([1,action,'NULL',0,reason])
                    # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']

                # calculate total public
                c1,c2=tmp_records[0][1],tmp_records[1][1]
                if c1=='J' and c2=='J':
                    e1,e2=40,40
                elif c1=='J' and c2=='F':
                    e1,e2=12,50
                elif c1=='F' and c2=='J':
                    e1,e2=50,12
                elif c1=='F' and c2=='F':
                    e1,e2=25,25
                else:
                    print(c1,c2)
                    assert False
                # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']
                tmp_records[0][2]=c2
                tmp_records[1][2]=c1
                tmp_records[0][3]=e1
                tmp_records[1][3]=e2

                # update records for all
                for i in range(N):
                    record=records[i]
                    record.loc[len(record.index)]=tmp_records[i]
                print('time:',time.time()-starttime)

                # save file
                recordfile=os.path.join(filepath,f'records_{expname}_sess{sess}.pkl')
                playerfile=os.path.join(filepath,f'players_{expname}_sess{sess}.pkl')
                with open(recordfile, 'wb') as f:
                    pickle.dump(records, f)
                with open(playerfile, 'wb') as f:
                    pickle.dump(players, f)
                break
            except Exception as e:
                print(e)
                print('error!sleep 30s...')
                time.sleep(30)   
    # NE choice rate (defect rate)

    allaction=[]

    for sess in range(1,1+nsess):    
        recordfile=os.path.join(filepath,f'records_{expname}_sess{sess}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)
        for record in records:
            allaction.extend(record['Your choice'].tolist())


    out=allaction.count('F')/len(allaction)
    print(out)

    ## 4. Finitely repeated prisoner’s dilemma
    nround=10

    system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
    """.strip()

    aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points in this round.'
    ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points in this round.'
    ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points in this round.'
    bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points in this round.'
    payoff=[aa,ab,ba,bb]
    payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]

    tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
    print(tmp_system_message)
    def get_history(record):
        history=''
        for i in record.index:
            round=i+1
            choice=record.loc[i,'Your choice']
            cochoice=record.loc[i,'Co-player choice']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{round}, you chose {choice} and the opponent chose {cochoice}, you earn {earning} points."""
            history+=tmp
            history+='\n'
        return history
    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                assert 0
        return out

    def get_action(bot,record,model,display=False):
        history=get_history(record)
        if len(record)==0:
            message=f"""
    It is round-{round} out of {nround} rounds now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. The 'action' should be F or J. 
    """.strip()
    #         For example, {{"thought": "xxx","action": x}}. The 'action' should be F or J. 
        else:
            message=f"""The history of the game is listed as follows delimited by triple backticks.
    ```
    {history}
    ```
    It is round-{round} out of {nround} rounds now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. The 'action' should be F or J. 
    """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(3)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        try:
            action=out['action']
            reason=out['thought']
        except:
            print('key error:',res)
            prompt=summarize_action.replace('<out>',res)
            out=gpt_completion(prompt)
            out=parse_response(out)
            print('corrected:',out)
            action=out['action']
            reason=out['thought']
    #         assert 0
        try:
            assert action in ['F','J']
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason
    nround=10
    nsess=10

    expname='llama13b'


    max_try=50
    display_result=0


    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
        """.strip()

        aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points in this round.'
        ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points in this round.'
        ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points in this round.'
        bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points in this round.'
        payoff=[aa,ab,ba,bb]
        
        ######################## init players and records
        N=2
        players=[]
        payoff=[aa,ab,ba,bb]
        for _ in tqdm(range(N)):
            random.shuffle(payoff)
            payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]
            tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
        #     print(tmp_system_message)
            bot=ChatApp(system_message=tmp_system_message)
            players.append(bot)
        records=[]
        for _ in range(N):    
            df = pd.DataFrame(columns=['Round','Your choice','Co-player choice','Earnings','Reason of choice'])
            records.append(df)

        ######################## start game

        starttime=time.time()
        for round in range(1,nround+1):
            print('-----------------------')
            print(f'round-{round}')

            # if this round done, continue
            recordfile=os.path.join(filepath,f'records_finPD_{expname}_{nround}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_finPD_{expname}_{nround}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

    #         for _ in range(max_try):
    #             try:
            if round>1:
                prev_round=round-1
                recordfile=os.path.join(filepath,f'records_finPD_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                playerfile=os.path.join(filepath,f'players_finPD_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
            # play
            tmp_records=[]
            for i in tqdm(range(N)):
                bot=players[i]
                record=records[i]
                action,reason=get_action(bot,record,model=model,display=display_result)
                tmp_records.append([round,action,'NULL',0,reason])
                # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']

            # calculate total public
            c1,c2=tmp_records[0][1],tmp_records[1][1]
            if c1=='J' and c2=='J':
                e1,e2=40,40
            elif c1=='J' and c2=='F':
                e1,e2=12,50
            elif c1=='F' and c2=='J':
                e1,e2=50,12
            elif c1=='F' and c2=='F':
                e1,e2=25,25
            else:
                print(c1,c2)
                assert False
            # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']
            tmp_records[0][2]=c2
            tmp_records[1][2]=c1
            tmp_records[0][3]=e1
            tmp_records[1][3]=e2

            # update records for all
            for i in range(N):
                record=records[i]
                record.loc[len(record.index)]=tmp_records[i]
            print('time:',time.time()-starttime)

            # save file
            recordfile=os.path.join(filepath,f'records_finPD_{expname}_{nround}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_finPD_{expname}_{nround}_sess{sess}_round{round}.pkl')
            with open(recordfile, 'wb') as f:
                pickle.dump(records, f)
            with open(playerfile, 'wb') as f:
                pickle.dump(players, f)
    # NE choice rate (defect rate)
    nsess=10
    nround=10
    round=nround

    allaction=[]

    for sess in range(1,1+nsess):    
        recordfile=os.path.join(filepath,f'records_finPD_{expname}_{nround}_sess{sess}_round{round}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)
        for record in records:
            allaction.extend(record['Your choice'].tolist())


    out=allaction.count('F')/len(allaction)
    print(out)


    ## 5. One-shot public goods game
    N,G,K,Z=4, 1.2, 0.3, 100 

    assert abs(G-N*K)<0.000001

    system_message = f"""
    Please act as a human player. You and {N-1} other players are playing a game for only one round. There is a shared public account and each player has a private account. Each player will be given {Z} points. Then, all players simultaneously allocate these points into private account and public account. 
    The points you allocate to your private account will be exchanged for earnings at a rate of 1:1, and these earnings will be received only by you. The total number of points in the public account equals to the sum of points allocated to the public account by all players (including yourself). These points will be exchanged for public earnings at a rate of 1:{G}, and these earnings will be equally shared among all {N} players, which means that each point in the public account will yield an earning of {K} points for each player. 
    In sum, your earnings can be described as: Your earnings = (Points in your private account * 1) + (Total points in public account * {K})
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then make a decision.
    """.strip()

    print(system_message)

    summarize_action=f"""Given the following paragraph delimited by triple backticks:
    ```
    <out>
    ```
    Please summarize how he thought, and the number of points allocated in private/public account in the first person from above paragraph in json format with keys 'thought', 'private' and 'public'. The value of 'private' and 'public' should be a number.
    """.strip()
    summarize_action=f"""Given the following paragraph delimited by triple backticks:
    ```
    <out>
    ```
    Please summarize how he thought, and the number of points allocated in private/public account in the first person from above paragraph in json format with keys 'thought', 'private' and 'public'. The value of 'private' and 'public' should be a number.
    """.strip()

    ###########################
    nsess=20

    expname='llama13b'

    max_try=50
    display_result=0

    system_message = f"""
    Please act as a human player. You and {N-1} other players are playing a game for only one round. There is a shared public account and each player has a private account. Each player will be given {Z} points. Then, all players simultaneously allocate these points into private account and public account. 
    The points you allocate to your private account will be exchanged for earnings at a rate of 1:1, and these earnings will be received only by you. The total number of points in the public account equals to the sum of points allocated to the public account by all players (including yourself). These points will be exchanged for public earnings at a rate of 1:{G}, and these earnings will be equally shared among all {N} players, which means that each point in the public account will yield an earning of {K} points for each player. 
    In sum, your earnings can be described as: Your earnings = (Points in your private account * 1) + (Total points in public account * {K})
    Your goal is to maximize your own earnings. You should first think carefully about what to do, and then make a decision.
    """.strip()

    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        
        ######################## init players and records
        players=[]
        for _ in tqdm(range(N)):
            bot=ChatApp(system_message=system_message)
            players.append(bot)
        records=[]
        cols=['Period','Private','Public','Total public','Earnings','Reason']
        for _ in range(N):    
            df = pd.DataFrame(columns=cols)
            records.append(df)

        ######################## start game
        # if this sess done, continue
        recordfile=os.path.join(filepath,f'records_{expname}_{N}_{G}_{K}_{Z}_sess{sess}.pkl')
        playerfile=os.path.join(filepath,f'players_{expname}_{N}_{G}_{K}_{Z}_sess{sess}.pkl')
        if os.path.exists(recordfile) and os.path.exists(playerfile):
            with open(recordfile, 'rb') as f:
                records = pickle.load(f)
            with open(playerfile, 'rb') as f:
                players = pickle.load(f)
            print('existed!')
            continue
        starttime=time.time()
        for _ in range(max_try):
            try:
                # play
                tmp_records=[]
                for i in tqdm(range(N)):
                    bot=players[i]
                    record=records[i]
                    private,public,reason=get_action(bot,model=model,display=display_result)
                    tmp_records.append([1,private,public,0,0,reason])
                    # ['Period','Private','Public','Total public','Earnings','Reason']

                # calculate total public
                total_public=sum([x[2] for x in tmp_records])

                # calculate earnings for all
                for r in tmp_records:
                    r[3]=total_public
                    r[4]=r[1]+total_public*K

                # update records for all
                for i in range(N):
                    record=records[i]
                    record.loc[len(record.index)]=tmp_records[i]

                print('time:',time.time()-starttime)


                # save file
                recordfile=os.path.join(filepath,f'records_{expname}_{N}_{G}_{K}_{Z}_sess{sess}.pkl')
                playerfile=os.path.join(filepath,f'players_{expname}_{N}_{G}_{K}_{Z}_sess{sess}.pkl')
                with open(recordfile, 'wb') as f:
                    pickle.dump(records, f)
                with open(playerfile, 'wb') as f:
                    pickle.dump(players, f)
                break
            except Exception as e:
                print(e)
                print('error!sleep 30s...')
                time.sleep(30)   

    nsess=20

    allprivate=[]
    for sess in range(1,nsess+1):
        recordfile=os.path.join(filepath,f'records_{expname}_{N}_{G}_{K}_{Z}_sess{sess}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)

        for record in records:
            tmp=record['Private'].tolist()
            allprivate.extend(tmp)
    out=np.mean(allprivate)
    print(out)


    ## 6. Finitely repeated public goods game
    N,G,K,Z=4, 1.2, 0.3, 100

    nperiod=10

    assert abs(G-N*K)<0.000001

    system_message = f"""
    Please act as a human player. You and {N-1} other players are playing a repeated game for {nperiod} rounds. In each round, there is a shared public account and each player has a private account. Each player will be given {Z} points. Then, all players simultaneously allocate these points into private account and public account. 
    The points you allocate to your private account will be exchanged for earnings at a rate of 1:1, and these earnings will be received only by you. The total number of points in the public account equals to the sum of points allocated to the public account by all players (including yourself). These points will be exchanged for public earnings at a rate of 1:{G}, and these earnings will be equally shared among all {N} players, which means that each point in the public account will yield an earning of {K} points for each player. 
    In sum, your earnings in each round can be described as: Your earnings = (Points in your private account * 1) + (Total points in public account * {K})
    Your goal is to maximize your own total earnings in all {nperiod} rounds. In each round, you should first think carefully about what to do, and then make a decision.
    """.strip()

    def get_history(record):
        history=''
        for i in record.index:
            period=i+1
            public=record.loc[i,'Public']
            total_public=record.loc[i,'Total public']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{period}, you allocated {public} points in public account, the total points in public account were {total_public}, your total earnings were {earning} points."""
            history+=tmp
            history+='\n'
        return history

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                assert 0
        return out
    def get_action(bot,record,model,display=False):
        history=get_history(record)
        if len(record)==0:
            message=f"""
    It is round-{period} out of {nperiod} rounds now. You are given {Z} points. Tell me how you think and how much you would like to allocate to the private and public account, respectively. Please answer in json format with keys 'thought', 'private', and 'public'. For example, {{"thought": "I think...","private": x,"public": 100-x }}. The sum of private and public should equal {Z}. 
    """.strip()
        else:
            message=f"""The history of the game is listed as follows delimited by triple backticks.
    ```
    {history}
    ```
    It is round-{period} out of {nperiod} rounds now. You are given {Z} points. Tell me how you think and how much you would like to allocate to the private and public account, respectively. Please answer in json format with keys 'thought', 'private', and 'public'. For example, {{"thought": "I think...","private": x,"public": 100-x }}. The sum of private and public should equal {Z}. 
    """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        
        try:
            private=out['private']
            public=out['public']
            reason=out['thought']
            private,public=int(private),int(public)
        except:
            print('key error:',res)
            prompt=summarize_action.replace('<out>',res)
            out=gpt_completion(prompt)
            out=parse_response(out)
            print('corrected:',out)
            private=out['private']
            public=out['public']
            reason=out['thought']
        
        try:
            assert abs(private+public-Z)<0.0001
        except:

            if private>=0 and private<=100:
                public = 100-private
            else:
                print(message)
                print('private+public incorrect\n',out)
                assert False
        
        return private,public,reason
    nsess=10

    expname='llama13b'

    max_try=50
    display_result=0

    N,G,K,Z=4, 1.2, 0.3, 100 

    nperiod=10

    assert abs(G-N*K)<0.000001

    system_message = f"""
    Please act as a human player. You and {N-1} other players are playing a repeated game for {nperiod} rounds. In each round, there is a shared public account and each player has a private account. Each player will be given {Z} points. Then, all players simultaneously allocate these points into private account and public account. 
    The points you allocate to your private account will be exchanged for earnings at a rate of 1:1, and these earnings will be received only by you. The total number of points in the public account equals to the sum of points allocated to the public account by all players (including yourself). These points will be exchanged for public earnings at a rate of 1:{G}, and these earnings will be equally shared among all {N} players, which means that each point in the public account will yield an earning of {K} points for each player. 
    In sum, your earnings in each round can be described as: Your earnings = (Points in your private account * 1) + (Total points in public account * {K})
    Your goal is to maximize your own total earnings in all {nperiod} rounds. In each round, you should first think carefully about what to do, and then make a decision.
    """.strip()


    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        
        players=[]
        for _ in tqdm(range(N)):
            bot=ChatApp(system_message=system_message)
            players.append(bot)
        records=[]
        cols=['Period','Private','Public','Total public','Earnings','Reason']
        for _ in range(N):    
            df = pd.DataFrame(columns=cols)
            records.append(df)

        ######################## start game

        starttime=time.time()
        for period in range(1,nperiod+1):
            print('-----------------------')
            print(f'period-{period}')

            # if this block done, continue
            recordfile=os.path.join(filepath,f'records_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{period}.pkl')
            playerfile=os.path.join(filepath,f'players_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{period}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            for _ in range(max_try):
                try:
                    if period>1:
                        prev_period=period-1
                        recordfile=os.path.join(filepath,f'records_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{prev_period}.pkl')
                        playerfile=os.path.join(filepath,f'players_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{prev_period}.pkl')
                        with open(recordfile, 'rb') as f:
                            records = pickle.load(f)
                        with open(playerfile, 'rb') as f:
                            players = pickle.load(f)
                    # play
                    tmp_records=[]
                    for i in tqdm(range(N)):
                        bot=players[i]
                        record=records[i]
                        private,public,reason=get_action(bot,record,model=model,display=display_result)
                        tmp_records.append([period,private,public,0,0,reason])
                        # ['Period','Private','Public','Total public','Earnings','Reason']

                    # calculate total public
                    total_public=sum([x[2] for x in tmp_records])

                    # calculate earnings for all
                    for r in tmp_records:
                        r[3]=total_public
                        r[4]=r[1]+total_public*K

                    # update records for all
                    for i in range(N):
                        record=records[i]
                        record.loc[len(record.index)]=tmp_records[i]

                    print('time:',time.time()-starttime)


                    # save file
                    recordfile=os.path.join(filepath,f'records_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{period}.pkl')
                    playerfile=os.path.join(filepath,f'players_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{period}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    with open(playerfile, 'wb') as f:
                        pickle.dump(players, f)

                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)

    nsess=10
    period=nperiod

    allprivate=[]
    for sess in range(1,nsess+1):
        recordfile=os.path.join(filepath,f'records_PGD_{expname}_{N}_{G}_{K}_{Z}_{nperiod}_sess{sess}_period{period}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)

        for record in records:
            tmp=record['Private'].tolist()
            allprivate.extend(tmp)
    out=np.mean(allprivate)
    print(out)



    ## 7. Infinitely repeated prisoner’s dilemma
    # init instruction

    delta,delta_str,p1,p2=0.5,'0.5','50%','50%'

    print('params=',delta,delta_str,p1,p2)


    system_message = f"""
    Please act as a human player. You are playing a game with another human player (called the opponent) for several rounds. After each round, there is a {p1} chance that the game will repeat for another round and the other {p2} chance that the game will end. In each round, you and the opponent will simultaneously choose an action between F and J.
    <payoff>
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
    """

    aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points in this round.'
    ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points in this round.'
    ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points in this round.'
    bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points in this round.'
    payoff=[aa,ab,ba,bb]
    payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]


    tmp_system_message=system_message.replace('<payoff>',payoff_str)
    def get_history(record):
        history=''
        for i in record.index:
            round=i+1
            choice=record.loc[i,'Your choice']
            cochoice=record.loc[i,'Co-player choice']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{round}, you chose {choice} and the opponent chose {cochoice}, you earn {earning} points."""
            history+=tmp
            history+='\n'
        return history
    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                print('Summarize json:',out)
                out=json.loads(out,strict=0)
        return out
    def get_action(bot,record,model,display=False):
        history=get_history(record)
        if len(record)==0:
            message=f"""
    It is round-{round} now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. For example, {{"thought": "xxx.","action": x }}. The 'action' should be F or J. 
    """.strip()
    # For example, {{"thought": "xxx.","action": x }}. The 'action' should be F or J. 
        else:
            message=f"""The history of the game is listed as follows delimited by triple backticks.
    ```
    {history}
    ```
    It is round-{round} now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. For example, {{"thought": "xxx.","action": x }}. The 'action' should be F or J. 
    """.strip()
    #         message=f"""The history of the game is listed as follows.
    # {history}
    # It is round-{round} now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. The 'action' should be F or J. 
    # """.strip()
        res=bot.chat_wo_update(message, model=model, temperature=0)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        try:
            action=out['action']
            reason=out['thought']
        except:
            print('key error:',res)
            prompt=summarize_action.replace('<out>',res)
            out=gpt_completion(prompt)
            out=parse_response(out)
            print('corrected:',out)
            action=out['action']
            reason=out['thought']
        try:
            assert action in ['F','J']
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason

    #################################################
    nsess=20

    expname='llama13b'

    max_try=50
    display_result=0
    max_round=100
    print('params=',delta,delta_str,p1,p2)


    system_message = f"""
    Please act as a human player. You are playing a game with another human player (called the opponent) for several rounds. After each round, there is a {p1} chance that the game will repeat for another round and the other {p2} chance that the game will end. In each round, you and the opponent will simultaneously choose an action between F and J.
    <payoff>
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
    """

    aa=f'If you choose J and the opponent chooses J, you earn 40 points and the opponent earns 40 points in this round.'
    ab=f'If you choose J and the opponent chooses F, you earn 12 points and the opponent earns 50 points in this round.'
    ba=f'If you choose F and the opponent chooses J, you earn 50 points and the opponent earns 12 points in this round.'
    bb=f'If you choose F and the opponent chooses F, you earn 25 points and the opponent earns 25 points in this round.'
    payoff=[aa,ab,ba,bb]

    for sess in range(1,nsess+1):
        print(f'==============sess{sess}==============')
        
        N=2
        players=[]
        payoff=[aa,ab,ba,bb]
        for _ in tqdm(range(N)):
            random.shuffle(payoff)
            payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]
            tmp_system_message=system_message.replace('<payoff>',payoff_str)
        #     print(tmp_system_message)
            bot=ChatApp(system_message=tmp_system_message)
            players.append(bot)
        records=[]
        for _ in range(N):    
            df = pd.DataFrame(columns=['Round','Your choice','Co-player choice','Earnings','Reason of choice'])
            records.append(df)

        round=1
        # if this block done, continue
        recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
        playerfile=os.path.join(filepath,f'players_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
        if os.path.exists(recordfile) and os.path.exists(playerfile):
            with open(recordfile, 'rb') as f:
                records = pickle.load(f)
            with open(playerfile, 'rb') as f:
                players = pickle.load(f)
            print('existed!')
            continue


        starttime=time.time()
        for round in range(1,max_round+1):
            print('-----------------------')
            print(f'round-{round}')

            # if this block done, continue
            recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            for _ in range(max_try):
                try:
                    if round>1:
                        prev_round=round-1
                        recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{prev_round}.pkl')
                        playerfile=os.path.join(filepath,f'players_{expname}_delta{delta_str}_sess{sess}_round{prev_round}.pkl')
                        with open(recordfile, 'rb') as f:
                            records = pickle.load(f)
                        with open(playerfile, 'rb') as f:
                            players = pickle.load(f)
                    # play
                    tmp_records=[]
                    for i in tqdm(range(N)):
                        bot=players[i]
                        record=records[i]
                        action,reason=get_action(bot,record,model=model,display=display_result)
                        tmp_records.append([round,action,'NULL',0,reason])
                        # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']

                    # calculate total public
                    c1,c2=tmp_records[0][1],tmp_records[1][1]
                    if c1=='J' and c2=='J':
                        e1,e2=40,40
                    elif c1=='J' and c2=='F':
                        e1,e2=12,50
                    elif c1=='F' and c2=='J':
                        e1,e2=50,12
                    elif c1=='F' and c2=='F':
                        e1,e2=25,25
                    else:
                        print(c1,c2)
                        assert False
                    # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']
                    tmp_records[0][2]=c2
                    tmp_records[1][2]=c1
                    tmp_records[0][3]=e1
                    tmp_records[1][3]=e2

                    # update records for all
                    for i in range(N):
                        record=records[i]
                        record.loc[len(record.index)]=tmp_records[i]

                    print('time:',time.time()-starttime)

                    # save file
                    recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
                    playerfile=os.path.join(filepath,f'players_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    with open(playerfile, 'wb') as f:
                        pickle.dump(players, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)
            if random.random()>delta:
                break


    nsess=20

    result=[]
    allaction=[]
    for sess in range(1,11):
        for round in range(1,101):
            recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile):
                maxround=round
            else:
                break
        recordfile=os.path.join(filepath,f'records_{expname}_delta{delta_str}_sess{sess}_round{maxround}.pkl')
        with open(recordfile, 'rb') as f:
            records = pickle.load(f)
        for i in range(len(records)):
            record=records[i]
            tmp=record['Earnings'].tolist()
            result.extend(tmp)
            allaction.extend(record['Your choice'].tolist())
            
    out=allaction.count('J')/len(allaction)
    print('cooperation rate:',out)

    eff=np.mean((np.array(result)-25)/15)
    print('efficiency:',eff)






    ## 8. Battle of the sexes
    nround=10

    system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
    """.strip()

    aa1=f'If you choose J and the opponent chooses J, you earn 10 points and the opponent earns 7 points in this round.'
    ab1=f'If you choose J and the opponent chooses F, you earn 0 points and the opponent earns 0 points in this round.'
    ba1=f'If you choose F and the opponent chooses J, you earn 0 points and the opponent earns 0 points in this round.'
    bb1=f'If you choose F and the opponent chooses F, you earn 7 points and the opponent earns 10 points in this round.'
    payoff1=[aa1,ab1,ba1,bb1]

    aa2=f'If you choose J and the opponent chooses J, you earn 7 points and the opponent earns 10 points in this round.'
    ab2=f'If you choose J and the opponent chooses F, you earn 0 points and the opponent earns 0 points in this round.'
    ba2=f'If you choose F and the opponent chooses J, you earn 0 points and the opponent earns 0 points in this round.'
    bb2=f'If you choose F and the opponent chooses F, you earn 10 points and the opponent earns 7 points in this round.'
    payoff2=[aa2,ab2,ba2,bb2]

    payoff_str=payoff1[0]+'\n'+payoff1[1]+'\n'+payoff1[2]+'\n'+payoff1[3]
    tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
    def get_history(record):
        history=''
        for i in record.index:
            round=i+1
            choice=record.loc[i,'Your choice']
            cochoice=record.loc[i,'Co-player choice']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{round}, you chose {choice} and the opponent chose {cochoice}, you earn {earning} points."""
            history+=tmp
            history+='\n'
        return history

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=get_completion(prompt, model,temperature=T)
                out=parse_response(out)
        return out
    def get_action(bot,record,model,display=False):
        history=get_history(record)
        if len(record)==0:
            message=f"""
    It is round-{round} out of {nround} rounds now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. For example, {{"thought": "xxx.","action": x }}. The 'action' should be F or J. 
    """.strip()
    # For example, {{"thought": "xxx.","action": x }}.
        else:
            message=f"""The history of the game is listed as follows delimited by triple backticks.
    ```
    {history}
    ```
    It is round-{round} out of {nround} rounds now. Tell me how you think and the action you would like to choose. Please answer in json format with keys 'thought' and 'action'. For example, {{"thought": "xxx.","action": x }}. The 'action' should be F or J. 
    """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display:
            print('-------get action---------')
            print(message)
            print(res)
        out=parse_response(res)
        try:
            action=out['action']
            reason=out['thought']
        except:
            print('key error:',res)
            prompt=summarize_action.replace('<out>',res)
            out=get_completion(prompt, model,temperature=T)
            out=parse_response(out)
            print('corrected:',out)
            action=out['action']
            reason=out['thought']
        try:
            assert action in ['F','J']
        except:
            print('Invalid action:',out)
            assert 0
        return action,reason

    ############################
    nround=10
    nsess=10

    expname='llama13b'


    max_try=50
    display_result=0


    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
        """.strip()

        aa1=f'If you choose J and the opponent chooses J, you earn 10 points and the opponent earns 7 points in this round.'
        ab1=f'If you choose J and the opponent chooses F, you earn 0 points and the opponent earns 0 points in this round.'
        ba1=f'If you choose F and the opponent chooses J, you earn 0 points and the opponent earns 0 points in this round.'
        bb1=f'If you choose F and the opponent chooses F, you earn 7 points and the opponent earns 10 points in this round.'
        payoff1=[aa1,ab1,ba1,bb1]

        aa2=f'If you choose J and the opponent chooses J, you earn 7 points and the opponent earns 10 points in this round.'
        ab2=f'If you choose J and the opponent chooses F, you earn 0 points and the opponent earns 0 points in this round.'
        ba2=f'If you choose F and the opponent chooses J, you earn 0 points and the opponent earns 0 points in this round.'
        bb2=f'If you choose F and the opponent chooses F, you earn 10 points and the opponent earns 7 points in this round.'
        payoff2=[aa2,ab2,ba2,bb2]
        
        ######################## init players and records
        N=2
        players=[]
        for i in tqdm(range(N)):
            if i==0:
                random.shuffle(payoff1)
                payoff_str=payoff1[0]+'\n'+payoff1[1]+'\n'+payoff1[2]+'\n'+payoff1[3]
            else:
                random.shuffle(payoff2)
                payoff_str=payoff2[0]+'\n'+payoff2[1]+'\n'+payoff2[2]+'\n'+payoff2[3]
            tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
    #         print(tmp_system_message)
            bot=ChatApp(system_message=tmp_system_message)
            players.append(bot)
        records=[]
        for _ in range(N):    
            df = pd.DataFrame(columns=['Round','Your choice','Co-player choice','Earnings','Reason of choice'])
            records.append(df)
        ######################## start game

        starttime=time.time()
        for round in range(1,nround+1):
            print('-----------------------')
            print(f'round-{round}')

            # if this round done, continue
            recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            for _ in range(max_try):
                try:
                    if round>1:
                        prev_round=round-1
                        recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                        playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                        with open(recordfile, 'rb') as f:
                            records = pickle.load(f)
                        with open(playerfile, 'rb') as f:
                            players = pickle.load(f)
                    # play
                    tmp_records=[]
                    for i in tqdm(range(N)):
                        bot=players[i]
                        record=records[i]
                        action,reason=get_action(bot,record,model=model,display=display_result)
                        tmp_records.append([round,action,'NULL',0,reason])
                        # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']

                    # calculate total public
                    c1,c2=tmp_records[0][1],tmp_records[1][1]
                    if c1=='J' and c2=='J':
                        e1,e2=10,7
                    elif c1=='J' and c2=='F':
                        e1,e2=0,0
                    elif c1=='F' and c2=='J':
                        e1,e2=0,0
                    elif c1=='F' and c2=='F':
                        e1,e2=7,10
                    else:
                        print(c1,c2)
                        assert False
                    # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']
                    tmp_records[0][2]=c2
                    tmp_records[1][2]=c1
                    tmp_records[0][3]=e1
                    tmp_records[1][3]=e2

                    # update records for all
                    for i in range(N):
                        record=records[i]
                        record.loc[len(record.index)]=tmp_records[i]
                    print('time:',time.time()-starttime)

                    # save file
                    recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
                    playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{round}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    with open(playerfile, 'wb') as f:
                        pickle.dump(players, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)

    nsess=10
    nround=10
    round=nround

    result=[]
    for sess in range(1,nsess+1):
        recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)

        for record in records:
            tmp=record['Earnings'].tolist()
            result.append(tmp)
    eff=np.mean((np.array(result)-0)/(8.5-0))
    print('efficiency:',eff)






    ## 9. Stag hunt
    nround=10

    system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
    """.strip()

    aa=f'If you choose J and the opponent chooses J, you earn 10 points and the opponent earns 10 points in this round.'
    ab=f'If you choose J and the opponent chooses F, you earn 1 points and the opponent earns 8 points in this round.'
    ba=f'If you choose F and the opponent chooses J, you earn 8 points and the opponent earns 1 points in this round.'
    bb=f'If you choose F and the opponent chooses F, you earn 5 points and the opponent earns 5 points in this round.'
    payoff=[aa,ab,ba,bb]
    payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]

    tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
    summarize_action=f"""Given the following paragraph delimited by triple backticks:
    ```
    <out>
    ```
    Please summarize the action and how he thought in the first person from above paragraph in json format with keys 'thought' and 'action'. The 'action' should be F or J. 
    """.strip()
    def get_history(record):
        history=''
        for i in record.index:
            round=i+1
            choice=record.loc[i,'Your choice']
            cochoice=record.loc[i,'Co-player choice']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{round}, you chose {choice} and the opponent chose {cochoice}, you earn {earning} points."""
            history+=tmp
            history+='\n'
        return history

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.find('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                out=parse_response(out)
                print('summarized:\n',out)
    #             start_index = out.find('{')  # 查找第一个'{'的索引
    #             end_index = out.rfind('}')  # 查找最后一个'}'的索引
    #             out = out[start_index:end_index + 1]
    #             out=json.loads(extracted_content,strict=0)
        return out
    def get_history(record):
        history=''
        for i in record.index:
            round=i+1
            choice=record.loc[i,'Your choice']
            cochoice=record.loc[i,'Co-player choice']
            earning=record.loc[i,'Earnings']
            tmp=f"""In round-{round}, you chose {choice} and the opponent chose {cochoice}, you earn {earning} points."""
            history+=tmp
            history+='\n'
        return history

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.find('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                out=parse_response(out)
                print('summarized:\n',out)
    #             start_index = out.find('{')  # 查找第一个'{'的索引
    #             end_index = out.rfind('}')  # 查找最后一个'}'的索引
    #             out = out[start_index:end_index + 1]
    #             out=json.loads(extracted_content,strict=0)
        return out
    nround=10
    nsess=10

    expname='llama13b'

    max_try=50
    display_result=0


    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        system_message = """
    Please act as a human player. You are playing a game with another human player (called the opponent) for {nround} rounds.
    In each round, you and the opponent will simultaneously choose an action between F and J.
    {payoff_str}
    Before you choose an action, your actions, the opponent’s actions and your earnings in each of the previous rounds will be shown. Your goal is to maximize your own total earnings in all {nround} rounds. In each round, you should first think carefully about what to do, and then choose one of the two actions: F or J.
        """.strip()

        aa=f'If you choose J and the opponent chooses J, you earn 10 points and the opponent earns 10 points in this round.'
        ab=f'If you choose J and the opponent chooses F, you earn 1 points and the opponent earns 8 points in this round.'
        ba=f'If you choose F and the opponent chooses J, you earn 8 points and the opponent earns 1 points in this round.'
        bb=f'If you choose F and the opponent chooses F, you earn 5 points and the opponent earns 5 points in this round.'
        payoff=[aa,ab,ba,bb]
        
        ######################## init players and records
        N=2
        players=[]
        payoff=[aa,ab,ba,bb]
        for _ in tqdm(range(N)):
            random.shuffle(payoff)
            payoff_str=payoff[0]+'\n'+payoff[1]+'\n'+payoff[2]+'\n'+payoff[3]
            tmp_system_message=system_message.format(nround=nround, payoff_str=payoff_str)
        #     print(tmp_system_message)
            bot=ChatApp(system_message=tmp_system_message)
            players.append(bot)
        records=[]
        for _ in range(N):    
            df = pd.DataFrame(columns=['Round','Your choice','Co-player choice','Earnings','Reason of choice'])
            records.append(df)

        ######################## start game

        starttime=time.time()
        for round in range(1,nround+1):
            print('-----------------------')
            print(f'round-{round}')

            # if this round done, continue
            recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            for _ in range(max_try):
                try:
                    if round>1:
                        prev_round=round-1
                        recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                        playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{prev_round}.pkl')
                        with open(recordfile, 'rb') as f:
                            records = pickle.load(f)
                        with open(playerfile, 'rb') as f:
                            players = pickle.load(f)
                    # play
                    tmp_records=[]
                    for i in tqdm(range(N)):
                        bot=players[i]
                        record=records[i]
                        action,reason=get_action(bot,record,model=model,display=display_result)
                        tmp_records.append([round,action,'NULL',0,reason])
                        # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']

                    # calculate total public
                    c1,c2=tmp_records[0][1],tmp_records[1][1]
                    if c1=='J' and c2=='J':
                        e1,e2=10,10
                    elif c1=='J' and c2=='F':
                        e1,e2=1,8
                    elif c1=='F' and c2=='J':
                        e1,e2=8,1
                    elif c1=='F' and c2=='F':
                        e1,e2=5,5
                    else:
                        print(c1,c2)
                        assert False
                    # ['Round','Your choice','Co-player choice','Earnings','Reason of choice']
                    tmp_records[0][2]=c2
                    tmp_records[1][2]=c1
                    tmp_records[0][3]=e1
                    tmp_records[1][3]=e2

                    # update records for all
                    for i in range(N):
                        record=records[i]
                        record.loc[len(record.index)]=tmp_records[i]
                    print('time:',time.time()-starttime)

                    # save file
                    recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
                    playerfile=os.path.join(filepath,f'players_{expname}_{nround}_sess{sess}_round{round}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    with open(playerfile, 'wb') as f:
                        pickle.dump(players, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)
    nsess=10
    nround=10
    round=nround

    result=[]
    for sess in range(1,nsess+1):
        recordfile=os.path.join(filepath,f'records_{expname}_{nround}_sess{sess}_round{round}.pkl')
        with open(recordfile, 'rb') as f:
            records= pickle.load(f)

        for record in records:
            tmp=record['Earnings'].tolist()
            result.append(tmp)
    eff=np.mean((np.array(result)-4.5)/(10-4.5))
    print('efficiency:',eff)




    ## 10. Minimum effort
    N=9
    nperiod=8

    system_message = f"""
    Please act as a human player. You and {N-1} other players are playing a game for {nperiod} periods. In each period, each of you will be asked to choose a value of X. The values of X you may choose are 1, 2, 3, 4, 5, 6, 7. The value you choose for X and the smallest value of X chosen by a decision maker will determine the payoff you receive for this period.
    You are provided with an EARNINGS TABLE that shows the potential number of points that you may receive, shown as follows delimited by triple backticks.
    ```
    Your Choice of X \ Smallest value of X chosen	7	6	5	4	3	2	1
    7	13	11	9	7	5	3	1
    6	x	12	10	8	6	4	2
    5	x	x	11	9	7	5	3
    4	x	x	x	10	8	6	4
    3	x	x	x	x	9	7	5
    2	x	x	x	x	x	8	6
    1	x	x	x	x	x	x	7
    ```
    Note that the EARNINGS TABLE is the same for every player. Only your earnings are shown in the EARNINGS TABLE. The earnings in each period may be found by looking across from the value you choose on the left-hand side of the table and down from the smallest value of X chosen by a decision maker. For example, if you choose 4 and the smallest value of X chosen is 4, you earn 10 points for this period. Your goal is to maximize your own total earnings in all {nperiod} periods. In each period, you should first think carefully about what to do, and then choose the value of X.
    """
    def get_earnings(choice,smallest):
        assert choice in [1,2,3,4,5,6,7]
        assert smallest<=choice
        A = {
            7: {7: 1.30, 6: 1.10, 5: 0.90, 4: 0.70, 3: 0.50, 2: 0.30, 1: 0.10},
            6: {6: 1.20, 5: 1.00, 4: 0.80, 3: 0.60, 2: 0.40, 1: 0.20},
            5: {5: 1.10, 4: 0.90, 3: 0.70, 2: 0.50, 1: 0.30},
            4: {4: 1.00, 3: 0.80, 2: 0.60, 1: 0.40},
            3: {3: 0.90, 2: 0.70, 1: 0.50},
            2: {2: 0.80, 1: 0.60},
            1: {1: 0.70}
        }
        return int(A[choice][smallest]*10)

    def get_display_table(record):
        display_table=''
        for i in record.index:
            period=record.loc[i,'Period']
            choice=record.loc[i,'Your choice X']
            smallest=record.loc[i,'Smallest X']
            earning=record.loc[i,'Earnings']   
            tmp=f"""In period-{period}, your chose {choice}, the smallest X chosen was {smallest}, your earnings were {earning} points."""
            display_table+=tmp
            display_table+='\n'
        balance=record['Earnings'].sum()
        balance='%.0f'%balance
        display_table+=f"""Your cumulative earnings so far are {balance} points."""
        return display_table

    def parse_response(response): # return a json
        response = response.replace('Action','action')
        response = response.replace('Thought','thought')
        try:
            out=json.loads(response,strict=0)
        except:
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if end_index==-1:
                response=response.strip()+'}'
                
            start_index = response.find('{')  # 查找第一个'{'的索引
            end_index = response.rfind('}')  # 查找最后一个'}'的索引
            if start_index != -1 and end_index != -1:
                extracted_content = response[start_index:end_index + 1]
                try:
                    out=json.loads(extracted_content,strict=0)
                except:
                    print('json error: ',extracted_content)
                    corrected_json=correct_json(extracted_content)
                    print('corrected:',corrected_json)
                    start_index = corrected_json.find('{')  # 查找第一个'{'的索引
                    end_index = corrected_json.rfind('}')  # 查找最后一个'}'的索引
                    extracted_content = corrected_json[start_index:end_index + 1]
                    out=json.loads(extracted_content,strict=0)
            else:
                print('No json found in:',response)
                prompt=summarize_action.replace('<out>',response)
                out=gpt_completion(prompt)
                out=json.loads(out)
        return out
    def get_action(bot,record,model,display=False):
        if period==1:
            message=f"""It is period-{period} out of {nperiod} periods now. Tell me how you think and the value of X you would like to choose. Please answer in json format with keys 'thought' and 'X'. For example, {{"thought": "xxx.","X": x }}. The 'X' should be in 1,2,3,4,5,6,7.
            """.strip()
        else:
            display_table=get_display_table(record)
            message=f"""The history of decisions and your earnings is listed as follows delimited by triple backticks.
    ```
    {display_table}
    ```
    It is period-{period} out of {nperiod} periods now. Tell me how you think and the value of X you would like to choose. Please answer in json format with keys 'thought' and 'X'. For example, {{"thought": "xxx.","X": x }}. The 'X' should be in 1,2,3,4,5,6,7.
        """.strip()
    #         message=f"""The history of decisions and your earnings is listed as follows delimited by triple backticks.
    # ```
    # {display_table}
    # ```
    # It is period-{period} out of {nperiod} periods now. Tell me how you think and the value of X you would like to choose. Your choice of X should be in 1,2,3,4,5,6,7.
    #     """.strip()
        res=bot.chat_wo_update(message, model=model)
        time.sleep(1)
        if display_result:
            print('------get action------')
            print(message)
            print(res)
        out=parse_response(res)
        try:
            X=int(out['X'])
            reason=out['thought']
        except:
            print('key error! messages:',message)
            print('response:',response)
            prompt=summarize_action.replace('<out>',response)
            out=gpt_completion(prompt)
            out=json.loads(out)
            X=int(out['X'])
            reason=out['thought']
            
        try:
            X in [1,2,3,4,5,6,7]
        except:
            print('Invalid action:',out)
            assert 0
        return X,reason
    nsess=10

    expname='llama13b'


    max_try=50
    display_result=0

    for sess in range(1,1+nsess):
        print(f'==============sess{sess}==============')
        players=[]
        for _ in tqdm(range(N)):
            bot=ChatApp(system_message=system_message)
            players.append(bot)
        records=[]
        cols=['Period','Your choice X','Smallest X','Earnings','Reason of choice']
        for _ in range(N):    
            df = pd.DataFrame(columns=cols)
            records.append(df)
            
        starttime=time.time()
        for period in range(1,nperiod+1):
            print('-----------------------')
            print(f'period-{period}')

            # if this block done, continue
            recordfile=os.path.join(filepath,f'records_ME_{expname}_{N}_{nperiod}_sess{sess}_period{period}.pkl')
            playerfile=os.path.join(filepath,f'players_ME_{expname}_{N}_{nperiod}_sess{sess}_period{period}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            for _ in range(max_try):
                try:
                    if period>1:
                        prev_period=period-1
                        recordfile=os.path.join(filepath,f'records_ME_{expname}_{N}_{nperiod}_sess{sess}_period{prev_period}.pkl')
                        playerfile=os.path.join(filepath,f'players_ME_{expname}_{N}_{nperiod}_sess{sess}_period{prev_period}.pkl')
                        with open(recordfile, 'rb') as f:
                            records = pickle.load(f)
                        with open(playerfile, 'rb') as f:
                            players = pickle.load(f)
                    # play
                    tmp_records=[]
                    for i in tqdm(range(N)):
                        bot=players[i]
                        record=records[i]
                        X,reason=get_action(bot,record,model=model,display=display_result)
                        tmp_records.append([period, X, 0.0, 0.0, reason])
                        # cols=['Period','Your choice X','Smallest X','Earnings','Reason of choice']

                    # calculate smallest X
                    smallest=min([x[1] for x in tmp_records])

                    # calculate earnings for all
                    for r in tmp_records:
                        earning=get_earnings(r[1],smallest)
                        r[2]=smallest
                        r[3]=earning

                    # update records for all
                    for i in range(N):
                        record=records[i]
                        record.loc[len(record.index)]=tmp_records[i]

                    print('time:',time.time()-starttime)

                    # save file
                    recordfile=os.path.join(filepath,f'records_ME_{expname}_{N}_{nperiod}_sess{sess}_period{period}.pkl')
                    playerfile=os.path.join(filepath,f'players_ME_{expname}_{N}_{nperiod}_sess{sess}_period{period}.pkl')
                    with open(recordfile, 'wb') as f:
                        pickle.dump(records, f)
                    with open(playerfile, 'wb') as f:
                        pickle.dump(players, f)
                    break
                except Exception as e:
                    print(e)
                    print('error!sleep 30s...')
                    time.sleep(30)
    # choices=np.zeros((N,nperiod))

    nsess=10
    period=nperiod

    result=[]
    for sess in range(1,1+nsess):
        recordfile=os.path.join(filepath,f'records_ME_{expname}_{N}_{nperiod}_sess{sess}_period{period}.pkl')
        with open(recordfile, 'rb') as f:
            records=pickle.load(f)
        for record in records:
            tmp=record['Earnings'].tolist()
            result.extend(tmp)
    eff=np.mean((np.array(result)-1)/(13-1))
    print('efficiency:',eff)

def run_collective_rationality(model):
    ## 1. MMLU
    filename="./data/mmlu.pkl"
    with open(filename, 'rb') as f:
        data=pickle.load(f)
    system_message = f"""You are an expert in biology, chemistry, computer science, mathematics, physics and are objective and unbiased, and you can be persuaded if other agent's answers make sense. Please keep this in mind.""".strip()


    question_template="Can you answer the following question as accurately as possible? {}: (A) {}, (B) {}, (C) {}, (D) {}. Explain your answer, putting the answer in the form (X) at the end of your response. You should choose exactly one of the choices A,B,C,D."

    debate_template="""
    These are the solutions to the problem from other agents: 
    One agent solution: {}
    Another agent solution: {}
    Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.
    """
    # debate_template
    N=3
    sess=1
    T=1.0
    nround=1

    expname='llama13b'



    for taskid in range(50):
        # init question
        question=data['task_info'][taskid]
        answer=data['answer'][taskid]
        print(f'===task-{taskid}===')
        print(question)
        print('Correct ans:',answer)
        question_prompt=question_template.format(*question)

        # init players
        players=[]
        for _ in tqdm(range(N)):
            bot=ChatApp(system_message=system_message)
            players.append(bot)

        # play
        for round in range(1+nround):
            print(f'round-{round}')
            # load if exist
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            if round==0:
                # QA
                for i in tqdm(range(N)):
                    bot=players[i]
                    res=bot.chat(question_prompt,model=model,temperature=T)
            else:
                anss=[x.messages[-1]['content'] for x in players]
                for i in tqdm(range(N)):
                    bot=players[i]
                    other_ans=[anss[j] for j in range(N) if j!=i]
                    debate_prompt=debate_template.format(*other_ans)
                    res=bot.chat(debate_prompt,model=model,temperature=T)
            # save
            records=[x.messages for x in players]
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            with open(recordfile, 'wb') as f:
                pickle.dump(records, f)
            with open(playerfile, 'wb') as f:
                pickle.dump(players, f)
    sess=1
    expname='llama13b'


    taskid=0
    round=1

    recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
    if not os.path.exists(recordfile):
        print(f"{recordfile} NOT EXIST!")

    with open(recordfile, 'rb') as f:
        records = pickle.load(f)
    records[2]
    def count_independent_occurrences(s):
        # 初始化计数器
        counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for i, char in enumerate(s):
            # 检查当前字符是否是 'A', 'B', 'C', 'D' 之一
            if char in counts:
                # 检查前一个和后一个字符是否为字母
                if (i == 0 or not s[i - 1].isalpha()) and (i == len(s) - 1 or not s[i + 1].isalpha()):
                    counts[char] += 1

        return counts

    def find_single_letter(s):
        lettercnt=count_independent_occurrences(s) # {'A': 2, 'B': 2, 'C': 2, 'D': 2}
        letters = 'ABCD'
        found_letter = None
        for letter in letters:
            if lettercnt[letter]>0:
                # 检查是否已经找到一个字母
                if found_letter is not None:
                    # 如果已经找到一个字母，说明有多于一个字母出现，返回 None
                    return None
                found_letter = letter
        return found_letter

    def parse_answer(content:str, task_info:tuple=None):
        x=find_single_letter(content)
        if x is not None:
            return x
        
        assert len(task_info) == 5
        if "Combined Answer".lower() in content.lower():
            return None
        pattern = r"\((\w+)\)|(\w+)\)"
        matches = re.findall(pattern, content)
        matches = [match[0] or match[1] for match in matches]
        solution_by_re = None
        # assert len(matches)<=1, str(len(matches))
        for match_str in matches[::-1]:
            solution_by_re = match_str.upper()
            if solution_by_re >= 'A' and solution_by_re <= 'D':
                break
            else:
                solution_by_re = None
    #     if len(matches) > 1:
    #         print("mike:",(content,),"parse:", solution_by_re)
        solution_by_item = [-1,-1,-1,-1]
        idx = 0
        for item in task_info[1:]:
            pos = content.lower().rfind(item.lower().strip())
            if pos >= 0:
                solution_by_item[idx] = pos
            idx += 1
        if max(solution_by_item) == -1:
            solution_by_item = None
        else:
            solution_by_item = ["A","B","C","D"][
                solution_by_item.index(max(solution_by_item))
            ]
        if solution_by_item is None and solution_by_re is not None:
            return solution_by_re
        elif solution_by_item is not None and solution_by_re is None:
            return solution_by_item
        elif solution_by_item is None and solution_by_re is None:
            return None
        elif solution_by_item is not None and solution_by_re is not None:
            if solution_by_item == solution_by_re:
                return solution_by_item
            else:
                return solution_by_item 
    content=records[2][2]['content']
    print(content)
    task_info=data['task_info'][taskid]
    parse_answer(content,task_info)
    sess=1
    nround=1

    expname='llama13b'

    allresults=[] # taskid,round,player
    flatten_results=[]
    for taskid in range(50):
        task_info=data['task_info'][taskid]
        result=[]
        for round in range(1+nround):
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            with open(recordfile, 'rb') as f:
                records = pickle.load(f)
            anss=[x[-1]['content'] for x in records]
            choices=[parse_answer(ans,task_info) for ans in anss]
            flatten_results.extend(choices)
            result.append(choices)
        allresults.append(result)

    print(len(flatten_results))
    valid_rate=len([x for x in flatten_results if x in ['A','B','C','D']])/len(flatten_results)
    print(valid_rate)

    def calculate_matching_ratio(list1, list2):
        # 确保两个列表长度相同
        if len(list1) != len(list2):
            return None

        # 计算对应位置相等的元素数量
        matching_count = sum(1 for x, y in zip(list1, list2) if x == y)

        # 计算占比
        ratio = matching_count / len(list1)
        return ratio

    for round in [0,1]:
        flatten_results=[]
        flatten_answers=[]
        for taskid in range(50):
            task_info=data['task_info'][taskid]
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            with open(recordfile, 'rb') as f:
                records = pickle.load(f)
            anss=[x[-1]['content'] for x in records]
            choices=[parse_answer(ans,task_info) for ans in anss]
            answers=[data['answer'][taskid] for _ in range(len(choices))]
            flatten_results.extend(choices)
            flatten_answers.extend(answers)
        correct_rate=calculate_matching_ratio(flatten_results,flatten_answers)
        print('%.3f'%correct_rate,end='\t')
    print('')


    ## 2. MATH
    filename="./data/math.pkl"
    with open(filename, 'rb') as f:
        data=pickle.load(f)
    system_message = f"""You are an expert skilled in solving mathematical problems and are objective and unbiased, and you can be persuaded if other agent's answers make sense. Please keep this in mind.""".strip()

    question_template="Here is a math problem written in LaTeX: {}\nPlease carefully consider it and explain your reasoning. Put your answer in the form \\boxed{{answer}}, at the end of your response.".strip()

    debate_template="""
    These are the solutions to the problem from other agents: 
    One agent solution: {}
    Another agent solution: {}
    Using the reasoning from other agents as additional information and referring to your historical answers, can you give an updated answer? Put your answer in the form \\boxed{{answer}}, at the end of your response.
    """.strip()
    # debate_template
    N=3
    sess=1
    T=1.0
    nround=1

    expname='llama13b'


    for taskid in range(50):
        # init question
        question=data['task_info'][taskid]
        answer=data['answer'][taskid]
        print(f'===task-{taskid}===')
        print(question)
        print('Correct ans:',answer)
        question_prompt=question_template.format(*question)

        # init players
        players=[]
        for _ in tqdm(range(N)):
            bot=ChatApp(system_message=system_message)
            players.append(bot)

        # play
        for round in range(1+nround):
            print(f'round-{round}')
            # load if exist
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            if os.path.exists(recordfile) and os.path.exists(playerfile):
                with open(recordfile, 'rb') as f:
                    records = pickle.load(f)
                with open(playerfile, 'rb') as f:
                    players = pickle.load(f)
                print('existed!')
                continue

            if round==0:
                # QA
                for i in tqdm(range(N)):
                    bot=players[i]
                    res=bot.chat(question_prompt,model=model,temperature=T)
            else:
                anss=[x.messages[-1]['content'] for x in players]
                for i in tqdm(range(N)):
                    bot=players[i]
                    other_ans=[anss[j] for j in range(N) if j!=i]
                    debate_prompt=debate_template.format(*other_ans)
                    res=bot.chat(debate_prompt,model=model,temperature=T)
            # save
            records=[x.messages for x in players]
            recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            playerfile=os.path.join(filepath,f'players_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
            with open(recordfile, 'wb') as f:
                pickle.dump(records, f)
            with open(playerfile, 'wb') as f:
                pickle.dump(players, f)

    sess=1
    expname='llama13b'


    taskid=46
    round=1

    recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}_round{round}.pkl')
    if not os.path.exists(recordfile):
        print(f"{recordfile} NOT EXIST!")

    with open(recordfile, 'rb') as f:
        records = pickle.load(f)
    def extract_math(string):
        results = []
        stack = []
        flag = r"\boxed{"
        string_copy = copy.copy(string)
        idx = string_copy.find(flag)
        if idx == -1:
    #         print(f"math parse failed: \"{string}\"")
            return []
            # assert False, f"math parse failed: \"{string}\""
        output_flag = 0
        idx += len(flag)
        while idx < len(string):
            if string[idx] == '{':
                output_flag += 1
            elif string[idx] == '}' and output_flag == 0:
                results.append("".join(stack))
                stack.clear()
                idx = string_copy.find(flag, idx)
                if idx == -1:
                    break
                else:
                    idx += len(flag)
                    continue
            elif string[idx] == '}' and output_flag != 0:
                output_flag -= 1
            stack.append(string[idx])
            idx += 1
        return results

    def parse_answer(content:str, task_info:tuple=None):
        matches = extract_math(string=content)
        if len(matches)==0:
            return None
            # assert False, f"math parse failed: \"{content}\""
        else:
            return matches[-1]
    answers=[]
    for x in data['answer']:
        if type(x)==tuple:
            answers.append(x[0])
        else:
            answers.append(x)
            
    def extract_integer(s):
        """Extracts the (integer) number from a string, returns None if there is more than one number (or no numbers).
        Ignores commas and spaces so 1,230 is 1230 and 12 345 678 is 12345678.
        Also ignores digits after decimal poitns so 1.23 is 1
        """
        num_strings = re.findall(r'-?[0-9][0-9 ,\.]*', s)
        numlist=[int(num_string.replace(",", "").replace(" ", "").split(".")[0]) for num_string in num_strings]
        return numlist




    ## 3. General knowledge questions

    KNOWLEDGE_QUESTIONS = [('How many bones does an adult human have?', 206),
    ('What is the melting temperature of aluminum (in degrees Celsius)?', 660),
    ('How many degrees Fahrenheit are 100 degrees Celsius?', 212),
    ('How many (earth) days has a year on the Mars?', 687),
    ('What is the speed of sound in the air (in meters per second)?', 343),
    ('How many ribs does a human have, total?', 24),
    ('What is the melting temperature of gold (in degrees Celsius)?', 1064),
    ('What is the speed of light in a vacuum (in meters per second)?', 299792458),
    ('How many keys does a typical piano have?', 88),
    ('How many chromosomes does a dog have, total?', 78)]

    # names
    race_surnames={"Black or African American": ["SMALLS", "JEANBAPTISTE", "DIALLO", "KAMARA", "PIERRELOUIS", "GADSON", "JEANLOUIS", "BAH", "DESIR", "MENSAH", "BOYKINS", "CHERY", "JEANPIERRE", "BOATENG", "OWUSU", "JAMA", "JALLOH", "SESAY", "NDIAYE", "ABDULLAHI", "WIGFALL", "BIENAIME", "DIOP", "EDOUARD", "TOURE", "GRANDBERRY", "FLUELLEN", "MANIGAULT", "ABEBE", "SOW", "TRAORE", "MONDESIR", "OKAFOR", "BANGURA", "LOUISSAINT", "CISSE", "OSEI", "CALIXTE", "CEPHAS", "BELIZAIRE", "FOFANA", "KOROMA", "CONTEH", "STRAUGHTER", "JEANCHARLES", "MWANGI", "KEBEDE", "MOHAMUD", "PRIOLEAU", "YEBOAH", "APPIAH", "AJAYI", "ASANTE", "FILSAIME", "HARDNETT", "HYPPOLITE", "SAINTLOUIS", "JEANFRANCOIS", "RAVENELL", "KEITA", "BEKELE", "TADESSE", "MAYWEATHER", "OKEKE", "ASARE", "ULYSSE", "SAINTIL", "TESFAYE", "JEANJACQUES", "OJO", "NWOSU", "OKORO", "FOBBS", "KIDANE", "PETITFRERE", "YOHANNES", "WARSAME", "LAWAL", "DESTA", "VEASLEY", "ADDO", "LEAKS", "GUEYE", "MEKONNEN", "STFLEUR", "BALOGUN", "ADJEI", "OPOKU", "COAXUM", "VASSELL", "PROPHETE", "LESANE", "METELLUS", "EXANTUS", "HAILU", "DORVIL", "FRIMPONG", "BERHANE", "NJOROGE", "BEYENE"], "White": ["OLSON", "SNYDER", "WAGNER", "MEYER", "SCHMIDT", "RYAN", "HANSEN", "HOFFMAN", "JOHNSTON", "LARSON", "CARLSON", "OBRIEN", "JENSEN", "HANSON", "WEBER", "WALSH", "SCHULTZ", "SCHNEIDER", "KELLER", "BECK", "SCHWARTZ", "BECKER", "WOLFE", "ZIMMERMAN", "MCCARTHY", "ERICKSON", "KLEIN", "OCONNOR", "SWANSON", "CHRISTENSEN", "FISCHER", "WOLF", "GALLAGHER", "SCHROEDER", "PARSONS", "BAUER", "MUELLER", "HARTMAN", "KRAMER", "FLYNN", "OWEN", "SHAFFER", "HESS", "OLSEN", "PETERSEN", "ROTH", "HOOVER", "WEISS", "DECKER", "YODER", "LARSEN", "SWEENEY", "FOLEY", "HENSLEY", "HUFFMAN", "CLINE", "ONEILL", "KOCH", "BRENNAN", "BERG", "RUSSO", "MACDONALD", "KLINE", "JACOBSON", "BERGER", "BLANKENSHIP", "BARTLETT", "ODONNELL", "STEIN", "STOUT", "SEXTON", "NIELSEN", "HOWE", "MORSE", "KNAPP", "HERMAN", "STARK", "HEBERT", "SCHAEFER", "REILLY", "CONRAD", "DONOVAN", "MAHONEY", "HAHN", "PECK", "BOYLE", "HURLEY", "MAYER", "MCMAHON", "CASE", "DUFFY", "FRIEDMAN", "FRY", "DOUGHERTY", "CRANE", "HUBER", "MOYER", "KRUEGER", "RASMUSSEN", "BRANDT"], "Asian and Native Hawaiian and Other Pacific Islander": ["NGUYEN", "KIM", "PATEL", "TRAN", "CHEN", "LI", "LE", "WANG", "YANG", "PHAM", "LIN", "LIU", "HUANG", "WU", "ZHANG", "SHAH", "HUYNH", "YU", "CHOI", "HO", "KAUR", "VANG", "CHUNG", "TRUONG", "PHAN", "XIONG", "LIM", "VO", "VU", "LU", "TANG", "CHO", "NGO", "CHENG", "KANG", "TAN", "NG", "DANG", "DO", "LY", "HAN", "HOANG", "BUI", "SHARMA", "CHU", "MA", "XU", "ZHENG", "SONG", "DUONG", "LIANG", "SUN", "ZHOU", "THAO", "ZHAO", "SHIN", "ZHU", "LEUNG", "HU", "JIANG", "LAI", "GUPTA", "CHEUNG", "DESAI", "OH", "HA", "CAO", "YI", "HWANG", "LO", "DINH", "HSU", "CHAU", "YOON", "LUU", "TRINH", "HE", "HER", "LUONG", "MEHTA", "MOUA", "TAM", "KO", "KWON", "YOO", "CHIU", "SU", "SHEN", "PAN", "DONG", "BEGUM", "GAO", "GUO", "CHOWDHURY", "VUE", "THAI", "JAIN", "LOR", "YAN", "DAO"], "American Indian and Alaska Native": ["BEGAY", "YAZZIE", "BENALLY", "TSOSIE", "NEZ", "BEGAYE", "ETSITTY", "BECENTI", "YELLOWHAIR", "MANYGOATS", "WAUNEKA", "MANUELITO", "APACHITO", "BEDONIE", "CALABAZA", "PESHLAKAI", "CLAW", "ROANHORSE", "GOLDTOOTH", "ETCITTY", "TSINNIJINNIE", "NOTAH", "CLAH", "ATCITTY", "TWOBULLS", "WERITO", "HOSTEEN", "YELLOWMAN", "ATTAKAI", "BITSUI", "DELGARITO", "HENIO", "GOSEYUN", "KEAMS", "SECATERO", "DECLAY", "TAPAHA", "BEYALE", "HASKIE", "CAYADITTO", "BLACKHORSE", "ETHELBAH", "TSINNIE", "WALKINGEAGLE", "ALTAHA", "BITSILLY", "WASSILLIE", "BENALLIE", "SMALLCANYON", "LITTLEDOG", "COSAY", "CLITSO", "TESSAY", "SECODY", "BIGCROW", "TABAHA", "CHASINGHAWK", "BLUEEYES", "OLANNA", "BLACKGOAT", "COWBOY", "KANUHO", "SHIJE", "GISHIE", "LITTLELIGHT", "LAUGHING", "WHITEHAT", "ERIACHO", "RUNNINGCRANE", "CHINANA", "KAMEROFF", "SPOTTEDHORSE", "ARCOREN", "WHITEPLUME", "DAYZIE", "SPOTTEDEAGLE", "HEAVYRUNNER", "STANDINGROCK", "POORBEAR", "GANADONEGRO", "AYZE", "WHITEFACE", "YEPA", "TALAYUMPTEWA", "MADPLUME", "BITSUIE", "TSETHLIKAI", "AHASTEEN", "DOSELA", "BIRDINGROUND", "TODACHEENIE", "BITSIE", "TODACHEENE", "BULLBEAR", "LASILOO", "KEYONNIE", "NOTAFRAID", "COLELAY", "KALLESTEWA", "LITTLEWHITEMAN"], "Hispanic or Latino": ["GARCIA", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ", "PEREZ", "SANCHEZ", "RAMIREZ", "TORRES", "FLORES", "RIVERA", "GOMEZ", "DIAZ", "MORALES", "GUTIERREZ", "ORTIZ", "CHAVEZ", "RUIZ", "ALVAREZ", "CASTILLO", "JIMENEZ", "VASQUEZ", "MORENO", "HERRERA", "MEDINA", "AGUILAR", "VARGAS", "GUZMAN", "MENDEZ", "MUNOZ", "SALAZAR", "GARZA", "SOTO", "VAZQUEZ", "ALVARADO", "DELGADO", "PENA", "CONTRERAS", "SANDOVAL", "GUERRERO", "RIOS", "ESTRADA", "ORTEGA", "NUNEZ", "MALDONADO", "DOMINGUEZ", "VEGA", "ESPINOZA", "ROJAS", "MARQUEZ", "PADILLA", "MEJIA", "JUAREZ", "FIGUEROA", "AVILA", "MOLINA", "CAMPOS", "AYALA", "CARRILLO", "CABRERA", "LARA", "ROBLES", "CERVANTES", "SOLIS", "SALINAS", "FUENTES", "VELASQUEZ", "AGUIRRE", "OCHOA", "CARDENAS", "CALDERON", "RIVAS", "SERRANO", "ROSALES", "CASTANEDA", "GALLEGOS", "IBARRA", "SUAREZ", "OROZCO", "SALAS", "ESCOBAR", "VELAZQUEZ", "MACIAS", "ZAMORA", "VILLARREAL", "BARRERA", "PINEDA", "SANTANA", "TREVINO", "LOZANO", "RANGEL", "ARIAS", "MORA", "VALENZUELA", "ZUNIGA", "MELENDEZ", "GALVAN", "VELEZ", "MEZA"]}
    TITLES = ["Mr.", "Ms.", "Mx."]
    names=[]
    for title in TITLES:
        for race in race_surnames:
            surnames = race_surnames[race][:10]
            names.extend([title + " " + surname.title() for surname in surnames])
    question_template = """$name was asked the following question. They were not allowed to consult any external sources and were instructed to make their best guess if they were unsure. Their answer was written as an integer using digits 0-9.

    Question (text): [$question]

    $name's answer (integer): ["""
    sess=1
    T=1.0

    expname='llama13b'

    print(model)

    for taskid in range(len(KNOWLEDGE_QUESTIONS)):
        question=KNOWLEDGE_QUESTIONS[taskid][0]
        print(f'Question {taskid}:',question)
        # load records
        recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}.json')
        if os.path.exists(recordfile):
            with open(recordfile, 'r') as f:
                records = json.load(f)
            print('existed!')
        else:
            records={name:{'response':None,'answer':None} for name in names}
            with open(recordfile, 'w') as f:
                json.dump(records, f)

        for name in tqdm(names):
            if records[name]['response'] is None:
                prompt = question_template.replace("$name", name).replace("$question", question.strip())
                res=get_completion(prompt, model,temperature=T)
                records[name]['response']=res
                with open(recordfile, 'w') as f:
                    json.dump(records, f)

    # from original code
    def extract_integer(s):
        """Extracts the (integer) number from a string, returns None if there is more than one number (or no numbers).
        Ignores commas and spaces so 1,230 is 1230 and 12 345 678 is 12345678.
        Also ignores digits after decimal poitns so 1.23 is 1
        
        Also reject numbers bigger than 1 googol :-)
        """
        num_strings = re.findall(r'-?[0-9][0-9 ,\.]*', s)
        if len(num_strings) == 1:
            n = int(num_strings[0].replace(",", "").replace(" ", "").split(".")[0])
            if abs(n) < 10**30:
                return n
        else:
            numlist=[int(num_string.replace(",", "").replace(" ", "").split(".")[0]) for num_string in num_strings]
            eliminate_set=set([0,9,100,1947])
            numlist=list(set(numlist)-eliminate_set)
            if len(set(numlist))==1:
                return numlist[0]
            else:
                return None
            if abs(n) < 10**30:
                return n
            else:
                return None
        return None
    s="""I think the answer is 206. which is 100."""
    # s="""I think the answer is 206."""
    print(extract_integer(s))
    for taskid in range(10):
        print(taskid)
        recordfile=os.path.join(filepath,f'records_{expname}_task{taskid}_sess{sess}.json')
        with open(recordfile, 'r') as f:
            records = json.load(f)
        # fill answer
        cnt=0
        for name in records:
            v=records[name]
            if v['answer'] is not None:
                continue
            res=v['response']
            ans=extract_integer(res)
            if ans is not None:
                records[name]['answer']=ans
            else:
                cnt+=1
        print('#None',cnt)
        with open(recordfile, 'w') as f:
            json.dump(records,f)
