# %% [markdown]
# CSET485 Assignment 1: German Credit, roll suffix 044
# This is an explanatory analysis, not a lending deployment model.
# Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
# UCI's good/bad risk labels are proxies for the assignment's approve/reject,
# not observed historical approval decisions. Positive always means approve.

# %%
from pathlib import Path
import sys, json, hashlib, platform, urllib.request, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 44
ROOT = Path.cwd()
if (ROOT / 'outputs' / 'data' / 'german.data').exists():
    ROOT = ROOT / 'outputs'
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
if not (DATA / 'german.data').exists():
    urllib.request.urlretrieve(URL, DATA / 'german.data')
data_hash = hashlib.sha256((DATA / 'german.data').read_bytes()).hexdigest()
print('Seed:', SEED, '| SHA256:', data_hash)

# %% [markdown]
# Preprocessing: validate all 1,000 rows; retain all valid outliers; no imputation
# is needed for this file. 'Unknown/no savings' is a documented category, not NaN.
# Fit numeric scaling and categorical vocabulary on training rows only. Drop one
# reference level per categorical variable; retain the same scales/references
# when dropping a confounder. No target, row ID, or derived sex is added to X.
# The derived sex group is only used for auditing (the original combined
# personal-status/sex feature remains in the required full model).

# %%
columns = ['checking', 'duration', 'history', 'purpose', 'amount', 'savings',
           'employment', 'installment', 'personal_status', 'guarantor',
           'residence', 'property', 'age', 'other_plans', 'housing',
           'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'label']
numeric = ['duration', 'amount', 'installment', 'residence', 'age', 'existing_credits', 'dependents']
categorical = [c for c in columns[:-1] if c not in numeric]
df = pd.read_csv(DATA / 'german.data', sep=r'\s+', names=columns)
df.index.name = 'source_row_zero_based'
assert df.shape == (1000, 21) and not df.isna().any().any()
assert set(df.label) == {1, 2}
df['approve'] = (df.label == 1).astype(int)
sex_map = {'A91': 'Male', 'A92': 'Female', 'A93': 'Male', 'A94': 'Male', 'A95': 'Female'}
df['sex'] = df.personal_status.map(sex_map)
assert df.sex.notna().all()
train_idx, test_idx = train_test_split(df.index, test_size=0.30, random_state=SEED, stratify=df.approve)
train, test = df.loc[train_idx].copy(), df.loc[test_idx].copy()
mean, scale = train[numeric].mean(), train[numeric].std(ddof=0)
levels = {c: sorted(train[c].unique()) for c in categorical}
def design(frame):
    z = (frame[numeric] - mean) / scale
    for c in categorical:
        unknown = set(frame[c]) - set(levels[c])
        if unknown:
            raise ValueError(f'Unseen category in {c}: {unknown}')
        for level in levels[c][1:]:
            z[f'{c}_{level}'] = (frame[c] == level).astype(float)
    return sm.add_constant(z.astype(float), has_constant='add')
Xtr, Xte = design(train), design(test)
ytr, yte = train.approve, test.approve
assert np.linalg.matrix_rank(Xtr.to_numpy()) == Xtr.shape[1]
outliers = []
for c in numeric:
    q1, q3 = train[c].quantile([.25, .75]); iqr = q3-q1
    outliers.append({'feature': c, 'train_min': train[c].min(), 'train_max': train[c].max(),
                     'train_1.5IQR_count': int(((train[c] < q1-1.5*iqr)|(train[c] > q3+1.5*iqr)).sum())})
pd.DataFrame(outliers).to_csv(RESULTS / 'outlier_audit.csv', index=False)
print('Train/test:', len(train), len(test), '| approvals:', ytr.sum(), yte.sum())
print('Design columns including intercept:', Xtr.shape[1])
print('References:', {c: levels[c][0] for c in categorical})

# %% [markdown]
# Part 1: OLS is a linear probability model; predictions may fall outside [0,1].
# Do not clip before the required median split. The assignment's comparison
# matrix compares model labels to each other, not either model to ground truth.
# We also report separate ground-truth confusion matrices to avoid confusion.

# %%
ols = sm.OLS(ytr, Xtr).fit()
# L2 regularization avoids quasi/perfect separation in rare purpose categories.
# C=1 is fixed in advance of evaluating this final model; no test-set tuning.
# The intercept is not penalized. The same C, rows and scaling are used in 2.2.
def fit_prediction_model(x,y):
    fitted = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, tol=1e-9).fit(x.drop(columns='const'),y)
    assert fitted.n_iter_[0] < 2000
    fitted.params = pd.Series(np.r_[fitted.intercept_[0],fitted.coef_[0]],index=x.columns)
    return fitted
logit = fit_prediction_model(Xtr,ytr)
linear_predictions = ols.predict(Xte)
prob = pd.Series(logit.predict_proba(Xte.drop(columns='const'))[:,1],index=test.index)
median = float(linear_predictions.median())
pred_a = (linear_predictions >= median).astype(int)
pred_b = (prob >= .5).astype(int)
agreement = pd.DataFrame(confusion_matrix(pred_a, pred_b, labels=[0,1]),
                         index=['OLS reject', 'OLS approve'], columns=['Logit reject', 'Logit approve'])
mismatch_n = int((pred_a != pred_b).sum())
comparison = test.copy()
comparison['ols_score'] = linear_predictions
comparison['logit_probability'] = prob
comparison['ols_decision'] = pred_a
comparison['logit_decision_0.5'] = pred_b
comparison.to_csv(RESULTS / 'test_predictions.csv')
agreement.to_csv(RESULTS / 'model_agreement.csv')
coef = pd.DataFrame({'coefficient_L2_C1': logit.params})
coef.to_csv(RESULTS / 'full_model_coefficients.csv')
summary = {'seed': SEED, 'sha256': data_hash, 'train_n': len(train), 'test_n': len(test),
           'median_ols': median, 'mismatch_n': mismatch_n, 'mismatch_pct': 100*mismatch_n/len(test),
           'ols_approved': int(pred_a.sum()), 'logit_approved': int(pred_b.sum()),
           'ols_accuracy': accuracy_score(yte,pred_a), 'logit_accuracy': accuracy_score(yte,pred_b),
           'logit_auc': roc_auc_score(yte,prob), 'logit_brier': brier_score_loss(yte,prob),
           'ols_outside_01': int(((linear_predictions<0)|(linear_predictions>1)).sum())}
print(agreement.to_string())
print(json.dumps(summary, indent=2))
print('Logit versus truth (rows actual 0,1; columns predicted 0,1):\n', confusion_matrix(yte,pred_b))

# %% [markdown]
# Part 2.1: exploratory subgroup hunt on all 1,000 rows, as requested, separately
# from held-out prediction evaluation. Use a stable, interpretable numeric-only
# model with all seven numeric predictors, globally standardized, in every fit.
# This is deliberately a different explanatory specification from the full
# prediction model. Search two declared partitions (sex; age <35 versus >=35)
# and seven numeric predictors. Export every result; use Bonferroni over all
# 14 between-group comparisons, rather than choosing an unadjusted p-value.
# Since the groups are disjoint, SE(beta_A-beta_B)=sqrt(SE_A^2+SE_B^2).
# These are conditional associations, not causal effects. A sign change alone
# with wide intervals is not established Simpson's paradox.

# %%
all_z = sm.add_constant(((df[numeric]-mean)/scale).astype(float), has_constant='add')
pooled = sm.GLM(df.approve, all_z, family=sm.families.Binomial()).fit(cov_type='HC0')
partitions = {'sex': df.sex == 'Female', 'age35': df.age < 35}
subgroup_fits, hunt_rows = {}, []
for partition, mask in partitions.items():
    fits = []
    for keep in [mask, ~mask]:
        fit = sm.GLM(df.loc[keep,'approve'], all_z.loc[keep], family=sm.families.Binomial()).fit(cov_type='HC0')
        assert fit.converged
        fits.append(fit)
    subgroup_fits[partition] = fits
    for feature in numeric:
        a,b = fits
        delta = a.params[feature]-b.params[feature]
        delta_se = np.sqrt(a.bse[feature]**2+b.bse[feature]**2)
        p = 2*norm.sf(abs(delta/delta_se))
        hunt_rows.append({'partition': partition, 'feature': feature,
                         'pooled_beta': pooled.params[feature],
                         'pooled_low': pooled.conf_int().loc[feature,0], 'pooled_high': pooled.conf_int().loc[feature,1],
                         'A_n': int(mask.sum()), 'B_n': int((~mask).sum()),
                         'A_beta': a.params[feature], 'A_low': a.conf_int().loc[feature,0], 'A_high': a.conf_int().loc[feature,1],
                         'B_beta': b.params[feature], 'B_low': b.conf_int().loc[feature,0], 'B_high': b.conf_int().loc[feature,1],
                         'difference': delta, 'difference_SE': delta_se, 'p':p})
hunt = pd.DataFrame(hunt_rows)
hunt['p_bonferroni'] = np.minimum(1, len(hunt)*hunt.p)
hunt = hunt.sort_values('p')
hunt.to_csv(RESULTS / 'subgroup_hunt_all_comparisons.csv', index=False)
selected = hunt.iloc[0].to_dict()
print(hunt.to_string(index=False, float_format=lambda x:f'{x:.5f}'))
print('Selected strongest evidence (exploratory):', json.dumps(selected, indent=2))

# %% [markdown]
# Part 2.2: omit duration, a plausible shared driver of credit amount and risk.
# The same training rows and original coding/scales are retained. Full means
# all 20 usable predictors, not the outcome/derived fields. Changes in logistic
# coefficients can also reflect non-collapsibility; they do not prove causal
# confounding. Percentage change uses 100*(reduced-full)/abs(full), with the
# denominator stated explicitly; near-zero coefficients make percentages unstable.

# %%
reduced = fit_prediction_model(Xtr.drop(columns='duration'),ytr)
ovb = pd.DataFrame({'full': logit.params, 'without_duration': reduced.params}).drop(index=['const','duration'])
ovb['change'] = ovb.without_duration-ovb.full
ovb['percent_change_abs_denominator'] = 100*ovb.change/ovb.full.abs().replace(0,np.nan)
ovb['absolute_percent_change'] = ovb.percent_change_abs_denominator.abs()
ovb.to_csv(RESULTS / 'omitted_variable_all_coefficients.csv')
ovb_four = ovb.loc[['amount','installment','age','existing_credits']]
print('Correlation of duration with amount (training):', train.duration.corr(train.amount))
print(ovb_four.to_string())
print('Largest relative shift among four:', ovb_four.absolute_percent_change.idxmax())
print('Largest relative shift among all retained predictors:', ovb.absolute_percent_change.idxmax())
print('Largest absolute coefficient shift among all:', ovb.change.abs().idxmax())

# %% [markdown]
# Part 2.3: identify confident mistakes using held-out outcomes only, then sample
# with seed 44 if more than 50 exist. This is a retrospective error audit, not a
# representative evaluation subset and not a new training set. Describing an
# enriched trait is evidence of a failure pattern, not proof of its mechanism.

# %%
wrong_mask = ((prob>=.75)&(yte==0)) | ((prob<=.25)&(yte==1))
wrong = comparison.loc[wrong_mask].copy()
adversarial = wrong.sample(n=min(50,len(wrong)), random_state=SEED)
adversarial.to_csv(RESULTS / 'adversarial_subset.csv')
wrong.to_csv(RESULTS / 'all_confident_errors.csv')
summary.update({'confident_wrong_n': len(wrong), 'confident_wrong_pct': 100*len(wrong)/len(test),
                'adversarial_n': len(adversarial),
                'confident_false_approvals': int(((prob>=.75)&(yte==0)).sum()),
                'confident_false_rejections': int(((prob<=.25)&(yte==1)).sum())})
demographics = []
for name,frame in [('Test set',test),('All confident errors',wrong),('Sampled subset',adversarial)]:
    demographics.append({'population':name, 'n':len(frame), 'mean_age':frame.age.mean(),
                         'median_age':frame.age.median(), 'age_min':frame.age.min(), 'age_max':frame.age.max(),
                         'female_pct':100*(frame.sex=='Female').mean(), 'male_pct':100*(frame.sex=='Male').mean(),
                         'actual_approve_pct':100*frame.approve.mean()})
demographics = pd.DataFrame(demographics)
demographics.to_csv(RESULTS / 'adversarial_demographics.csv',index=False)
traits = {'checking A14 (no checking account)':lambda f:f.checking=='A14',
          'checking A11/A12 (balance below 200 DM)':lambda f:f.checking.isin(['A11','A12']),
          'duration <= 24 months':lambda f:f.duration<=24,
          'age <35':lambda f:f.age<35,
          'savings A61 (below 100 DM)':lambda f:f.savings=='A61',
          'existing credits paid duly A32':lambda f:f.history=='A32'}
trait_rows = []
for name,fn in traits.items():
    trait_rows.append({'trait':name, 'subset_pct':100*fn(adversarial).mean(),
                       'test_pct':100*fn(test).mean(), 'subset_count':int(fn(adversarial).sum())})
trait_table = pd.DataFrame(trait_rows)
trait_table['enrichment_percentage_points'] = trait_table.subset_pct-trait_table.test_pct
trait_table.to_csv(RESULTS / 'adversarial_traits.csv', index=False)
print(json.dumps({k:v for k,v in summary.items() if 'confident' in k or 'adversarial' in k},indent=2))
print(demographics.to_string(index=False)); print(trait_table.to_string(index=False))
# Local explanation: actual contribution to one confidently incorrect approval.
example = wrong.loc[wrong.approve==0].sort_values('logit_probability', ascending=False).iloc[0]
contributions = (Xte.loc[example.name] * logit.params).sort_values()
contributions.to_csv(RESULTS / 'confident_error_logit_contributions.csv', header=['log_odds_contribution'])
print('Example source row:', example.name, '| p:',example.logit_probability,'| actual:', example.approve)
print('Largest positive contributions:',contributions.tail(6).to_string())

# %% [markdown]
# Part 2.4: Group A=Female, Group B=Male from the documented combined sex/status
# field. It is a limited historical binary coding, not a complete gender measure.
# Compute the assignment's exact directional ratios; do not silently symmetrize.
# EO=min(TPR_A/TPR_B,FPR_A/FPR_B) is an assignment-specific summary and EO=1 alone
# does NOT establish equalized odds. Export both component ratios and raw rates.
# Undefined denominators yield NaN (NA in CSV); never replace 0/0 with 0 or 1.

# %%
def divide(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan
def group_rates(actual, decision):
    tn,fp,fn,tp = confusion_matrix(actual,decision,labels=[0,1]).ravel()
    return {'n':len(actual),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
            'selection':float(np.mean(decision)), 'TPR':divide(tp,tp+fn),
            'FPR':divide(fp,fp+tn), 'PPV':divide(tp,tp+fp), 'accuracy':float(np.mean(actual==decision))}
fair_rows, raw_rates = [], []
for t in np.round(np.arange(.05,.951,.1),2):
    decision = (prob>=t).astype(int)
    groups={}
    for g in ['Female','Male']:
        mask = test.sex==g
        groups[g]=group_rates(yte[mask],decision[mask])
        raw_rates.append({'threshold':t,'group':g,**groups[g]})
    a,b=groups['Female'],groups['Male']
    tr,fr=divide(a['TPR'],b['TPR']),divide(a['FPR'],b['FPR'])
    eo=min(tr,fr) if np.isfinite(tr) and np.isfinite(fr) else np.nan
    fair_rows.append({'threshold':t,'DP':divide(a['selection'],b['selection']),
                      'EO':eo,'PP':divide(a['PPV'],b['PPV']), 'TPR_ratio':tr,'FPR_ratio':fr,
                      'accuracy':accuracy_score(yte,decision),
                      'Female_selection':a['selection'],'Male_selection':b['selection'],
                      'Female_TPR':a['TPR'],'Male_TPR':b['TPR'],
                      'Female_FPR':a['FPR'],'Male_FPR':b['FPR']})
fairness=pd.DataFrame(fair_rows)
rates=pd.DataFrame(raw_rates)
fairness.to_csv(RESULTS/'fairness_thresholds.csv',index=False,na_rep='NA')
rates.to_csv(RESULTS/'fairness_group_counts_rates.csv',index=False,na_rep='NA')
print(fairness.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
fig,ax=plt.subplots(figsize=(8,6))
valid=fairness.dropna(subset=['DP','EO'])
ax.plot(valid.DP,valid.EO,'o-',color='#166b8f',linewidth=1,alpha=.8)
offsets = {.05:(15,20), .15:(25,-20), .25:(-65,-10), .35:(-65,18)}
for _,r in valid.iterrows():
    ax.annotate(f"{r.threshold:.2f}",(r.DP,r.EO),xytext=offsets.get(r.threshold,(6,6)),
                textcoords='offset points',fontsize=9,
                arrowprops={'arrowstyle':'-','color':'gray','linewidth':.6})
ax.axvline(1,color='gray',linestyle='--',linewidth=.8); ax.axhline(1,color='gray',linestyle='--',linewidth=.8)
ax.set(xlabel='Demographic parity: Female / Male selection rate',
       ylabel='Assignment EO: min(TPR ratio, FPR ratio)',xlim=(0,1.5),ylim=(0,1.5),
       title='German Credit | seed 44 | test n=300')
ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(RESULTS/'fairness_tradeoff.png',dpi=180); plt.close(fig)
# Also preserve unbounded values in a separate view to disclose clipping.
fig,ax=plt.subplots(figsize=(8,5))
for metric in ['DP','EO','PP']:
    ax.plot(fairness.threshold,fairness[metric],marker='o',label=metric)
ax.axhline(1,color='gray',linestyle='--',linewidth=.8)
ax.set(xlabel='Approval threshold',ylabel='Female / Male ratio',title='All thresholds; missing values are undefined')
ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(RESULTS/'fairness_by_threshold.png',dpi=180); plt.close(fig)

# %% [markdown]
# Quantify an observed trade-off without assuming that a larger ratio is always
# fairer: define parity proximity = min(r,1/r) for r>0 (1=parity). Find all pairs
# where DP proximity improves and EO proximity worsens; export them all. Also
# print the raw-ratio changes for a pair entirely below 1, if one exists, which
# matches the assignment's percentage-gain wording directly.
# Threshold selection on this test set is exploratory, not an unbiased estimate
# of a tuned policy. A future deployment needs separate validation and testing.

# %%
def proximity(r):
    return min(r,1/r) if np.isfinite(r) and r>0 else np.nan
tradeoffs=[]
for _,a in fairness.iterrows():
    for _,b in fairness.iterrows():
        da,db,ea,eb=map(proximity,[a.DP,b.DP,a.EO,b.EO])
        if np.isfinite([da,db,ea,eb]).all() and db>da+1e-8 and eb<ea-1e-8:
            gain=100*(db-da)/da; loss=100*(ea-eb)/ea
            tradeoffs.append({'from_threshold':a.threshold,'to_threshold':b.threshold,
                             'DP_from':a.DP,'DP_to':b.DP,'EO_from':a.EO,'EO_to':b.EO,
                             'DP_proximity_gain_pct':gain,'EO_proximity_loss_pct':loss,'cost':loss/gain})
trades=pd.DataFrame(tradeoffs)
trades.to_csv(RESULTS/'tradeoff_pairs.csv',index=False)
print('Trade-off pairs:\n',trades.to_string(index=False))
# Two real, anonymous records whose approvals change from t=.65 to t=.75.
scenario = comparison[(prob>=.65)&(prob<.75)].groupby('sex',sort=True).head(1).copy()
scenario['decision_at_0.65'] = (scenario.logit_probability>=.65).astype(int)
scenario['decision_at_0.75'] = (scenario.logit_probability>=.75).astype(int)
scenario.to_csv(RESULTS/'two_person_scenario.csv')
print('Two-person scenario:\n',scenario[['sex','age','approve','logit_probability','decision_at_0.65','decision_at_0.75']].to_string())
summary['subgroup_selected']=selected
summary['example_wrong_row']=int(example.name)
summary['example_wrong_probability']=float(example.logit_probability)
summary['duration_amount_correlation']=float(train.duration.corr(train.amount))
summary['versions']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                     'scipy':scipy.__version__,'statsmodels':sm.__version__,'sklearn':sklearn.__version__,
                     'matplotlib':matplotlib.__version__}
(RESULTS/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame({'train_index':pd.Series(train_idx),'test_index':pd.Series(test_idx)}).to_csv(RESULTS/'split_indices.csv',index=False)
print('Saved all results to:',RESULTS)
print('Versions:',summary['versions'])

# %% [markdown]
# Interpretation checklist for the written responses:
# 1. Distinguish model agreement from ground-truth accuracy.
# 2. Quote subgroup coefficients, intervals, adjusted p-value, and model scope.
# 3. Quote four before/after coefficients; do not infer causality from shifts.
# 4. Propose a prospective audit including applicants the bank previously rejected.
# 5. Separate observed error patterns from untested explanations of their cause.
# 6. Use actual test rows when giving the two-person threshold scenario.
# 7. Name a defensible threshold with at least three metrics and a counterargument.
# 8. State whose harms matter when favoring one fairness criterion.
# Reflection: use the evidence to justify a deployment decision and its limits.
# Do not claim that every dataset must show a paradox or a fairness conflict.
